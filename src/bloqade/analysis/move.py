import abc
import math
from typing import Dict, List, Tuple, Sequence
from dataclasses import dataclass

from kirin import ir
from bloqade import qasm2
from bloqade.noise import native
from kirin.rewrite import abc as result_abc, result
from kirin.dialects import py, ilist
from bloqade.analysis import address
from bloqade.qasm2.dialects import uop, core, glob, parallel


@qasm2.extended
def mt():
    q = qasm2.qreg(4)

    qasm2.h(q[0])
    qasm2.h(q[1])
    qasm2.h(q[2])
    qasm2.h(q[3])
    qasm2.parallel.cz(ctrls=[q[0], q[2]], qargs=[q[1], q[3]])
    qasm2.h(q[0])
    qasm2.h(q[1])
    qasm2.h(q[2])
    qasm2.h(q[3])

    return q


frame, _ = address.AddressAnalysis(mt.dialects).run_analysis(mt)


@dataclass(frozen=True)
class PauliNoiseParams:
    move_px_rate: float  # rate = prob/distance
    move_py_rate: float
    move_pz_rate: float
    move_loss_rate: float

    pick_loss_prob: float
    pick_px: float
    pick_py: float
    pick_pz: float

    local_px: float
    local_py: float
    local_pz: float
    local_loss_prob: float

    global_px: float
    global_py: float
    global_pz: float
    global_loss_prob: float

    cz_paired_gate_px: float
    cz_paired_gate_py: float
    cz_paired_gate_pz: float
    cz_gate_loss_prob: float

    cz_unpaired_gate_px: float
    cz_unpaired_gate_py: float
    cz_unpaired_gate_pz: float
    cz_ungate_loss_prob: float

    move_speed: float
    lattice_spacing: float


@dataclass
class NoiseRewriteRuleABC(result_abc.RewriteRule, abc.ABC):
    address_analysis: Dict[ir.SSAValue, address.Address]
    qubit_ssa_value: Dict[int, ir.SSAValue]
    noise_params: PauliNoiseParams

    @staticmethod
    def poisson_pauli_prob(rate: float, duration: float) -> float:
        """Calculate the number of noise events and their probabilities for a given rate and duration."""
        return 0.5 * (1 - math.exp(-2 * rate * duration))

    @classmethod
    def join_pauli_probs(cls, p1: float, *arg: float) -> float:
        if len(arg) == 0:
            return p1
        else:
            p2 = cls.join_pauli_probs(*arg)
            return p1 * (1 - p2) + p2 * (1 - p1)

    @classmethod
    @abc.abstractmethod
    def move_deconflictor(
        cls, ctrls: List[int], qargs: List[int]
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Takes a set of ctrls and qargs and returns a dictionary of (ctrl, qarg) -> (ctrl_distance, qarg_distance)"""
        pass

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if isinstance(node, (uop.RZ, uop.UGate)):
            return self.rewrite_single_qubit_gate(node)
        elif isinstance(node, uop.CZ):
            return self.rewrite_cz_gate(node)
        elif isinstance(node, (parallel.UGate, parallel.RZ)):
            return self.rewrite_parallel_single_qubit_gate(node)
        elif isinstance(node, parallel.CZ):
            return self.rewrite_parallel_cz_gate(node)
        elif isinstance(node, glob.UGate):
            return self.rewrite_global_single_qubit_gate(node)
        else:
            return result.RewriteResult()

    def insert_single_qubit_noise(
        self,
        node: ir.Statement,
        qargs: Sequence[ir.SSAValue],
        probs: Tuple[float, float, float, float],
    ):
        const_px = py.constant.Constant(value=probs[0])
        const_py = py.constant.Constant(value=probs[1])
        const_pz = py.constant.Constant(value=probs[2])

        local_loss = py.constant.Constant(value=probs[3])
        const_px.insert_before(node)
        const_py.insert_before(node)
        const_pz.insert_before(node)
        local_loss.insert_before(node)

        for qarg in qargs:
            noise_node = native.PauliChannel(
                const_px.result, const_py.result, const_pz.result, qarg
            )
            loss_node = native.AtomLossChannel(local_loss.result, qarg)

            noise_node.insert_before(node)
            loss_node.insert_before(node)

        return result.RewriteResult(has_done_something=True)

    def rewrite_single_qubit_gate(self, node: uop.RZ | uop.UGate):
        probs = (
            self.noise_params.local_px,
            self.noise_params.local_py,
            self.noise_params.local_pz,
            self.noise_params.local_loss_prob,
        )
        return self.insert_single_qubit_noise(node, [node.qarg], probs)

    def insert_move_noise_channels(
        self,
        node: ir.Statement,
        qarg: ir.SSAValue,
        distance: float,
        insert_before: bool,
    ):
        duration = distance / self.noise_params.move_speed

        def move_noise(rate: float, distance: float, pick_prob: float):
            return self.join_pauli_probs(
                pick_prob,
                self.poisson_pauli_prob(rate, distance),
            )

        px_value = move_noise(
            self.noise_params.move_px_rate, duration, self.noise_params.pick_px
        )

        py_value = move_noise(
            self.noise_params.move_py_rate, duration, self.noise_params.pick_py
        )

        pz_value = move_noise(
            self.noise_params.move_pz_rate, duration, self.noise_params.pick_pz
        )

        p_loss_value = move_noise(
            self.noise_params.move_loss_rate, duration, self.noise_params.pick_loss_prob
        )

        px_node = py.constant.Constant(value=px_value)
        py_node = py.constant.Constant(value=py_value)
        pz_node = py.constant.Constant(value=pz_value)
        p_loss_node = py.constant.Constant(value=p_loss_value)

        if insert_before:
            p_loss_node.insert_before(node)
            native.AtomLossChannel(p_loss_node.result, qarg).insert_before(node)

            px_node.insert_before(node)
            py_node.insert_before(node)
            pz_node.insert_before(node)
            native.PauliChannel(
                px_node.result, py_node.result, pz_node.result, qarg
            ).insert_before(node)
        else:
            native.AtomLossChannel(p_loss_node.result, qarg).insert_after(node)
            p_loss_node.insert_after(node)

            native.PauliChannel(
                px_node.result, py_node.result, pz_node.result, qarg
            ).insert_after(node)
            px_node.insert_after(node)
            py_node.insert_after(node)
            pz_node.insert_after(node)

    def insert_cz_gate_noise(
        self,
        node: ir.Statement,
        qargs: Sequence[ir.SSAValue],
        ctrls: Sequence[ir.SSAValue],
        ctrl_distances: Sequence[float],
        qarg_distances: Sequence[float],
    ):

        for qarg, qarg_distance in zip(qargs, qarg_distances):
            self.insert_move_noise_channels(
                node, qarg, qarg_distance, insert_before=True
            )
        for ctrl, ctrl_distance in zip(ctrls, ctrl_distances):
            self.insert_move_noise_channels(
                node, ctrl, ctrl_distance, insert_before=True
            )

        (
            paired_px := py.constant.Constant(value=self.noise_params.cz_paired_gate_px)
        ).insert_before(node)
        (
            paired_py := py.constant.Constant(value=self.noise_params.cz_paired_gate_py)
        ).insert_before(node)
        (
            paired_pz := py.constant.Constant(value=self.noise_params.cz_paired_gate_pz)
        ).insert_before(node)

        for ctrl, qarg in zip(ctrls, qargs):
            native.CZPauliChannel(
                paired_px.result,
                paired_py.result,
                paired_pz.result,
                paired_px.result,
                paired_py.result,
                paired_pz.result,
                ctrl,
                qarg,
                paired=True,
            ).insert_before(node)

        (
            unpaired_px := py.constant.Constant(
                value=self.noise_params.cz_unpaired_gate_px
            )
        ).insert_before(node)
        (
            unpaired_py := py.constant.Constant(
                value=self.noise_params.cz_unpaired_gate_py
            )
        ).insert_before(node)
        (
            unpaired_pz := py.constant.Constant(
                value=self.noise_params.cz_unpaired_gate_pz
            )
        ).insert_before(node)
        for ctrl, qarg in zip(ctrls, qargs):
            native.CZPauliChannel(
                unpaired_px.result,
                unpaired_py.result,
                unpaired_pz.result,
                unpaired_px.result,
                unpaired_py.result,
                unpaired_pz.result,
                ctrl,
                qarg,
                paired=False,
            ).insert_before(node)

        (
            loss_prob := py.constant.Constant(value=self.noise_params.cz_gate_loss_prob)
        ).insert_before(node)

        for ctrl, qarg in zip(ctrls, qargs):
            native.AtomLossChannel(loss_prob.result, ctrl).insert_before(node)
            native.AtomLossChannel(loss_prob.result, qarg).insert_before(node)

        for qarg, qarg_distance in zip(qargs, qarg_distances):
            self.insert_move_noise_channels(
                node, qarg, qarg_distance, insert_before=False
            )

        for ctrl, ctrl_distance in zip(ctrls, ctrl_distances):
            self.insert_move_noise_channels(
                node, ctrl, ctrl_distance, insert_before=False
            )

    def rewrite_cz_gate(self, node: uop.CZ):
        qarg_addr = self.address_analysis[node.qarg]
        ctrl_addr = self.address_analysis[node.ctrl]

        if not isinstance(qarg_addr, address.AddressQubit) or not isinstance(
            ctrl_addr, address.AddressQubit
        ):
            return result.RewriteResult()

        distance = (
            abs(qarg_addr.data - ctrl_addr.data)
            * self.noise_params.lattice_spacing
            / 2.0
        )
        self.insert_cz_gate_noise(
            node, [node.qarg], [node.ctrl], [distance], [distance]
        )
        return result.RewriteResult(has_done_something=True)

    def rewrite_global_single_qubit_gate(self, node: glob.UGate):
        addrs = self.address_analysis[node.registers]
        if not isinstance(addrs, address.AddressTuple):
            return result.RewriteResult()

        if not all(isinstance(addr, address.AddressReg) for addr in addrs.data):
            return result.RewriteResult()

        ssa_values = []

        for addr in addrs.data:
            assert isinstance(addr, address.AddressReg)
            assert isinstance(node.registers, ir.ResultValue)
            assert isinstance(node.registers.stmt, ilist.New)
            for pos, idx in enumerate(addr.data):
                # insert new `get` instruction
                if idx not in self.qubit_ssa_value:
                    reg_ssa_value = node.registers.stmt.values[pos]
                    node.insert_before(index := py.constant.Constant(value=pos))
                    node.insert_before(
                        (new_qubit := core.QRegGet(reg_ssa_value, index.result))
                    )
                    self.qubit_ssa_value[idx] = new_qubit.result

                ssa_values.append(self.qubit_ssa_value[idx])

        probs = (
            self.noise_params.global_px,
            self.noise_params.global_py,
            self.noise_params.global_pz,
            self.noise_params.global_loss_prob,
        )
        return self.insert_single_qubit_noise(node, ssa_values, probs)

    def rewrite_parallel_single_qubit_gate(self, node: parallel.RZ | parallel.UGate):
        addrs = self.address_analysis[node.qargs]
        if not isinstance(addrs, address.AddressTuple):
            return result.RewriteResult()

        if not all(isinstance(addr, address.AddressQubit) for addr in addrs.data):
            return result.RewriteResult()

        probs = (
            self.noise_params.local_px,
            self.noise_params.local_py,
            self.noise_params.local_pz,
            self.noise_params.local_loss_prob,
        )
        assert isinstance(node.qargs, ir.ResultValue)
        assert isinstance(node.qargs.stmt, ilist.New)
        return self.insert_single_qubit_noise(node, node.qargs.stmt.values, probs)

    def rewrite_parallel_cz_gate(self, node: parallel.CZ):
        ctrls = self.address_analysis[node.ctrls]
        qargs = self.address_analysis[node.qargs]

        if not isinstance(ctrls, address.AddressTuple) or not isinstance(
            qargs, address.AddressTuple
        ):
            return result.RewriteResult()

        if not all(
            isinstance(addr, address.AddressQubit) for addr in ctrls.data
        ) or not all(isinstance(addr, address.AddressQubit) for addr in qargs.data):
            return result.RewriteResult()

        ctrl_qubits = list(map(lambda addr: addr.data, ctrls.data))
        qarg_qubits = list(map(lambda addr: addr.data, qargs.data))

        groups = self.move_deconflictor(ctrl_qubits, qarg_qubits)

        ctrl_distances = []
        qarg_distances = []
        ctrls = []
        qargs = []
        for ctrl, qarg in zip(ctrl_qubits, qarg_qubits):
            ctrl_distance, qarg_distance = groups[(ctrl, qarg)]
            ctrl_distances.append(ctrl_distance)
            qarg_distances.append(qarg_distance)
            ctrls.append(self.qubit_ssa_value[ctrl])
            qargs.append(self.qubit_ssa_value[qarg])

        self.insert_cz_gate_noise(node, qargs, ctrls, ctrl_distances, qarg_distances)
        return result.RewriteResult(has_done_something=True)
