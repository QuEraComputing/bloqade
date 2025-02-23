import abc
import math
from typing import Dict, List, Tuple
from dataclasses import field, dataclass

from kirin import ir
from bloqade import qasm2
from bloqade.noise import native
from kirin.rewrite import abc as result_abc, walk, result
from kirin.dialects import py, ilist
from bloqade.analysis import address
from bloqade.qasm2.passes import UOpToParallel
from bloqade.qasm2.dialects import uop, core, glob, parallel


@dataclass
class NoiseModelABC(abc.ABC):
    # rate = prob/time

    move_px_rate: float = field(default=1e-6, kw_only=True)
    move_py_rate: float = field(default=1e-6, kw_only=True)
    move_pz_rate: float = field(default=1e-6, kw_only=True)
    move_loss_rate: float = field(default=1e-6, kw_only=True)

    pick_loss_prob: float = field(default=1e-4, kw_only=True)
    pick_px: float = field(default=1e-3, kw_only=True)
    pick_py: float = field(default=1e-3, kw_only=True)
    pick_pz: float = field(default=1e-3, kw_only=True)

    local_px: float = field(default=1e-3, kw_only=True)
    local_py: float = field(default=1e-3, kw_only=True)
    local_pz: float = field(default=1e-3, kw_only=True)
    local_loss_prob: float = field(default=1e-4, kw_only=True)

    global_px: float = field(default=1e-3, kw_only=True)
    global_py: float = field(default=1e-3, kw_only=True)
    global_pz: float = field(default=1e-3, kw_only=True)
    global_loss_prob: float = field(default=1e-3, kw_only=True)

    cz_paired_gate_px: float = field(default=1e-3, kw_only=True)
    cz_paired_gate_py: float = field(default=1e-3, kw_only=True)
    cz_paired_gate_pz: float = field(default=1e-3, kw_only=True)
    cz_gate_loss_prob: float = field(default=1e-3, kw_only=True)

    cz_unpaired_gate_px: float = field(default=1e-3, kw_only=True)
    cz_unpaired_gate_py: float = field(default=1e-3, kw_only=True)
    cz_unpaired_gate_pz: float = field(default=1e-3, kw_only=True)
    cz_ungate_loss_prob: float = field(default=1e-3, kw_only=True)

    move_speed: float = field(default=1e-2, kw_only=True)
    lattice_spacing: float = field(default=4.0, kw_only=True)

    @classmethod
    @abc.abstractmethod
    def parallel_cz_errors(
        cls, ctrls: List[int], qargs: List[int], rest: List[int]
    ) -> Dict[Tuple[float, float, float, float], List[int]]:
        """Takes a set of ctrls and qargs and returns a noise model for all qubits."""
        pass

    @staticmethod
    def poisson_pauli_prob(rate: float, duration: float) -> float:
        """Calculate the number of noise events and their probabilities for a given rate and duration."""
        return 0.5 * (1 - math.exp(-2 * rate * duration))

    @classmethod
    def join_binary_probs(cls, p1: float, *arg: float) -> float:
        if len(arg) == 0:
            return p1
        else:
            p2 = cls.join_binary_probs(*arg)
            return p1 * (1 - p2) + p2 * (1 - p1)


@dataclass
class FixLocationNoiseModel(NoiseModelABC):

    gate_zone_y_offset: float = 20.0
    gate_spacing: float = 20.0

    def deconflict(
        self, ctrls: List[int], qargs: List[int]
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:

        # sort by ctrl qubit first to guarantee that they will be in ascending order
        sorted_pairs = sorted(zip(ctrls, qargs), key=lambda x: x[0])

        groups = []
        # group by qarg only putting it in a group if the qarg is greater than the last qarg in the group
        # thus ensuring that the qargs are in ascending order
        while len(sorted_pairs) > 0:
            ctrl, qarg = sorted_pairs.pop(0)

            found = False
            for group in groups:
                if group[-1][1] < qarg:
                    group.append((ctrl, qarg))
                    found = True
                    break
            if not found:
                groups.append([(ctrl, qarg)])

        return [tuple(zip(*group)) for group in groups]

    def calculate_move_duration(self, ctrls: List[int], qargs: List[int]) -> float:
        """Calculate the time it takes to move the qubits from the ctrl to the qarg qubits."""

        position_pairs = list(zip(qargs, ctrls))
        # sort by the distance between the ctrl and qarg qubits
        sorted(position_pairs, key=lambda ele: abs(ele[0] - ele[1]))

        # greedy algorithm to find the best slot for each qubit pair
        slots = {}
        while position_pairs:
            ctrl, qarg = position_pairs.pop()

            mid = (ctrl + qarg) * self.lattice_spacing / 2
            slot = int(mid / self.gate_spacing)

            if slot not in slots:
                slots[slot] = (ctrl, qarg)
                continue

            # find the first slot that is not in slots that is close to the mid point
            for i in range(1, len(slots) + 1):
                if slot + i not in slots:
                    slots[slot + i] = (ctrl, qarg)
                    found = True
                    break
                elif slot - i not in slots:
                    slots[slot - i] = (ctrl, qarg)
                    found = True
                    break

            assert found, "No slot found"

        qarg_x_distance = float("-inf")
        ctrl_x_distance = float("-inf")

        for slot, (ctrl, qarg) in slots.items():
            qarg_x_distance = max(
                qarg_x_distance,
                abs(qarg * self.lattice_spacing - slot * self.gate_spacing),
            )
            ctrl_x_distance = max(
                ctrl_x_distance,
                abs(ctrl * self.lattice_spacing - slot * self.gate_spacing),
            )

        qarg_max_distance = math.sqrt(qarg_x_distance**2 + self.gate_zone_y_offset**2)
        ctrl_max_distance = math.sqrt(
            ctrl_x_distance**2 + (self.gate_zone_y_offset - 3) ** 2
        )

        return (qarg_max_distance + ctrl_max_distance) / self.move_speed

        return

    def parallel_cz_errors(
        self, ctrls: List[int], qargs: List[int], rest: List[int]
    ) -> Dict[Tuple[float, float, float, float], List[int]]:
        """Apply parallel gates by moving ctrl qubits to qarg qubits.

        Deconfict the ctrl moves by finding subsets in which both the
        ctrl and the qarg qubits are in ascending order.

        """
        groups = self.deconflict(ctrls, qargs)

        move_duration = sum(map(self.calculate_move_duration, *zip(*groups)))

        px_time = self.poisson_pauli_prob(self.move_px_rate, move_duration)
        py_time = self.poisson_pauli_prob(self.move_py_rate, move_duration)
        px_time = self.poisson_pauli_prob(self.move_pz_rate, move_duration)
        p_loss_time = self.poisson_pauli_prob(self.move_loss_rate, move_duration)

        errors = {(px_time, py_time, px_time, p_loss_time): rest}

        px_moved = self.join_binary_probs(self.pick_px, px_time)
        py_moved = self.join_binary_probs(self.pick_py, py_time)
        pz_moved = self.join_binary_probs(self.pick_pz, px_time)
        p_loss_moved = self.join_binary_probs(self.pick_loss_prob, p_loss_time)

        errors[(px_moved, py_moved, pz_moved, p_loss_moved)] = sorted(ctrls + qargs)

        return errors


@dataclass
class NoiseRewriteRule(result_abc.RewriteRule):
    address_analysis: Dict[ir.SSAValue, address.Address]
    qubit_ssa_value: Dict[int, ir.SSAValue]
    noise_model: NoiseModelABC = field(default_factory=FixLocationNoiseModel)

    def __post_init__(self):
        for ssa, addr in self.address_analysis.items():
            if not isinstance(ssa, ir.ResultValue):
                continue
            # insert all qubit statements
            if isinstance(addr, address.AddressReg):
                node = ssa.stmt
                assert isinstance(node, core.QRegNew)
                for pos, idx in enumerate(addr.data):
                    if idx not in self.qubit_ssa_value:
                        pos_stmt = py.constant.Constant(value=pos)
                        qubit_stmt = core.QRegGet(node.result, pos_stmt.result)
                        qubit_stmt.insert_after(node)
                        pos_stmt.insert_after(node)

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if isinstance(node, uop.SingleQubitGate):
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
        qargs: ir.SSAValue,
        probs: Tuple[float, float, float, float],
    ):
        native.PauliChannel(qargs, px=probs[0], py=probs[1], pz=probs[2]).insert_before(
            node
        )
        native.AtomLossChannel(qargs, prob=probs[3]).insert_before(node)

        return result.RewriteResult(has_done_something=True)

    def rewrite_single_qubit_gate(self, node: uop.RZ | uop.UGate):
        probs = (
            self.noise_model.local_px,
            self.noise_model.local_py,
            self.noise_model.local_pz,
            self.noise_model.local_loss_prob,
        )
        (qargs := ilist.New(values=(node.qarg,))).insert_before(node)
        return self.insert_single_qubit_noise(node, qargs.result, probs)

    def insert_move_noise_channels(
        self,
        node: ir.Statement,
        errors: Dict[Tuple[float, float, float, float], List[int]],
        insert_before: bool,
    ):

        nodes = []

        for probs, qubits in errors.items():
            nodes.append(
                qargs := ilist.New(values=[self.qubit_ssa_value[q] for q in qubits])
            )
            nodes.append(native.AtomLossChannel(qargs.result, prob=probs[3]))
            nodes.append(
                native.PauliChannel(qargs.result, px=probs[0], py=probs[1], pz=probs[2])
            )

        if insert_before:
            for n in nodes:
                n.insert_before(node)
        else:
            for n in reversed(nodes):
                n.insert_after(node)

    def insert_cz_gate_noise(
        self,
        node: ir.Statement,
        ctrls: ir.SSAValue,
        qargs: ir.SSAValue,
    ):
        native.CZPauliChannel(
            ctrls,
            qargs,
            px_ctrl=self.noise_model.cz_paired_gate_px,
            py_ctrl=self.noise_model.cz_paired_gate_py,
            pz_ctrl=self.noise_model.cz_paired_gate_pz,
            px_qarg=self.noise_model.cz_paired_gate_px,
            py_qarg=self.noise_model.cz_paired_gate_py,
            pz_qarg=self.noise_model.cz_paired_gate_pz,
            paired=True,
        ).insert_before(node)

        native.CZPauliChannel(
            ctrls,
            qargs,
            px_ctrl=self.noise_model.cz_unpaired_gate_px,
            py_ctrl=self.noise_model.cz_unpaired_gate_py,
            pz_ctrl=self.noise_model.cz_unpaired_gate_pz,
            px_qarg=self.noise_model.cz_unpaired_gate_px,
            py_qarg=self.noise_model.cz_unpaired_gate_py,
            pz_qarg=self.noise_model.cz_unpaired_gate_pz,
            paired=False,
        ).insert_before(node)

    def rewrite_cz_gate(self, node: uop.CZ):
        qarg_addr = self.address_analysis[node.qarg]
        ctrl_addr = self.address_analysis[node.ctrl]

        if not isinstance(qarg_addr, address.AddressQubit) or not isinstance(
            ctrl_addr, address.AddressQubit
        ):
            return result.RewriteResult()

        other_qubits = sorted(set(self.qubit_ssa_value.keys()) - {node.qarg, node.ctrl})
        errors = self.noise_model.parallel_cz_errors(
            [ctrl_addr.data], [qarg_addr.data], other_qubits
        )
        self.insert_move_noise_channels(node, errors, insert_before=True)
        (
            ctrls := ilist.New(values=[self.qubit_ssa_value[ctrl_addr.data]])
        ).insert_before(node)
        (
            qargs := ilist.New(values=[self.qubit_ssa_value[qarg_addr.data]])
        ).insert_before(node)
        self.insert_cz_gate_noise(
            node,
            ctrls.result,
            qargs.result,
        )
        self.insert_move_noise_channels(node, errors, insert_before=False)
        return result.RewriteResult(has_done_something=True)

    def rewrite_global_single_qubit_gate(self, node: glob.UGate):
        addrs = self.address_analysis[node.registers]
        if not isinstance(addrs, address.AddressTuple):
            return result.RewriteResult()

        qargs = []

        for addr in addrs.data:
            if not isinstance(addr, address.AddressReg):
                return result.RewriteResult()

            qargs.extend(self.qubit_ssa_value[q] for q in addr.data)

        probs = (
            self.noise_model.global_px,
            self.noise_model.global_py,
            self.noise_model.global_pz,
            self.noise_model.global_loss_prob,
        )
        (qargs := ilist.New(values=tuple(qargs))).insert_before(node)
        return self.insert_single_qubit_noise(node, qargs.result, probs)

    def rewrite_parallel_single_qubit_gate(self, node: parallel.RZ | parallel.UGate):
        addrs = self.address_analysis[node.qargs]
        if not isinstance(addrs, address.AddressTuple):
            return result.RewriteResult()

        if not all(isinstance(addr, address.AddressQubit) for addr in addrs.data):
            return result.RewriteResult()

        probs = (
            self.noise_model.local_px,
            self.noise_model.local_py,
            self.noise_model.local_pz,
            self.noise_model.local_loss_prob,
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
        rest = sorted(set(self.qubit_ssa_value.keys()) - set(ctrl_qubits + qarg_qubits))
        groups = self.noise_model.parallel_cz_errors(ctrl_qubits, qarg_qubits, rest)

        self.insert_move_noise_channels(node, groups, insert_before=True)
        self.insert_cz_gate_noise(node, node.ctrls, node.qargs)
        self.insert_move_noise_channels(node, groups, insert_before=False)
        return result.RewriteResult(has_done_something=True)


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


UOpToParallel(mt.dialects)(mt)
address_analysis = address.AddressAnalysis(mt.dialects)
frame, _ = address_analysis.run_analysis(mt)
print(address_analysis.qubit_ssa_value)
noise = walk.Walk(NoiseRewriteRule(frame.entries, address_analysis.qubit_ssa_value))
noise.rewrite(mt.code)


mt.print()
