from typing import Dict, List, Tuple
from dataclasses import field, dataclass

from kirin import ir
from bloqade.noise import native
from kirin.rewrite import abc as result_abc, result
from kirin.dialects import py, ilist
from bloqade.analysis import address
from bloqade.qasm2.dialects import uop, core, glob, parallel


@dataclass
class NoiseRewriteRule(result_abc.RewriteRule):
    address_analysis: Dict[ir.SSAValue, address.Address]
    qubit_ssa_value: Dict[int, ir.SSAValue]
    noise_model: native.NoiseModelABC = field(default_factory=native.TwoRowZoneModel)

    def __post_init__(self):
        for ssa, addr in self.address_analysis.items():
            if not isinstance(ssa, ir.ResultValue):
                continue
            # insert any missing qubit statements
            if isinstance(addr, address.AddressReg):
                node = ssa.stmt
                assert isinstance(node, core.QRegNew)
                for pos, idx in enumerate(addr.data):
                    if idx not in self.qubit_ssa_value:
                        pos_stmt = py.constant.Constant(value=pos)
                        self.qubit_ssa_value[idx] = (
                            qubit_stmt := core.QRegGet(node.result, pos_stmt.result)
                        ).result
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

    def move_noise_stmts(
        self,
        errors: Dict[Tuple[float, float, float, float], List[int]],
    ) -> list[ir.Statement]:

        nodes = []

        for probs, qubits in errors.items():
            if len(qubits) == 0:
                continue

            nodes.append(
                qargs := ilist.New(tuple(self.qubit_ssa_value[q] for q in qubits))
            )
            nodes.append(native.AtomLossChannel(qargs.result, prob=probs[3]))
            nodes.append(
                native.PauliChannel(qargs.result, px=probs[0], py=probs[1], pz=probs[2])
            )

        return nodes

    def cz_gate_noise(
        self,
        ctrls: ir.SSAValue,
        qargs: ir.SSAValue,
    ) -> list[ir.Statement]:
        return [
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
            ),
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
            ),
            native.AtomLossChannel(ctrls, prob=self.noise_model.cz_gate_loss_prob),
            native.AtomLossChannel(qargs, prob=self.noise_model.cz_gate_loss_prob),
        ]

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
        self.move_noise_stmts(node, errors, insert_before=True)
        self.move_noise_stmts(node, errors, insert_before=False)
        (ctrls := ilist.New([self.qubit_ssa_value[ctrl_addr.data]])).insert_after(node)
        (qargs := ilist.New([self.qubit_ssa_value[qarg_addr.data]])).insert_after(node)
        move_noise_nodes = self.move_noise_stmts(errors)
        gate_noise_nodes = self.cz_gate_noise(ctrls.result, qargs.result)

        for new_node in move_noise_nodes:
            new_node.insert_before(node)

        for new_node in gate_noise_nodes:
            new_node.insert_before(node)

        for new_node in reversed(move_noise_nodes):
            new_node.from_stmt(new_node).insert_after(node)

        return result.RewriteResult(has_done_something=True)

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
        errors = self.noise_model.parallel_cz_errors(ctrl_qubits, qarg_qubits, rest)

        move_noise_nodes = self.move_noise_stmts(errors)
        gate_noise_nodes = self.cz_gate_noise(node.ctrls, node.qargs)

        for new_node in move_noise_nodes:
            new_node.insert_before(node)

        for new_node in gate_noise_nodes:
            new_node.insert_before(node)

        for new_node in reversed(move_noise_nodes):
            new_node.from_stmt(new_node).insert_after(node)

        return result.RewriteResult(has_done_something=True)
