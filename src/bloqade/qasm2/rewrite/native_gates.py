from functools import cached_property
from dataclasses import field, dataclass

import cirq
import cirq.transformers
import cirq.contrib.qasm_import
import cirq.transformers.target_gatesets
import cirq.transformers.target_gatesets.compilation_target_gateset
from kirin import ir
from kirin.rewrite import abc, result
from bloqade.qasm2.dialects import uop, expr
from cirq.transformers.target_gatesets.compilation_target_gateset import (
    CompilationTargetGateset,
)


# rydeberg gate sets
class RydbergTargetGateset(cirq.CZTargetGateset):
    def __init__(self, *, cnz_max_size: int = 2, atol: float = 1e-8):
        additional = [cirq.Z.controlled(cn) for cn in range(2, cnz_max_size)]
        super().__init__(atol=atol, additional_gates=additional)
        self.cnz_max_size = cnz_max_size

    @property
    def num_qubits(self) -> int:
        return max(2, self.cnz_max_size)


def one_qubit_gate_to_u3_angles(op: cirq.Operation) -> tuple[float, float, float]:
    mat = cirq.unitary(op)
    phi0, phi1, phi2 = (  # Z angle, Y angle, then Z angle
        cirq.deconstruct_single_qubit_matrix_into_angles(mat)
    )
    return phi0, phi1, phi2


@dataclass
class RydbergGateSetRewriteRule(abc.RewriteRule):
    # NOTE, this can only rewrite qasm2.main and qasm2.gate!
    gateset: CompilationTargetGateset = field(default_factory=RydbergTargetGateset)

    @cached_property
    def cached_qubits(self) -> tuple[cirq.LineQubit, ...]:

        # qasm2 stmts only have up to 3 qubits gates, so we cached only 3.
        return tuple(cirq.LineQubit(i) for i in range(3))

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:

        # only deal with uop
        if node.dialect == uop.dialect:
            return getattr(self, f"rewrite_{node.name}")(node)

        return result.RewriteResult()

    def rewrite_CZ(self, node: uop.CZ) -> result.RewriteResult:
        return result.RewriteResult()

    def rewrite_CX(self, node: uop.CX) -> result.RewriteResult:
        return self._rewrite_2q_ctrl_gates(cirq.CX, node)

    def _rewrite_2q_ctrl_gates(
        self, cirq_gate: type[cirq.Operation], node: uop.TwoQubitCtrlGate
    ) -> result.RewriteResult:

        target_gates = self.gateset.decompose_to_target_gateset(
            cirq_gate(self.cached_qubits[0], self.cached_qubits[1]), 0
        )

        qubits_ssa = [node.ctrl, node.qarg]
        new_stmts = []
        for new_gate in target_gates:
            if len(new_gate.qubits) == 1:
                # 1q
                phi0, phi1, phi2 = one_qubit_gate_to_u3_angles(new_gate)
                phi0_stmt = expr.ConstFloat(value=phi0)
                phi1_stmt = expr.ConstFloat(value=phi1)
                phi2_stmt = expr.ConstFloat(value=phi2)

                new_stmts.append(phi0_stmt)
                new_stmts.append(phi1_stmt)
                new_stmts.append(phi2_stmt)
                new_stmts.append(
                    uop.UGate(
                        qarg=qubits_ssa[new_gate.qubits[0].x],
                        theta=phi0_stmt.result,
                        phi=phi1_stmt.result,
                        lam=phi2_stmt.result,
                    )
                )
            else:
                # 2q
                new_stmts.append(uop.CZ(ctrl=node.ctrl, qarg=node.qarg))

        # add new stmt:
        node.replace_by(new_stmts[0])
        node = new_stmts[0]

        if len(new_stmts) > 1:
            for stmt in new_stmts[1:]:
                stmt.insert_after(node)
                node = stmt

        return result.RewriteResult(has_done_something=True)
