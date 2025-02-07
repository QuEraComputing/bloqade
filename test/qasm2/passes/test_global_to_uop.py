from typing import Tuple
from dataclasses import dataclass

from kirin import ir
from bloqade import qasm2
from kirin.rewrite import abc, walk, result
from kirin.passes.abc import Pass
from bloqade.qasm2.dialects import uop, glob
from bloqade.analysis.address import AddressQubit, AddressAnalysis


class GlobalToUOpRule(abc.RewriteRule):

    def __init__(self, qubit_ssa_vals: Tuple[ir.SSAValue]) -> None:
        self.qubit_ssa_vals = qubit_ssa_vals

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if node.dialect == glob.dialect:
            getattr(self, f"rewrite_{node.name}")(node)
            return result.RewriteResult(has_done_something=True)

        return result.RewriteResult()

    def rewrite_ugate(self, node: ir.Statement) -> None:
        assert isinstance(node, glob.UGate)

        for qubit_ssa_val in self.qubit_ssa_vals:
            new_node = uop.UGate(
                qarg=qubit_ssa_val, theta=node.theta, phi=node.phi, lam=node.lam
            )
            new_node.insert_after(node)

        node.delete()


@dataclass
class GlobalToUOp(Pass):

    def extract_qubit_ssa_vals(self, mt: ir.Method) -> Tuple[ir.SSAValue]:

        results, result = AddressAnalysis(mt.dialects).run_analysis(mt)

        qubits = {}

        # GOAL: Get the ssa value for the first reference of each qubit.
        for ssa, result in results.items():
            if not isinstance(result, AddressQubit):
                # skip any stmts that are not qubits
                continue

            # get qubit id from analysis result
            qubit_id = result.data

            # check if id has already been found
            # if so, skip this ssa value
            if qubit_id in qubits:
                continue

            qubits[qubit_id] = ssa

        return qubits.values()

    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:

        qubit_ssa_vals = self.extract_qubit_ssa_vals(mt)
        rewriter = walk.Walk(GlobalToUOpRule(qubit_ssa_vals))

        return rewriter.rewrite(mt.code)


if __name__ == "__main__":

    @qasm2.main
    def main():
        qasm2.qreg(3)

        qasm2.glob.u(theta=1.0, phi=1.0, lam=1.0)

    main.print()

    # run through rewriter

    GlobalToUOp(main.dialects)(main)

    # Verify successful rewrite

    main.print()
