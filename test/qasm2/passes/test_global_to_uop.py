# from typing import Dict
# from dataclasses import dataclass

# from kirin import ir
from bloqade import qasm2

# from kirin.rewrite import abc, cse, dce, walk, result
from bloqade.analysis import address

# from kirin.passes.abc import Pass
# from bloqade.qasm2.dialects import uop, glob

"""
@dataclass
class GlobalToUOpRule(abc.RewriteRule):
    id_map: Dict[int, ir.SSAValue]
    address_analysis: Dict[ir.SSAValue, address.Address]

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if node.dialect == glob.dialect:
            return getattr(self, f"rewrite_{node.name}")(node)

    def rewrite_u(self, node: ir.Statement):
        assert isinstance(node, glob.UGate)
        # To-do, need to feed in qubit ssa values and just repeatedly spit out standard ugates


class GlobalToUOP(Pass):

    id_map: Dict[int, ir.SSAValue]
    address_analysis: Dict[ir.SSAValue, address.Address]

    def generate_rule(self, mt: ir.Method) -> GlobalToUOpRule:
        results, _ = address.AddressAnalysis(mt.dialects).run_analysis(mt)

        id_map = {}

        # GOAL: Get the ssa value for the first reference of each qubit.
        for ssa, addr in results.items():
            if not isinstance(addr, address.AddressQubit):
                # skip any stmts that are not qubits
                continue

            # get qubit id from analysis result
            qubit_id = addr.data

            # check if id has already been found
            # if so, skip this ssa value
            if qubit_id in id_map:
                continue

            id_map[qubit_id] = ssa

        return GlobalToUOpRule(id_map=id_map, address_analysis=results)

    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:
        rewriter = walk.Walk(self.generate_rule(mt))
        result = rewriter.rewrite(mt.code)

        result = walk.Walk(dce.DeadCodeElimination()).rewrite(mt.code)
        result = walk.Walk(cse.CommonSubexpressionElimination()).rewrite(mt.code)

        return result
"""


if __name__ == "__main__":

    @qasm2.extended
    def main():
        q1 = qasm2.qreg(5)
        q2 = qasm2.qreg(10)

        qasm2.glob.u(theta=1.0, phi=1.0, lam=1.0, registers=[q1, q2])

        return q1

    results, stuff = address.AddressAnalysis(main.dialects).run_analysis(main)

    main.print(analysis=results)

    print(stuff)
