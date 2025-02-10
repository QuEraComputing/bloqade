from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import cse, dce, walk, result
from bloqade.analysis import address
from kirin.passes.abc import Pass

from ..rewrite.parallel_to_uop import ParallelToUOpRule

@dataclass
class ParallelToUOp(Pass):

    def generate_rule(self, mt: ir.Method) -> ParallelToUOpRule:
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

        return ParallelToUOpRule(id_map=id_map, address_analysis=results)

    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:
        rewriter = walk.Walk(self.generate_rule(mt))
        result = rewriter.rewrite(mt.code)

        result = walk.Walk(dce.DeadCodeElimination()).rewrite(mt.code)
        result = walk.Walk(cse.CommonSubexpressionElimination()).rewrite(mt.code)

        return result
