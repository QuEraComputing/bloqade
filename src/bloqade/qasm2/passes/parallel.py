from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import dce, walk, chain, result, fixpoint
from bloqade.analysis import address, schedule
from kirin.passes.abc import Pass
from bloqade.qasm2.rewrite import ParallelToUOpRule, UOpToParallelRule


@dataclass
class ParallelToUOp(Pass):

    def generate_rule(self, mt: ir.Method) -> ParallelToUOpRule:
        frame, _ = address.AddressAnalysis(mt.dialects).run_analysis(mt)

        id_map = {}

        # GOAL: Get the ssa value for the first reference of each qubit.
        for ssa, addr in frame.entries.items():
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

        return ParallelToUOpRule(id_map=id_map, address_analysis=frame.entries)

    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:
        return fixpoint.Fixpoint(
            chain.Chain(
                walk.Walk(self.generate_rule(mt)), walk.Walk(dce.DeadCodeElimination())
            )
        ).rewrite(mt.code)


@dataclass
class UOpToParallel(Pass):

    def generate_rule(self, mt: ir.Method):
        frame, _ = address.AddressAnalysis(mt.dialects).run_analysis(mt)
        dags = schedule.DagScheduleAnalysis(
            mt.dialects, address_analysis=frame.entries
        ).get_dags(mt)
        return UOpToParallelRule(dags=dags, address_analysis=frame.entries)

    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:
        return fixpoint.Fixpoint(
            chain.Chain(
                walk.Walk(self.generate_rule(mt)), walk.Walk(dce.DeadCodeElimination())
            )
        ).rewrite(mt.code)
