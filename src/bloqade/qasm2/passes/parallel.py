"""
Passes for converting parallel gates into multiple single gates as well as
converting multiple single gates to parallel gates.
"""

from typing import Type
from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import dce, walk, chain, result, fixpoint
from bloqade.analysis import address, schedule
from kirin.passes.abc import Pass
from bloqade.qasm2.rewrite import (
    MergePolicyABC,
    ParallelToUOpRule,
    UOpToParallelRule,
    SimpleGreedyMergePolicy,
)


@dataclass
class ParallelToUOp(Pass):
    """Pass to convert parallel gates into single gates.

    This pass rewrites any parallel gates from the `qasm2.parallel` dialect into multiple
    single gates in the `qasm2.uop` dialect, bringing the program closer to
    conforming to standard QASM2 syntax.

    ## Usage Examples
    ```
    # Define kernel
    @qasm2.extended
    def main():
        q = qasm2.qreg(4)

        qasm2.parallel.cz(ctrls=[q[0], q[2]], qargs=[q[1], q[3]])

    # Run rewrite
    ParallelToUOp(main.dialects)(main)
    ```

    The `qasm2.parallel.cz` statement has been rewritten to individual gates:

    ```
    qasm2.uop.cz(ctrl=q[0], qarg=q[1])
    qasm2.uop.cz(ctrl=q[2], qarg=q[3])
    ```

    """

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
    """Pass to convert single gates into parallel gates.

    This pass looks for single gates from the `qasm2.uop` dialect that can be combined
    into parallel gates from the `qasm2.parallel` dialect and performs a rewrite to do so.

    ## Usage Examples
    ```
    # Define kernel
    @qasm2.main
    def test():
        q = qasm2.qreg(4)

        theta = 0.1
        phi = 0.2
        lam = 0.3

        qasm2.u(q[1], theta, phi, lam)
        qasm2.u(q[3], theta, phi, lam)
        qasm2.cx(q[1], q[3])
        qasm2.u(q[2], theta, phi, lam)
        qasm2.u(q[0], theta, phi, lam)
        qasm2.cx(q[0], q[2])

    # Run rewrite
    UOpToParallel(main.dialects)(main)
    ```

    The individual `qasm2.u` statements have now been combined
    into a single `qasm2.parallel.u` statement.

    ```
    qasm2.parallel.u(qargs = [q[0], q[1], q[2], q[3]], theta, phi, lam)
    qasm2.uop.CX(q[1], q[3])
    qasm2.uop.CX(q[0], q[2])
    ```

    """

    merge_policy_type: Type[MergePolicyABC] = SimpleGreedyMergePolicy

    def generate_rule(self, mt: ir.Method):
        frame, _ = address.AddressAnalysis(mt.dialects).run_analysis(mt)
        dags = schedule.DagScheduleAnalysis(
            mt.dialects, address_analysis=frame.entries
        ).get_dags(mt)

        return UOpToParallelRule(
            {
                block: self.merge_policy_type.from_analysis(dag, frame.entries)
                for block, dag in dags.items()
            }
        )

    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:
        return fixpoint.Fixpoint(
            chain.Chain(
                walk.Walk(self.generate_rule(mt)), walk.Walk(dce.DeadCodeElimination())
            )
        ).rewrite(mt.code)
