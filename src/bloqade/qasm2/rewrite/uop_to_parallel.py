from typing import Dict, List, Tuple, Callable, Iterable
from dataclasses import field, dataclass

from kirin import ir
from kirin.rewrite import abc, result
from kirin.dialects import ilist
from bloqade.analysis import address
from kirin.analysis.const import lattice
from bloqade.qasm2.dialects import uop, core, parallel
from bloqade.analysis.schedule import StmtDag


def same_id_checker(ssa1: ir.SSAValue, ssa2: ir.SSAValue):
    if ssa1 is ssa2:
        return True
    elif (hint1 := ssa1.hints.get("const")) and (hint2 := ssa2.hints.get("const")):
        assert isinstance(hint1, lattice.Result) and isinstance(hint2, lattice.Result)
        return hint1.is_equal(hint2)
    else:
        return False


class MergeResults:
    def __init__(self, merge_groups, group_numbers):
        self.merge_groups = merge_groups
        self.group_numbers = group_numbers


@dataclass
class GreedyParallelGrouping:
    ssa_value_checker: Callable[[ir.SSAValue, ir.SSAValue], bool] = field(
        default=same_id_checker
    )

    def check_equiv_args(
        self,
        args1: Iterable[ir.SSAValue],
        args2: Iterable[ir.SSAValue],
    ):
        try:
            return all(
                self.ssa_value_checker(ssa1, ssa2)
                for ssa1, ssa2 in zip(args1, args2, strict=True)
            )
        except ValueError:
            return False

    def policy(self, stmt1: ir.Statement, stmt2: ir.Statement):

        match stmt1, stmt2:
            case (
                (uop.UGate(), uop.UGate())
                | (uop.RZ(), uop.RZ())
                | (parallel.UGate(), parallel.UGate())
                | (parallel.UGate(), uop.UGate())
                | (uop.UGate(), parallel.UGate())
                | (uop.UGate(), parallel.UGate())
                | (uop.UGate(), parallel.UGate())
                | (parallel.RZ(), parallel.RZ())
                | (uop.RZ(), parallel.RZ())
                | (parallel.RZ(), uop.RZ())
            ):

                return self.check_equiv_args(stmt1.args[1:], stmt2.args[1:])
            case (
                (parallel.CZ(), parallel.CZ())
                | (parallel.CZ(), uop.CZ())
                | (uop.CZ(), parallel.CZ())
                | (uop.CZ(), uop.CZ())
            ):
                return True

            case _:
                return False

    def merge_gates(self, gate_stmts: Iterable[ir.Statement]):
        groups = []

        for gate in gate_stmts:
            grouped = False
            for group in groups:

                if any(self.policy(gate, group_gate) for group_gate in group):
                    group.append(gate)
                    grouped = True
                    break

            if not grouped:
                groups.append([gate])

        return groups

    def topological_groups(self, dag: StmtDag):
        inc_edges = {k: set(v) for k, v in dag.inc_edges.items()}
        # worse case is a linear dag,
        # so we can use len(dag.stmts) as the limit
        for _ in range(len(dag.stmts)):
            if len(inc_edges) == 0:
                break
            # get edges with no dependencies
            group = [
                node_id for node_id, inc_edges in inc_edges.items() if not inc_edges
            ]
            # remove nodes in group from inc_edges
            for n in group:
                inc_edges.pop(n)
                for m in dag.out_edges[n]:
                    inc_edges[m].remove(n)

            yield group

        if inc_edges:
            raise ValueError("Cyclic dependency detected")

    def __call__(
        self,
        dag: StmtDag,
    ):
        merge_groups: List[List[ir.Statement]] = []
        group_numbers: Dict[ir.Statement, int] = {}

        for topological_group in self.topological_groups(dag):
            if len(topological_group) == 1:
                continue

            stmts = map(dag.stmts.__getitem__, topological_group)
            gate_groups = self.merge_gates(stmts)

            for group in gate_groups:
                if len(group) == 1:
                    continue

                for gate in group:
                    group_numbers[gate] = len(merge_groups)

                merge_groups.append(group)

        return MergeResults(merge_groups, group_numbers)


@dataclass
class UOpToParallelRule(abc.RewriteRule):
    address_analysis: Dict[ir.SSAValue, address.Address]
    dags: Dict[ir.Block, StmtDag]
    grouping_results: Dict[ir.Block, MergeResults] = field(
        init=False, default_factory=dict
    )
    parallel_grouping: Callable[[StmtDag], MergeResults] = field(init=False)

    def __post_init__(self):
        self.parallel_grouping = GreedyParallelGrouping()

    def get_merge_results(self, block: ir.Block | None) -> MergeResults | None:
        if block is None or block not in self.dags:
            return None

        if block not in self.grouping_results and block in self.dags:
            self.grouping_results[block] = self.parallel_grouping(self.dags[block])

        return self.grouping_results[block]

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        merge_results = self.get_merge_results(node.parent_block)

        if merge_results is None:
            return result.RewriteResult()

        if node not in merge_results.group_numbers:
            return result.RewriteResult()

        group_number = merge_results.group_numbers[node]
        group = merge_results.merge_groups[group_number]
        if node is group[0]:
            method = getattr(self, f"rewrite_group_{node.name}")
            method(node, group)

        node.delete()

        return result.RewriteResult(has_done_something=True)

    def merge_and_move_qubits(
        self, node: ir.Statement, qargs: List[ir.SSAValue]
    ) -> Tuple[ir.SSAValue, ...]:

        qubits = []

        for qarg in qargs:
            addr = self.address_analysis[qarg]

            if isinstance(addr, address.AddressQubit):
                qubits.append(qarg)

            elif isinstance(addr, address.AddressTuple):
                assert isinstance(qarg, ir.ResultValue)
                assert isinstance(qarg.stmt, ilist.New)
                qubits.extend(qarg.stmt.values)

        for qarg in qubits:
            if (
                isinstance(qarg, ir.ResultValue)
                and isinstance(qarg.owner, core.QRegGet)
                and isinstance(qarg.owner.idx, ir.ResultValue)
            ):
                idx = qarg.owner.idx
                idx.owner.delete(safe=False)
                idx.owner.insert_before(node)
                qarg.owner.delete(safe=False)
                qarg.owner.insert_before(node)

        return tuple(qubits)

    def rewrite_group_cz(self, node: ir.Statement, group: List[ir.Statement]):
        ctrls = []
        qargs = []

        for stmt in group:
            if isinstance(stmt, uop.CZ):
                ctrls.append(stmt.ctrl)
                qargs.append(stmt.qarg)
            elif isinstance(stmt, parallel.CZ):
                ctrls.append(stmt.ctrls)
                qargs.append(stmt.qargs)
            else:
                raise RuntimeError(f"Unexpected statement {stmt}")

        ctrls_values = self.merge_and_move_qubits(node, ctrls)
        qargs_values = self.merge_and_move_qubits(node, qargs)

        new_ctrls = ilist.New(values=ctrls_values)
        new_qargs = ilist.New(values=qargs_values)
        new_gate = parallel.CZ(ctrls=new_ctrls.result, qargs=new_qargs.result)

        new_ctrls.insert_before(node)
        new_qargs.insert_before(node)
        new_gate.insert_before(node)

    def rewrite_group_U(self, node: ir.Statement, group: List[ir.Statement]):
        self.rewrite_group_u(node, group)

    def rewrite_group_u(self, node: ir.Statement, group: List[ir.Statement]):
        qargs = []

        for stmt in group:
            if isinstance(stmt, uop.UGate):
                qargs.append(stmt.qarg)
            elif isinstance(stmt, parallel.UGate):
                qargs.append(stmt.qargs)
            else:
                raise RuntimeError(f"Unexpected statement {stmt}")

        assert isinstance(node, (uop.UGate, parallel.UGate))

        qargs_values = self.merge_and_move_qubits(node, qargs)

        new_qargs = ilist.New(values=qargs_values)
        new_gate = parallel.UGate(
            qargs=new_qargs.result,
            theta=node.theta,
            phi=node.phi,
            lam=node.lam,
        )
        new_qargs.insert_before(node)
        new_gate.insert_before(node)

    def rewrite_group_rz(self, node: ir.Statement, group: List[ir.Statement]):
        qargs = []

        for stmt in group:
            if isinstance(stmt, uop.RZ):
                qargs.append(stmt.qarg)
            elif isinstance(stmt, parallel.RZ):
                qargs.append(stmt.qargs)
            else:
                raise RuntimeError(f"Unexpected statement {stmt}")

        assert isinstance(node, (uop.RZ, parallel.RZ))

        qargs_values = self.merge_and_move_qubits(node, qargs)
        new_qargs = ilist.New(values=qargs_values)
        new_gate = parallel.RZ(
            qargs=new_qargs.result,
            theta=node.theta,
        )
        new_qargs.insert_before(node)
        new_gate.insert_before(node)
