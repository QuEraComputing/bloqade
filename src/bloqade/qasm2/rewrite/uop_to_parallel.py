import abc
from typing import Dict, List, Tuple, Callable, Iterable
from dataclasses import field, dataclass

from kirin import ir
from kirin.rewrite import abc as rewrite_abc, result
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


@dataclass
class MergeRewriterABC(abc.ABC):
    address_analysis: Dict[ir.SSAValue, address.Address]
    merge_groups: List[List[ir.Statement]]
    group_numbers: Dict[ir.Statement, int]

    @abc.abstractmethod
    def apply(self, node: ir.Statement) -> result.RewriteResult:
        pass


@dataclass
class SimpleMergeRewriter(MergeRewriterABC):
    def apply(self, node: ir.Statement) -> result.RewriteResult:
        if node not in self.group_numbers:
            return result.RewriteResult()

        group = self.merge_groups[self.group_numbers[node]]
        if node is group[0]:
            method = getattr(self, f"rewrite_group_{node.name}")
            method(node, group)

        node.delete()

        return result.RewriteResult(has_done_something=True)

    def move_and_collect_qubit_list(
        self, node: ir.Statement, qargs: List[ir.SSAValue]
    ) -> Tuple[ir.SSAValue, ...]:

        qubits = []
        # collect references to qubits
        for qarg in qargs:
            addr = self.address_analysis[qarg]

            if isinstance(addr, address.AddressQubit):
                qubits.append(qarg)

            elif isinstance(addr, address.AddressTuple):
                assert isinstance(qarg, ir.ResultValue)
                assert isinstance(qarg.stmt, ilist.New)
                qubits.extend(qarg.stmt.values)

        # for qubits coming from QRegGet, move both the get statement
        # and the index statement before the current node, we do not need
        # to move the register because the current node has some dependency
        # on it.
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

        ctrls_values = self.move_and_collect_qubit_list(node, ctrls)
        qargs_values = self.move_and_collect_qubit_list(node, qargs)

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

        qargs_values = self.move_and_collect_qubit_list(node, qargs)

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

        qargs_values = self.move_and_collect_qubit_list(node, qargs)
        new_qargs = ilist.New(values=qargs_values)
        new_gate = parallel.RZ(
            qargs=new_qargs.result,
            theta=node.theta,
        )
        new_qargs.insert_before(node)
        new_gate.insert_before(node)


@dataclass
class GreedyMergePolicy:
    ssa_value_checker: Callable[[ir.SSAValue, ir.SSAValue], bool]

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

    def can_merge(self, stmt1: ir.Statement, stmt2: ir.Statement) -> bool:
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

    def __call__(self, gate_stmts: Iterable[ir.Statement]) -> List[List[ir.Statement]]:
        """Group gates that can be merged together based on the merge_gate_policy

        Args:
            gate_stmts (Generator[ir.Statement]): The gates to try and group

        Returns:
            List[List[ir.Statement]]: A list of groups of gates that can be merged together

        """
        iterable = iter(gate_stmts)
        groups = [[next(iterable)]]
        for gate in iterable:
            if self.can_merge(gate, groups[-1][-1]):
                groups[-1].append(gate)
            else:
                groups.append([gate])

        return groups


@dataclass
class Parallelize:
    merge_gate_policy: Callable[[Iterable[ir.Statement]], List[List[ir.Statement]]] = (
        field(default_factory=lambda: GreedyMergePolicy(same_id_checker))
    )
    merge_rewriter_type: type[MergeRewriterABC] = field(default=SimpleMergeRewriter)

    def topological_groups(self, dag: StmtDag):
        """Split the dag into topological groups where each group
        contains nodes that have no dependencies on each other, but
        have dependencies on nodes in one or more previous groups.

        Args:
            dag (StmtDag): The dag to split into groups


        Yields:
            List[str]: A list of node ids in a topological group


        Raises:
            ValueError: If a cyclic dependency is detected


        The idea is to yield all nodes with no dependencies, then remove
        those nodes from the graph repeating until no nodes are left
        or we reach some upper limit. Worse case is a linear dag,
        so we can use len(dag.stmts) as the upper limit

        If we reach the limit and there are still nodes left, then we
        have a cyclic dependency.
        """

        inc_edges = {k: set(v) for k, v in dag.inc_edges.items()}

        for _ in range(len(dag.stmts)):
            if len(inc_edges) == 0:
                break
            # get nodes with no dependencies
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
        address_analysis: Dict[ir.SSAValue, address.Address],
    ):
        merge_groups: List[List[ir.Statement]] = []
        group_numbers: Dict[ir.Statement, int] = {}

        for topological_group in self.topological_groups(dag):
            if len(topological_group) == 1:
                continue

            stmts = map(dag.stmts.__getitem__, topological_group)
            gate_groups = self.merge_gate_policy(stmts)

            for group in gate_groups:
                if len(group) == 1:
                    continue

                for gate in group:
                    group_numbers[gate] = len(merge_groups)

                merge_groups.append(group)

        return self.merge_rewriter_type(address_analysis, merge_groups, group_numbers)


@dataclass
class UOpToParallelRule(rewrite_abc.RewriteRule):
    address_analysis: Dict[ir.SSAValue, address.Address]
    dags: Dict[ir.Block, StmtDag]
    grouping_results: Dict[ir.Block, MergeRewriterABC] = field(
        init=False, default_factory=dict
    )
    parallel_policy: Callable[
        [StmtDag, Dict[ir.SSAValue, address.Address]], MergeRewriterABC
    ] = field(init=False)

    def __post_init__(self):
        self.parallel_policy = Parallelize()

    def get_merge_rewriter(self, block: ir.Block | None) -> MergeRewriterABC | None:
        if block is None or block not in self.dags:
            return None

        if block in self.grouping_results:
            return self.grouping_results[block]

        self.grouping_results[block] = self.parallel_policy(
            self.dags[block], self.address_analysis
        )
        return self.grouping_results[block]

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        merge_rewriter = self.get_merge_rewriter(node.parent_block)

        if merge_rewriter is None:
            return result.RewriteResult()

        return merge_rewriter.apply(node)
