from typing import Any, Set, Dict, Iterable, Optional, final
from itertools import chain
from collections import OrderedDict
from dataclasses import field, dataclass
from collections.abc import Sequence

from kirin import ir, graph, interp, idtable
from kirin.lattice import (
    SingletonMeta,
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
)
from kirin.analysis import Forward, ForwardFrame
from kirin.dialects import func
from bloqade.analysis import address
from bloqade.qasm2.parse.print import Printer


@dataclass
class GateSchedule(
    SimpleJoinMixin["GateSchedule"],
    SimpleMeetMixin["GateSchedule"],
    BoundedLattice["GateSchedule"],
):

    @classmethod
    def bottom(cls) -> "GateSchedule":
        return NotQubit()

    @classmethod
    def top(cls) -> "GateSchedule":
        return Qubit()


@final
@dataclass
class NotQubit(GateSchedule, metaclass=SingletonMeta):

    def is_subseteq(self, other: GateSchedule) -> bool:
        return True


@final
@dataclass
class Qubit(GateSchedule, metaclass=SingletonMeta):

    def is_subseteq(self, other: GateSchedule) -> bool:
        return isinstance(other, Qubit)


# Treat global gates as terminators for this analysis, e.g. split block in half.


@dataclass(slots=True)
class StmtDag(graph.Graph[ir.Statement]):
    id_table: idtable.IdTable[ir.Statement] = field(
        default_factory=lambda: idtable.IdTable()
    )
    stmts: Dict[str, ir.Statement] = field(default_factory=OrderedDict)
    out_edges: Dict[str, Set[str]] = field(default_factory=OrderedDict)
    inc_edges: Dict[str, Set[str]] = field(default_factory=OrderedDict)

    def add_node(self, node: ir.Statement):
        node_id = self.id_table[node]
        self.stmts[node_id] = node
        self.out_edges.setdefault(node_id, set())
        self.inc_edges.setdefault(node_id, set())
        return node_id

    def add_edge(self, src: ir.Statement, dst: ir.Statement):
        src_id = self.add_node(src)
        dst_id = self.add_node(dst)

        self.out_edges[src_id].add(dst_id)
        self.inc_edges[dst_id].add(src_id)

    def get_parents(self, node: ir.Statement) -> Iterable[ir.Statement]:
        return (
            self.stmts[node_id]
            for node_id in self.inc_edges.get(self.id_table[node], set())
        )

    def get_children(self, node: ir.Statement) -> Iterable[ir.Statement]:
        return (
            self.stmts[node_id]
            for node_id in self.out_edges.get(self.id_table[node], set())
        )

    def get_neighbors(self, node: ir.Statement) -> Iterable[ir.Statement]:
        return chain(self.get_parents(node), self.get_children(node))

    def get_nodes(self) -> Iterable[ir.Statement]:
        return self.stmts.values()

    def get_edges(self) -> Iterable[tuple[ir.Statement, ir.Statement]]:
        return (
            (self.stmts[src], self.stmts[dst])
            for src, dsts in self.out_edges.items()
            for dst in dsts
        )

    def print(
        self,
        printer: Optional["Printer"] = None,
        analysis: dict["ir.SSAValue", Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def topological_groups(self):
        """Split the dag into topological groups where each group
        contains nodes that have no dependencies on each other, but
        have dependencies on nodes in one or more previous groups.

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

        inc_edges = {k: set(v) for k, v in self.inc_edges.items()}

        check_next = inc_edges.keys()

        for _ in range(len(self.stmts)):
            if len(inc_edges) == 0:
                break

            group = [node_id for node_id in check_next if len(inc_edges[node_id]) == 0]
            yield group

            check_next = set()
            for n in group:
                inc_edges.pop(n)
                for m in self.out_edges[n]:
                    check_next.add(m)
                    inc_edges[m].remove(n)

        if inc_edges:
            raise ValueError("Cyclic dependency detected")


@dataclass
class DagScheduleAnalysis(Forward[GateSchedule]):
    keys = ["qasm2.schedule.dag"]
    lattice = GateSchedule

    address_analysis: Dict[ir.SSAValue, address.Address]
    use_def: Dict[int, ir.Statement] = field(init=False)
    stmt_dag: StmtDag = field(init=False)
    stmt_dags: Dict[ir.Block, StmtDag] = field(init=False)

    def initialize(self):
        self.use_def = {}
        self.stmt_dag = StmtDag()
        self.stmt_dags = {}
        return super().initialize()

    def push_current_dag(self, block: ir.Block):
        # run when hitting terminator statements
        assert block not in self.stmt_dags, "Block already in stmt_dags"

        for node in self.use_def.values():
            self.stmt_dag.add_node(node)

        self.stmt_dags[block] = self.stmt_dag
        self.stmt_dag = StmtDag()
        self.use_def = {}

    def run_method(self, method: ir.Method, args: tuple[GateSchedule, ...]):
        # NOTE: we do not support dynamic calls here, thus no need to propagate method object
        return self.run_callable(method.code, (self.lattice.bottom(),) + args)

    def eval_stmt_fallback(self, frame: ForwardFrame, stmt: ir.Statement):
        if stmt.has_trait(ir.IsTerminator):
            assert (
                stmt.parent_block is not None
            ), "Terminator statement has no parent block"
            self.push_current_dag(stmt.parent_block)

        return tuple(self.lattice.top() for _ in stmt.results)

    def _update_dag(self, stmt: ir.Statement, addr: address.Address):
        if isinstance(addr, address.AddressQubit):
            old_stmt = self.use_def.get(addr.data, None)
            if old_stmt is not None:
                self.stmt_dag.add_edge(old_stmt, stmt)
            self.use_def[addr.data] = stmt
        elif isinstance(addr, address.AddressReg):
            for idx in addr.data:
                old_stmt = self.use_def.get(idx, None)
                if old_stmt is not None:
                    self.stmt_dag.add_edge(old_stmt, stmt)
                self.use_def[idx] = stmt
        elif isinstance(addr, address.AddressTuple):
            for sub_addr in addr.data:
                self._update_dag(stmt, sub_addr)

    def update_dag(self, stmt: ir.Statement, args: Sequence[ir.SSAValue]):
        for arg in args:
            self._update_dag(
                stmt, self.address_analysis.get(arg, address.Address.bottom())
            )

    def get_dags(self, mt: ir.Method, args=None, kwargs=None):
        if args is None:
            args = tuple(self.lattice.top() for _ in mt.args)

        self.run(mt, args, kwargs).expect()
        return self.stmt_dags


@func.dialect.register(key="qasm2.schedule.dag")
class FuncImpl(interp.MethodTable):
    @interp.impl(func.Invoke)
    @interp.impl(func.Call)
    def invoke(
        self,
        interp: DagScheduleAnalysis,
        frame: ForwardFrame,
        stmt: func.Invoke | func.Call,
    ):
        interp.update_dag(stmt, stmt.inputs)
        return tuple(interp.lattice.top() for _ in stmt.results)
