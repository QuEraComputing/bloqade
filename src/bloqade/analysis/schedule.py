from typing import Any, Set, Dict, List, Iterable, Optional, final
from itertools import chain
from dataclasses import field, dataclass
from collections.abc import Sequence

from kirin import ir, graph, idtable
from kirin.lattice import (
    SingletonMeta,
    BoundedLattice,
    SimpleJoinMixin,
    SimpleMeetMixin,
)
from kirin.analysis import Forward, ForwardFrame
from kirin.dialects import ilist
from bloqade.analysis import address
from kirin.interp.exceptions import InterpreterError
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
    nodes: idtable.IdTable[ir.Statement] = field(
        default_factory=lambda: idtable.IdTable()
    )
    stmts: Dict[str, ir.Statement] = field(default_factory=dict)
    fwd_edges: Dict[str, Set[str]] = field(default_factory=dict)
    bwd_edges: Dict[str, Set[str]] = field(default_factory=dict)

    def add_node(self, node: ir.Statement):
        node_id = self.nodes.add(node)
        self.stmts[node_id] = node
        return node_id

    def add_edge(self, src: ir.Statement, dst: ir.Statement):
        src_id = self.add_node(src)
        dst_id = self.add_node(dst)

        self.fwd_edges.setdefault(src_id, set()).add(dst_id)
        self.bwd_edges.setdefault(dst_id, set()).add(src_id)

    def get_parents(self, node: ir.Statement) -> Iterable[ir.Statement]:
        return (
            self.stmts[node_id]
            for node_id in self.bwd_edges.get(self.nodes[node], set())
        )

    def get_children(self, node: ir.Statement) -> Iterable[ir.Statement]:
        return (
            self.stmts[node_id]
            for node_id in self.fwd_edges.get(self.nodes[node], set())
        )

    def get_neighbors(self, node: ir.Statement) -> graph.Iterable[ir.Statement]:
        return chain(self.get_parents(node), self.get_children(node))

    def get_nodes(self) -> graph.Iterable[ir.Statement]:
        return self.stmts.values()

    def get_edges(self) -> Iterable[tuple[ir.Statement, ir.Statement]]:
        return (
            (self.stmts[src], self.stmts[dst])
            for src, dsts in self.fwd_edges.items()
            for dst in dsts
        )

    def print(
        self,
        printer: Optional["Printer"] = None,
        analysis: dict["ir.SSAValue", Any] | None = None,
    ) -> None:
        raise NotImplementedError


@dataclass
class DagScheduleAnalysis(Forward[GateSchedule]):
    keys = ["qasm2.schedule.dag"]
    lattice = GateSchedule

    address_analysis: Dict[ir.SSAValue, address.Address]
    use_def: Dict[int, ir.Statement] = field(init=False)
    stmt_dag: StmtDag = field(init=False)
    stmt_dags: List[StmtDag] = field(init=False)

    def initialize(self):
        super().initialize()
        self.use_def = {}
        self.stmt_dag = StmtDag()
        self.stmt_dags = []

    def push_current_dag(self):
        # run when hitting terminator statements
        self.stmt_dags.append(self.stmt_dag)
        self.stmt_dag = StmtDag()
        self.use_def = {}

    def eval_stmt_fallback(self, frame: ForwardFrame[GateSchedule], stmt: ir.Statement):
        if stmt.has_trait(ir.IsTerminator):
            self.push_current_dag()

        return tuple(self.lattice.top() for _ in stmt.results)

    def update_dag(self, stmt: ir.Statement, args: Sequence[ir.SSAValue]):
        addrs = [
            self.address_analysis.get(arg, address.Address.bottom()) for arg in args
        ]

        for arg, addr in zip(args, addrs):
            if not isinstance(addr, address.AddressQubit):
                raise InterpreterError(f"Expected AddressQubit for result {arg}")

            if addr.data in self.use_def:
                self.stmt_dag.add_edge(self.use_def[addr.data], stmt)

        for addr in addrs:
            assert isinstance(addr, address.AddressQubit)
            self.use_def[addr.data] = stmt

    def get_ilist_ssa(self, value: ir.SSAValue):
        addr = self.address_analysis[value]

        if not isinstance(addr, address.AddressTuple):
            raise InterpreterError(f"Expected AddressTuple, got {addr}")

        if not all(isinstance(addr, address.AddressQubit) for addr in addr.data):
            raise InterpreterError("Expected AddressQubit")

        assert isinstance(value, ir.ResultValue)
        assert isinstance(value.stmt, ilist.New)

        return value.stmt.values

    def get_dag(self, mt: ir.Method, args=None, kwargs=None):
        if args is None:
            args = tuple(self.lattice.top() for _ in mt.args)

        self.run(mt, args, kwargs).expect()
        return self.stmt_dags
