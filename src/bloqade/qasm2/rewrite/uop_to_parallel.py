import abc
from typing import Dict, List, Tuple, Iterable
from dataclasses import field, dataclass

from kirin import ir
from kirin.rewrite import abc as rewrite_abc
from kirin.dialects import py, ilist
from bloqade.analysis import address
from kirin.analysis.const import lattice
from bloqade.qasm2.dialects import uop, core, parallel
from bloqade.squin.analysis.schedule import StmtDag


class MergePolicyABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        pass

    @classmethod
    @abc.abstractmethod
    def can_merge(cls, stmt1: ir.Statement, stmt2: ir.Statement) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def merge_gates(
        cls, gate_stmts: Iterable[ir.Statement]
    ) -> List[List[ir.Statement]]:
        pass

    @classmethod
    @abc.abstractmethod
    def from_analysis(
        cls, dag: StmtDag, address_analysis: Dict[ir.SSAValue, address.Address]
    ) -> "MergePolicyABC":
        pass


@dataclass
class SimpleMergePolicy(MergePolicyABC):
    """General merge policy for merging gates based on their type and arguments.

    Base class to implement a merge policy for CZ, U and RZ gates, To completed the policy implement the
    `merge_gates` class method. This will take an iterable of statements and return a list
    of groups of statements that can be merged together. There are two mix-in classes
    that can be used to implement the `merge_gates` method. The `GreedyMixin` will merge
    gates together greedily, while the `OptimalMixIn` will merge gates together optimally.

    """

    address_analysis: Dict[ir.SSAValue, address.Address]
    """Mapping from SSA values to their address analysis results. Needed for rewrites"""
    merge_groups: List[List[ir.Statement]]
    """List of groups of statements that can be merged together"""
    group_numbers: Dict[ir.Statement, int]
    """Mapping from statements to their group number"""
    group_has_merged: Dict[int, bool] = field(default_factory=dict)
    """Mapping from group number to whether the group has been merged"""

    @staticmethod
    def same_id_checker(ssa1: ir.SSAValue, ssa2: ir.SSAValue):
        if ssa1 is ssa2:
            return True
        elif (hint1 := ssa1.hints.get("const")) and (hint2 := ssa2.hints.get("const")):
            assert isinstance(hint1, lattice.Result) and isinstance(
                hint2, lattice.Result
            )
            return hint1.is_equal(hint2)
        else:
            return False

    @classmethod
    def check_equiv_args(
        cls,
        args1: Iterable[ir.SSAValue],
        args2: Iterable[ir.SSAValue],
    ):
        try:
            return all(
                cls.same_id_checker(ssa1, ssa2)
                for ssa1, ssa2 in zip(args1, args2, strict=True)
            )
        except ValueError:
            return False

    @classmethod
    def can_merge(cls, stmt1: ir.Statement, stmt2: ir.Statement) -> bool:
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
                return cls.check_equiv_args(stmt1.args[1:], stmt2.args[1:])
            case (
                (parallel.CZ(), parallel.CZ())
                | (parallel.CZ(), uop.CZ())
                | (uop.CZ(), parallel.CZ())
                | (uop.CZ(), uop.CZ())
                | (uop.Barrier(), uop.Barrier())
            ):
                return True

            case _:
                return False

    @classmethod
    def from_analysis(
        cls,
        dag: StmtDag,
        address_analysis: Dict[ir.SSAValue, address.Address],
    ):

        merge_groups = []
        group_numbers = {}

        for group in dag.topological_groups():
            gate_groups = cls.merge_gates(map(dag.stmts.__getitem__, group))
            gate_groups_iter = (group for group in gate_groups if len(group) > 1)

            for gate_group in gate_groups_iter:
                group_number = len(merge_groups)
                merge_groups.append(gate_group)
                for stmt in gate_group:
                    group_numbers[stmt] = group_number

        for group in merge_groups:
            group.sort(key=lambda stmt: dag.stmt_index[stmt])

        return cls(
            address_analysis=address_analysis,
            merge_groups=merge_groups,
            group_numbers=group_numbers,
        )

    def __call__(self, node: ir.Statement) -> rewrite_abc.RewriteResult:

        if node not in self.group_numbers:
            return rewrite_abc.RewriteResult()

        group_number = self.group_numbers[node]
        group = self.merge_groups[group_number]
        if node is group[0]:
            result = getattr(self, f"rewrite_group_{node.name}")(node, group)

            self.group_has_merged[group_number] = result.has_done_something
            return result

        if self.group_has_merged[group_number]:
            node.delete()

        return rewrite_abc.RewriteResult(
            has_done_something=self.group_has_merged[group_number]
        )

    def move_and_collect_qubit_list(
        self, qargs: List[ir.SSAValue], node: ir.Statement
    ) -> Tuple[ir.SSAValue, ...] | None:

        qubits: List[ir.SSAValue] = []
        # collect references to qubits
        for qarg in qargs:
            addr = self.address_analysis[qarg]

            if isinstance(addr, address.AddressQubit):
                qubits.append(qarg)

            elif isinstance(addr, address.AddressTuple):
                assert isinstance(qarg, ir.ResultValue)
                assert isinstance(qarg.stmt, ilist.New)
                qubits.extend(qarg.stmt.values)
            else:
                # give up if we cannot determine the address
                return None

        new_qubits = []

        # the registers must be moved to the top of the block
        # before this pass can be applied
        for qubit_ref in qubits:
            qubit = qubit_ref.owner
            match qubit:
                case ir.BlockArgument():  # do not need to move the qubit
                    new_qubits.append(qubit)
                case core.QRegGet(reg=reg, idx=ir.BlockArgument() as idx):
                    new_qubit = core.QRegGet(reg=reg, idx=idx)
                    new_qubit.insert_before(node)
                    new_qubits.append(new_qubit.result)
                case core.QRegGet(
                    reg=reg, idx=ir.ResultValue(stmt=py.Constant() as idx)
                ):
                    (new_idx := idx.from_stmt(idx)).insert_before(node)
                    (
                        new_qubit := core.QRegGet(reg=reg, idx=new_idx.result)
                    ).insert_before(node)
                    new_qubits.append(new_qubit.result)
                case _:
                    return None

        return tuple(new_qubits)

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
                return rewrite_abc.RewriteResult(has_done_something=False)

        ctrls_values = self.move_and_collect_qubit_list(ctrls, node)
        qargs_values = self.move_and_collect_qubit_list(qargs, node)

        if ctrls_values is None or qargs_values is None:
            # give up if we cannot determine the address or cannot move the qubits
            return rewrite_abc.RewriteResult(has_done_something=False)

        new_ctrls = ilist.New(values=ctrls_values)
        new_qargs = ilist.New(values=qargs_values)
        new_gate = parallel.CZ(ctrls=new_ctrls.result, qargs=new_qargs.result)

        new_ctrls.insert_before(node)
        new_qargs.insert_before(node)
        new_gate.insert_before(node)

        node.delete()

        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_group_U(self, node: ir.Statement, group: List[ir.Statement]):
        return self.rewrite_group_u(node, group)

    def rewrite_group_u(self, node: ir.Statement, group: List[ir.Statement]):
        qargs = []

        for stmt in group:
            if isinstance(stmt, uop.UGate):
                qargs.append(stmt.qarg)
            elif isinstance(stmt, parallel.UGate):
                qargs.append(stmt.qargs)
            else:
                return rewrite_abc.RewriteResult(has_done_something=False)

        assert isinstance(node, (uop.UGate, parallel.UGate))
        qargs_values = self.move_and_collect_qubit_list(qargs, node)

        if qargs_values is None:
            return rewrite_abc.RewriteResult(has_done_something=False)

        new_qargs = ilist.New(values=qargs_values)
        new_gate = parallel.UGate(
            qargs=new_qargs.result,
            theta=node.theta,
            phi=node.phi,
            lam=node.lam,
        )
        new_qargs.insert_before(node)
        new_gate.insert_before(node)
        node.delete()

        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_group_rz(self, node: ir.Statement, group: List[ir.Statement]):
        qargs = []

        for stmt in group:
            if isinstance(stmt, uop.RZ):
                qargs.append(stmt.qarg)
            elif isinstance(stmt, parallel.RZ):
                qargs.append(stmt.qargs)
            else:
                return rewrite_abc.RewriteResult(has_done_something=False)

        assert isinstance(node, (uop.RZ, parallel.RZ))

        qargs_values = self.move_and_collect_qubit_list(qargs, node)

        if qargs_values is None:
            return rewrite_abc.RewriteResult(has_done_something=False)

        new_qargs = ilist.New(values=qargs_values)
        new_gate = parallel.RZ(
            qargs=new_qargs.result,
            theta=node.theta,
        )
        new_qargs.insert_before(node)
        new_gate.insert_before(node)
        node.delete()

        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_group_barrier(self, node: uop.Barrier, group: List[uop.Barrier]):
        qargs = []
        for stmt in group:
            qargs.extend(stmt.qargs)

        qargs_values = self.move_and_collect_qubit_list(qargs, node)

        if qargs_values is None:
            return rewrite_abc.RewriteResult(has_done_something=False)

        new_node = uop.Barrier(qargs=qargs_values)
        new_node.insert_before(node)
        node.delete()

        return rewrite_abc.RewriteResult(has_done_something=True)


class GreedyMixin(MergePolicyABC):
    """Merge policy that greedily merges gates together.

    The `merge_gates` method will merge policy will try greedily merge gates together.
    This policy has a worst case complexity of O(n) where n is the
    number of gates in the input iterable.
    """

    @classmethod
    def merge_gates(
        cls, gate_stmts: Iterable[ir.Statement]
    ) -> List[List[ir.Statement]]:

        iterable = iter(gate_stmts)
        groups = [[next(iterable)]]

        for stmt in gate_stmts:
            if cls.can_merge(groups[-1][-1], stmt):
                groups[-1].append(stmt)
            else:
                groups.append([stmt])

        return groups


class OptimalMixIn(MergePolicyABC):
    """Merge policy that merges gates together optimally.

    The `merge_gates` method will merge policy will try to merge every gate into every
    group of gates, terminating when it finds a group that can be merged with the current
    gate. This policy has a worst case complexity of O(n^2) where n is the number of gates
    in the input iterable.

    """

    @classmethod
    def merge_gates(
        cls, gate_stmts: Iterable[ir.Statement]
    ) -> List[List[ir.Statement]]:

        groups = []
        for stmt in gate_stmts:
            found = False
            for group in groups:
                if cls.can_merge(group[-1], stmt):
                    group.append(stmt)
                    found = True
                    break

            if not found:
                groups.append([stmt])

        return groups


@dataclass
class SimpleGreedyMergePolicy(GreedyMixin, SimpleMergePolicy):
    pass


@dataclass
class SimpleOptimalMergePolicy(OptimalMixIn, SimpleMergePolicy):
    pass


@dataclass
class UOpToParallelRule(rewrite_abc.RewriteRule):
    merge_rewriters: Dict[ir.Block | None, MergePolicyABC]

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        merge_rewriter = self.merge_rewriters.get(
            node.parent_block, lambda _: rewrite_abc.RewriteResult()
        )
        return merge_rewriter(node)
