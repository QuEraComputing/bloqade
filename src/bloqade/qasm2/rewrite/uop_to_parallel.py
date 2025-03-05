import abc
from typing import Dict, List, Tuple, Iterable
from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc as rewrite_abc, result
from kirin.dialects import ilist
from bloqade.analysis import address
from kirin.analysis.const import lattice
from bloqade.qasm2.dialects import uop, core, parallel
from bloqade.analysis.schedule import StmtDag


class MergePolicyABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, node: ir.Statement) -> result.RewriteResult:
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

    def __call__(self, node: ir.Statement) -> result.RewriteResult:
        if node not in self.group_numbers:
            return result.RewriteResult()

        group = self.merge_groups[self.group_numbers[node]]
        if node is group[-1]:
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

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        merge_rewriter = self.merge_rewriters.get(
            node.parent_block, lambda _: result.RewriteResult()
        )
        return merge_rewriter(node)
