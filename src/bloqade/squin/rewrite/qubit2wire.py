from dataclasses import dataclass

from kirin import ir, types
from bloqade.squin import wire, qubit
from kirin.rewrite import abc
from kirin.dialects import py, func, ilist


@dataclass
class Qubit2WireRule(abc.RewriteRule):
    """
    This rewrite rule is intended to replace the qubit dialect with the wire dialect.

    The pass keeps track of the map using the `defs` and `inv_defs`
    dictionaries keep track of the mapping between the qubit reference and wire values.

    During the rewrite, there are some notable edge cases:

    1. The qubit value is an argument of the function.
    2. The qubit reference is passed into a subroutine.
    3. There is a container of qubit references as an argument of a function

    Cases 1 and 2 are supported. Case 3 supported if the container size is known at compile time.

    """

    defs: dict[ir.SSAValue, ir.SSAValue]
    inv_defs: dict[ir.SSAValue, ir.SSAValue]
    SUPPORTED_STMTS = frozenset({func.Invoke, func.Call, qubit.Apply})

    @staticmethod
    def infer_qubit_ilist_size(typ: types.TypeAttribute) -> int | None:
        """Given a type attribute, infer the size of the container of qubits."""
        if typ.is_subseteq(ilist.IListType[qubit.QubitType, types.Any]):
            assert isinstance(typ, types.Generic)

            if isinstance(size_hint := typ.vars[1], types.Literal) and isinstance(
                size := size_hint.data, int
            ):
                return size

        return None

    def replace_wire(self, old_wire: ir.SSAValue, new_wire: ir.SSAValue):
        """This function is used to replace an old wire with a new wire in the
        defs/inv_defs dictionaries. Note that the new wire can't be assigned to a qubit.

        Args:
            old_wire (ir.SSAValue): The old wire that needs to be replaced.
            new_wire (ir.SSAValue): The new wire that will replace the old wire.


        """
        assert old_wire in self.inv_defs, "old_wire is not attached to any qubit"
        assert new_wire not in self.defs, "new_wire is already attached to a qubit"

        self.defs[(qubit_ref := self.inv_defs[old_wire])] = new_wire
        self.inv_defs[new_wire] = qubit_ref

    def get_wire(self, value: ir.SSAValue) -> ir.SSAValue | None:
        """Get the wire associated with the ssa value. If it doesn't exist, create a new wire and insert it into the IR.

        Args:
            value (ir.SSAValue): The qubit value for which we want to get the wire.

        Returns:
            ir.SSAValue | None: The wire associated with the qubit value, or None if the value is not a qubit.


        """
        if value.type.is_subseteq(qubit.QubitType):
            if value in self.defs:
                return self.defs[value]

            owner = value.owner
            new_wire = wire.Unwrap(value)
            if isinstance(owner, ir.Block):
                if owner.first_stmt is None:
                    owner.stmts.append(new_wire)
                else:
                    new_wire.insert_before(owner.first_stmt)
            else:
                new_wire.insert_after(owner)

            self.defs[value] = (result := new_wire.result)
            self.inv_defs[result] = value

            return result
        else:
            return None

    def get_wires(self, value: ir.SSAValue) -> tuple[ir.SSAValue, ...] | None:
        """Take a container of qubits and return a tuple of wires. If the container size is not known at compile time, return None.

        Currently this function only handles the IListType of qubits.

        Args:
            value (ir.SSAValue): The ssa value pointing to the container of qubits.

        Returns:
            tuple[ir.SSAValue, ...] | None: A tuple of wires if all qubits can be converted, otherwise None.

        """
        value_type = value.type
        size = self.infer_qubit_ilist_size(value_type)

        if size is None:
            return None

        wires: list[ir.SSAValue] = []
        new_stmts: list[ir.Statement] = []
        for i in range(size):

            new_stmts.append(index := py.Constant(i))
            new_stmts.append(get_qubit := py.GetItem(value, index.result))
            new_stmts.append(unpwrapped := wire.Unwrap(get_qubit.result))
            self.defs[get_qubit.result] = unpwrapped.result
            self.inv_defs[unpwrapped.result] = get_qubit.result
            wires.append(unpwrapped.result)

        owner = value.owner
        if isinstance(owner, ir.Block):
            # is argument
            if owner.first_stmt is None:
                owner.stmts.extend(new_stmts)
            else:
                for stmt in new_stmts:
                    stmt.insert_before(owner.first_stmt)
        else:
            for stmt in new_stmts:
                stmt.insert_after(owner)

        return tuple(wires)

    def rewrite_Statement(self, node: ir.Statement):
        if type(node) not in self.SUPPORTED_STMTS:
            return abc.RewriteResult()

        return getattr(self, f"rewrite_{type(node).__name__}")(node)

    def write_Apply(self, node: qubit.Apply):
        wires = self.get_wires(node.operator)

        if wires is None:
            return abc.RewriteResult()

        new_apply = wire.Apply(node.operator, *wires)

        node.replace_by(new_apply)

        for old_wire, new_wire in zip(wires, new_apply.results):
            self.replace_wire(old_wire, new_wire)

    def _rewrite_call_like(self, node: func.Call | func.Invoke):
        """Wrap wire before call-like node."""
        has_done_something = False
        for arg in node.inputs:
            w = self.get_wire(arg)
            if w is None:
                continue

            has_done_something = True
            wire.Wrap(w, arg).insert_before(node)

        return abc.RewriteResult(has_done_something=has_done_something)

    def rewrite_Invoke(self, node: func.Invoke):
        return self._rewrite_call_like(node)

    def rewrite_Call(self, node: func.Call):
        return self._rewrite_call_like(node)
