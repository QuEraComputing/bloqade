from dataclasses import dataclass

from kirin import ir, types
from bloqade.squin import wire, qubit
from kirin.rewrite import abc
from kirin.dialects import func, ilist


@dataclass
class Qubit2WireRule(abc.RewriteRule):
    """
    This rewrite rule is intended to replace qubit dialect with the wire dialect.

    In most cases this pass keeps track of the this map using the `defs` dictionary.
    The `defs` dictionary maps the qubit SSA values to the corresponding wire SSA values.

    There are some notable edge cases related to:

    1. The qubit value is an argument of the function.
    2. The qubit reference is passed into a subroutine.
    3. The ilist of qubits as a block argument.

    This rewrite rule will handle those cases as well.

    Case 1: The argument is unwrapped at the top level of the function and added to the dictionary.
    Case 2: The wire is wrapped and the qubit reference is passed into the invoke.

    Case 3: The qubits can't be unwrapped currently without the ability to unwrap the entire list.

    """

    defs: dict[ir.SSAValue, ir.SSAValue]
    inv_defs: dict[ir.SSAValue, ir.SSAValue]
    SUPPORTED_STMTS = frozenset({func.Invoke, func.Call, qubit.Apply})

    def replace_wire(self, old_wire: ir.SSAValue, new_wire: ir.SSAValue):
        assert old_wire in self.inv_defs
        assert new_wire not in self.defs

        self.defs[(qubit_ref := self.inv_defs[old_wire])] = new_wire
        self.inv_defs[new_wire] = qubit_ref

    def get_wire(self, value: ir.SSAValue) -> ir.SSAValue | None:

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
        value_type = value.type

        if value_type.is_subseteq(ilist.IListType[qubit.QubitType, types.Any]):
            owner = value.owner
            if isinstance(owner, ir.Block):
                return None
            else:
                assert isinstance(owner, ilist.New)
                wires: list[ir.SSAValue] = []
                for value in owner.values:
                    w = self.get_wire(value)
                    if w is None:
                        return None

                    wires.append(w)

                return tuple(wires)

        else:
            return None

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
