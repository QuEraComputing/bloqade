from typing import Optional
from dataclasses import field, dataclass

from kirin import ir, types
from bloqade.squin import wire, qubit
from kirin.rewrite import abc
from kirin.dialects import py, scf, func, ilist


@dataclass(frozen=True)
class QubitRef:
    reg: ir.SSAValue | None
    index: int | ir.BlockArgument
    qubit_ssa: ir.SSAValue = field(hash=False, compare=False, repr=False)


def wid(wire_value):
    return f"{id(wire_value)} : {wire_value.uses}"


@dataclass
class QubitFrame:
    parent: Optional["QubitFrame"] = field(default=None)
    qubits_to_wires: dict[QubitRef, ir.SSAValue] = field(default_factory=dict)
    wires_to_qubits: dict[ir.SSAValue, QubitRef] = field(default_factory=dict)

    @staticmethod
    def infer_qubit_container_size(value: ir.SSAValue) -> int | None:
        if isinstance(value.owner, qubit.New):
            n_qubits = value.owner.n_qubits.owner
            if not isinstance(n_qubits, py.Constant) or not isinstance(
                n_qubits.value, int
            ):
                return None

            return n_qubits.value

        typ = value.type

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
        wire to qubit maps. Note that the new wire can't be assigned to a qubit.

        Args:
            old_wire (ir.SSAValue): The old wire that needs to be replaced.
            new_wire (ir.SSAValue): The new wire that will replace the old wire.


        """
        assert old_wire in self.wires_to_qubits, "old_wire is not attached to any qubit"
        assert (
            new_wire not in self.qubits_to_wires
        ), "new_wire is already attached to a qubit"
        print("replace_wire", wid(old_wire), " with ", wid(new_wire))
        self.qubits_to_wires[qubit_ref := self.wires_to_qubits.pop(old_wire)] = new_wire
        self.wires_to_qubits[new_wire] = qubit_ref

    def get_qubit_ref(self, qubit_ssa: ir.SSAValue) -> QubitRef | None:
        """Construct a qubit reference from the ssa value."""
        if not qubit_ssa.type.is_subseteq(qubit.QubitType):
            return None

        owner = qubit_ssa.owner
        if isinstance(owner, ir.BlockArgument):
            return QubitRef(reg=None, index=owner, qubit_ssa=qubit_ssa)
        elif isinstance(owner, py.GetItem):
            reg = owner.obj
            index = owner.index

            if isinstance(index, ir.BlockArgument):
                return QubitRef(reg=reg, index=index, qubit_ssa=qubit_ssa)
            elif isinstance(index.owner, py.Constant) and isinstance(
                index.owner.value, int
            ):
                return QubitRef(reg=reg, index=index.owner.value, qubit_ssa=qubit_ssa)

        return None

    def get_wire(self, qubit_ssa: ir.SSAValue) -> ir.SSAValue | None:
        """Get the wire associated with the ssa value. If it doesn't exist, create a new wire and insert it into the IR.

        Args:
            qubit_ref (ir.SSAValue): The qubit value for which we want to get the wire.

        Returns:
            ir.SSAValue | None: The wire associated with the qubit value, or None if the value is not a qubit.


        """
        qubit_ref = self.get_qubit_ref(qubit_ssa)
        if qubit_ref is None:
            return None

        wire_value = self.qubits_to_wires.get(qubit_ref, None)
        if wire_value is not None:
            return wire_value

        # create a new wire
        owner = qubit_ssa.owner
        new_wire = wire.Unwrap(qubit_ref.qubit_ssa)
        if isinstance(owner, ir.Block):
            if owner.first_stmt is None:
                owner.stmts.append(new_wire)
            else:
                new_wire.insert_before(owner.first_stmt)
        else:
            new_wire.insert_after(owner)

        self.qubits_to_wires[qubit_ref] = (result := new_wire.result)
        self.wires_to_qubits[result] = qubit_ref
        print("create new wire", wid(result))

        return result

    def get_wires(self, value: ir.SSAValue) -> tuple[ir.SSAValue, ...] | None:
        """Take a container of qubits and return a tuple of wires. If the container size is not known at compile time, return None.

        Currently this function only handles the IListType of qubits.

        Args:
            value (ir.SSAValue): The ssa value pointing to the container of qubits.

        Returns:
            tuple[ir.SSAValue, ...] | None: A tuple of wires if all qubits can be converted, otherwise None.

        """
        owner = value.owner
        if isinstance(owner, ilist.New):
            wires = []

            for value in owner.values:
                wire_value = self.get_wire(value)
                if wire_value is None:
                    return None
                wires.append(wire_value)

            return tuple(wires)

        size = self.infer_qubit_container_size(value)
        if size is None:
            return None

        wires: list[ir.SSAValue] = []
        new_stmts: list[ir.Statement] = []
        for i in range(size):

            new_stmts.append(index := py.Constant(i))
            new_stmts.append(get_qubit := py.GetItem(value, index.result))
            new_stmts.append(unpwrapped := wire.Unwrap(get_qubit.result))
            get_qubit.result.type = qubit.QubitType
            wires.append(wire_value := unpwrapped.result)
            qubit_ref = self.get_qubit_ref(get_qubit.result)
            assert qubit_ref is not None, "qubit_ref is None"
            self.qubits_to_wires[qubit_ref] = wire_value
            self.wires_to_qubits[wire_value] = qubit_ref

        if isinstance(owner, ir.Block):
            # is argument
            if owner.first_stmt is None:
                owner.stmts.extend(new_stmts)
            else:
                for stmt in new_stmts:
                    stmt.insert_before(owner.first_stmt)
        else:
            for stmt in reversed(new_stmts):
                stmt.insert_after(owner)

    def pop_wire(self, wire_value: ir.SSAValue):
        """Pop a wire from the qubit frame."""
        qubit_ref = self.wires_to_qubits.pop(wire_value)
        self.qubits_to_wires.pop(qubit_ref)
        return qubit_ref

    @property
    def curr_wires(self):
        return list(self.qubits_to_wires.values())


@dataclass
class Qubit2WireRule(abc.RewriteRule):
    """
    This rewrite rule is intended to replace the qubit dialect with the wire dialect.

    The pass keeps track of the map using the `defs` and `inv_defs`
    dictionaries keep track of the mapping between the qubit reference and wire values.

    During the rewrite, there are some notable edge cases:

    [x] The qubit value is an argument of the function.
    [x] The qubit reference is passed into a subroutine.
    [x] There is a container of qubit references as an argument of a function
    [x] All wires must be unwrapped before a return statement.
    [ ] All wires associated with a register must be wrapped before an invoke/call statement.

    Cases 1 and 2 are supported. Case 3 supported if the container size is known at compile time.

    """

    frame: QubitFrame = field(default_factory=QubitFrame)

    old_wires: set[ir.SSAValue] = field(default_factory=set)

    def push_frame(self):
        self.frame = QubitFrame(
            self.frame,
            self.frame.qubits_to_wires.copy(),
            self.frame.wires_to_qubits.copy(),
        )

    def pop_frame(self):
        if self.frame.parent is None:
            raise ValueError("Cannot pop the root frame.")

        self.frame = self.frame.parent

    def wrap_before(self, node: ir.Statement, wire_value: ir.SSAValue):
        """Wrap wire before return node. updates defs and inv_defs."""

        print("wrap", wid(wire_value))
        qubit_ref = self.frame.pop_wire(wire_value)
        wire.Wrap(wire_value, qubit_ref.qubit_ssa).insert_before(node)

    def rewrite_Statement(self, node: ir.Statement):
        if isinstance(node, qubit.Apply):
            return self.rewrite_Apply(node)
        elif isinstance(node, (func.Call, func.Invoke)):
            return self.rewrite_call_like(node)
        elif isinstance(node, (scf.IfElse,)):
            return self.rewrite_IfElse(node)
        elif isinstance(node, (scf.Yield,)):
            return self.rewrite_Yield(node)
        elif isinstance(node, func.Return):
            return self.rewrite_return(node)
        else:
            return abc.RewriteResult()

    def rewrite_IfElse(self, node: scf.IfElse):
        print("rewrite_IfElse")

        new_node = scf.IfElse(node.cond, node.then_body.clone(), node.else_body.clone())

        num_old_results = len(node.results)
        num_new_results = len(new_node.results)
        wires = list(self.frame.curr_wires)

        assert num_old_results + len(wires) == num_new_results

        node.replace_by(new_node)

        for old_wire, new_wire in zip(wires, new_node.results[num_old_results:]):
            assert new_wire.type.is_subseteq(wire.WireType)
            has_done_something = True
            self.frame.replace_wire(old_wire, new_wire)

        return abc.RewriteResult(has_done_something=has_done_something)

    def rewrite_Yield(self, node: scf.Yield):
        print("rewrite_Yield")
        # TODO: use current region to get the wires to add here
        new_values = node.values + tuple(self.frame.curr_wires)
        node.replace_by(scf.Yield(*new_values))

        if len(new_values) > len(node.values):
            return abc.RewriteResult(has_done_something=True)
        else:
            return abc.RewriteResult()

    def rewrite_Apply(self, node: qubit.Apply):
        print("rewrite_Apply")
        wires = self.frame.get_wires(node.qubits)
        if wires is None:
            return abc.RewriteResult()

        new_apply = wire.Apply(node.operator, *wires)

        node.replace_by(new_apply)

        for old_wire, new_wire in zip(wires, new_apply.results):
            self.frame.replace_wire(old_wire, new_wire)
        return abc.RewriteResult(has_done_something=True)

    def rewrite_call_like(self, node: func.Call | func.Invoke):
        """Wrap wire before call-like node."""
        has_done_something = False
        for arg in node.inputs:

            if (wire_value := self.frame.get_wire(arg)) is not None:
                has_done_something = True
                self.wrap_before(node, wire_value=wire_value)
                continue

            if (wire_values := self.frame.get_wires(arg)) is not None:
                has_done_something = True
                for wire_value in wire_values:
                    self.wrap_before(node, wire_value=wire_value)
                continue

        return abc.RewriteResult(has_done_something=has_done_something)

    def rewrite_return(self, node: func.Return):
        """Wrap all wires before return node."""
        for wire_value in self.frame.curr_wires:
            self.wrap_before(node, wire_value)

        return abc.RewriteResult(has_done_something=True)
