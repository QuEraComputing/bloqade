"""SQuIN IQP circuits and placement on the Gemini physical grid."""

import random
import inspect
from functools import wraps

from kirin import ir, passes
from kirin.dialects import py, func, ilist
from bloqade.lanes.dialects import arch
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.validation.recursion import CallGraph, check_call_graph

from bloqade import qubit, squin
from bloqade.gemini.physical.group import kernel as physical_kernel

GridAddress = tuple[int, int, int]
Layout = tuple[tuple[GridAddress, ...], ...]


def _hypercube_dimension(n_qubits: int) -> int:
    if type(n_qubits) is not int or n_qubits < 1 or n_qubits & (n_qubits - 1):
        raise ValueError("n_qubits must be a positive power of two")
    return n_qubits.bit_length() - 1


def build_iqp_circuit(n_qubits: int = 64, seed: int = 42) -> ir.Method:
    """Build a measured hypercube circuit. The seed selects each layer's T sites."""
    dimension = _hypercube_dimension(n_qubits)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    phase_width = max(1, 3 * n_qubits // 4)
    masks = tuple(
        tuple(sorted(rng.sample(range(n_qubits), phase_width)))
        for _ in range(dimension + 1)
    )

    @squin.kernel
    def parity_iqp():
        reg = squin.qalloc(n_qubits)
        squin.broadcast.h(reg)
        for layer in range(dimension + 1):
            if layer > 0:
                stride = 2 ** (layer - 1)
                left = reg[:stride]
                right = reg[stride : 2 * stride]
                for start in range(2 * stride, n_qubits, 2 * stride):
                    left = left + reg[start : start + stride]
                    right = right + reg[start + stride : start + 2 * stride]
                squin.broadcast.cx(left, right)

            selected = ilist.IList([reg[masks[layer][0]]])
            for j in range(1, phase_width):
                selected = selected + ilist.IList([reg[masks[layer][j]]])
            squin.broadcast.t(selected)
        squin.broadcast.h(reg)
        return squin.broadcast.measure(reg)

    return parity_iqp


def get_iqp_layout(n_qubits: int = 64, layout_type: str = "hypercube") -> Layout:
    """Return one (zone, row, column) address per slot, in circuit-qubit order."""
    _hypercube_dimension(n_qubits)
    if layout_type != "hypercube":
        raise ValueError(f"Unknown IQP layout type: {layout_type!r}")
    if n_qubits > 64:
        raise ValueError("The Gemini hypercube layout supports up to 64 qubits")
    words = get_arch_spec().words
    slots = []
    for i in range(n_qubits):
        # Each group of eight occupies an even-numbered word in the storage zone.
        col, row = words[2 * (i // 8)].sites[i % 8]
        slots.append(((0, row, col),))
    return tuple(slots)


def _pin_before(location: LocationAddress, before: ir.Statement) -> arch.stmts.NewAt:
    constants = [
        py.Constant(value)
        for value in (location.zone_id, location.word_id, location.site_id)
    ]
    for const in constants:
        const.insert_before(before)
    pin = arch.stmts.NewAt(*(const.result for const in constants))
    pin.insert_before(before)
    return pin


def _bind_register(
    method: ir.Method, reg: ir.BlockArgument, pins: list[LocationAddress]
) -> None:
    assert isinstance(method.code, func.Function)
    block = reg.block
    first = block.first_stmt
    assert first is not None
    pins_for_reg = [_pin_before(location, first) for location in pins]
    register = ilist.New([pin.qubit for pin in pins_for_reg])
    if not register.result.type.is_subseteq(reg.type):
        raise ValueError("Layout does not match the type or width of reg")
    register.insert_before(first)
    reg.replace_by(register.result)
    block.args.delete(reg)
    method.nargs -= 1
    method.arg_names = [arg for arg in method.arg_names or () if arg != "reg"]
    method.code.slots = tuple(name for name in method.code.slots if name != "reg")
    method.code.signature = func.Signature(method.arg_types, method.return_type)
    block.args[0].type = method.self_type

    # TaskBuilder binds arguments using the Python signature as well as the IR.
    if method.py_func is not None:
        signature = inspect.signature(method.py_func)

        @wraps(method.py_func)
        def placed(*args, **kwargs):
            return method(*args, **kwargs)

        setattr(
            placed,
            "__signature__",
            signature.replace(
                parameters=[p for p in signature.parameters.values() if p.name != "reg"]
            ),
        )
        method.py_func = placed


def compose_layout_circuit(
    layout: Layout, circuit: ir.Method, *, arch_spec: ArchSpec | None = None
) -> ir.Method:
    """Pin a copy of a SQuIN circuit to grid sites in slot-major order.

    Addresses within each slot are consumed in order. A circuit may allocate
    its own qubits in a statically unrollable sequence, or receive them in an
    IList argument named reg. Other arguments and the return value are preserved.
    """
    spec = get_arch_spec() if arch_spec is None else arch_spec
    coordinates = [coord for slot in layout for coord in slot]
    if not coordinates:
        raise ValueError("The layout must contain at least one address")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("The layout contains a duplicate grid address")

    pins = []
    for coord in coordinates:
        if len(coord) != 3 or any(type(i) is not int or i < 0 for i in coord):
            raise ValueError(f"Invalid (zone, row, column) address: {coord!r}")
        location = spec.location_at(*coord)
        if location is None:
            raise ValueError(f"No site at grid address {coord}")
        pins.append(location)

    method = circuit.similar(dialects=physical_kernel)
    block = method.callable_region.blocks[0]
    reg = next((arg for arg in method.args if arg.name == "reg"), None)

    if reg is not None:
        _bind_register(method, reg, pins)

    check_call_graph(method)
    AggressiveUnroll(method.dialects).fixpoint(method)
    allocations = [
        stmt
        for stmt in method.callable_region.walk()
        if isinstance(stmt, (qubit.stmts.New, arch.stmts.NewAt))
    ]
    # Higher-order instructions such as ilist.map can still hide allocations
    # when a register size or loop bound is supplied at run time.
    if CallGraph(method).callees(method) or any(
        isinstance(stmt, func.Call) for stmt in method.callable_region.walk()
    ):
        raise ValueError(
            "Layout composition requires statically unrollable kernel calls and allocations"
        )
    if len(method.callable_region.blocks) != 1 or any(
        stmt.parent_block is not block for stmt in allocations
    ):
        raise ValueError("Layout composition requires a static allocation order")
    if len(allocations) != len(pins):
        raise ValueError(
            f"Layout has {len(pins)} sites but the circuit allocates {len(allocations)} qubits"
        )
    if reg is None:
        for old, location in zip(allocations, pins, strict=True):
            pin = _pin_before(location, old)
            old.results[0].replace_by(pin.qubit)
            old.delete()

    passes.Fold(method.dialects).fixpoint(method)
    method.verify()
    return method
