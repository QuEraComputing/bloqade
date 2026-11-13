"""Tests for the ``move.LogicalInitialize`` -> Gemini state-injection bridge."""

from typing import Any

import pytest
import bloqade.squin as squin
from kirin import ir, rewrite
from kirin.analysis import forward
from kirin.dialects import py, ilist
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.arch.gemini import logical, physical
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.rewrite.move2squin import gates
from bloqade.lanes.arch.gemini.state_injection import (
    N_ROWS,
    CZWindow,
    resolve_col_group_and_row,
    make_state_injection_args_getter,
    theta_phi_to_axis_rotation_angle,
    logical_initialize_to_state_injection_args,
)

import bloqade.gemini as gemini

WINDOW = CZWindow.from_bounds(-20.0, 20.0, keep_out_buffer=10.0)


def _const_value(ssa: ir.SSAValue) -> Any:
    owner = ssa.owner
    assert isinstance(owner, py.Constant), owner
    return owner.value.unwrap()


def _ilist_values(ssa: ir.SSAValue) -> tuple[ir.SSAValue, ...]:
    owner = ssa.owner
    assert isinstance(owner, ilist.New), owner
    return tuple(owner.values)


def _mask(ssa: ir.SSAValue) -> list[bool]:
    return list(_const_value(ssa))


def _make_node(
    block: ir.Block,
    locations: tuple[LocationAddress, ...],
    thetas: tuple[ir.SSAValue, ...],
    phis: tuple[ir.SSAValue, ...],
) -> move.LogicalInitialize:
    lams = tuple(ir.TestValue() for _ in locations)
    node = move.LogicalInitialize(
        current_state=ir.TestValue(),
        thetas=thetas,
        phis=phis,
        lams=lams,
        location_addresses=locations,
    )
    block.stmts.append(node)
    return node


# --- geometry -----------------------------------------------------------------


@pytest.mark.parametrize(
    "word_id, expected",
    [
        (0, (0, 0)),  # column group 0, left site
        (2, (1, 0)),  # column group 1, left site
        (4, (0, 1)),
        (10, (1, 2)),
        (16, (0, 4)),
        (18, (1, 4)),
    ],
)
def test_resolve_col_group_and_row_logical_spec(word_id, expected):
    spec = logical.get_arch_spec()
    assert resolve_col_group_and_row(spec, LocationAddress(word_id, 0)) == expected


@pytest.mark.parametrize("word_id", [1, 3, 5, 19])
def test_resolve_col_group_and_row_rejects_right_site(word_id):
    """The pulse only addresses left sites; a right site must not alias its
    left partner's slot."""
    with pytest.raises(ValueError, match="right site"):
        resolve_col_group_and_row(logical.get_arch_spec(), LocationAddress(word_id, 0))


@pytest.mark.parametrize(
    "location", [LocationAddress(0, 0), LocationAddress(0, 7), LocationAddress(6, 3)]
)
def test_resolve_col_group_and_row_rejects_physical_spec(location):
    with pytest.raises(ValueError, match="Gemini logical grid"):
        resolve_col_group_and_row(physical.get_arch_spec(), location)


def test_resolve_col_group_and_row_rejects_invalid_location():
    with pytest.raises(ValueError):
        resolve_col_group_and_row(logical.get_arch_spec(), LocationAddress(99, 0))


def test_resolve_col_group_and_row_rejects_unknown_zone():
    with pytest.raises(ValueError, match="zone"):
        resolve_col_group_and_row(logical.get_arch_spec(), LocationAddress(0, 0, 1))


# --- angle convention ---------------------------------------------------------


def test_rotation_angle_is_theta_in_turns_and_axis_is_phi_plus_quarter():
    """Logical |1> is theta = 0.5 turns; the bridge must pass it through
    unchanged (the compiler-services bridge divided by 2*pi here)."""
    block = ir.Block()
    block.stmts.append(theta := py.Constant(0.5))
    block.stmts.append(phi := py.Constant(0.0))
    node = _make_node(block, (LocationAddress(0, 0),), (theta.result,), (phi.result,))

    args = logical_initialize_to_state_injection_args(
        node, logical.get_arch_spec(), WINDOW
    )
    _, axis_col0, rotation_col0, *_ = args

    rotation = _ilist_values(rotation_col0)[0]
    assert rotation is theta.result
    assert _const_value(rotation) == 0.5

    axis = _ilist_values(axis_col0)[0]
    add = axis.owner
    assert isinstance(add, py.Add)
    assert add.lhs is phi.result
    assert _const_value(add.lhs) + _const_value(add.rhs) == 0.25


def test_shared_phi_reuses_axis_angle_statement():
    block = ir.Block()
    theta = ir.TestValue()
    phi = ir.TestValue()
    node = _make_node(
        block,
        (LocationAddress(0, 0), LocationAddress(4, 0)),
        (theta, theta),
        (phi, phi),
    )

    args = logical_initialize_to_state_injection_args(
        node, logical.get_arch_spec(), WINDOW
    )
    axis_values = _ilist_values(args[1])
    assert axis_values[0] is axis_values[1]
    assert len([s for s in block.stmts if isinstance(s, py.Add)]) == 1


def test_standalone_helper_inserts_own_quarter_turn():
    block = ir.Block()
    block.stmts.append(theta := py.Constant(0.125))
    block.stmts.append(phi := py.Constant(0.3))
    block.stmts.append(anchor := py.Constant(0.0))

    axis, rotation = theta_phi_to_axis_rotation_angle(theta.result, phi.result, anchor)

    assert rotation is theta.result
    add = axis.owner
    assert isinstance(add, py.Add)
    quarter = add.rhs.owner
    assert isinstance(quarter, py.Constant) and quarter.value.unwrap() == 0.25
    # Both new statements land before the insertion point, constant first.
    stmts = list(block.stmts)
    assert stmts.index(quarter) < stmts.index(add) < stmts.index(anchor)


def _eval(ssa: ir.SSAValue) -> float:
    owner = ssa.owner
    if isinstance(owner, py.Constant):
        return owner.value.unwrap()
    if isinstance(owner, py.Add):
        return _eval(owner.lhs) + _eval(owner.rhs)
    if isinstance(owner, py.Sub):
        return _eval(owner.lhs) - _eval(owner.rhs)
    raise AssertionError(f"cannot evaluate {owner!r}")


@pytest.mark.parametrize("theta_value, phi_value", [(0.5, 0.0), (0.2, 0.35)])
def test_axis_angle_round_trips_through_move2squin_global_r(theta_value, phi_value):
    """Pin the bridge to move2squin's GlobalR mapping (phi = axis - 0.25): feeding
    the bridge's (axis, rotation) back through it must recover (theta, phi)."""
    block = ir.Block()
    block.stmts.append(theta := py.Constant(theta_value))
    block.stmts.append(phi := py.Constant(phi_value))
    gate = move.GlobalR(
        current_state=ir.TestValue(),
        rotation_angle=theta.result,
        axis_angle=theta.result,  # placeholder, replaced below
    )
    block.stmts.append(gate)
    axis, rotation = theta_phi_to_axis_rotation_angle(theta.result, phi.result, gate)
    gate.axis_angle = axis
    gate.rotation_angle = rotation

    frame: forward.ForwardFrame[atom.MoveExecution] = forward.ForwardFrame(
        gate,
        entries={
            gate.result: atom.AtomState(
                atom.AtomStateData.new({0: LocationAddress(0, 0)})
            )
        },
    )
    rule = gates.InsertGates(
        arch_spec=logical.get_arch_spec(),
        physical_ssa_values={0: ir.TestValue()},  # type: ignore
        move_exec_analysis=frame,
    )
    rewrite.Walk(rule).rewrite(block)

    (u3,) = [s for s in block.stmts if isinstance(s, squin.gate.stmts.U3)]
    assert _eval(u3.theta) == pytest.approx(theta_value)
    assert _eval(u3.phi) == pytest.approx(phi_value)


# --- argument layout ----------------------------------------------------------


def test_args_layout_covers_both_column_groups():
    block = ir.Block()
    locations = (
        LocationAddress(0, 0),  # group 0, row 0
        LocationAddress(2, 0),  # group 1, row 0
        LocationAddress(8, 0),  # group 0, row 2
        LocationAddress(18, 0),  # group 1, row 4
    )
    thetas = tuple(ir.TestValue() for _ in locations)
    phis = tuple(ir.TestValue() for _ in locations)
    node = _make_node(block, locations, thetas, phis)

    args = logical_initialize_to_state_injection_args(
        node, logical.get_arch_spec(), WINDOW
    )
    assert len(args) == 10
    mask0, axis0, rot0, mask1, axis1, rot1, ymin, ymax, ymin_ko, ymax_ko = args

    assert _mask(mask0) == [True, False, True, False, False]
    assert _mask(mask1) == [True, False, False, False, True]

    rot0_values = _ilist_values(rot0)
    rot1_values = _ilist_values(rot1)
    assert rot0_values[0] is thetas[0]
    assert rot0_values[2] is thetas[2]
    assert rot1_values[0] is thetas[1]
    assert rot1_values[4] is thetas[3]

    # Rows the mask switches off share a single 0.0 filler constant.
    fillers = {rot0_values[1], rot0_values[3], rot0_values[4]}
    fillers |= {rot1_values[1], rot1_values[2], rot1_values[3]}
    fillers |= {_ilist_values(axis0)[1], _ilist_values(axis1)[2]}
    assert len(fillers) == 1
    assert _const_value(next(iter(fillers))) == 0.0

    assert len(_ilist_values(axis0)) == len(_ilist_values(axis1)) == N_ROWS
    assert (
        _const_value(ymin),
        _const_value(ymax),
        _const_value(ymin_ko),
        _const_value(ymax_ko),
    ) == (-20.0, 20.0, -30.0, 30.0)

    # Everything the bridge produced sits in the block ahead of the node.
    stmts = list(block.stmts)
    assert stmts[-1] is node
    for arg in args:
        assert arg.owner in stmts[:-1]


def test_duplicate_slot_raises():
    block = ir.Block()
    node = _make_node(
        block,
        (LocationAddress(0, 0), LocationAddress(0, 0)),
        (ir.TestValue(), ir.TestValue()),
        (ir.TestValue(), ir.TestValue()),
    )
    with pytest.raises(ValueError, match="both resolve to column group 0, row 0"):
        logical_initialize_to_state_injection_args(
            node, logical.get_arch_spec(), WINDOW
        )


def test_rejected_node_leaves_block_untouched():
    """A failure on a later address must not leave IR from earlier ones."""
    block = ir.Block()
    node = _make_node(
        block,
        (LocationAddress(0, 0), LocationAddress(1, 0)),  # second is a right site
        (ir.TestValue(), ir.TestValue()),
        (ir.TestValue(), ir.TestValue()),
    )
    with pytest.raises(ValueError, match="right site"):
        logical_initialize_to_state_injection_args(
            node, logical.get_arch_spec(), WINDOW
        )
    assert list(block.stmts) == [node]


def test_empty_node_raises():
    block = ir.Block()
    node = _make_node(block, (), (), ())
    with pytest.raises(ValueError, match="no locations"):
        logical_initialize_to_state_injection_args(
            node, logical.get_arch_spec(), WINDOW
        )


def test_multi_zone_node_raises():
    block = ir.Block()
    node = _make_node(
        block,
        (LocationAddress(0, 0, 0), LocationAddress(0, 0, 1)),
        (ir.TestValue(), ir.TestValue()),
        (ir.TestValue(), ir.TestValue()),
    )
    with pytest.raises(ValueError, match="single zone"):
        logical_initialize_to_state_injection_args(
            node, logical.get_arch_spec(), WINDOW
        )


def test_mismatched_argument_lengths_raise():
    block = ir.Block()
    node = _make_node(
        block,
        (LocationAddress(0, 0), LocationAddress(2, 0)),
        (ir.TestValue(),),
        (ir.TestValue(), ir.TestValue()),
    )
    with pytest.raises(ValueError):
        logical_initialize_to_state_injection_args(
            node, logical.get_arch_spec(), WINDOW
        )


def test_row_beyond_kernel_rows_raises():
    block = ir.Block()
    node = _make_node(
        block, (LocationAddress(18, 0),), (ir.TestValue(),), (ir.TestValue(),)
    )
    with pytest.raises(ValueError, match="only addresses 2 rows"):
        logical_initialize_to_state_injection_args(
            node, logical.get_arch_spec(), WINDOW, n_rows=2
        )


def test_malformed_window_raises():
    block = ir.Block()
    node = _make_node(
        block, (LocationAddress(0, 0),), (ir.TestValue(),), (ir.TestValue(),)
    )
    with pytest.raises(ValueError, match="CZWindow"):
        logical_initialize_to_state_injection_args(
            node, logical.get_arch_spec(), CZWindow(0.0, 40.0, 5.0, 50.0)
        )


def test_make_getter_matches_direct_call():
    block = ir.Block()
    node = _make_node(
        block, (LocationAddress(0, 0),), (ir.TestValue(),), (ir.TestValue(),)
    )
    getter = make_state_injection_args_getter(logical.get_arch_spec(), WINDOW)
    args = getter(node)
    assert len(args) == 10
    assert _mask(args[0]) == [True, False, False, False, False]


# --- end to end ---------------------------------------------------------------


def test_bridge_applies_to_pipeline_output():
    """Every LogicalInitialize the logical pipeline emits can be bridged, and
    the masks account for exactly the qubits the node initialises."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(3)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        squin.cx(reg[1], reg[2])
        gemini.logical.terminal_measure(reg)

    spec = logical.get_arch_spec()
    out = LogicalPipeline(
        arch_spec=spec, transversal_rewrite=True, simulation=False
    ).emit(kernel)

    inits = [
        s for s in out.callable_region.walk() if isinstance(s, move.LogicalInitialize)
    ]
    assert inits, "pipeline emitted no move.LogicalInitialize"

    for node in inits:
        args = logical_initialize_to_state_injection_args(node, spec, WINDOW)
        assert len(args) == 10
        active = sum(_mask(args[0])) + sum(_mask(args[3]))
        assert active == len(node.location_addresses)
        for arg in args:
            owner = arg.owner
            assert isinstance(owner, ir.Statement) and owner.parent is not None

    out.verify()
