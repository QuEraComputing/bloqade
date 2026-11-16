"""Stepping through ``move.PhysicalInitialize`` in the move debuggers.

``move.PhysicalInitialize`` stands for an initialization kernel. Given that
kernel, the debuggers show its gates and, for every CZ, one move bringing the
paired atoms together, the pulse, and one move back.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from matplotlib import pyplot as plt
from kirin.dialects import ilist
from bloqade.lanes.prelude import kernel
from bloqade.lanes.dialects import move
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.word import Word
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    Mode as RustMode,
    Zone as RustZone,
    LocationAddress as RustLocAddr,
)
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.arch.gemini.logical.upstream import (
    steane7_initialize_broadcast,
    steane7_initialize_with_noise,
)

from bloqade import qubit, squin
from bloqade.gemini.visualize import debug, plotly_debug
from bloqade.gemini.visualize.artist import (
    INITIALIZE_MOVE_LABEL,
    DebugStep,
    MovingAtomsScatter,
    _hop_path,
    get_drawer,
    get_state_artist,
    render_generator,
    collect_debug_steps,
)
from bloqade.gemini.visualize.initialize import (
    InitializeR,
    InitializeCZ,
    InitializeRz,
    InitializeOperation,
    InitializeKernelError,
    _merge_rotations,
    initialize_operations,
)

THETA, PHI, LAM = 0.11, 0.07, 0.31  # turns

# The debuggers move the controls of each CZ. The noisy Steane kernel lists the
# movers first in every CZ; the clean one only pairs the qubits, so the noisy
# kernel is the one to step through.
_, STEANE_INITIALIZE = steane7_initialize_with_noise()

# The Steane kernel on one block, after its U3 on site 6.
STEANE_OPERATIONS: list[InitializeOperation] = [
    InitializeR(0.25, -0.25, (0, 1, 2, 3, 4, 5)),
    InitializeCZ((2, 4, 6), (1, 3, 5)),
    InitializeR(0.25, 0.25, (6,)),
    InitializeCZ((0, 2, 4), (3, 5, 6)),
    InitializeR(0.25, 0.25, (2, 3, 4, 5, 6)),
    InitializeCZ((1, 3, 5), (0, 2, 4)),
    InitializeR(0.25, 0.25, (1, 2, 4)),
    InitializeR(0.0, 0.5, (3,)),
    InitializeRz(0.5, (1, 5)),
]


def _block(word_id: int) -> tuple[LocationAddress, ...]:
    return tuple(LocationAddress(word_id, site_id) for site_id in range(7))


# Kirin lowers a call's keyword arguments as constants the body closes over.
BLOCKS = (_block(0), _block(2), _block(4))
FILLED = BLOCKS[0] + BLOCKS[1] + BLOCKS[2]

# Words 0 and 1 are CZ partners, so the sites the movers of one block would
# travel to are occupied by the other block.
PARTNER_BLOCKS = (_block(0), _block(1))
PARTNER_FILLED = PARTNER_BLOCKS[0] + PARTNER_BLOCKS[1]


@kernel
def initialize_program():
    state = move.load()
    state = move.fill(state, location_addresses=FILLED)
    state = move.physical_initialize(
        state,
        (THETA, 0.0, 0.0),
        (PHI, 0.0, 0.0),
        (LAM, 0.0, 0.0),
        location_addresses=BLOCKS,
    )
    move.store(state)


@kernel
def crowded_initialize_program():
    state = move.load()
    state = move.fill(state, location_addresses=PARTNER_FILLED)
    state = move.physical_initialize(
        state,
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        location_addresses=PARTNER_BLOCKS,
    )
    move.store(state)


@pytest.fixture(scope="module")
def arch_spec() -> ArchSpec:
    return get_arch_spec()


@pytest.fixture(scope="module")
def steps(arch_spec: ArchSpec) -> list[DebugStep]:
    return collect_debug_steps(initialize_program, arch_spec, STEANE_INITIALIZE)


def _kinds(steps: list[DebugStep]) -> list[str]:
    return [type(step.statement).__name__ for step in steps]


# ── initialize_operations ────────────────────────────────────────────


def test_operations_follow_the_steane_kernel() -> None:
    operations = initialize_operations(STEANE_INITIALIZE, [0.0], [0.0], [0.0], 7)

    # The U3 of a block prepared in |0> is three rotations by zero.
    assert operations == STEANE_OPERATIONS


def test_noise_channels_are_not_operations() -> None:
    _, noisy = steane7_initialize_with_noise(local_px=0.01, mover_pz=0.02)

    assert initialize_operations(noisy, [0.0], [0.0], [0.0], 7) == STEANE_OPERATIONS


def test_clean_kernel_applies_the_same_gates_to_the_same_pairs() -> None:
    """The clean and noisy kernels differ at most in which side of a CZ is named first."""

    def unordered(operations: list[InitializeOperation]) -> list[object]:
        return [
            (
                {frozenset(pair) for pair in zip(op.controls, op.targets)}
                if isinstance(op, InitializeCZ)
                else op
            )
            for op in operations
        ]

    clean = initialize_operations(steane7_initialize_broadcast, [0.0], [0.0], [0.0], 7)

    assert unordered(clean) == unordered(STEANE_OPERATIONS)


def test_injected_state_is_a_u3_on_the_injection_site() -> None:
    operations = initialize_operations(
        STEANE_INITIALIZE, [THETA, 0.0], [PHI, 0.0], [LAM, 0.0], 7
    )

    # U3 = Rz(phi) Ry(theta) Rz(lam); the second block stays in |0>.
    assert operations[:3] == [
        InitializeRz(LAM, (6,)),
        InitializeR(0.25, THETA, (6,)),
        InitializeRz(PHI, (6,)),
    ]
    # Qubits are numbered block by block.
    assert operations[3] == InitializeR(
        0.25, -0.25, (0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12)
    )


def test_blocks_injected_alike_share_their_operations() -> None:
    operations = initialize_operations(
        STEANE_INITIALIZE, [THETA, THETA], [PHI, PHI], [LAM, LAM], 7
    )

    assert operations[:3] == [
        InitializeRz(LAM, (6, 13)),
        InitializeR(0.25, THETA, (6, 13)),
        InitializeRz(PHI, (6, 13)),
    ]


def test_merge_rotations_only_slides_past_independent_operations() -> None:
    assert _merge_rotations(
        [
            InitializeRz(0.5, (0,)),
            InitializeR(0.25, 0.25, (0,)),
            InitializeRz(0.5, (1,)),  # slides back past the R on qubit 0
            InitializeRz(0.5, (0,)),  # blocked by the R on qubit 0
            InitializeRz(0.0, (2,)),  # identity
            InitializeCZ((0,), (1,)),
            InitializeRz(0.5, (2,)),  # blocked by the CZ
        ]
    ) == [
        InitializeRz(0.5, (0, 1)),
        InitializeR(0.25, 0.25, (0,)),
        InitializeRz(0.5, (0,)),
        InitializeCZ((0,), (1,)),
        InitializeRz(0.5, (2,)),
    ]


def test_kernel_with_a_branch_cannot_be_listed() -> None:
    @squin.kernel
    def branching(
        theta: ilist.IList[float, Any],
        phi: ilist.IList[float, Any],
        lam: ilist.IList[float, Any],
        qubits: ilist.IList[ilist.IList[qubit.Qubit, Any], Any],
    ):
        if squin.measure(qubits[0][0]):
            squin.x(qubits[0][1])

    with pytest.raises(InitializeKernelError, match="fixed sequence of gates"):
        initialize_operations(branching, [0.0], [0.0], [0.0], 2)


# ── collect_debug_steps ──────────────────────────────────────────────


def test_initialize_is_one_step_without_a_kernel(arch_spec: ArchSpec) -> None:
    steps = collect_debug_steps(initialize_program, arch_spec)

    assert _kinds(steps) == ["Load", "Fill", "PhysicalInitialize"]
    assert all(step.moves is None for step in steps)


def test_initialize_steps_through_the_kernel(steps: list[DebugStep]) -> None:
    assert [step.title.split(": ", 1)[1] for step in steps] == [
        "Load()",
        "Fill()",
        f"PhysicalInitialize > LocalRz({LAM})",
        f"PhysicalInitialize > LocalR(0.25, {THETA})",
        f"PhysicalInitialize > LocalRz({PHI})",
        "PhysicalInitialize > LocalR(0.25, -0.25)",
        "PhysicalInitialize > Move()",
        "PhysicalInitialize > CZ()",
        "PhysicalInitialize > Move()",
        "PhysicalInitialize > LocalR(0.25, 0.25)",
        "PhysicalInitialize > Move()",
        "PhysicalInitialize > CZ()",
        "PhysicalInitialize > Move()",
        "PhysicalInitialize > LocalR(0.25, 0.25)",
        "PhysicalInitialize > Move()",
        "PhysicalInitialize > CZ()",
        "PhysicalInitialize > Move()",
        "PhysicalInitialize > LocalR(0.25, 0.25)",
        "PhysicalInitialize > LocalR(0.0, 0.5)",
        "PhysicalInitialize > LocalRz(0.5)",
    ]
    assert steps[-1].title.startswith(f"Step {len(steps)} / {len(steps)}: ")


def test_gate_steps_act_on_the_sites_of_every_block(steps: list[DebugStep]) -> None:
    injected, sqrt_y_adj = steps[2], steps[5]

    # Only the first block is injected with a state other than |0>.
    assert isinstance(injected.statement, move.LocalRz)
    assert injected.statement.location_addresses == (BLOCKS[0][6],)
    assert injected.parameter_values == (LAM,)

    assert isinstance(sqrt_y_adj.statement, move.LocalR)
    assert set(sqrt_y_adj.statement.location_addresses) == {
        block[site] for block in BLOCKS for site in range(6)
    }
    assert sqrt_y_adj.parameter_values == (0.25, -0.25)
    # A gate step shows no route, whatever the state it sits on recorded.
    assert sqrt_y_adj.moves == ()


def test_each_cz_is_one_move_in_a_pulse_and_one_move_back(
    arch_spec: ArchSpec, steps: list[DebugStep]
) -> None:
    home = dict(steps[1].state.data.qubit_to_locations)
    qubit_at = {location: qubit_id for qubit_id, location in home.items()}
    cz_operations = [
        operation
        for operation in STEANE_OPERATIONS
        if isinstance(operation, InitializeCZ)
    ]
    cz_indices = [
        index for index, step in enumerate(steps) if isinstance(step.statement, move.CZ)
    ]
    assert len(cz_indices) == len(cz_operations) == 3

    for cz_index, operation in zip(cz_indices, cz_operations):
        move_in, cz, move_out = steps[cz_index - 1 : cz_index + 2]
        movers = {
            qubit_at[block[control]]: qubit_at[block[target]]
            for block in BLOCKS
            for control, target in zip(operation.controls, operation.targets)
        }

        # One move puts every control on the CZ partner site of its target.
        assert isinstance(move_in.statement, move.PhysicalInitialize)
        assert move_in.moves is not None
        assert {path.qubit_id for path in move_in.moves} == set(movers)
        for path in move_in.moves:
            assert path.source == home[path.qubit_id]
            assert path.destination == arch_spec.get_cz_partner(
                home[movers[path.qubit_id]]
            )
            assert path.label == INITIALIZE_MOVE_LABEL
            assert path.waypoints[0] == arch_spec.get_position(path.source)
            assert path.waypoints[-1] == arch_spec.get_position(path.destination)
            assert (
                move_in.state.data.qubit_to_locations[path.qubit_id] == path.destination
            )

        # The pulse then pairs exactly the qubits the kernel pairs.
        assert isinstance(cz.statement, move.CZ)
        assert cz.state is move_in.state
        controls, targets, _ = cz.state.data.get_qubit_pairing(
            cz.statement.zone_address, arch_spec
        )
        assert {frozenset(pair) for pair in zip(controls, targets)} == {
            frozenset(pair) for pair in movers.items()
        }

        # One move takes every control back to where it started.
        assert move_out.moves is not None
        assert {
            (path.qubit_id, path.source, path.destination) for path in move_out.moves
        } == {(path.qubit_id, path.destination, path.source) for path in move_in.moves}
        assert dict(move_out.state.data.qubit_to_locations) == home


def test_expansion_leaves_the_program_untouched(arch_spec: ArchSpec) -> None:
    before = initialize_program.code.print_str()

    collect_debug_steps(initialize_program, arch_spec, STEANE_INITIALIZE)

    assert initialize_program.code.print_str() == before
    for stmt in initialize_program.callable_region.walk():
        if isinstance(stmt, (move.Fill, move.PhysicalInitialize)):
            # Nothing built for the debugger is left using the program's values.
            assert all(use.stmt.parent is not None for use in stmt.result.uses)


def test_initialize_that_cannot_be_laid_out_stays_one_step(arch_spec: ArchSpec) -> None:
    with pytest.warns(UserWarning, match="single step.*already occupied"):
        steps = collect_debug_steps(
            crowded_initialize_program, arch_spec, STEANE_INITIALIZE
        )

    assert _kinds(steps) == ["Load", "Fill", "PhysicalInitialize"]


def test_hop_is_a_straight_line_in_a_single_row_zone() -> None:
    word = Word(sites=((0, 0), (1, 0)))
    zone = RustZone(
        name="test",
        grid=RustGrid.from_positions([0.0, 1.0], [0.0]),
        site_buses=[],
        word_buses=[],
        words_with_site_buses=[],
        sites_with_word_buses=[],
        entangling_pairs=[],
    )
    mode = RustMode(
        name="all",
        zones=[0],
        bitstring_order=[RustLocAddr(0, 0, 0), RustLocAddr(0, 0, 1)],
    )
    one_row = ArchSpec.from_components(words=(word,), zones=(zone,), modes=[mode])

    path = _hop_path(one_row, 0, LocationAddress(0, 0), LocationAddress(0, 1))

    assert path.waypoints == ((0.0, 0.0), (1.0, 0.0))


# ── renderers ────────────────────────────────────────────────────────


def test_matplotlib_debuggers_render_every_step(
    arch_spec: ArchSpec, steps: list[DebugStep]
) -> None:
    figure, ax = plt.subplots()
    draw, num_steps = get_drawer(
        initialize_program,
        arch_spec,
        ax,
        initialize_kernel=STEANE_INITIALIZE,
    )
    assert num_steps == len(steps)
    for step_index in range(num_steps):
        ax.cla()
        draw(step_index)
        assert ax.get_title() == steps[step_index].title
    plt.close(figure)

    figure, ax = plt.subplots()
    get_renderer, num_steps = render_generator(
        initialize_program,
        arch_spec,
        ax,
        initialize_kernel=STEANE_INITIALIZE,
    )
    assert num_steps == len(steps)
    for step_index in range(num_steps):
        ax.cla()
        num_frames, renderer = get_renderer(step_index)
        for frame in (0, num_frames // 2, num_frames):
            renderer(frame)
    plt.close(figure)


def test_move_renderer_carries_the_atoms_along_the_explicit_routes(
    arch_spec: ArchSpec, steps: list[DebugStep]
) -> None:
    step = next(step for step in steps if step.moves)
    assert step.moves is not None
    figure, ax = plt.subplots()
    state_artist = get_state_artist(arch_spec, ax)

    renderer = state_artist.move_renderer(step.state, speed=2.0, moves=step.moves)

    assert renderer is not None
    (moving,) = [
        part for part in renderer.renderers if isinstance(part, MovingAtomsScatter)
    ]
    assert sorted(moving.moving_atom_labels) == sorted(
        path.qubit_id for path in step.moves
    )
    # The state itself records no lanes, so without the routes nothing moves.
    assert state_artist.move_renderer(step.state, speed=2.0) is None

    renderer.update(renderer.total_time)
    arrived = {
        (float(x), float(y)) for x, y in np.asarray(moving.scatter.get_offsets())
    }
    assert arrived == {arch_spec.get_position(path.destination) for path in step.moves}
    plt.close(figure)


def test_plotly_debugger_names_the_initialize_moves(
    plotly, arch_spec: ArchSpec, steps: list[DebugStep]
) -> None:
    figure = plotly_debug.plotly_debugger(
        initialize_program,
        arch_spec,
        show=False,
        initialize_kernel=STEANE_INITIALIZE,
    )
    assert figure is not None
    assert [frame.layout.title.text for frame in figure.frames] == [
        step.title for step in steps
    ]

    def route_labels(frame) -> set[str]:
        return {
            segment["busLabel"]
            for trace in frame.data
            if (trace.meta or {}).get("bloqadeTraceKind") == "movePath"
            for segment in trace.meta["segments"]
        }

    for step, frame in zip(steps, figure.frames):
        expected = {INITIALIZE_MOVE_LABEL} if step.moves else set()
        assert route_labels(frame) == expected


@pytest.mark.parametrize(
    ("debugger_name", "builder_name"),
    [("debugger", "get_drawer"), ("animated_debugger", "render_generator")],
)
def test_matplotlib_debuggers_pass_the_kernel_on(
    monkeypatch, arch_spec: ArchSpec, debugger_name: str, builder_name: str
) -> None:
    passed = {}

    def builder(*_args, **kwargs):
        passed.update(kwargs)
        return (lambda _index: (0, lambda _frame: None)), 0

    monkeypatch.setattr(debug, builder_name, builder)
    figure, ax = plt.subplots()

    getattr(debug, debugger_name)(
        initialize_program,
        arch_spec,
        interactive=False,
        ax=ax,
        initialize_kernel=STEANE_INITIALIZE,
    )

    assert passed == {"initialize_kernel": STEANE_INITIALIZE}
    plt.close(figure)
