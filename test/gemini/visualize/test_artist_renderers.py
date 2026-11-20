"""Coverage for the two matplotlib renderers built on ``collect_debug_steps``.

``get_drawer`` (the static debugger) and ``render_generator`` (the animated
one) were rewritten to consume :func:`collect_debug_steps` instead of
interpreting the kernel and formatting statement text themselves. Neither had a
direct test, so the rewrite was carried entirely by the Plotly tests, which
drive a different renderer over the same steps.

One kernel reaches every branch of both renderers in a single interpretation: a
gate step (dispatched through the ``methods`` table), a transport step (played
frame by frame), and steps that are neither.
"""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from matplotlib import pyplot as plt, animation
from bloqade.lanes.prelude import kernel
from bloqade.lanes.dialects import move
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.word import Word
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    Mode as RustMode,
    Zone as RustZone,
    SiteBus,
    LocationAddress as RustLocAddr,
)
from bloqade.lanes.bytecode.encoding import Direction, SiteLaneAddress

from bloqade.gemini.visualize import debug as debug_module
from bloqade.gemini.visualize.debug import debugger, animated_debugger
from bloqade.gemini.visualize.artist import (
    get_drawer,
    render_generator,
    collect_debug_steps,
)

# Kirin lowers a call's keyword arguments as statement operands, so a
# ``SiteLaneAddress`` cannot be built inside a kernel body; it is a constant the
# body closes over instead.
_SITE_LANE = SiteLaneAddress(
    word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD, zone_id=0
)


@pytest.fixture
def transport_arch_spec() -> ArchSpec:
    """One word of two sites joined by a site bus with a curved path.

    The path carries an interior waypoint so the animated renderer has a real
    trajectory to interpolate; a straight two-point hop would not distinguish
    its transport branch from its static one.
    """
    word = Word(sites=((0, 0), (1, 0)))
    rust_zone = RustZone(
        name="test",
        grid=RustGrid.from_positions([0.0, 1.0], [0.0]),
        site_buses=[SiteBus(src=[0], dst=[1])],
        word_buses=[],
        words_with_site_buses=[0],
        sites_with_word_buses=[],
        entangling_pairs=[],
    )
    rust_mode = RustMode(
        name="all",
        zones=[0],
        bitstring_order=[RustLocAddr(0, 0, 0), RustLocAddr(0, 0, 1)],
    )
    return ArchSpec.from_components(
        words=(word,),
        zones=(rust_zone,),
        modes=[rust_mode],
        paths={_SITE_LANE: ((0.0, 0.0), (0.5, 0.75), (1.0, 0.0))},
    )


@kernel
def _gate_then_move_kernel():
    state = move.load()
    state = move.fill(state, location_addresses=(move.LocationAddress(0, 0, 0),))
    state = move.local_r(
        state, 0.25, 1.5, location_addresses=(move.LocationAddress(0, 0, 0),)
    )
    state = move.move(state, lanes=(_SITE_LANE,))
    move.end_measure(state, zone_addresses=(move.ZoneAddress(0),))


@kernel
def _no_atom_state_kernel():
    return 1


@pytest.fixture
def axes():
    figure, ax = plt.subplots()
    yield ax
    plt.close(figure)


def _step_kinds(arch_spec: ArchSpec) -> list[str]:
    return [
        type(step.statement).__name__
        for step in collect_debug_steps(_gate_then_move_kernel, arch_spec)
    ]


def test_kernel_reaches_every_renderer_branch(transport_arch_spec: ArchSpec) -> None:
    """Guard the fixture itself: a silent reshaping would hollow out the tests."""
    assert _step_kinds(transport_arch_spec) == [
        "Load",
        "Fill",
        "LocalR",
        "Move",
        "EndMeasure",
    ]


def test_static_debugger_exports_mp4(transport_arch_spec: ArchSpec, tmp_path) -> None:
    if not animation.writers.is_available("ffmpeg"):
        pytest.skip("FFmpeg is not installed")
    output = tmp_path / "static.mp4"
    debugger(
        _gate_then_move_kernel,
        transport_arch_spec,
        pause_time=0,
        to_mp4=output,
    )
    assert output.read_bytes()[4:8] == b"ftyp"

    with pytest.raises(FileExistsError, match="already exists"):
        debugger(
            _gate_then_move_kernel,
            transport_arch_spec,
            pause_time=0,
            to_mp4=output,
        )
    assert output.read_bytes()[4:8] == b"ftyp"


def test_animated_debugger_exports_mp4(transport_arch_spec: ArchSpec, tmp_path) -> None:
    if not animation.writers.is_available("ffmpeg"):
        pytest.skip("FFmpeg is not installed")
    output = tmp_path / "animated.mp4"
    animated_debugger(
        _gate_then_move_kernel,
        transport_arch_spec,
        fps=2,
        to_mp4=output,
    )
    assert output.read_bytes()[4:8] == b"ftyp"


@pytest.mark.parametrize("exporter", [debugger, animated_debugger])
def test_mp4_export_records_one_frame_without_steps(
    exporter, transport_arch_spec: ArchSpec, tmp_path, monkeypatch
) -> None:
    writer = Mock()
    writer.saving.return_value = nullcontext()
    monkeypatch.setattr(
        debug_module,
        "mp4_writer",
        lambda path, *, fps: (str(path), writer),
    )

    exporter(_no_atom_state_kernel, transport_arch_spec, to_mp4=tmp_path / "empty.mp4")

    writer.grab_frame.assert_called_once_with()


@pytest.mark.parametrize("exporter", [debugger, animated_debugger])
def test_mp4_export_removes_partial_file_on_failure(
    exporter, transport_arch_spec: ArchSpec, tmp_path, monkeypatch
) -> None:
    output = tmp_path / "partial.mp4"
    writer = Mock()

    def saving(*_args, **_kwargs):
        output.write_bytes(b"partial")
        return nullcontext()

    writer.saving.side_effect = saving
    writer.grab_frame.side_effect = RuntimeError("encoding failed")
    monkeypatch.setattr(
        debug_module,
        "mp4_writer",
        lambda path, *, fps: (str(path), writer),
    )

    with pytest.raises(RuntimeError, match="encoding failed"):
        exporter(_gate_then_move_kernel, transport_arch_spec, to_mp4=output)

    assert not output.exists()


# ── get_drawer ───────────────────────────────────────────────────


def test_get_drawer_titles_every_step(transport_arch_spec: ArchSpec, axes) -> None:
    """Each step draws, and its title is the one ``collect_debug_steps`` built."""
    steps = collect_debug_steps(_gate_then_move_kernel, transport_arch_spec)
    draw, num_steps = get_drawer(_gate_then_move_kernel, transport_arch_spec, axes)

    assert num_steps == len(steps)

    for step_index, step in enumerate(steps):
        draw(step_index)
        assert axes.get_title() == step.title

    # The gate step's title carries the operands resolved from the frame, which
    # is the behaviour that moved out of this module into ``collect_debug_steps``.
    local_r_title = next(
        step.title for step in steps if type(step.statement).__name__ == "LocalR"
    )
    assert local_r_title.endswith("LocalR(0.25, 1.5)")
    assert local_r_title.startswith(f"Step 3 / {num_steps}:")


def test_get_drawer_on_kernel_without_atom_states(
    transport_arch_spec: ArchSpec, axes
) -> None:
    """No steps means ``draw`` is a no-op rather than an ``IndexError``."""
    draw, num_steps = get_drawer(_no_atom_state_kernel, transport_arch_spec, axes)

    assert num_steps == 0
    draw(0)
    assert axes.get_title() == ""


# ── render_generator ─────────────────────────────────────────────


def test_render_generator_holds_gate_steps_for_a_fixed_duration(
    transport_arch_spec: ArchSpec, axes
) -> None:
    """A statement in the ``methods`` table renders once and holds."""
    get_renderer, num_steps = render_generator(
        _gate_then_move_kernel, transport_arch_spec, axes, fps=30
    )
    gate_index = _step_kinds(transport_arch_spec).index("LocalR")

    total_frames, advance = get_renderer(gate_index)

    assert num_steps == 5
    assert total_frames == 90  # 3 seconds at 30 fps
    # Name the branch, not just the frame count: a hold and a transport step can
    # coincide on duration, so the count alone would not tell them apart.
    assert advance.__name__ == "_no_op"
    assert advance(0) is None
    assert axes.get_title().endswith("LocalR(0.25, 1.5)")


def test_render_generator_animates_a_transport_step(
    transport_arch_spec: ArchSpec, axes
) -> None:
    """A ``Move`` step yields a frame count and a callable that interpolates."""
    get_renderer, _ = render_generator(
        _gate_then_move_kernel, transport_arch_spec, axes, fps=30
    )
    move_index = _step_kinds(transport_arch_spec).index("Move")

    total_frames, advance = get_renderer(move_index)

    assert advance.__name__ == "_move_renderer"
    # One second at 30 fps: the lane is short, so the 1s floor applies.
    assert total_frames == 30
    for frame in (0, total_frames // 2, total_frames):
        assert advance(frame) is None
    # Out-of-range frames are dropped rather than extrapolating off the path.
    assert advance(-1) is None
    assert advance(total_frames + 1) is None


def test_render_generator_holds_steps_that_neither_gate_nor_move(
    transport_arch_spec: ArchSpec, axes
) -> None:
    """``Fill`` has no gate renderer and no lanes, so it takes the hold path."""
    get_renderer, _ = render_generator(
        _gate_then_move_kernel, transport_arch_spec, axes, fps=10
    )
    fill_index = _step_kinds(transport_arch_spec).index("Fill")

    total_frames, advance = get_renderer(fill_index)

    assert advance.__name__ == "_no_op"
    assert total_frames == 30  # 3 seconds at 10 fps
    assert advance(0) is None


@pytest.mark.xfail(
    strict=True,
    reason=(
        "EndMeasure re-animates the preceding transport. collect_debug_steps "
        "records the statement's *input* state, which is the same AtomState "
        "object the preceding Move produced -- prev_lanes included -- so "
        "move_renderer builds a renderer for a move that already played. "
        "Delete this marker once the measurement step stops carrying the "
        "previous step's lanes."
    ),
)
def test_render_generator_does_not_replay_the_move_on_end_measure(
    transport_arch_spec: ArchSpec, axes
) -> None:
    """A measurement moves no atoms, so it should hold rather than animate."""
    get_renderer, _ = render_generator(
        _gate_then_move_kernel, transport_arch_spec, axes, fps=30
    )
    kinds = _step_kinds(transport_arch_spec)
    move_frames, _ = get_renderer(kinds.index("Move"))

    total_frames, advance = get_renderer(kinds.index("EndMeasure"))

    # Today this is ("_move_renderer", move_frames): the identical animation.
    assert advance.__name__ == "_no_op"
    assert total_frames != move_frames


def test_render_generator_on_kernel_without_atom_states(
    transport_arch_spec: ArchSpec, axes
) -> None:
    get_renderer, num_steps = render_generator(
        _no_atom_state_kernel, transport_arch_spec, axes, fps=24
    )

    assert num_steps == 0
    total_frames, advance = get_renderer(0)
    assert total_frames == 72
    assert advance(0) is None
