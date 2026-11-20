from __future__ import annotations

from typing import Any, cast

import pytest
import bloqade.squin as squin
from kirin import ir
from kirin.dialects import py
from bloqade.lanes.dialects import move
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.transform import PhysicalPipeline
from bloqade.lanes.analysis.atom import AtomState
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.analysis.atom.atom_state_data import AtomStateData

from bloqade.gemini.visualize import plotly_debug, plotly_circuit
from bloqade.gemini.visualize.artist import DebugStep, collect_debug_steps
from bloqade.gemini.visualize.plotly_circuit import (
    CircuitColumn,
    circuit_cursor,
    circuit_columns,
    cz_pair_offsets,
    circuit_window_range,
)


def _column(step_index: int, kind: Any = "local_r") -> CircuitColumn:
    return CircuitColumn(
        step_index=step_index,
        kind=kind,
        label="R",
        description="LocalR()",
        qubit_ids=(0,),
    )


def test_circuit_cursor_sits_on_gate_columns_and_between_them() -> None:
    columns = [_column(2), _column(5), _column(6)]
    assert circuit_cursor(columns, 0) == -0.5
    assert circuit_cursor(columns, 2) == 0.0
    assert circuit_cursor(columns, 3) == 0.5
    assert circuit_cursor(columns, 5) == 1.0
    assert circuit_cursor(columns, 6) == 2.0
    assert circuit_cursor(columns, 9) == 2.5
    assert circuit_cursor([], 4) == -0.5


def test_circuit_window_range_shows_everything_when_it_fits() -> None:
    assert circuit_window_range(3.0, 10, 24) == (-0.6, 9.6)
    assert circuit_window_range(0.0, 0, 24) == (-0.6, 0.6)


def test_circuit_window_range_follows_cursor_and_clamps_to_edges() -> None:
    low, high = circuit_window_range(50.0, 100, 10)
    assert (low, high) == (45.0, 55.0)
    assert circuit_window_range(0.0, 100, 10) == (-0.6, 9.4)
    low, high = circuit_window_range(99.0, 100, 10)
    assert high == pytest.approx(99.6)
    assert high - low == pytest.approx(10.0)


def _local_gate_steps(arch_spec: ArchSpec) -> list[DebugStep]:
    """Two atoms on one word; a load, a local R on qubit 1, and a global Rz."""
    locations = {
        qubit_id: LocationAddress(word_id=0, site_id=site_id, zone_id=0)
        for qubit_id, site_id in enumerate(range(2))
    }
    state = AtomState(AtomStateData.new(locations))
    load = move.Load()
    axis_angle = py.Constant(value=ir.PyAttr(0.25))
    rotation_angle = py.Constant(value=ir.PyAttr(0.5))
    local_r = move.LocalR(
        current_state=load.result,
        axis_angle=axis_angle.result,
        rotation_angle=rotation_angle.result,
        location_addresses=(locations[1],),
    )
    global_rz = move.GlobalRz(
        current_state=local_r.result, rotation_angle=rotation_angle.result
    )
    return [
        DebugStep(load, state, "Step 1 / 3: Load()"),
        DebugStep(local_r, state, "Step 2 / 3: LocalR(0.25, 0.5)", (0.25, 0.5)),
        DebugStep(global_rz, state, "Step 3 / 3: GlobalRz(0.5)", (0.5,)),
    ]


def test_circuit_columns_resolve_qubits_from_atom_state() -> None:
    arch_spec = get_arch_spec()
    steps = _local_gate_steps(arch_spec)

    columns = circuit_columns(steps, arch_spec)

    assert [column.step_index for column in columns] == [1, 2]
    assert columns[0].kind == "local_r"
    assert columns[0].qubit_ids == (1,)
    assert columns[0].description == "LocalR(axis_angle=0.25, rotation_angle=0.5)"
    assert columns[1].kind == "global_rz"
    assert columns[1].qubit_ids == (0, 1)
    assert plotly_circuit.circuit_qubit_ids(steps) == [0, 1]


def test_highlight_trace_is_a_band_on_gates_and_a_cursor_elsewhere(plotly) -> None:
    arch_spec = get_arch_spec()
    steps = _local_gate_steps(arch_spec)
    columns = circuit_columns(steps, arch_spec)
    colors = plotly_debug._theme_colors("light")

    cursor = plotly_circuit.circuit_highlight_trace(columns, 0, 2, colors)
    band = plotly_circuit.circuit_highlight_trace(columns, 1, 2, colors)

    assert cursor.fill == "none"
    assert list(cursor.x) == [-0.5, -0.5]
    assert cursor.xaxis == "x2" and cursor.yaxis == "y2"
    assert band.fill == "toself"
    assert band.fillcolor == colors["local_r"]
    assert min(band.x) == pytest.approx(-0.45)
    assert max(band.x) == pytest.approx(0.45)


@pytest.fixture(scope="module")
def compiled_program() -> tuple[ir.Method, ArchSpec]:
    @squin.kernel
    def kernel():
        reg = squin.qalloc(3)
        squin.h(reg[0])
        squin.cz(reg[0], reg[1])
        squin.rz(0.3, reg[2])
        squin.cz(reg[1], reg[2])
        squin.qubit.measure(reg)  # type: ignore[arg-type]

    return PhysicalPipeline().emit(kernel), get_arch_spec()


def test_compiled_program_circuit_matches_gate_steps(compiled_program) -> None:
    mt, arch_spec = compiled_program
    steps = collect_debug_steps(mt, arch_spec)
    gate_types = (
        move.LocalR,
        move.LocalRz,
        move.GlobalR,
        move.GlobalRz,
        move.StarRz,
        move.CZ,
        move.EndMeasure,
    )
    gate_steps = [
        index
        for index, step in enumerate(steps)
        if isinstance(step.statement, gate_types)
    ]

    columns = circuit_columns(steps, arch_spec)

    assert [column.step_index for column in columns] == gate_steps
    cz_columns = [column for column in columns if column.kind == "cz"]
    assert len(cz_columns) == 2
    assert {pair for column in cz_columns for pair in column.pairs} == {
        (0, 1),
        (1, 2),
    } or {frozenset(pair) for column in cz_columns for pair in column.pairs} == {
        frozenset({0, 1}),
        frozenset({1, 2}),
    }
    assert columns[-1].kind == "measure"
    assert columns[-1].qubit_ids == (0, 1, 2)


def test_debugger_figure_hosts_a_step_synced_circuit_panel(
    plotly, compiled_program
) -> None:
    mt, arch_spec = compiled_program
    steps = collect_debug_steps(mt, arch_spec)
    columns = circuit_columns(steps, arch_spec)

    figure = plotly_debug.build_plotly_debugger_figure(
        mt,
        arch_spec,
        interactive=True,
        pause_time=0.5,
        atom_marker="o",
        theme="light",
        height=600,
        circuit_window=4,
    )

    layout = cast(Any, figure.layout)
    meta = layout.meta["bloqadePlotlyDebugger"]
    assert meta["circuitColumnCount"] == len(columns)
    assert meta["circuitStepColumns"] == [column.step_index for column in columns]
    wires, connectors, gates = (
        cast(Any, figure.data)[index] for index in meta["circuitTraceIndices"]
    )
    assert wires.meta["bloqadeTraceKind"] == "circuitWire"
    assert connectors.meta["bloqadeTraceKind"] == "circuitConnector"
    assert gates.meta["bloqadeTraceKind"] == "circuitGate"
    assert gates.xaxis == "x2" and gates.yaxis == "y2"
    assert "click to jump" in gates.hovertemplate
    # Each gate marker carries the zero-based step index the JS jumps to.
    gate_steps = sorted({row[0] for row in gates.customdata})
    assert gate_steps == [column.step_index for column in columns]

    # The circuit sits on its own axis pair above the architecture.
    assert layout.yaxis.domain[1] < 1
    assert layout.yaxis2.domain[0] >= layout.yaxis.domain[1]
    assert layout.yaxis2.domain[1] == pytest.approx(1.0)
    assert list(layout.yaxis2.ticktext) == ["q0", "q1", "q2"]
    assert layout.height > 600

    # Frames highlight the active column and window the x-range around it.
    highlight_index = len(cast(Any, figure.frames[0].data)) - 3
    for step_index, frame in enumerate(figure.frames):
        highlight = cast(Any, frame.data)[highlight_index]
        assert highlight.meta["bloqadeTraceKind"] == "circuitHighlight"
        low, high = cast(Any, frame.layout).xaxis2.range
        assert high - low == pytest.approx(4.0)
        cursor = circuit_cursor(columns, step_index)
        assert low - 0.5 <= cursor <= high + 0.5
    last_range = cast(Any, figure.frames[-1].layout).xaxis2.range
    assert last_range[1] == pytest.approx(len(columns) - 0.4)

    html = figure.to_html(full_html=False, include_plotlyjs=False)
    assert "circuitGate" in html
    assert "jumpToDebuggerStep" in html


def test_debugger_figure_can_omit_the_circuit_panel(plotly, compiled_program) -> None:
    mt, arch_spec = compiled_program

    figure = plotly_debug.build_plotly_debugger_figure(
        mt,
        arch_spec,
        interactive=True,
        pause_time=0.5,
        atom_marker="o",
        theme="light",
        height=600,
        show_circuit=False,
    )

    layout = cast(Any, figure.layout)
    meta = layout.meta["bloqadePlotlyDebugger"]
    assert meta["circuitTraceIndices"] == []
    assert meta["circuitColumnCount"] == 0
    assert layout.height == 600
    assert "xaxis2" not in layout.to_plotly_json()
    assert "xaxis2" not in cast(Any, figure.frames[0].layout).to_plotly_json()
    assert all(trace.xaxis in (None, "x") for trace in cast(Any, figure.data))


def test_circuit_window_must_be_positive(compiled_program) -> None:
    mt, arch_spec = compiled_program
    with pytest.raises(ValueError, match="circuit_window"):
        plotly_debug.build_plotly_debugger_figure(
            mt,
            arch_spec,
            interactive=True,
            pause_time=0.5,
            atom_marker="o",
            theme="light",
            height=600,
            circuit_window=0,
        )


def test_gate_highlight_variants_share_attribute_keys(plotly, compiled_program) -> None:
    """Frame merging would otherwise leak a CZ fill into LocalR marker frames."""
    mt, arch_spec = compiled_program
    colors = plotly_debug._theme_colors("light")
    steps = collect_debug_steps(mt, arch_spec)
    by_type = {type(step.statement).__name__: step for step in steps}
    local = plotly_debug._gate_trace(by_type["LocalR"], arch_spec, colors)
    cz = plotly_debug._gate_trace(by_type["CZ"], arch_spec, colors)
    none = plotly_debug._gate_trace(by_type["Move"], arch_spec, colors)

    expected_keys = {
        "mode",
        "marker",
        "line",
        "fill",
        "fillcolor",
        "hoveron",
        "opacity",
        "name",
        "showlegend",
        "visible",
        "zorder",
        "x",
        "y",
    }
    for trace in (local, cz, none):
        assert expected_keys <= set(trace.to_plotly_json())
    assert local.mode == "markers" and local.fill == "none"
    assert cz.mode == "lines" and cz.fill == "toself" and cz.fillcolor == colors["cz"]
    assert none.fill == "none" and list(none.x) == []


def test_cz_pair_offsets_separate_overlapping_pairs_only() -> None:
    rows = {qubit_id: qubit_id for qubit_id in range(6)}

    # Disjoint spans share the column centre.
    assert cz_pair_offsets([(0, 1), (2, 3), (4, 5)], rows) == [0.0, 0.0, 0.0]
    # Two overlapping spans are pushed to opposite sides of the column.
    two = cz_pair_offsets([(0, 2), (1, 3)], rows)
    assert two[0] < 0.0 < two[1]
    assert two[0] == pytest.approx(-two[1])
    # A third pair that is disjoint from the first reuses its sub-column.
    three = cz_pair_offsets([(0, 2), (1, 3), (4, 5)], rows)
    assert three[2] == three[0]
    assert cz_pair_offsets([], rows) == []


def test_cz_layers_draw_one_connector_per_pair_with_partner_hover(plotly) -> None:
    colors = plotly_debug._theme_colors("light")
    column = CircuitColumn(
        step_index=3,
        kind="cz",
        label="CZ",
        description="CZ()",
        qubit_ids=(0, 1, 2, 3),
        pairs=((0, 2), (1, 3)),
    )

    _wires, connectors, gates = plotly_circuit.circuit_static_traces(
        [column], [0, 1, 2, 3], colors
    )

    # Two pairs, two connectors (each segment is two points plus a break).
    connector_xs = [x for x in connectors.x if x is not None]
    assert len(connector_xs) == 4
    assert len(set(connector_xs)) == 2, "overlapping pairs must be staggered"
    # Every endpoint marker sits on its own connector's x and names its partner.
    assert len(gates.x) == 4
    partner_text = {row[3]: row[4] for row in gates.customdata}
    assert partner_text[0] == "CZ control, paired with qubit 2"
    assert partner_text[2] == "CZ target, paired with qubit 0"
    assert partner_text[1] == "CZ control, paired with qubit 3"
    assert "%{customdata[4]}" in gates.hovertemplate


def test_star_rz_is_labelled_apart_from_a_plain_rz() -> None:
    """``StarRz`` shares ``local_rz``'s colour but not its label.

    Both land in the same circuit row style, so the label is the only thing
    telling a Steane star apart from an ordinary local Rz.
    """
    load = move.Load()
    rotation_angle = py.Constant(value=ir.PyAttr(0.5))
    star_rz = move.StarRz(
        current_state=load.result,
        rotation_angle=rotation_angle.result,
        location_addresses=(LocationAddress(0, 0, 0),),
        qubit_indices=(0, 1, 2),
    )
    local_rz = move.LocalRz(
        current_state=load.result,
        rotation_angle=rotation_angle.result,
        location_addresses=(LocationAddress(0, 0, 0),),
    )

    assert plotly_circuit._gate_kind_and_label(star_rz) == ("local_rz", "Rz*")
    assert plotly_circuit._gate_kind_and_label(local_rz) == ("local_rz", "Rz")
    # A statement that is not a gate gets no column at all.
    assert plotly_circuit._gate_kind_and_label(load) is None


def test_circuit_gates_skip_qubits_without_a_wire(plotly) -> None:
    """A column may name a qubit the diagram has no row for; it is dropped.

    ``circuit_qubit_ids`` collects the qubits that occupy a site at some step,
    so a column naming anything else -- a qubit that was never placed -- has
    nowhere to draw and must not shift the rows of the qubits that were.
    """
    colors = plotly_debug._theme_colors("light")
    column = CircuitColumn(
        step_index=0,
        kind="local_r",
        label="R",
        description="LocalR()",
        qubit_ids=(0, 99),
    )

    _, _, gates = plotly_circuit.circuit_static_traces([column], [0], colors)

    # Only qubit 0 has a wire, so only qubit 0 gets a marker.
    assert len(cast(Any, gates).x) == 1
    assert [data[3] for data in cast(Any, gates).customdata] == [0]
