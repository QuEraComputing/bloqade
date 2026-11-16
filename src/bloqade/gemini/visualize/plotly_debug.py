from __future__ import annotations

import math
import itertools
from typing import TYPE_CHECKING, Any, Literal, cast
from dataclasses import dataclass
from collections.abc import Sequence

from kirin import ir
from bloqade.lanes.dialects import move
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.analysis.atom import AtomState
from bloqade.lanes.bytecode.encoding import MoveType, LaneAddress, LocationAddress

from bloqade.gemini.visualize.arch import ArchVisualizer, bus_preview_label
from bloqade.gemini.visualize.artist import MovePath, DebugStep, collect_debug_steps
from bloqade.gemini.visualize.plotly_circuit import (
    CIRCUIT_X_AXIS,
    CIRCUIT_Y_AXIS,
    CircuitColumn,
    _plotly,
    circuit_row_y,
    circuit_cursor,
    circuit_columns,
    circuit_y_range,
    gate_description,
    circuit_qubit_ids,
    circuit_window_range,
    circuit_static_traces,
    circuit_highlight_trace,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure  # type: ignore[reportMissingImports]


Theme = Literal["light", "dark"]

_ATOM_SYMBOLS = {
    "o": "circle",
    ".": "circle",
    "s": "square",
    "+": "cross",
    "x": "x",
    "^": "triangle-up",
    "v": "triangle-down",
    "d": "diamond",
    "D": "diamond",
    "*": "star",
    "p": "pentagon",
    "h": "hexagon",
}


@dataclass(frozen=True)
class _MovePathSegment:
    """Plotly data for one segment of an atom's latest move."""

    qubit_id: int
    source: LocationAddress
    destination: LocationAddress
    start: tuple[float, float]
    end: tuple[float, float]
    color_value: float
    bus_label: str


def _lane_bus_label(lane: LaneAddress) -> str:
    """Name the bus a lane belongs to, as ``arch`` names it.

    Delegates so a move-path tooltip and a site-hover preview cannot end up
    calling one bus two things; see :func:`bus_preview_label`. An inter-zone
    lane carries a ``zone_id`` that names only where it starts, so it maps to
    the zone-less case rather than being labelled with that zone.
    """
    zone_id = None if lane.move_type == MoveType.ZONE else lane.zone_id
    return bus_preview_label(lane.move_type, zone_id, lane.bus_id)


def _lane_moves(state: AtomState, arch_spec: ArchSpec) -> list[MovePath]:
    """The routes recorded in ``state`` as the lanes its atoms last took."""
    moves: list[MovePath] = []
    for qubit_id, lane in sorted(state.data.prev_lanes.items()):
        source, destination = arch_spec.get_endpoints(lane)
        moves.append(
            MovePath(
                qubit_id,
                source,
                destination,
                arch_spec.get_path(lane),
                _lane_bus_label(lane),
            )
        )
    return moves


def _move_path_segments(
    state: AtomState,
    arch_spec: ArchSpec,
    moves: Sequence[MovePath] | None = None,
) -> list[_MovePathSegment]:
    """Translate the routes into ``state`` into Plotly path segments.

    The routes are ``moves``, or the atom-analysis lanes recorded in the state
    when ``moves`` is ``None``.
    """
    if moves is None:
        moves = _lane_moves(state, arch_spec)

    segments: list[_MovePathSegment] = []
    for move_path in moves:
        path_segments = list(itertools.pairwise(move_path.waypoints))
        denominator = max(1, len(path_segments) - 1)
        segments.extend(
            _MovePathSegment(
                qubit_id=move_path.qubit_id,
                source=move_path.source,
                destination=move_path.destination,
                start=start,
                end=end,
                color_value=index / denominator,
                bus_label=move_path.label,
            )
            for index, (start, end) in enumerate(path_segments)
        )
    return segments


def _theme_colors(theme: Theme) -> dict[str, str]:
    if theme == "light":
        return {
            "paper": "#ffffff",
            "plot": "#f8fafc",
            "text": "#1e293b",
            "site": "#ffffff",
            "site_edge": "#475569",
            "atom": "#6437ff",
            "route": "#7c3aed",
            "local_r": "#2563eb",
            "local_rz": "#16a34a",
            "cz": "#dc2626",
            "measure": "#d97706",
        }
    if theme == "dark":
        return {
            "paper": "#0f172a",
            "plot": "#111827",
            "text": "#f8fafc",
            "site": "#111827",
            "site_edge": "#cbd5e1",
            "atom": "#8f78ff",
            "route": "#c4b5fd",
            "local_r": "#60a5fa",
            "local_rz": "#4ade80",
            "cz": "#f87171",
            "measure": "#fbbf24",
        }
    raise ValueError("theme must be 'light' or 'dark'")


def _empty_trace(name: str) -> Any:
    go = _plotly()
    return go.Scatter(x=[], y=[], mode="lines", name=name, visible=False)


def _atom_trace(
    step: DebugStep,
    arch_spec: ArchSpec,
    colors: dict[str, str],
    atom_marker: str,
) -> Any:
    go = _plotly()
    state = step.state
    atoms = [
        (qubit_id, location, arch_spec.get_position(location))
        for qubit_id, location in sorted(state.data.qubit_to_locations.items())
    ]
    atom_customdata = [
        [
            qubit_id,
            location.zone_id,
            location.word_id,
            location.site_id,
            *arch_spec.words[location.word_id].sites[location.site_id],
            (
                f"<br><b>gate:</b> {gate_description(step)}"
                if _gate_applies_to_location(step.statement, location)
                else ""
            ),
        ]
        for qubit_id, location, _ in atoms
    ]
    return go.Scatter(
        x=[position[0] for _, _, position in atoms],
        y=[position[1] for _, _, position in atoms],
        customdata=atom_customdata,
        text=[str(qubit_id) for qubit_id, _, _ in atoms],
        mode="markers+text",
        textposition="middle center",
        textfont={"color": "white", "size": 10},
        marker={
            "color": colors["atom"],
            "size": 18,
            "symbol": _ATOM_SYMBOLS.get(atom_marker, atom_marker),
            "line": {"color": colors["paper"], "width": 1},
        },
        meta={"bloqadeTraceKind": "atom"},
        name="Atoms",
        showlegend=False,
        hovertemplate=(
            "<b>atom %{customdata[0]}</b><br>"
            "(zone, word, site): (%{customdata[1]}, %{customdata[2]}, "
            "%{customdata[3]})<br>"
            "grid (x, y): (%{customdata[4]}, %{customdata[5]})<br>"
            "position (x, y): (%{x:.3f}, %{y:.3f}) µm"
            "%{customdata[6]}<extra></extra>"
        ),
    )


def _gate_applies_to_location(
    statement: ir.Statement, location: LocationAddress
) -> bool:
    if isinstance(statement, (move.LocalR, move.LocalRz, move.StarRz)):
        return location in statement.location_addresses
    if isinstance(statement, (move.GlobalR, move.GlobalRz)):
        return True
    if isinstance(statement, move.CZ):
        return location.zone_id == statement.zone_address.zone_id
    if isinstance(statement, move.EndMeasure):
        return any(
            location.zone_id == address.zone_id for address in statement.zone_addresses
        )
    return False


def _viridis_color(color_value: float) -> str:
    from plotly.colors import sample_colorscale

    return cast(str, sample_colorscale("Viridis", [color_value])[0])


def _group_route_segments(
    route_segments: Sequence[_MovePathSegment],
) -> list[list[_MovePathSegment]]:
    """Group equally colored segments so simultaneous moves share traces."""
    segments_by_color: dict[float, list[_MovePathSegment]] = {}
    for segment in route_segments:
        segments_by_color.setdefault(segment.color_value, []).append(segment)
    return list(segments_by_color.values())


def _route_trace(segments: Sequence[_MovePathSegment] | None) -> Any:
    """Render equally colored route segments with visible arrowheads."""
    go = _plotly()
    if not segments:
        return go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Move path",
            showlegend=False,
            visible=False,
            hoverinfo="skip",
            meta={"bloqadeTraceKind": "movePath", "segments": []},
        )

    x_values: list[float | None] = []
    y_values: list[float | None] = []
    marker_sizes: list[int] = []
    marker_symbols: list[str] = []
    marker_angles: list[float] = []
    for segment in segments:
        delta_x = segment.end[0] - segment.start[0]
        delta_y = segment.end[1] - segment.start[1]
        # Plotly marker angles are clockwise from screen-up. With equal axis
        # scaling, atan2(dx, dy) maps data-space segments to that convention.
        arrow_angle = math.degrees(math.atan2(delta_x, delta_y))
        x_values.extend([segment.start[0], segment.end[0], None])
        y_values.extend([segment.start[1], segment.end[1], None])
        marker_sizes.extend([0, 16, 0])
        marker_symbols.extend(["circle", "arrow", "circle"])
        marker_angles.extend([0, arrow_angle, 0])

    color = _viridis_color(segments[0].color_value)
    segment_metadata = [
        {
            "atomId": segment.qubit_id,
            "busLabel": segment.bus_label,
            "source": (
                f"({segment.source.zone_id}, {segment.source.word_id}, "
                f"{segment.source.site_id})"
            ),
            "destination": (
                f"({segment.destination.zone_id}, {segment.destination.word_id}, "
                f"{segment.destination.site_id})"
            ),
            "color": color,
            "start": list(segment.start),
            "end": list(segment.end),
        }
        for segment in segments
    ]
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        line={"color": color, "width": 4},
        marker={
            "color": color,
            "size": marker_sizes,
            "symbol": marker_symbols,
            "angle": marker_angles,
            "angleref": "up",
            "line": {"color": color, "width": 1.5},
        },
        name="Move path",
        meta={
            "bloqadeTraceKind": "movePath",
            "segments": segment_metadata,
        },
        showlegend=False,
        visible=True,
        cliponaxis=False,
        # A custom SVG hit target spans the entire segment and anchors its
        # single tooltip at the segment center. Native marker hover would add
        # a second tooltip specifically at each arrowhead.
        hoverinfo="skip",
    )


def _global_gate_bounds(arch_spec: ArchSpec) -> tuple[float, float, float, float]:
    visualizer = ArchVisualizer(arch_spec)
    x_min, x_max = visualizer.x_bounds
    y_min, y_max = visualizer.y_bounds
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_padding = 0.5 * x_span if x_span else 1.0
    y_padding = 0.5 * y_span if y_span else 1.0
    return (
        x_min - x_padding,
        x_max + x_padding,
        y_min - y_padding,
        y_max + y_padding,
    )


def _zone_gate_bounds(
    zone_ids: set[int], arch_spec: ArchSpec
) -> tuple[float, float, float, float] | None:
    zone_positions = [
        position
        for location, position in ArchVisualizer(arch_spec)._iter_locations()
        if location.zone_id in zone_ids
    ]
    if not zone_positions:
        return None

    x_values = [position[0] for position in zone_positions]
    y_values = [position[1] for position in zone_positions]
    y_span = max(y_values) - min(y_values)
    y_padding = 0.1 * y_span if y_span else 1.0
    return (
        min(x_values) - 10.0,
        max(x_values) + 10.0,
        min(y_values) - y_padding,
        max(y_values) + y_padding,
    )


def _gate_trace_base(name: str, colors: dict[str, str]) -> dict[str, Any]:
    """Attribute set shared by every gate-highlight variant.

    Plotly animation merges frame trace attributes into the live trace rather
    than replacing them. Every variant therefore sets the same keys, otherwise
    a CZ frame's ``fill="toself"`` would persist into a later LocalR frame and
    fill the polygon spanned by that gate's marker positions.
    """
    return {
        "mode": "lines",
        "marker": {"color": colors["local_r"], "size": 0, "opacity": 0.0},
        "line": {"width": 0, "color": colors["local_r"]},
        "fill": "none",
        "fillcolor": "rgba(0,0,0,0)",
        "hoveron": "points",
        "opacity": 1.0,
        "hovertemplate": None,
        "hoverinfo": "skip",
        "name": name,
        "showlegend": False,
        "visible": True,
        "zorder": -10,
    }


def _gate_trace(
    step: DebugStep,
    arch_spec: ArchSpec,
    colors: dict[str, str],
) -> Any:
    go = _plotly()
    statement = step.statement
    description = gate_description(step)
    local_gate_types = (move.LocalR, move.LocalRz, move.StarRz)
    if isinstance(statement, local_gate_types):
        positions = [
            arch_spec.get_position(location)
            for location in statement.location_addresses
        ]
        color = (
            colors["local_r"]
            if isinstance(statement, move.LocalR)
            else colors["local_rz"]
        )
        return go.Scatter(
            x=[position[0] for position in positions],
            y=[position[1] for position in positions],
            **{
                **_gate_trace_base(type(statement).__name__, colors),
                "mode": "markers",
                "marker": {"color": color, "size": 34, "opacity": 0.3},
                "hoverinfo": None,
                "hovertemplate": f"{description}<extra></extra>",
            },
        )

    region: tuple[float, float, float, float] | None = None
    color = colors["local_r"]
    if isinstance(statement, (move.GlobalR, move.GlobalRz)):
        region = _global_gate_bounds(arch_spec)
        color = (
            colors["local_r"]
            if isinstance(statement, move.GlobalR)
            else colors["local_rz"]
        )
    elif isinstance(statement, move.CZ):
        region = _zone_gate_bounds({statement.zone_address.zone_id}, arch_spec)
        color = colors["cz"]
    elif isinstance(statement, move.EndMeasure):
        region = _zone_gate_bounds(
            {address.zone_id for address in statement.zone_addresses},
            arch_spec,
        )
        color = colors["measure"]

    if region is None:
        # No gate at this step: an empty trace with the shared attribute set
        # so it never inherits a previous frame's fill or markers.
        return go.Scatter(x=[], y=[], **_gate_trace_base("Gate highlight", colors))

    x_min, x_max, y_min, y_max = region
    return go.Scatter(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        **{
            **_gate_trace_base(type(statement).__name__, colors),
            "line": {"width": 0, "color": color},
            "fill": "toself",
            "fillcolor": color,
            "hoveron": "fills",
            "opacity": 0.25,
            "hoverinfo": None,
            "hovertemplate": f"{description}<extra></extra>",
        },
    )


@dataclass(frozen=True)
class _CircuitPanel:
    """Executed-circuit data shared by the static traces and every frame."""

    columns: tuple[CircuitColumn, ...]
    qubit_ids: tuple[int, ...]
    window: int

    @property
    def column_count(self) -> int:
        return len(self.columns)

    def x_range(self, step_index: int) -> tuple[float, float]:
        return circuit_window_range(
            circuit_cursor(self.columns, step_index), self.column_count, self.window
        )


def _dynamic_traces(
    step: DebugStep,
    step_index: int,
    route_groups: Sequence[Sequence[_MovePathSegment]],
    route_trace_count: int,
    arch_spec: ArchSpec,
    colors: dict[str, str],
    atom_marker: str,
    circuit: _CircuitPanel | None,
) -> list[Any]:
    # The circuit highlight sits before the gate and atom traces so the last
    # two dynamic traces stay the architecture gate and atom traces.
    circuit_traces = (
        [
            circuit_highlight_trace(
                circuit.columns, step_index, len(circuit.qubit_ids), colors
            )
        ]
        if circuit is not None
        else []
    )
    return [
        *(
            _route_trace(route_groups[index] if index < len(route_groups) else None)
            for index in range(route_trace_count)
        ),
        *circuit_traces,
        _gate_trace(step, arch_spec, colors),
        _atom_trace(step, arch_spec, colors, atom_marker),
    ]


def _frame_layout(
    step: DebugStep, step_index: int, circuit: _CircuitPanel | None
) -> dict[str, Any]:
    layout: dict[str, Any] = {"title": {"text": step.title}}
    if circuit is not None:
        layout[f"xaxis{CIRCUIT_X_AXIS[1:]}"] = {
            "range": list(circuit.x_range(step_index))
        }
    return layout


# Pixel budget for the executed-circuit panel above the architecture.
_CIRCUIT_ROW_PX = 24
_CIRCUIT_MIN_PX = 130
_CIRCUIT_MAX_PX = 380
_CIRCUIT_GAP_PX = 50
_DEBUGGER_MARGIN = {"l": 70, "r": 270, "t": 165, "b": 65}


def _circuit_panel_layout(
    circuit: _CircuitPanel, height: int, colors: dict[str, str]
) -> tuple[int, dict[str, Any]]:
    """Return the total figure height and axis layout hosting the circuit."""
    qubit_count = len(circuit.qubit_ids)
    circuit_px = min(
        max(_CIRCUIT_ROW_PX * qubit_count + 70, _CIRCUIT_MIN_PX), _CIRCUIT_MAX_PX
    )
    total_height = height + circuit_px + _CIRCUIT_GAP_PX
    plot_area = total_height - _DEBUGGER_MARGIN["t"] - _DEBUGGER_MARGIN["b"]
    circuit_fraction = circuit_px / plot_area
    gap_fraction = _CIRCUIT_GAP_PX / plot_area
    x_key = f"xaxis{CIRCUIT_X_AXIS[1:]}"
    y_key = f"yaxis{CIRCUIT_Y_AXIS[1:]}"
    # The circuit occupies the top of the plot area, directly under the
    # control row; the gap below it holds the circuit's axis title.
    layout = {
        "yaxis": {"domain": [0.0, 1.0 - circuit_fraction - gap_fraction]},
        x_key: {
            "anchor": CIRCUIT_Y_AXIS,
            "domain": [0.0, 1.0],
            "range": list(circuit.x_range(0)),
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "title": {
                "text": (
                    "Executed circuit (synced to the step slider; "
                    "click a gate to jump to its step)"
                ),
                "font": {"size": 12},
            },
        },
        y_key: {
            "anchor": CIRCUIT_X_AXIS,
            "domain": [1.0 - circuit_fraction, 1.0],
            "range": list(circuit_y_range(qubit_count)),
            "tickvals": [circuit_row_y(index) for index in range(qubit_count)],
            "ticktext": [f"q{qubit_id}" for qubit_id in circuit.qubit_ids],
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"size": 11, "color": colors["text"]},
        },
    }
    return total_height, layout


def _slider_step(frame_names: Sequence[str], label: str) -> dict[str, object]:
    return {
        "label": label,
        "method": "animate",
        "args": [
            list(frame_names),
            {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            },
        ],
    }


def build_plotly_debugger_figure(
    mt: ir.Method,
    arch_spec: ArchSpec,
    *,
    interactive: bool,
    pause_time: float,
    atom_marker: str,
    theme: Theme,
    height: int,
    show_circuit: bool = True,
    circuit_window: int = 24,
    initialize_kernel: ir.Method | None = None,
) -> Figure:
    """Build the Plotly figure without displaying it."""
    if pause_time < 0:
        raise ValueError("pause_time must be non-negative")
    if circuit_window < 1:
        raise ValueError("circuit_window must be at least 1")
    go = _plotly()
    steps = collect_debug_steps(mt, arch_spec, initialize_kernel)
    colors = _theme_colors(theme)
    circuit = (
        _CircuitPanel(
            columns=tuple(circuit_columns(steps, arch_spec)),
            qubit_ids=tuple(circuit_qubit_ids(steps)),
            window=circuit_window,
        )
        if show_circuit
        else None
    )
    route_groups_by_step = [
        _group_route_segments(_move_path_segments(step.state, arch_spec, step.moves))
        for step in steps
    ]
    route_trace_count = max(
        1,
        max((len(groups) for groups in route_groups_by_step), default=0),
    )

    # Start with the complete interactive architecture rather than rebuilding
    # its sites and buses here. This preserves one source of truth for site
    # metadata, bus selectors, hover highlighting, and transport paths.
    figure = ArchVisualizer(arch_spec).plot_interactive(
        show_site_ids=False,
        show_all_buses=False,
        show_bus_legend=False,
        path_style="exact",
        site_lane_preview="hover",
        bus_line_style="dashed",
        theme=theme,
        height=height,
    )
    architecture_trace_count = len(cast(Any, figure.data))
    circuit_trace_indices: list[int] = []
    if circuit is not None:
        circuit_trace_indices = list(
            range(architecture_trace_count, architecture_trace_count + 3)
        )
        figure.add_traces(
            circuit_static_traces(circuit.columns, circuit.qubit_ids, colors)
        )
    dynamic_trace_start = len(cast(Any, figure.data))
    dynamic_trace_count = route_trace_count + (1 if circuit is not None else 0) + 2
    dynamic_trace_indices = list(
        range(dynamic_trace_start, dynamic_trace_start + dynamic_trace_count)
    )
    frames: list[Any] = []

    for step_index, step in enumerate(steps):
        frames.append(
            go.Frame(
                name=f"step-{step_index}",
                data=_dynamic_traces(
                    step,
                    step_index,
                    route_groups_by_step[step_index],
                    route_trace_count,
                    arch_spec,
                    colors,
                    atom_marker,
                    circuit,
                ),
                traces=dynamic_trace_indices,
                layout=_frame_layout(step, step_index, circuit),
            )
        )

    initial_data = (
        _dynamic_traces(
            steps[0],
            0,
            route_groups_by_step[0],
            route_trace_count,
            arch_spec,
            colors,
            atom_marker,
            circuit,
        )
        if steps
        else [
            *(_route_trace(None) for _ in range(route_trace_count)),
            *([_empty_trace("Circuit highlight")] if circuit is not None else []),
            _empty_trace("Gate highlight"),
            _empty_trace("Atoms"),
        ]
    )
    title = steps[0].title if steps else "Plotly move debugger: no atom-state steps"
    figure.add_traces(initial_data)
    figure.frames = frames

    sliders = []
    debugger_update_menus = []
    if interactive and frames:
        sliders = [
            {
                "active": 0,
                "currentvalue": {"prefix": "Step: "},
                "pad": {"t": 45},
                "steps": [
                    _slider_step([frame.name], str(index + 1))
                    for index, frame in enumerate(frames)
                ],
            }
        ]
        if len(frames) > 1:
            debugger_update_menus = [
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0,
                    "xanchor": "left",
                    # Keep playback on the same row as the architecture
                    # controls in the upper margin.
                    "y": 1.09,
                    "yanchor": "top",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play steps",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "fromcurrent": True,
                                    "mode": "immediate",
                                    "frame": {
                                        "duration": max(1, round(pause_time * 1000)),
                                        "redraw": True,
                                    },
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ]

    architecture_meta = dict(cast(Any, figure.layout.meta) or {})
    architecture_update_menus = list(figure.layout.updatemenus or ())
    # Make room for playback at the left, then center the complete four-group
    # control row over the plot. ``plot_interactive`` only has the latter
    # three groups, so this debugger-specific alignment belongs here.
    for menu, x_position in zip(
        architecture_update_menus,
        (0.18, 0.47, 0.75),
    ):
        menu.x = x_position
        menu.y = 1.09
    figure.layout.annotations = ()
    total_height = height
    circuit_layout: dict[str, Any] = {}
    if circuit is not None:
        total_height, circuit_layout = _circuit_panel_layout(circuit, height, colors)
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=total_height,
        margin=dict(_DEBUGGER_MARGIN),
        showlegend=False,
        uirevision="bloqade-plotly-debugger",
        sliders=sliders,
        updatemenus=[*architecture_update_menus, *debugger_update_menus],
        meta={
            **architecture_meta,
            "bloqadePlotlyDebugger": {
                "stepCount": len(steps),
                "frameCount": len(frames),
                "frameNames": [frame.name for frame in frames],
                "routeTraceCount": route_trace_count,
                "architectureTraceCount": architecture_trace_count,
                "circuitTraceIndices": circuit_trace_indices,
                "circuitColumnCount": circuit.column_count if circuit else 0,
                "circuitStepColumns": (
                    [column.step_index for column in circuit.columns] if circuit else []
                ),
            },
        },
        **circuit_layout,
    )
    return figure


def _in_jupyter_kernel() -> bool:
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def _show_plotly_debugger(
    figure: Figure,
    *,
    renderer: str | None,
    auto_play: bool,
    frame_duration_ms: int,
) -> None:
    selected_renderer = renderer
    if selected_renderer is None and not _in_jupyter_kernel():
        selected_renderer = "browser"
    figure.show(
        renderer=selected_renderer,
        config={"responsive": True, "scrollZoom": True},
        auto_play=auto_play,
        animation_opts={
            "frame": {"duration": frame_duration_ms, "redraw": True},
            "transition": {"duration": 0},
            "fromcurrent": True,
            "mode": "immediate",
        },
    )


def plotly_debugger(
    mt: ir.Method,
    arch_spec: ArchSpec,
    interactive: bool = True,
    pause_time: float = 1.0,
    atom_marker: str = "o",
    *,
    show: bool = True,
    renderer: str | None = None,
    theme: Theme = "light",
    height: int = 720,
    show_circuit: bool = True,
    circuit_window: int = 24,
    initialize_kernel: ir.Method | None = None,
) -> Figure | None:
    """Display a browser-native, discrete-step move-program debugger.

    This is a Plotly alternative to :func:`debugger`; it does not replace the
    existing Matplotlib debugger or :func:`animated_debugger`. In a notebook,
    Plotly embeds the controls in the cell output. In a regular Python process,
    the default renderer opens the figure in a browser.

    Hover over an empty site or an atom to preview the transport lanes
    available at that site, using the architecture's exact/cartoon path view.

    Args:
        mt: Compiled physical move program to visualize.
        arch_spec: Architecture used to compile ``mt``.
        interactive: Include the step slider and play/pause controls. When
            false, displaying the figure automatically plays all steps.
        pause_time: Seconds between steps during playback.
        atom_marker: Matplotlib-style marker name for atoms (for example,
            ``"o"`` or ``"s"``), or a Plotly marker symbol.
        show: Display the figure immediately. Set false to customize it first.
        renderer: Explicit Plotly renderer. The default embeds in Jupyter and
            uses ``"browser"`` in a regular Python process.
        theme: Initial light or dark theme.
        height: Height of the architecture panel in pixels. The executed
            circuit panel, when shown, is added above it.
        show_circuit: Render the executed circuit above the architecture, one
            column per gate step. The active step's column is highlighted and
            kept in view; clicking a gate jumps the debugger to its step.
        circuit_window: Maximum number of circuit columns visible at once.
            Longer circuits scroll horizontally to follow the active step.
        initialize_kernel: Kernel that ``move.PhysicalInitialize`` stands for.
            When given, each such statement is stepped through instead of
            shown as one step; see
            :func:`~bloqade.gemini.visualize.artist.collect_debug_steps`.

    Returns:
        The constructed Plotly figure when ``show=False``; otherwise ``None``.
    """
    figure = build_plotly_debugger_figure(
        mt,
        arch_spec,
        interactive=interactive,
        pause_time=pause_time,
        atom_marker=atom_marker,
        theme=theme,
        height=height,
        show_circuit=show_circuit,
        circuit_window=circuit_window,
        initialize_kernel=initialize_kernel,
    )
    if show:
        _show_plotly_debugger(
            figure,
            renderer=renderer,
            auto_play=not interactive,
            frame_duration_ms=max(1, round(pause_time * 1000)),
        )
        return None
    return figure
