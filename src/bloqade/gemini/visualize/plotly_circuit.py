"""Executed-circuit reconstruction and Plotly traces for the move debugger.

The move dialect program that :func:`~bloqade.gemini.visualize.plotly_debug.plotly_debugger`
interprets already describes the executed circuit: every gate statement carries
the physical locations (or zone) it acts on, and the interpreted atom state maps
those locations back to qubit ids. This module turns the debugger steps into
circuit columns (one per gate step) and renders them as a compact circuit
diagram on a secondary Plotly axis pair, so the diagram stays in lockstep with
the atom animation.
"""

from __future__ import annotations

from typing import Any, Literal
from dataclasses import dataclass
from collections.abc import Sequence

from kirin import ir
from bloqade.lanes.dialects import move
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress

from bloqade.gemini.visualize.artist import DebugStep

GateKind = Literal["local_r", "local_rz", "global_r", "global_rz", "cz", "measure"]

CIRCUIT_X_AXIS = "x2"
CIRCUIT_Y_AXIS = "y2"


def _plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The Plotly move debugger requires the 'atom-vis' extra"
        ) from exc
    return go


def gate_parameter_names(statement: ir.Statement) -> tuple[str, ...]:
    if isinstance(statement, (move.LocalR, move.GlobalR)):
        return ("axis_angle", "rotation_angle")
    if isinstance(statement, (move.LocalRz, move.StarRz, move.GlobalRz)):
        return ("rotation_angle",)
    return ()


def gate_description(step: DebugStep) -> str:
    """Format an interpreted gate with named, evaluated parameters."""
    if isinstance(step.statement, (move.EndMeasure, move.Measure)):
        zones = tuple(address.zone_id for address in step.statement.zone_addresses)
        return f"{type(step.statement).__name__}(zones={zones})"

    parameters = ", ".join(
        f"{name}={value!r}"
        for name, value in zip(
            gate_parameter_names(step.statement),
            step.parameter_values,
            strict=False,
        )
    )
    return f"{type(step.statement).__name__}({parameters})"


@dataclass(frozen=True)
class CircuitColumn:
    """One executed gate layer of the circuit, tied to a debugger step."""

    step_index: int
    kind: GateKind
    label: str
    description: str
    qubit_ids: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...] = ()


def _gate_kind_and_label(statement: ir.Statement) -> tuple[GateKind, str] | None:
    if isinstance(statement, move.LocalR):
        return "local_r", "R"
    if isinstance(statement, move.GlobalR):
        return "global_r", "R"
    if isinstance(statement, move.StarRz):
        return "local_rz", "Rz*"
    if isinstance(statement, move.LocalRz):
        return "local_rz", "Rz"
    if isinstance(statement, move.GlobalRz):
        return "global_rz", "Rz"
    if isinstance(statement, move.CZ):
        return "cz", "CZ"
    if isinstance(statement, (move.EndMeasure, move.Measure)):
        return "measure", "M"
    return None


def _qubits_at(
    step: DebugStep, locations: Sequence[LocationAddress]
) -> tuple[int, ...]:
    qubit_ids = (step.state.data.get_qubit(location) for location in locations)
    return tuple(qubit_id for qubit_id in qubit_ids if qubit_id is not None)


def _qubits_in_zones(step: DebugStep, zone_ids: set[int]) -> tuple[int, ...]:
    return tuple(
        sorted(
            qubit_id
            for location, qubit_id in step.state.data.locations_to_qubit.items()
            if location.zone_id in zone_ids
        )
    )


def circuit_columns(
    steps: Sequence[DebugStep], arch_spec: ArchSpec
) -> list[CircuitColumn]:
    """Reconstruct the executed circuit, one column per gate step.

    The qubit ids are resolved from the interpreted atom state exactly the way
    ``move2squin`` resolves them when it emits the final squin program, so the
    result is the executed circuit by construction (after ASAP/ALAP reordering
    and gate fusion).
    """
    columns: list[CircuitColumn] = []
    for step_index, step in enumerate(steps):
        statement = step.statement
        kind_and_label = _gate_kind_and_label(statement)
        if kind_and_label is None:
            continue
        kind, label = kind_and_label
        pairs: tuple[tuple[int, int], ...] = ()

        if isinstance(statement, (move.LocalR, move.LocalRz, move.StarRz)):
            qubit_ids = _qubits_at(step, statement.location_addresses)
        elif isinstance(statement, (move.GlobalR, move.GlobalRz)):
            qubit_ids = tuple(sorted(step.state.data.qubit_to_locations))
        elif isinstance(statement, move.CZ):
            controls, targets, _ = step.state.data.get_qubit_pairing(
                statement.zone_address, arch_spec
            )
            pairs = tuple(zip(controls, targets, strict=True))
            qubit_ids = tuple(sorted({*controls, *targets}))
        else:
            assert isinstance(statement, (move.EndMeasure, move.Measure))
            qubit_ids = _qubits_in_zones(
                step, {address.zone_id for address in statement.zone_addresses}
            )

        columns.append(
            CircuitColumn(
                step_index=step_index,
                kind=kind,
                label=label,
                description=gate_description(step),
                qubit_ids=qubit_ids,
                pairs=pairs,
            )
        )
    return columns


def circuit_qubit_ids(steps: Sequence[DebugStep]) -> list[int]:
    """Every qubit id that occupies a site at any step, in ascending order."""
    qubit_ids: set[int] = set()
    for step in steps:
        qubit_ids.update(step.state.data.qubit_to_locations)
    return sorted(qubit_ids)


def circuit_cursor(columns: Sequence[CircuitColumn], step_index: int) -> float:
    """Position of ``step_index`` along the circuit's column axis.

    Gate steps sit on their own column. Steps without a gate (moves, fills,
    loads) sit halfway between the previously executed column and the next one,
    which is where the atoms are physically travelling to.
    """
    cursor = -0.5
    for column_index, column in enumerate(columns):
        if column.step_index == step_index:
            return float(column_index)
        if column.step_index < step_index:
            cursor = column_index + 0.5
        else:
            break
    return cursor


def circuit_window_range(
    cursor: float, column_count: int, window: int
) -> tuple[float, float]:
    """Visible column range keeping ``cursor`` centered inside a window."""
    left_edge = -0.6
    right_edge = max(column_count, 1) - 0.4
    span = right_edge - left_edge
    if window <= 0 or span <= window:
        return (left_edge, right_edge)
    low = min(max(cursor - window / 2, left_edge), right_edge - window)
    return (low, low + window)


def circuit_row_y(row_index: int) -> float:
    # ``0.0 - row`` avoids a ``-0.0`` tick for the first wire.
    return 0.0 - row_index


def circuit_y_range(qubit_count: int) -> tuple[float, float]:
    return (circuit_row_y(max(qubit_count, 1) - 1) - 0.7, 0.7)


def _circuit_axes() -> dict[str, str]:
    return {"xaxis": CIRCUIT_X_AXIS, "yaxis": CIRCUIT_Y_AXIS}


# Half-width of a column available for staggering simultaneous CZ pairs.
_CZ_STAGGER_HALF_WIDTH = 0.36


def cz_pair_offsets(
    pairs: Sequence[tuple[int, int]], rows: dict[int, int]
) -> list[float]:
    """Horizontal offset within a column for each CZ pair.

    Pairs whose wire spans overlap would draw collinear connectors and become
    indistinguishable, so they are assigned to separate sub-columns (a greedy
    interval colouring). Pairs with disjoint spans share a sub-column. The
    sub-columns are spread symmetrically around the column centre.
    """
    spans = [
        (min(rows[control], rows[target]), max(rows[control], rows[target]))
        for control, target in pairs
    ]
    order = sorted(range(len(pairs)), key=lambda index: spans[index])
    sub_column_of: dict[int, int] = {}
    sub_column_end: list[int] = []  # last occupied row of each sub-column
    for index in order:
        low, high = spans[index]
        for sub_column, end in enumerate(sub_column_end):
            if end < low:
                sub_column_end[sub_column] = high
                sub_column_of[index] = sub_column
                break
        else:
            sub_column_of[index] = len(sub_column_end)
            sub_column_end.append(high)

    sub_column_count = len(sub_column_end)
    if sub_column_count <= 1:
        return [0.0] * len(pairs)
    spacing = 2 * _CZ_STAGGER_HALF_WIDTH / (sub_column_count - 1)
    return [
        -_CZ_STAGGER_HALF_WIDTH + sub_column_of[index] * spacing
        for index in range(len(pairs))
    ]


def circuit_static_traces(
    columns: Sequence[CircuitColumn],
    qubit_ids: Sequence[int],
    colors: dict[str, str],
) -> list[Any]:
    """Wires, CZ connectors, and gate markers for the executed circuit.

    CZ layers draw one connector per pair, staggered into sub-columns when
    pairs overlap on the wire axis, so each two-qubit gate is legible. Gate
    markers carry their CZ partner in hover text.
    """
    go = _plotly()
    rows = {qubit_id: index for index, qubit_id in enumerate(qubit_ids)}
    column_count = max(len(columns), 1)

    wire_x: list[float | None] = []
    wire_y: list[float | None] = []
    for row_index in range(len(qubit_ids)):
        wire_x.extend([-0.5, column_count - 0.5, None])
        wire_y.extend([circuit_row_y(row_index)] * 2 + [None])
    wires = go.Scatter(
        x=wire_x,
        y=wire_y,
        mode="lines",
        line={"color": colors["site_edge"], "width": 1},
        hoverinfo="skip",
        name="Circuit wires",
        showlegend=False,
        meta={"bloqadeTraceKind": "circuitWire"},
        **_circuit_axes(),
    )

    connector_x: list[float | None] = []
    connector_y: list[float | None] = []
    gate_x: list[float] = []
    gate_y: list[float] = []
    gate_text: list[str] = []
    gate_symbol: list[str] = []
    gate_size: list[int] = []
    gate_color: list[str] = []
    gate_customdata: list[list[Any]] = []
    for column_index, column in enumerate(columns):
        base = [column.step_index, column.step_index + 1, column.description]
        if column.kind == "cz":
            pairs = [
                (control, target)
                for control, target in column.pairs
                if control in rows and target in rows
            ]
            offsets = cz_pair_offsets(pairs, rows)
            for (control, target), offset in zip(pairs, offsets, strict=True):
                x_value = column_index + offset
                control_y = circuit_row_y(rows[control])
                target_y = circuit_row_y(rows[target])
                connector_x.extend([x_value, x_value, None])
                connector_y.extend([control_y, target_y, None])
                for qubit_id, y_value, partner, role in (
                    (control, control_y, target, "control"),
                    (target, target_y, control, "target"),
                ):
                    gate_x.append(x_value)
                    gate_y.append(y_value)
                    gate_text.append("")
                    gate_symbol.append("circle")
                    gate_size.append(11)
                    gate_color.append(colors["cz"])
                    gate_customdata.append(
                        [*base, qubit_id, f"CZ {role}, paired with qubit {partner}"]
                    )
            continue

        for qubit_id in column.qubit_ids:
            if qubit_id not in rows:
                continue
            gate_x.append(float(column_index))
            gate_y.append(circuit_row_y(rows[qubit_id]))
            if column.kind in ("global_r", "global_rz"):
                gate_text.append(column.label)
                gate_symbol.append("diamond")
                gate_size.append(26)
                role_text = "global pulse"
            else:
                gate_text.append(column.label)
                gate_symbol.append("square")
                gate_size.append(22)
                role_text = "local pulse"
            gate_color.append(colors[_color_key(column.kind)])
            gate_customdata.append([*base, qubit_id, role_text])

    connectors = go.Scatter(
        x=connector_x,
        y=connector_y,
        mode="lines",
        line={"color": colors["cz"], "width": 2},
        hoverinfo="skip",
        name="CZ connectors",
        showlegend=False,
        meta={"bloqadeTraceKind": "circuitConnector"},
        **_circuit_axes(),
    )
    gates = go.Scatter(
        x=gate_x,
        y=gate_y,
        text=gate_text,
        customdata=gate_customdata,
        mode="markers+text",
        textposition="middle center",
        textfont={"color": "white", "size": 10},
        marker={
            "symbol": gate_symbol,
            "size": gate_size,
            "color": gate_color,
            "line": {"color": colors["paper"], "width": 1},
        },
        name="Circuit gates",
        showlegend=False,
        meta={"bloqadeTraceKind": "circuitGate"},
        hovertemplate=(
            "<b>step %{customdata[1]}</b>: %{customdata[2]}<br>"
            "qubit %{customdata[3]} · %{customdata[4]}<br>"
            "<i>click to jump to this step</i><extra></extra>"
        ),
        **_circuit_axes(),
    )
    return [wires, connectors, gates]


def _color_key(kind: GateKind) -> str:
    if kind in ("local_r", "global_r"):
        return "local_r"
    if kind in ("local_rz", "global_rz"):
        return "local_rz"
    return kind


def circuit_highlight_trace(
    columns: Sequence[CircuitColumn],
    step_index: int,
    qubit_count: int,
    colors: dict[str, str],
) -> Any:
    """Translucent band over the active column, or a cursor between columns.

    Plotly animation merges frame trace attributes into the live trace instead
    of replacing them, so both variants set the same attribute keys.
    """
    go = _plotly()
    y_low, y_high = circuit_y_range(qubit_count)
    y_low += 0.2
    y_high -= 0.2
    cursor = circuit_cursor(columns, step_index)
    active_column = next(
        (column for column in columns if column.step_index == step_index), None
    )
    if active_column is None:
        x_values = [cursor, cursor]
        y_values = [y_low, y_high]
        line = {"color": colors["route"], "width": 2, "dash": "dot"}
        fill = "none"
        fillcolor = "rgba(0,0,0,0)"
        opacity = 1.0
    else:
        half_width = 0.45
        x_values = [
            cursor - half_width,
            cursor + half_width,
            cursor + half_width,
            cursor - half_width,
            cursor - half_width,
        ]
        y_values = [y_low, y_low, y_high, y_high, y_low]
        color = colors[_color_key(active_column.kind)]
        line = {"color": color, "width": 0, "dash": "solid"}
        fill = "toself"
        fillcolor = color
        opacity = 0.25

    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        line=line,
        fill=fill,
        fillcolor=fillcolor,
        opacity=opacity,
        hoverinfo="skip",
        name="Circuit highlight",
        showlegend=False,
        visible=True,
        meta={"bloqadeTraceKind": "circuitHighlight"},
        **_circuit_axes(),
    )
