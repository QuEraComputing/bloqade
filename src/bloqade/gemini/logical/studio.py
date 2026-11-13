"""Export a static Gemini logical kernel as a Gemini Studio share link.

The link describes the logical circuit, not its compiled atom moves. Gemini
Studio compiles the imported circuit using its own compiler and settings.
"""

import json
import math
from typing import Any, cast

from kirin import ir
from bloqade.qubit import stmts as qubit_stmts
from kirin.analysis import const
from kirin.dialects import py, func, ilist
from bloqade.analysis import address
from bloqade.squin.gate import stmts as gate_stmts
from bloqade.decoders.dialects import annotate
from bloqade.lanes.dialects.arch import stmts as common_qubit_stmts
from bloqade.analysis.address.lattice import AddressReg, AddressQubit
from bloqade.lanes.dialects.steane713 import stmts as logical_stmts

_STUDIO_URL = "https://bloqade.quera.com/studio/gemini/"
# Studio accepts v13 links in both the current deployment and newer builds.
_LINK_VERSION = 13
_MAX_QUBITS = 10
_MAX_COLUMNS = 1_000
_SPANNED_GATES = frozenset({"h", "cx", "cy"})
_SINGLE_GATES: dict[type[ir.Statement], str] = {
    gate_stmts.H: "h",
    gate_stmts.X: "x",
    gate_stmts.Y: "y",
    gate_stmts.Z: "z",
}
_ADJOINT_GATES: dict[type[ir.Statement], str] = {
    gate_stmts.S: "s",
    gate_stmts.SqrtX: "sqrt_x",
    gate_stmts.SqrtY: "sqrt_y",
}
_TWO_QUBIT_GATES: dict[type[ir.Statement], str] = {
    gate_stmts.CX: "cx",
    gate_stmts.CY: "cy",
    gate_stmts.CZ: "cz",
}


def _constant(value: ir.SSAValue) -> int | float:
    owner = value.owner
    if isinstance(owner, py.Constant) and type(owner.value) in (int, float):
        return cast(int | float, owner.value)
    hint = value.hints.get("const")
    if isinstance(hint, const.Value) and type(hint.data) in (int, float):
        return cast(int | float, hint.data)
    raise ValueError(
        "Gemini Studio requires constant allocation and preparation values"
    )


def _qubit_ids(
    value: ir.SSAValue, entries: dict[ir.SSAValue, address.Address]
) -> tuple[int, ...]:
    resolved = entries.get(value)
    if isinstance(resolved, AddressReg):
        return tuple(resolved.data)
    if isinstance(resolved, AddressQubit):
        return (resolved.data,)
    raise ValueError("Gemini Studio requires statically resolved qubit operands")


def _placement(pinned: dict[int, int], count: int) -> list[dict[str, int]]:
    if len(set(pinned.values())) != len(pinned):
        raise ValueError("Gemini Studio requires distinct pinned home slots")
    used = set(pinned.values())
    slots = dict(pinned)
    for qubit in range(count):
        if qubit not in slots:
            slot = next(
                candidate for candidate in range(_MAX_QUBITS) if candidate not in used
            )
            slots[qubit] = slot
            used.add(slot)
    # Studio assigns every omitted qubit the lowest free slot, not its index.
    return [{"qubit": qubit, "slot": slots[qubit]} for qubit in range(count)]


def to_studio_url(kernel: ir.Method) -> str:
    """Return a Gemini Studio link for a static, Studio-supported logical kernel.

    Supported operations are H, X, Y, Z, S, sqrt(X), sqrt(Y), CX, CY, CZ,
    their broadcasts, constant logical initialization, ``qalloc_at`` home
    slots, and one terminal measurement of the whole register. Each source
    gate statement gets its own timestep group; H, CX, and CY occupy three
    columns in Studio. The link shows Studio's own default measurement and
    post-processing, not custom detector/observable annotations in ``kernel``.

    The kernel must already be flattened, normally by using
    ``@logical.kernel(aggressive_unroll=True)``. Unsupported quantum operations
    and dynamic control flow raise ``ValueError`` instead of being omitted.

    Studio's link format pins either every qubit or none. If a kernel pins only
    some qubits, the remaining qubits are pinned to the lowest free slots in
    Studio. Identity-only pins (``qalloc_at([0, 1, ...])``) cannot be preserved:
    Studio renders those as an unpinned allocation and may choose another layout.
    """
    if kernel.args:
        raise ValueError("Gemini Studio links require a kernel without arguments")
    blocks = kernel.callable_region.blocks
    if len(blocks) != 1:
        raise ValueError("Gemini Studio links require a single straight-line block")

    analysis = address.AddressAnalysis(kernel.dialects)
    frame, _ = analysis.run(kernel)
    count = analysis.qubit_count
    if not 1 <= count <= _MAX_QUBITS:
        raise ValueError(f"Gemini Studio supports 1–{_MAX_QUBITS} logical qubits")

    pinned: dict[int, int] = {}
    prep: dict[int, dict[str, int | float]] = {}
    prepared_ids: set[int] = set()
    gates: list[dict[str, Any]] = []
    column = 0
    measured = False

    for stmt in blocks[0].stmts:
        if stmt.regions:
            raise ValueError("Gemini Studio links do not support dynamic control flow")

        if isinstance(stmt, qubit_stmts.New):
            continue
        if isinstance(stmt, common_qubit_stmts.NewAt):
            (qubit,) = _qubit_ids(stmt.qubit, frame.entries)
            zone = _constant(stmt.zone_id)
            word = _constant(stmt.word_id)
            site = _constant(stmt.site_id)
            if (
                type(zone) is not int
                or type(word) is not int
                or type(site) is not int
                or zone != 0
                or site != 0
                or word not in range(0, 2 * _MAX_QUBITS, 2)
            ):
                raise ValueError("Gemini Studio only supports logical home slots 0–9")
            pinned[qubit] = word // 2
            continue

        if isinstance(stmt, (logical_stmts.Initialize, gate_stmts.U3)):
            if gates or measured:
                raise ValueError("Gemini Studio only supports preparation before gates")
            values = (_constant(stmt.theta), _constant(stmt.phi), _constant(stmt.lam))
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Gemini Studio preparation angles must be finite")
            for qubit in _qubit_ids(stmt.qubits, frame.entries):
                if qubit in prepared_ids:
                    raise ValueError(
                        "Gemini Studio cannot represent repeated preparation"
                    )
                prepared_ids.add(qubit)
                if any(values):
                    # SQuIN's low-level U3 stores turns; Studio stores radians.
                    prep[qubit] = {
                        "qubit": qubit,
                        "theta": round(values[0] * math.tau, 6),
                        "phi": round(values[1] * math.tau, 6),
                        "lam": round(values[2] * math.tau, 6),
                    }
            continue

        if isinstance(stmt, logical_stmts.TerminalLogicalMeasurement):
            measured_ids = _qubit_ids(stmt.qubits, frame.entries)
            if (
                measured
                or len(measured_ids) != count
                or set(measured_ids) != set(range(count))
            ):
                raise ValueError(
                    "Gemini Studio requires one terminal measurement of all qubits"
                )
            if measured_ids != tuple(range(count)):
                raise ValueError(
                    "Gemini Studio requires terminal measurement in allocation order"
                )
            measured = True
            continue

        if isinstance(stmt, (gate_stmts.H, gate_stmts.X, gate_stmts.Y, gate_stmts.Z)):
            name = _SINGLE_GATES[type(stmt)]
            if measured:
                raise ValueError("Gemini Studio cannot place gates after measurement")
            ids = _qubit_ids(stmt.qubits, frame.entries)
            if len(set(ids)) != len(ids):
                raise ValueError("A broadcast gate repeats a qubit")
            targets = [(qubit,) for qubit in ids]
        elif isinstance(stmt, (gate_stmts.S, gate_stmts.SqrtX, gate_stmts.SqrtY)):
            name = _ADJOINT_GATES[type(stmt)] + ("_adj" if stmt.adjoint else "")
            if measured:
                raise ValueError("Gemini Studio cannot place gates after measurement")
            ids = _qubit_ids(stmt.qubits, frame.entries)
            if len(set(ids)) != len(ids):
                raise ValueError("A broadcast gate repeats a qubit")
            targets = [(qubit,) for qubit in ids]
        elif isinstance(stmt, (gate_stmts.CX, gate_stmts.CY, gate_stmts.CZ)):
            name = _TWO_QUBIT_GATES[type(stmt)]
            if measured:
                raise ValueError("Gemini Studio cannot place gates after measurement")
            controls = _qubit_ids(stmt.controls, frame.entries)
            other = _qubit_ids(stmt.targets, frame.entries)
            if len(controls) != len(other):
                raise ValueError("A two-qubit broadcast has unequal operand lengths")
            targets = list(zip(controls, other))
            if len(set(controls + other)) != 2 * len(targets):
                raise ValueError("Two-qubit gates in one broadcast must be disjoint")
        else:
            if isinstance(
                stmt,
                (func.Return, annotate.stmts.SetDetector, annotate.stmts.SetObservable),
            ):
                continue
            if isinstance(stmt, (py.Constant, ilist.New)) or type(
                stmt
            ).__module__.startswith("kirin.dialects.py."):
                continue
            raise ValueError(
                f"Gemini Studio cannot represent {stmt.name} ({type(stmt).__name__})"
            )

        if targets:
            span = 3 if name in _SPANNED_GATES else 1
            if column + span > _MAX_COLUMNS:
                raise ValueError("Gemini Studio's 1000-column limit was exceeded")
            gates.extend(
                {"gate": name, "column": column, "targets": list(pair)}
                for pair in targets
            )
            column += span

    if not measured:
        raise ValueError("Gemini Studio links require terminal logical measurement")

    snapshot: dict[str, Any] = {"version": _LINK_VERSION, "qubits": count}
    if gates:
        snapshot["gates"] = gates
    if prep:
        snapshot["prep"] = [prep[qubit] for qubit in sorted(prep)]
    if pinned:
        snapshot["placement"] = _placement(pinned, count)
    # Studio's link codec removes quotes from identifier-shaped JSON tokens.
    payload = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=True).replace(
        '"', ""
    )
    return f"{_STUDIO_URL}#circuit={payload}"
