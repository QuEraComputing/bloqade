"""Tests for PhysicalTerminalMeasurementValidation."""

import bloqade.squin as squin
from kirin.dialects import ilist
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.native.upstream.squin2native import SquinToNative
from bloqade.lanes.validation.terminal_measure import (
    PhysicalTerminalMeasurementValidation,
)

from bloqade.gemini.common import impl as _common_impl  # noqa: F401


def _validate(kernel):
    """Run SquinToNative + AggressiveUnroll then validate — mirrors the pipeline."""
    out = SquinToNative().emit(kernel)
    AggressiveUnroll(out.dialects, no_raise=True).fixpoint(out)
    _, errors = PhysicalTerminalMeasurementValidation().run(out)
    return errors


def test_valid_kernel_passes():
    """Single measure consuming all qubits — no errors."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.qubit.measure(ilist.IList([reg[0], reg[1]]))  # type: ignore[arg-type]

    assert _validate(kernel) == []


def test_missing_measure_raises():
    """No measure statement — validation error about missing terminal measure."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)  # noqa: F841

    errors = _validate(kernel)
    assert len(errors) >= 1
    assert any(
        "terminal" in str(e).lower() or "measure" in str(e).lower() for e in errors
    )


def test_partial_measure_raises():
    """Measure covers only 1 of 2 allocated qubits — validation error."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.qubit.measure(ilist.IList([reg[0]]))  # type: ignore[arg-type]

    errors = _validate(kernel)
    assert len(errors) >= 1
    assert any("2" in str(e) or "allocated" in str(e).lower() for e in errors)


def test_multiple_measures_raises():
    """Two measure statements — validation error."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.qubit.measure(ilist.IList([reg[0]]))  # type: ignore[arg-type]
        squin.qubit.measure(ilist.IList([reg[1]]))  # type: ignore[arg-type]

    errors = _validate(kernel)
    assert len(errors) >= 1
    assert any(
        "multiple" in str(e).lower() or "more than one" in str(e).lower()
        for e in errors
    )
