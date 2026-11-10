from kirin import types as kirin_types, interp
from kirin.dialects import ilist
from bloqade.lanes.dialects.steane713 import stmts
from bloqade.analysis.measure_id.lattice import (
    MeasureId,
    AnyMeasureId,
    RawMeasureId,
    MeasureIdTuple,
)
from bloqade.analysis.measure_id.analysis import MeasureIDFrame, MeasurementIDAnalysis
from bloqade.lanes.dialects.steane713._dialect import dialect


@dialect.register(key="measure_id")
class LogicalQubit(interp.MethodTable):
    @interp.impl(stmts.TerminalLogicalMeasurement)
    def terminal_measurement(
        self,
        interp_: MeasurementIDAnalysis,
        frame: MeasureIDFrame,
        stmt: stmts.TerminalLogicalMeasurement,
    ):

        qubits_type = stmt.qubits.type
        if qubits_type.is_structurally_equal(kirin_types.Bottom):
            return (AnyMeasureId(),)

        # NOTE: an unresolved register type is a *reason to give up*, not a bug,
        # and every other unknowable case below says so by returning
        # `AnyMeasureId`. An `assert` here instead turned an ordinary untyped
        # parameter -- `def sub(reg): return terminal_measure(reg)` -- into an
        # `AssertionError` that `ValidationSuite` caught and re-reported as
        # "Validation pass '...' failed:" plus a traceback. The arity check
        # guards `vars[1]` below, which would raise `IndexError` on a `Generic`
        # that is not an `IList[T, Len]`.
        if (
            not isinstance(qubits_type, kirin_types.Generic)
            or len(qubits_type.vars) < 2
        ):
            return (AnyMeasureId(),)

        if not isinstance(len_var := qubits_type.vars[1], kirin_types.Literal):
            return (AnyMeasureId(),)

        if not isinstance(num_logical_qubits := len_var.data, int):
            return (AnyMeasureId(),)

        if (num_physical_qubits := stmt.num_physical_qubits) is not None:

            def logical_to_physical(
                logical_address: int,
            ) -> MeasureId:
                raw_measure_ids = map(
                    RawMeasureId,
                    range(
                        interp_.measure_count + 1,
                        interp_.measure_count + num_physical_qubits + 1,
                    ),
                )
                interp_.measure_count += num_physical_qubits
                return MeasureIdTuple(tuple(raw_measure_ids), ilist.IList)

        else:

            def logical_to_physical(
                logical_address: int,
            ) -> MeasureId:
                return AnyMeasureId()

        return (
            MeasureIdTuple(
                tuple(map(logical_to_physical, range(num_logical_qubits))), ilist.IList
            ),
        )
