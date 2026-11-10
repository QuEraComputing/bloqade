from kirin import types
from kirin.interp import Frame, MethodTable, impl
from bloqade.types import QubitType, MeasurementResultType
from kirin.dialects import ilist
from kirin.analysis.typeinfer import TypeInference
from bloqade.lanes.dialects.steane713 import stmts
from bloqade.lanes.dialects.steane713._dialect import dialect


@dialect.register(key="typeinfer")
class TypeInfer(MethodTable):
    @impl(stmts.TerminalLogicalMeasurement)
    def terminal_logical_measurement(
        self,
        interp_: TypeInference,
        frame: Frame[types.TypeAttribute],
        stmt: stmts.TerminalLogicalMeasurement,
    ):
        qubits_type = frame.get(stmt.qubits)
        if qubits_type.is_structurally_equal(
            types.Bottom
        ) or not qubits_type.is_subseteq(ilist.IListType[QubitType, types.Any]):
            return (types.Bottom,)

        assert isinstance(qubits_type, types.Generic)
        outer_len_type = qubits_type.vars[1]
        if stmt.num_physical_qubits is not None:
            inner_len_type = types.Literal(stmt.num_physical_qubits)
        else:
            inner_len_type = types.Any

        inner_type = ilist.IListType[MeasurementResultType, inner_len_type]
        return (ilist.IListType[inner_type, outer_len_type],)
