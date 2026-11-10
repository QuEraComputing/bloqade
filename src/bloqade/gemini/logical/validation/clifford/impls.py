"""Gemini method tables for compiler validation."""

from __future__ import annotations

from kirin import ir, interp as _interp
from bloqade.squin import gate
from kirin.analysis import ForwardFrame, const
from kirin.dialects import scf
from bloqade.analysis import address
from bloqade.lanes.validation.steane import _SteaneLogicalValidationAnalysis
from bloqade.lanes.dialects.steane713 import extensions as extensions
from bloqade.lanes.validation.static_call import (
    UnrollFixableError,
)

from bloqade import qubit


@scf.dialect.register(key="lanes.validate.logical")
class __ScfSteaneLogicalValidation(_interp.MethodTable):
    @_interp.impl(scf.IfElse)
    def if_else(
        self,
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: scf.IfElse,
    ):
        interp.add_validation_error(
            stmt,
            ir.ValidationError(
                stmt, "If statements are not supported in logical Gemini programs!"
            ),
        )
        return (interp.lattice.bottom(),)

    @_interp.impl(scf.For)
    def for_loop(
        self,
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: scf.For,
    ):
        if not isinstance(stmt.iterable.hints.get("const"), const.Value):
            interp.add_validation_error(
                stmt,
                UnrollFixableError(
                    stmt,
                    "Non-constant iterable in for loop is not supported in Gemini logical programs!",
                ),
            )

        return (interp.lattice.bottom(),)


@gate.dialect.register(key="lanes.validate.logical")
class __GateSteaneLogicalValidation(_interp.MethodTable):
    @_interp.impl(gate.stmts.U3)
    @_interp.impl(gate.stmts.T)
    @_interp.impl(gate.stmts.Rx)
    @_interp.impl(gate.stmts.Ry)
    @_interp.impl(gate.stmts.Rz)
    def non_clifford(
        self,
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.SingleQubitGate | gate.stmts.RotationGate,
    ):
        if interp.check_first_gate(stmt.qubits):
            return ()

        interp.add_validation_error(
            stmt,
            ir.ValidationError(
                stmt,
                f"Non-clifford gate {stmt.name} can only be used for initial state preparation, i.e. as the first gate!",
            ),
        )
        return ()

    @_interp.impl(gate.stmts.X)
    @_interp.impl(gate.stmts.Y)
    @_interp.impl(gate.stmts.SqrtX)
    @_interp.impl(gate.stmts.SqrtY)
    @_interp.impl(gate.stmts.Z)
    @_interp.impl(gate.stmts.H)
    @_interp.impl(gate.stmts.S)
    def clifford(
        self,
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.SingleQubitGate,
    ):
        # NOTE: ignore result, but make sure the first gate flag is set to False
        interp.check_first_gate(stmt.qubits)

        return ()

    @_interp.impl(gate.stmts.CX)
    @_interp.impl(gate.stmts.CY)
    @_interp.impl(gate.stmts.CZ)
    @_interp.impl(gate.stmts.Swap)
    def two_qubit_gate(
        self,
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.ControlledGate | gate.stmts.TwoQubitGate,
    ):
        # NOTE: both register operands are acted on; iterate over `args` so this
        # covers the (controls, targets) and (qubits1, qubits2) spellings alike
        for qubits in stmt.args:
            interp.check_first_gate(qubits)

        return ()


@extensions.dialect.register(key="lanes.validate.logical")
class __ExtensionsSteaneLogicalValidation(_interp.MethodTable):
    @_interp.impl(extensions.stmts.StarRz)
    def star_rz(
        self,
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: extensions.stmts.StarRz,
    ):
        interp.check_first_gate(stmt.qubits)
        return ()


@qubit.dialect.register(key="lanes.validate.logical")
class QubitMethods(_interp.MethodTable):
    @_interp.impl(qubit.stmts.New)
    def check_qubit_allocation(
        self,
        interp: _SteaneLogicalValidationAnalysis,
        frame: ForwardFrame,
        stmt: qubit.stmts.New,
    ):
        qubit_val = interp.addr_frame.get(stmt.result)
        if not isinstance(qubit_val, address.AddressQubit):
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Cannot determine qubit address location.",
                ),
            )
            return (interp.lattice.bottom(),)

        if qubit_val.data >= interp.max_qubits:
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    f"Qubit allocations exceeded {interp.max_qubits}.",
                ),
            )

        return (interp.lattice.bottom(),)
