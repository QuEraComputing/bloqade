from kirin import interp
from bloqade.qasm2.parse import ast
from bloqade.qasm2.emit.main import EmitQASM2Main, EmitQASM2Frame

from . import stmts
from ._dialect import dialect


@dialect.register(key="emit.qasm2.main")
class Core(interp.MethodTable):

    @interp.impl(stmts.CRegNew)
    def emit_creg_new(
        self, emit: EmitQASM2Main, frame: EmitQASM2Frame, stmt: stmts.CRegNew
    ):
        n_bits = emit.assert_node(ast.Number, frame.get(stmt.n_bits))
        # check if its int first, because Int.is_integer() is added for >=3.12
        assert isinstance(n_bits.value, int), "expected integer"
        name = emit.ssa_id[stmt.result]
        frame.body.append(ast.CReg(name=name, size=int(n_bits.value)))
        return (ast.Name(name),)

    @interp.impl(stmts.QRegNew)
    def emit_qreg_new(
        self, emit: EmitQASM2Main, frame: EmitQASM2Frame, stmt: stmts.QRegNew
    ):
        n_bits = emit.assert_node(ast.Number, frame.get(stmt.n_qubits))
        assert isinstance(n_bits.value, int), "expected integer"
        name = emit.ssa_id[stmt.result]
        frame.body.append(ast.QReg(name=name, size=int(n_bits.value)))
        return (ast.Name(name),)

    @interp.impl(stmts.Reset)
    def emit_reset(self, emit: EmitQASM2Main, frame: EmitQASM2Frame, stmt: stmts.Reset):
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(ast.Reset(qarg=qarg))
        return ()

    @interp.impl(stmts.Measure)
    def emit_measure(
        self, emit: EmitQASM2Main, frame: EmitQASM2Frame, stmt: stmts.Measure
    ):
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        carg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.carg))
        frame.body.append(ast.Measure(qarg=qarg, carg=carg))
        return ()

    @interp.impl(stmts.CRegEq)
    def emit_creg_eq(
        self, emit: EmitQASM2Main, frame: EmitQASM2Frame, stmt: stmts.CRegEq
    ):
        lhs = emit.assert_node(ast.Expr, frame.get(stmt.lhs))
        rhs = emit.assert_node(ast.Expr, frame.get(stmt.rhs))
        return (ast.Cmp(lhs=lhs, rhs=rhs),)

    @interp.impl(stmts.CRegGet)
    @interp.impl(stmts.QRegGet)
    def emit_qreg_get(
        self,
        emit: EmitQASM2Main,
        frame: EmitQASM2Frame,
        stmt: stmts.QRegGet | stmts.CRegGet,
    ):
        reg = emit.assert_node(ast.Name, frame.get(stmt.reg))
        idx = emit.assert_node(ast.Number, frame.get(stmt.idx))
        assert isinstance(idx.value, int)
        return (ast.Bit(reg, int(idx.value)),)
