from kirin import interp
from bloqade.qasm2.parse import ast
from bloqade.qasm2.emit.gate import EmitQASM2Gate, EmitQASM2Frame

from . import stmts
from ._dialect import dialect


@dialect.register(key="emit.qasm2.gate")
class UOp(interp.MethodTable):

    @interp.impl(stmts.CX)
    def emit_cx(
        self,
        emit: EmitQASM2Gate,
        frame: EmitQASM2Frame,
        stmt: stmts.CX,
    ):
        ctrl = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(ast.CXGate(ctrl=ctrl, qarg=qarg))
        return ()

    @interp.impl(stmts.UGate)
    def emit_ugate(
        self,
        emit: EmitQASM2Gate,
        frame: EmitQASM2Frame,
        stmt: stmts.UGate,
    ):
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        phi = emit.assert_node(ast.Expr, frame.get(stmt.phi))
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(ast.UGate(theta=theta, phi=phi, lam=lam, qarg=qarg))
        return ()

    @interp.impl(stmts.Barrier)
    def emit_barrier(
        self,
        emit: EmitQASM2Gate,
        frame: EmitQASM2Frame,
        stmt: stmts.Barrier,
    ):
        qargs = [
            emit.assert_node((ast.Bit, ast.Name), frame.get(qarg))
            for qarg in stmt.qargs
        ]
        frame.body.append(ast.Barrier(qargs=qargs))
        return ()

    @interp.impl(stmts.SX)
    @interp.impl(stmts.SXdag)
    @interp.impl(stmts.Id)
    @interp.impl(stmts.H)
    @interp.impl(stmts.X)
    @interp.impl(stmts.Y)
    @interp.impl(stmts.Z)
    @interp.impl(stmts.S)
    @interp.impl(stmts.Sdag)
    @interp.impl(stmts.T)
    @interp.impl(stmts.Tdag)
    def emit_single_qubit_gate(
        self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.SingleQubitGate
    ):
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(name=ast.Name(stmt.name), params=[], qargs=[qarg])
        )
        return ()

    @interp.impl(stmts.RX)
    @interp.impl(stmts.RY)
    @interp.impl(stmts.RZ)
    def emit_rotation(
        self,
        emit: EmitQASM2Gate,
        frame: EmitQASM2Frame,
        stmt: stmts.RX | stmts.RY | stmts.RZ,
    ):
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(name=ast.Name(stmt.name), params=[theta], qargs=[qarg])
        )
        return ()

    @interp.impl(stmts.U1)
    def emit_u1(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.U1):
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(name=ast.Name(stmt.name), params=[lam], qargs=[qarg])
        )
        return ()

    @interp.impl(stmts.U2)
    def emit_u2(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.U2):
        phi = emit.assert_node(ast.Expr, frame.get(stmt.phi))
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(name=ast.Name(stmt.name), params=[phi, lam], qargs=[qarg])
        )
        return ()

    @interp.impl(stmts.Swap)
    @interp.impl(stmts.CSX)
    @interp.impl(stmts.CZ)
    @interp.impl(stmts.CY)
    @interp.impl(stmts.CH)
    def emit_two_qubit_gate(
        self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.CZ
    ):
        ctrl = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(name=ast.Name(stmt.name), params=[], qargs=[ctrl, qarg])
        )
        return ()

    @interp.impl(stmts.CCX)
    def emit_ccx(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.CCX):
        ctrl1 = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl1))
        ctrl2 = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl2))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(
                name=ast.Name(stmt.name), params=[], qargs=[ctrl1, ctrl2, qarg]
            )
        )
        return ()

    @interp.impl(stmts.CSwap)
    def emit_cswap(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.CSwap):
        ctrl = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl))
        qarg1 = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg1))
        qarg2 = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg2))
        frame.body.append(
            ast.Instruction(
                name=ast.Name(stmt.name), params=[], qargs=[ctrl, qarg1, qarg2]
            )
        )
        return ()

    @interp.impl(stmts.CRZ)
    @interp.impl(stmts.CRY)
    @interp.impl(stmts.CRX)
    def emit_cr(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.CRX):
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        ctrl = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(name=ast.Name(stmt.name), params=[lam], qargs=[ctrl, qarg])
        )
        return ()

    @interp.impl(stmts.CU1)
    def emit_cu1(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.CU1):
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        ctrl = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(name=ast.Name(stmt.name), params=[lam], qargs=[ctrl, qarg])
        )
        return ()

    @interp.impl(stmts.CU3)
    def emit_cu3(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.CU3):
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        phi = emit.assert_node(ast.Expr, frame.get(stmt.phi))
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        ctrl = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(
                name=ast.Name(stmt.name), params=[theta, phi, lam], qargs=[ctrl, qarg]
            )
        )
        return ()

    @interp.impl(stmts.CU)
    def emit_cu(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.CU):
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        phi = emit.assert_node(ast.Expr, frame.get(stmt.phi))
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        gamma = emit.assert_node(ast.Expr, frame.get(stmt.gamma))
        ctrl = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(
                name=ast.Name(stmt.name),
                params=[theta, phi, lam, gamma],
                qargs=[ctrl, qarg],
            )
        )
        return ()

    @interp.impl(stmts.RZZ)
    @interp.impl(stmts.RXX)
    def emit_r2q(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: stmts.RZZ):
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        ctrl = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.ctrl))
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        frame.body.append(
            ast.Instruction(
                name=ast.Name(stmt.name), params=[theta], qargs=[ctrl, qarg]
            )
        )
        return ()
