from kirin import ir, interp
from kirin.decl import statement, info
from bloqade.qasm2.types import QubitType
from bloqade.qasm2.parse import ast
from bloqade.qasm2.emit.gate import EmitQASM2Frame, EmitQASM2Gate

dialect = ir.Dialect("qasm2.parallel")

@statement(dialect=dialect)
class CZ(ir.Statement):
    name = "cz"
    traits = frozenset({ir.FromPythonCall()})
    ctrls: tuple[ir.SSAValue, ...] = info.argument(QubitType)
    qargs: tuple[ir.SSAValue, ...] = info.argument(QubitType)


@statement(dialect=dialect)
class UGate(ir.Statement):
    name = "u"
    traits = frozenset({ir.FromPythonCall()})
    theta: ir.SSAValue = info.argument(ir.types.Float)
    phi: ir.SSAValue = info.argument(ir.types.Float)
    lam: ir.SSAValue = info.argument(ir.types.Float)
    qargs: tuple[ir.SSAValue, ...] = info.argument(QubitType)


@statement(dialect=dialect)
class RZ(ir.Statement):
    name = "rz"
    traits = frozenset({ir.FromPythonCall()})
    theta: ir.SSAValue = info.argument(ir.types.Float)
    qargs: tuple[ir.SSAValue, ...] = info.argument(QubitType)


@dialect.register(key="emit.qasm2.gate")
class Parallel(interp.MethodTable):

    def _emit_parallel_qargs(
        self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, qargs: tuple[ir.SSAValue, ...]
    ):
        return [
            (emit.assert_node((ast.Name, ast.Bit), frame.get(qarg)),) for qarg in qargs
        ]

    @interp.impl(UGate)
    def ugate(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: UGate):
        qargs = self._emit_parallel_qargs(emit, frame, stmt.qargs)
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        phi = emit.assert_node(ast.Expr, frame.get(stmt.phi))
        lam = emit.assert_node(ast.Expr, frame.get(stmt.lam))
        frame.body.append(
            ast.ParaU3Gate(
                theta=theta, phi=phi, lam=lam, qargs=ast.ParallelQArgs(qargs=qargs)
            )
        )
        return ()

    @interp.impl(RZ)
    def rz(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: RZ):
        qargs = self._emit_parallel_qargs(emit, frame, stmt.qargs)
        theta = emit.assert_node(ast.Expr, frame.get(stmt.theta))
        frame.body.append(
            ast.ParaRzGate(theta=theta, qargs=ast.ParallelQArgs(qargs=qargs))
        )
        return ()

    @interp.impl(CZ)
    def cz(self, emit: EmitQASM2Gate, frame: EmitQASM2Frame, stmt: CZ):
        ctrls = self._emit_parallel_qargs(emit, frame, stmt.ctrls)
        qargs = self._emit_parallel_qargs(emit, frame, stmt.qargs)
        frame.body.append(
            ast.ParaCZGate(
                qargs=ast.ParallelQArgs(
                    qargs=[ctrl + qarg for ctrl, qarg in zip(ctrls, qargs)]
                )
            )
        )
        return ()
