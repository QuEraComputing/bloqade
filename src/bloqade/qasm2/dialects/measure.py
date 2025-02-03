from kirin import ir, interp
from kirin.decl import info, statement
from bloqade.qasm2.parse import ast
from bloqade.qasm2.types import BitType, QubitType
from bloqade.qasm2.emit.main import EmitQASM2Main, EmitQASM2Frame

dialect = ir.Dialect("qasm2.measure")


@statement(dialect=dialect)
class Measure(ir.Statement):
    """Measure a qubit and store the result in a bit."""

    name = "measure"
    traits = frozenset({ir.FromPythonCall()})
    qarg: ir.SSAValue = info.argument(QubitType)
    """qarg (Qubit): The qubit to measure."""
    carg: ir.SSAValue = info.argument(BitType)
    """carg (Bit): The bit to store the result in."""


@dialect.register(key="emit.qasm2.main")
class Core(interp.MethodTable):

    @interp.impl(Measure)
    def emit_measure(self, emit: EmitQASM2Main, frame: EmitQASM2Frame, stmt: Measure):
        qarg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.qarg))
        carg = emit.assert_node((ast.Bit, ast.Name), frame.get(stmt.carg))
        frame.body.append(ast.Measure(qarg=qarg, carg=carg))
        return ()
