import io

from kirin import ir
from rich.console import Console
from kirin.analysis import CallGraph
from bloqade.qasm2.parse import ast, pprint
from bloqade.qasm2.passes.fold import QASM2Fold
from bloqade.qasm2.passes.py2qasm import Py2QASM

from .gate import EmitQASM2Gate
from .main import EmitQASM2Main


class QASM2:
    """QASM2 target for Bloqade kernels.

    QASM2 target that accepts a Bloqade kernel and produces an AST that you can then obtain a string for printing or saving as a file.
    """

    def __init__(
        self,
        main_target: ir.DialectGroup | None = None,
        gate_target: ir.DialectGroup | None = None,
        qelib1: bool = True,
        custom_gate: bool = True,
    ) -> None:
        """Initialize the QASM2 target.

        Args:
            main_target (ir.DialectGroup | None):
                The dialects that were used in the definition of the kernel. This is used to
                generate the correct header for the resulting QASM2 AST. Argument defaults to `None`.
                Internally set to the `qasm2.main` group of dialects.
            gate_target (ir.DialectGroup | None):
                The dialects involved in defining any custom gates in the kernel. Argument defaults to `None`.
                Internally set to the `qasm2.gate` group of dialects.
            qelib1 (bool):
                Include the `include "qelib1.inc"` line in the resulting QASM2 AST that's
                submitted to qBraid. Defaults to `True`.
            custom_gate (bool):
                Include the custom gate definitions in the resulting QASM2 AST. Defaults to `True`. If `False`, all the qasm2.gate will be inlined.

        """
        from bloqade import qasm2

        self.main_target = main_target or qasm2.main
        self.gate_target = gate_target or qasm2.gate
        self.qelib1 = qelib1
        self.custom_gate = custom_gate

    def emit(self, entry: ir.Method) -> ast.MainProgram:
        """Emit a QASM2 AST from the Bloqade kernel.

        Args:
            entry (ir.Method):
                The Bloqade kernel to convert to the QASM2 AST

        Returns:
            ast.MainProgram:
                A QASM2 AST object

        """
        assert len(entry.args) == 0, "entry method should not have arguments"
        entry = entry.similar()
        QASM2Fold(entry.dialects, inline_gate_subroutine=not self.custom_gate).fixpoint(
            entry
        )
        Py2QASM(entry.dialects)(entry)
        target_main = EmitQASM2Main(self.main_target)
        target_main.run(
            entry, tuple(ast.Name(name) for name in entry.arg_names[1:])
        ).expect()

        main_program = target_main.output
        assert main_program is not None, f"failed to emit {entry.sym_name}"

        extra = []
        if self.qelib1:
            extra.append(ast.Include("qelib1.inc"))

        if self.custom_gate:
            cg = CallGraph(entry)
            target_gate = EmitQASM2Gate(self.gate_target)

            for _, fn in cg.defs.items():
                if fn is entry:
                    continue

                fn = fn.similar(self.gate_target)
                QASM2Fold(fn.dialects).fixpoint(fn)
                Py2QASM(fn.dialects)(fn)
                target_gate.run(
                    fn, tuple(ast.Name(name) for name in fn.arg_names[1:])
                ).expect()
                assert target_gate.output is not None, f"failed to emit {fn.sym_name}"
                extra.append(target_gate.output)

        main_program.statements = extra + main_program.statements
        return main_program

    def emit_str(self, entry: ir.Method) -> str:
        """Emit a QASM2 AST from the Bloqade kernel.

        Args:
            entry (ir.Method):
                The Bloqade kernel to convert to the QASM2 AST

        Returns:
            str:
                A string with the QASM2 representation of the kernel

        """
        console = Console(
            file=io.StringIO(),
            force_terminal=False,
            force_interactive=False,
            force_jupyter=False,
            record=True,
        )
        pprint(self.emit(entry), console=console)
        return console.export_text()
