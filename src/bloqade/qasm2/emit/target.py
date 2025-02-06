import io

from kirin import ir
from rich.console import Console
from kirin.analysis import CallGraph
from bloqade.qasm2.parse import ast, pprint

from .gate import EmitQASM2Gate
from .main import EmitQASM2Main


class QASM2:

    def __init__(
        self,
        custom_gate: bool = False,
        qelib1: bool = False,
    ) -> None:
        self.qelib1 = qelib1
        self.custom_gate = custom_gate

    def emit(self, entry: ir.Method):
        main = EmitQASM2Main(entry.dialects)
        main.run(entry, tuple(ast.Name(name) for name in entry.arg_names[1:])).expect()
        main = main.output
        assert main is not None, f"failed to emit {entry.sym_name}"

        extra = []
        if self.qelib1:
            extra.append(ast.Include("qelib1.inc"))
        if self.custom_gate:
            cg = CallGraph(entry)
            gate = EmitQASM2Gate(entry.dialects)

            for _, fn in cg.defs.items():
                if fn is entry:
                    continue

                gate.run(
                    fn, tuple(ast.Name(name) for name in fn.arg_names[1:])
                ).expect()
                assert gate.output is not None, f"failed to emit {fn.sym_name}"
                extra.append(gate.output)

        main.statements = extra + main.statements
        return main

    def emit_str(self, entry: ir.Method) -> str:
        console = Console(
            file=io.StringIO(),
            force_terminal=False,
            force_interactive=False,
            force_jupyter=False,
            record=True,
        )
        pprint(QASM2(custom_gate=False, qelib1=True).emit(entry), console=console)
        return console.export_text()
