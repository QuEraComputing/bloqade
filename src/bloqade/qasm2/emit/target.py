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

    def __init__(
        self,
        custom_gate: bool = False,
        qelib1: bool = False,
    ) -> None:
        self.qelib1 = qelib1
        self.custom_gate = custom_gate

    def emit(self, entry: ir.Method):
        from bloqade.qasm2.groups import gate, main

        assert len(entry.args) == 0, "entry method should not have arguments"
        entry = entry.similar()
        QASM2Fold(entry.dialects).fixpoint(entry)
        Py2QASM(entry.dialects)(entry)
        target_main = EmitQASM2Main(main)
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
            target_gate = EmitQASM2Gate(gate)

            for _, fn in cg.defs.items():
                if fn is entry:
                    continue

                target_gate.run(
                    fn, tuple(ast.Name(name) for name in fn.arg_names[1:])
                ).expect()
                assert target_gate.output is not None, f"failed to emit {fn.sym_name}"
                extra.append(target_gate.output)

        main_program.statements = extra + main_program.statements
        return main_program

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
