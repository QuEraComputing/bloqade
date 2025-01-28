from kirin import ir
from kirin.analysis import CallGraph
from bloqade.qasm2.parse import ast

from .gate import EmitQASM2Gate
from .main import EmitQASM2Main


class QASM2:

    def __init__(
        self,
        *,
        qelib1: bool = True,
        custom_gate: bool = True,
    ) -> None:
        self.gate: EmitQASM2Gate = EmitQASM2Gate()
        self.main: EmitQASM2Main = EmitQASM2Main()
        self.qelib1 = qelib1
        self.custom_gate = custom_gate

    def emit(self, entry: ir.Method):
        self.main.run(
            entry, tuple(ast.Name(name) for name in entry.arg_names[1:])
        ).expect()
        main = self.main.output
        assert main is not None, f"failed to emit {entry.sym_name}"

        extra = []
        if self.qelib1:
            extra.append(ast.Include("qelib1.inc"))
        if self.custom_gate:
            cg = CallGraph(entry)
            for _, fn in cg.defs.items():
                if fn is entry:
                    continue

                self.gate.run(
                    fn, tuple(ast.Name(name) for name in fn.arg_names[1:])
                ).expect()
                assert self.gate.output is not None, f"failed to emit {fn.sym_name}"
                extra.append(self.gate.output)

        main.statements = extra + main.statements
        return main
