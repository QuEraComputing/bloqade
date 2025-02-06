import io

import rich
from kirin import ir
from bloqade import qasm2
from bloqade.noise import native
from bloqade.qasm2.emit import QASM2
from bloqade.qasm2.parse import pprint
from bloqade.noise.native.rewrite import RemoveNoisePass

simulation = qasm2.main.add(native)


@simulation
def test_atom_loss():
    q = qasm2.qreg(2)
    native.atom_loss_channel(0.7, q[0])
    native.atom_loss_channel(0.7, q[1])
    native.cz_pauli_channel(
        0.1, 0.4, 0.3, 0.2, 0.2, 0.2, q[0], q[1], paired=False
    )  # no noise here
    qasm2.cz(q[0], q[1])
    native.atom_loss_channel(0.4, q[0])
    native.atom_loss_channel(0.7, q[1])
    native.cz_pauli_channel(0.1, 0.4, 0.3, 0.2, 0.2, 0.2, q[0], q[1], paired=False)
    qasm2.cz(q[0], q[1])
    return q


RemoveNoisePass(simulation)(test_atom_loss)

new_dialects = set()

for stmt in test_atom_loss.code.walk():
    new_dialects.add(stmt.dialect)


new_dialect_group = ir.DialectGroup(new_dialects)
test_atom_loss.dialects = new_dialect_group

out = io.StringIO()
console = rich.console.Console(
    record=True,
    file=out,
    force_terminal=False,
    force_interactive=False,
    force_jupyter=False,
)
pprint(QASM2(custom_gate=False, qelib1=True).emit(test_atom_loss), console=console)
print(console.export_text())
