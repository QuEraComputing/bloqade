from kirin import ir
from bloqade import qasm2
from bloqade.noise import native
from kirin.dialects import func
from bloqade.qasm2.emit import QASM2
from bloqade.qasm2.dialects import indexing
from bloqade.noise.native.rewrite import RemoveNoisePass

simulation = ir.DialectGroup(
    [func, indexing, qasm2.core, qasm2.uop, qasm2.expr, native]
)


def test():

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

    expected = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
cz q[0], q[1];
cz q[0], q[1];
"""

    assert QASM2(qelib1=True).emit_str(test_atom_loss) == expected
