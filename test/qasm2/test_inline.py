import textwrap

from bloqade import qasm2
from kirin.dialects import ilist
from bloqade.qasm2.dialects import glob, noise, inline, parallel


def test_inline():

    lines = textwrap.dedent(
        """
    OPENQASM 2.0;

    qreg q[2];
    creg c[2];

    h q[0];
    CX q[0], q[1];
    barrier q[0], q[1];
    CX q[0], q[1];
    rx(pi/2) q[0];
    """
    )

    @qasm2.main.add(inline)
    def qasm2_inline_code():
        qasm2.inline(lines)

    qasm2_inline_code.print()


def test_inline_ext():

    lines = textwrap.dedent(
        r"""
    KIRIN {qasm2.uop, qasm2.expr, qasm2.parallel, qasm2.glob, qasm2.noise};

    qreg q[3];
    creg c[3];

    h q[0];
    CX q[0], q[1];
    barrier q[0], q[1];
    CX q[0], q[1];
    rx(pi/2) q[0];
    glob.U(1.0, 2.0, 3.0) {q}
    noise.PAULI1(1.0, 2.0, 3.0) q[0];
    parallel.U(1.0, 2.0, 3.0) {q[0]; q[1]; q[2];}
    parallel.CZ {
    q[0], q[1];
    q[2], q[3];
    }
    parallel.RZ(2.1) {
        q[0];
        q[2];
    }
    """
    )
    print(lines)

    @qasm2.main.add(inline).add(parallel).add(glob).add(noise).add(ilist)
    def qasm2_inline_code():
        qasm2.inline(lines)

    qasm2_inline_code.print()


if __name__ == "__main__":
    # test_inline()

    lines = textwrap.dedent(
        """
    KIRIN {qasm2.glob, qasm2.uop};
    include "qelib1.inc";

    qreg q1[2];
    qreg q2[3];

    glob.U(1.0, 2.0, 3.0) {q1, q2}
    """
    )

    print(lines)

    @qasm2.extended.add(inline)
    def qasm2_inline_code():
        qasm2.inline(lines)

    qasm2_inline_code.print()
