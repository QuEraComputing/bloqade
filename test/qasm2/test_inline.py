import textwrap

from bloqade import qasm2


def test_inline():

    lines = textwrap.dedent(
        """
    OPENQASM 2.0;

    qreg q[2];
    creg c[2];

    h q[0];
    CX q[0], q[1];
    barrier q;
    CX q[0], q[1];
    rx(pi/2) q[0];
    """
    )

    @qasm2.main
    def qasm2_inline_code():
        qasm2.inline(lines)

    qasm2_inline_code.print()
