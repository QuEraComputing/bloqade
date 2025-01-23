import textwrap

import pytest
from bloqade.qasm2.parse import ast, loads
from bloqade.qasm2.parse.build import BuildError


def test_mainprogram():
    assert (
        loads(
            textwrap.dedent(
                """
    OPENQASM 2.0;
    include "qelib1.inc";
    """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0), statements=[ast.Include("qelib1.inc")]
        )
    )

    assert (
        loads(
            textwrap.dedent(
                """
    OPENQASM 2.0-atom;
    include "qelib1.inc";
    """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0, "atom"), statements=[ast.Include("qelib1.inc")]
        )
    )


def test_reg():
    assert (
        loads(
            textwrap.dedent(
                """
    OPENQASM 2.0;
    qreg q[5];
    """
            )
        )
        == ast.MainProgram(version=ast.Version(2, 0), statements=[ast.QReg("q", 5)])
    )

    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                creg c[5];
                """
            )
        )
        == ast.MainProgram(version=ast.Version(2, 0), statements=[ast.CReg("c", 5)])
    )


def test_gate():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                gate cx a, b { CX a, b; }
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0),
            statements=[
                ast.Gate(
                    name="cx",
                    cparams=[],
                    qparams=["a", "b"],
                    body=[ast.CXGate(ast.Name("a"), ast.Name("b"))],
                )
            ],
        )
    )


def test_opaque():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                opaque cx a, b;
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0),
            statements=[
                ast.Opaque(
                    name="cx",
                    cparams=[],
                    qparams=[ast.Name("a"), ast.Name("b")],
                )
            ],
        )
    )


def test_if():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                if (c==0) x q;
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0),
            statements=[
                ast.IfStmt(
                    cond=ast.Cmp(ast.Name("c"), ast.Number(0)),
                    body=[
                        ast.Instruction(
                            name=ast.Name("x"), params=[], qargs=[ast.Name("q")]
                        )
                    ],
                )
            ],
        )
    )


def test_barrier():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                barrier q;
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0), statements=[ast.Barrier(qargs=[ast.Name("q")])]
        )
    )


def measure():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                measure q -> c;
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0),
            statements=[ast.Measure(ast.Name("q"), ast.Name("c"))],
        )
    )


def test_reset():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                reset q;
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0), statements=[ast.Reset(ast.Name("q"))]
        )
    )


def test_uop():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                U(0.1, 0.2, 0.3) q;
                CX a, b;
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0),
            statements=[
                ast.UGate(
                    ast.Number(0.1), ast.Number(0.2), ast.Number(0.3), ast.Name("q")
                ),
                ast.CXGate(ast.Name("a"), ast.Name("b")),
            ],
        )
    )


def test_parallel_ugate():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0-atom;
                parallel.U(theta, phi, lam) {q[0]; q[1]; q[2];}
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0, "atom"),
            statements=[
                ast.ParaU3Gate(
                    theta=ast.Name("theta"),
                    phi=ast.Name("phi"),
                    lam=ast.Name("lam"),
                    qargs=ast.ParallelQArgs(
                        [
                            (ast.Bit(ast.Name("q"), 0),),
                            (ast.Bit(ast.Name("q"), 1),),
                            (ast.Bit(ast.Name("q"), 2),),
                        ]
                    ),
                )
            ],
        )
    )

    with pytest.raises(BuildError):
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                parallel.U(theta, phi, lam) {q[0]; q[1]; q[2];}
                """
            )
        )


def test_parallel_czgate():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0-atom;
                parallel.CZ {q[0]; q[1]; q[2];}
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0, "atom"),
            statements=[
                ast.ParaCZGate(
                    qargs=ast.ParallelQArgs(
                        [
                            (ast.Bit(ast.Name("q"), 0),),
                            (ast.Bit(ast.Name("q"), 1),),
                            (ast.Bit(ast.Name("q"), 2),),
                        ]
                    )
                )
            ],
        )
    )

    with pytest.raises(BuildError):
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                parallel.CZ {
                q[0], q[1];
                q[2], q[3];
                }
                """
            )
        )


def test_parallel_rz_gate():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0-atom;
                parallel.RZ(theta) {q[0]; q[1];}
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0, "atom"),
            statements=[
                ast.ParaRZGate(
                    theta=ast.Name("theta"),
                    qargs=ast.ParallelQArgs(
                        [
                            (ast.Bit(ast.Name("q"), 0),),
                            (ast.Bit(ast.Name("q"), 1),),
                        ]
                    ),
                )
            ],
        )
    )

    with pytest.raises(BuildError):
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                parallel.RZ(theta) {q[0]; q[1]; q[2];}
                """
            )
        )


def test_expr():
    assert (
        loads(
            textwrap.dedent(
                """
                OPENQASM 2.0;
                U(1 + 1, -pi, sin(2.0)) q;
                """
            )
        )
        == ast.MainProgram(
            version=ast.Version(2, 0),
            statements=[
                ast.UGate(
                    ast.BinOp("+", ast.Number(1), ast.Number(1)),
                    ast.UnaryOp("-", ast.Pi()),
                    ast.Call("sin", [ast.Number(2.0)]),
                    ast.Name("q"),
                )
            ],
        )
    )
