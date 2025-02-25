from typing import List

from kirin import ir, types
from bloqade import qasm2
from kirin.rewrite import Walk, Fixpoint, CommonSubexpressionElimination
from kirin.dialects import py, func, ilist
from bloqade.qasm2.passes.glob import GlobalToParallel


def assert_with_print(expected_mt: ir.Method, mt: ir.Method):
    import io
    import difflib

    from rich.console import Console

    try:
        assert expected_mt.code.is_equal(mt.code)
    except AssertionError as e:

        gn_con = Console(record=True, file=io.StringIO())
        mt.print(console=gn_con)

        gn = gn_con.export_text()

        expected_con = Console(record=True, file=io.StringIO())
        expected_mt.print(console=expected_con)

        expected = expected_con.export_text()

        diff = difflib.Differ().compare(
            expected.splitlines(),
            gn.splitlines(),
        )

        print("\n".join(diff))

        raise e


def as_int(value: int):
    return py.constant.Constant(value=value)


def as_float(value: float):
    return py.constant.Constant(value=value)


def test_global2para_rewrite():

    @qasm2.extended
    def main():
        q1 = qasm2.qreg(1)
        q2 = qasm2.qreg(2)

        qasm2.glob.u(theta=1.3, phi=1.1, lam=1.2, registers=[q1, q2])

    GlobalToParallel(dialects=main.dialects)(main)

    # post-rewrite expected function
    expected: List[ir.Statement] = [
        (first_n_qubits := as_int(1)),
        (reg1 := qasm2.core.QRegNew(n_qubits=first_n_qubits.result)),
        (second_n_qubits := as_int(2)),
        (reg2 := qasm2.core.QRegNew(n_qubits=second_n_qubits.result)),
        (theta := as_float(1.3)),
        (phi := as_float(1.1)),
        (lam := as_float(1.2)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg1.result, idx=idx0.result)),
        (idx1 := as_int(0)),
        (q1 := qasm2.core.QRegGet(reg2.result, idx=idx1.result)),
        (idx2 := as_int(1)),
        (q2 := qasm2.core.QRegGet(reg2.result, idx=idx2.result)),
        (lt := ilist.New(values=[q0.result, q1.result, q2.result])),
        (
            qasm2.parallel.parallel.UGate(
                qargs=lt.result, theta=theta.result, phi=phi.result, lam=lam.result
            )
        ),
        (return_none := func.ConstantNone()),
        (func.Return(return_none)),
    ]

    reg1.result.name = "q1"
    reg2.result.name = "q2"

    block = ir.Block(expected)
    block.args.append_from(types.MethodType[[], types.NoneType], "main_self")
    expected_func_stmt = func.Function(
        sym_name="main",
        signature=func.Signature(inputs=(), output=types.NoneType),
        body=ir.Region(blocks=block),
    )

    expected_method = ir.Method(
        mod=None,
        py_func=None,
        sym_name="main",
        dialects=qasm2.main,
        code=expected_func_stmt,
        arg_names=[],
    )
    qasm2.main.run_pass(expected_method)  # type: ignore
    Fixpoint(Walk(CommonSubexpressionElimination())).rewrite(expected_method.code)
    assert_with_print(expected_method, main)


def test_global2para_rewrite2():

    @qasm2.extended
    def main():
        q1 = qasm2.qreg(1)
        q2 = qasm2.qreg(2)

        qasm2.glob.u(theta=1.3, phi=1.1, lam=1.2, registers=[q1])
        qasm2.glob.u(theta=0.3, phi=0.1, lam=0.2, registers=[q2])

    GlobalToParallel(dialects=main.dialects)(main)

    main.print()

    # post-rewrite expected function
    expected: List[ir.Statement] = [
        (first_n_qubits := as_int(1)),
        (reg1 := qasm2.core.QRegNew(n_qubits=first_n_qubits.result)),
        (second_n_qubits := as_int(2)),
        (reg2 := qasm2.core.QRegNew(n_qubits=second_n_qubits.result)),
        (theta := as_float(1.3)),
        (phi := as_float(1.1)),
        (lam := as_float(1.2)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg1.result, idx=idx0.result)),
        (lt := ilist.New(values=[q0.result])),
        (
            qasm2.parallel.parallel.UGate(
                qargs=lt.result, theta=theta.result, phi=phi.result, lam=lam.result
            )
        ),
        (theta2 := as_float(0.3)),
        (phi2 := as_float(0.1)),
        (lam2 := as_float(0.2)),
        (idx1 := as_int(0)),
        (q1 := qasm2.core.QRegGet(reg2.result, idx=idx1.result)),
        (idx2 := as_int(1)),
        (q2 := qasm2.core.QRegGet(reg2.result, idx=idx2.result)),
        (lt := ilist.New(values=[q1.result, q2.result])),
        (
            qasm2.parallel.parallel.UGate(
                qargs=lt.result, theta=theta2.result, phi=phi2.result, lam=lam2.result
            )
        ),
        (return_none := func.ConstantNone()),
        (func.Return(return_none)),
    ]
    reg1.result.name = "q1"
    reg2.result.name = "q2"

    block = ir.Block(expected)
    block.args.append_from(types.MethodType[[], types.NoneType], "main_self")
    expected_func_stmt = func.Function(
        sym_name="main",
        signature=func.Signature(inputs=(), output=types.NoneType),
        body=ir.Region(blocks=block),
    )

    expected_method = ir.Method(
        mod=None,
        py_func=None,
        sym_name="main",
        dialects=qasm2.main,
        code=expected_func_stmt,
        arg_names=[],
    )
    qasm2.main.run_pass(expected_method)  # type: ignore
    Fixpoint(Walk(CommonSubexpressionElimination())).rewrite(expected_method.code)
    assert_with_print(expected_method, main)
