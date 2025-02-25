from typing import List

from kirin import ir, types
from bloqade import qasm2
from kirin.rewrite import Walk, Fixpoint, CommonSubexpressionElimination
from kirin.dialects import py, func, ilist
from bloqade.qasm2.passes.glob import GlobalToParallel


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
        (idx1 := as_int(1)),
        (q1 := qasm2.core.QRegGet(reg2.result, idx=idx1.result)),
        (idx2 := as_int(2)),
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
    qasm2.main.run_pass(expected_method)
    Fixpoint(Walk(CommonSubexpressionElimination())).rewrite(expected_method.code)
    assert expected_method.code.is_equal(main.code)


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
        (theta := as_float(0.3)),
        (phi := as_float(0.1)),
        (lam := as_float(0.2)),
        (idx1 := as_int(1)),
        (q1 := qasm2.core.QRegGet(reg2.result, idx=idx1.result)),
        (idx2 := as_int(2)),
        (q2 := qasm2.core.QRegGet(reg2.result, idx=idx2.result)),
        (lt := ilist.New(values=[q1.result, q2.result])),
        (
            qasm2.parallel.parallel.UGate(
                qargs=lt.result, theta=theta.result, phi=phi.result, lam=lam.result
            )
        ),
        (return_none := func.ConstantNone()),
        (func.Return(return_none)),
    ]
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
    qasm2.main.run_pass(expected_method)
    Fixpoint(Walk(CommonSubexpressionElimination())).rewrite(expected_method.code)
    assert expected_method.code.is_equal(main.code)


test_global2para_rewrite2()
