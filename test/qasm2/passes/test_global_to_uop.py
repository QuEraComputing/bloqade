from typing import List

import pytest
from kirin import ir
from bloqade import qasm2
from kirin.dialects import func
from bloqade.qasm2.passes.glob import GlobalToUOP


def as_int(value: int):
    return qasm2.expr.ConstInt(value=value)


def as_float(value: float):
    return qasm2.expr.ConstFloat(value=value)


@pytest.mark.xfail(reason="bug in is_equal")
def test_global_rewrite():

    @qasm2.extended
    def main():
        q1 = qasm2.qreg(1)
        q2 = qasm2.qreg(2)

        qasm2.glob.u(theta=1.3, phi=1.1, lam=1.2, registers=[q1, q2])

    GlobalToUOP(dialects=main.dialects)(main)

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
        (
            qasm2.uop.UGate(
                qarg=q2.result, theta=theta.result, phi=phi.result, lam=lam.result
            )
        ),
        (
            qasm2.uop.UGate(
                qarg=q1.result, theta=theta.result, phi=phi.result, lam=lam.result
            )
        ),
        (
            qasm2.uop.UGate(
                qarg=q0.result, theta=theta.result, phi=phi.result, lam=lam.result
            )
        ),
        (return_none := func.ConstantNone()),
        (func.Return(return_none)),
    ]
    block = ir.Block(expected)
    block.args.append_from(ir.types.PyClass(ir.Method), "main_self")
    expected_func_stmt = func.Function(
        sym_name="main",
        signature=func.Signature(inputs=(), output=ir.types.NoneType),
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

    try:
        assert expected_method.code.is_equal(main.code)
    except AssertionError as e:
        print("Expected:")
        expected_method.print()
        print("Actual:")
        main.print()
        raise e
