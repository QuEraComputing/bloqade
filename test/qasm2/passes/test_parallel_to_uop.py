# from typing import List

# from kirin import ir
# from bloqade import qasm2
# from kirin.dialects import func
# from bloqade.qasm2.passes.parallel import ParallelToUOp


# def as_int(value: int):
#     return qasm2.expr.ConstInt(value=value)


# def as_float(value: float):
#     return qasm2.expr.ConstFloat(value=value)


# def test_cz_rewrite():

#     @qasm2.main
#     def main():
#         q = qasm2.qreg(4)

#         qasm2.parallel.cz(ctrls=(q[0], q[2]),qargs=(q[1], q[3]))

#     # Run rewrite
#     ParallelToUOp(main.dialects)(main)

#     # post-rewrite expected function
#     expected: List[ir.Statement] = [
#         (n_qubits := as_int(4)),
#         (reg := qasm2.core.QRegNew(n_qubits=n_qubits.result)),
#         (idx0 := as_int(0)),
#         (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
#         (idx2 := as_int(2)),
#         (q2 := qasm2.core.QRegGet(reg.result, idx=idx2.result)),
#         (idx1 := as_int(1)),
#         (q1 := qasm2.core.QRegGet(reg.result, idx=idx1.result)),
#         (idx3 := as_int(3)),
#         (q3 := qasm2.core.QRegGet(reg.result, idx=idx3.result)),
#         (qasm2.uop.CZ(ctrl=q0.result, qarg=q2.result)),
#         (qasm2.uop.CZ(ctrl=q1.result, qarg=q3.result)),
#         (return_none := func.ConstantNone()),
#         (func.Return(return_none))
#     ]

#     expected_func_stmt = func.Function(
#         sym_name="main",
#         signature=func.Signature(
#             inputs=(),
#             output=func.ConstantNone()
#         ),
#         body=ir.Region(ir.Block(expected)),
#     )

#     assert expected_func_stmt.is_structurally_equal(main.code)
