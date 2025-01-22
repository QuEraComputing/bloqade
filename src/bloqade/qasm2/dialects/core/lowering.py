import ast

from kirin import types
from kirin.lowering import Result, FromPythonAST, LoweringState
from kirin.exceptions import DialectLoweringError
from bloqade.qasm2.types import BitType, CRegType, QRegType, QubitType

from . import stmts
from ._dialect import dialect


@dialect.register
class QASMCoreLowering(FromPythonAST):
    def lower_Compare(self, state: LoweringState, node: ast.Compare) -> Result:
        lhs = state.visit(node.left).expect_one()
        if len(node.ops) != 1:
            raise DialectLoweringError(
                "only one comparison operator and == is supported for qasm2 lowering"
            )
        rhs = state.visit(node.comparators[0]).expect_one()
        if isinstance(node.ops[0], ast.Eq):
            stmt = stmts.CRegEq(lhs, rhs)
        else:
            raise DialectLoweringError(
                f"unsupported comparison operator {node.ops[0]} only Eq is supported."
            )

        return Result(state.append_stmt(stmt))

    def lower_Subscript(self, state: LoweringState, node: ast.Subscript) -> Result:
        value = state.visit(node.value).expect_one()
        index = state.visit(node.slice).expect_one()

        if not index.type.is_subseteq(types.Int):
            raise DialectLoweringError(
                f"unsupported subscript index type {index.type},"
                " only integer indices are supported in QASM 2.0"
            )

        if not isinstance(node.ctx, ast.Load):
            raise DialectLoweringError(
                f"unsupported subscript context {node.ctx},"
                " cannot write to subscript in QASM 2.0"
            )

        if value.type.is_subseteq(QRegType):
            stmt = stmts.QRegGet(reg=value, idx=index)
            stmt.result.type = QubitType
        elif value.type.is_subseteq(CRegType):
            stmt = stmts.CRegGet(reg=value, idx=index)
            stmt.result.type = BitType
        else:
            raise DialectLoweringError(
                f"unsupported subscript value type {value.type},"
                " only QReg and CReg are supported in QASM 2.0"
            )

        return Result(state.append_stmt(stmt))
