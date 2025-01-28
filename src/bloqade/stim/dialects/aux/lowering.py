import ast

from kirin.lowering import Result, FromPythonAST, LoweringState
from kirin.exceptions import DialectLoweringError

from . import stmts
from ._dialect import dialect


@dialect.register
class StimAuxLowering(FromPythonAST):

    def _const_stmt(
        self, state: LoweringState, value: int | float | str | bool
    ) -> stmts.ConstInt | stmts.ConstFloat | stmts.ConstStr | stmts.ConstBool:


        if isinstance(value, bool):
            return stmts.ConstBool(value=value)
        elif isinstance(value, int):
            return stmts.ConstInt(value=value)
        elif isinstance(value, float):
            return stmts.ConstFloat(value=value)
        elif isinstance(value, str):
            return stmts.ConstStr(value=value)
        else:
            raise DialectLoweringError(f"unsupported Stim constant type {type(value)}")

    def lower_Constant(self, state: LoweringState, node: ast.Constant) -> Result:
        stmt = self._const_stmt(state, node.value)
        return Result(state.append_stmt(stmt))

    def lower_Expr(self, state: LoweringState, node: ast.Expr) -> Result:
        return state.visit(node.value)

    def lower_UnaryOp(self, state: LoweringState, node: ast.UnaryOp) -> Result:
        if isinstance(node.op, ast.USub):
            value = state.visit(node.operand).expect_one()
            stmt = stmts.Neg(operand=value)
            return Result(state.append_stmt(stmt))
        else:
            raise DialectLoweringError(f"unsupported Stim unaryop {node.op}")
