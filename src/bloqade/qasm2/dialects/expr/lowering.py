import ast

from kirin.exceptions import DialectLoweringError
from kirin.ir import SSAValue, types
from kirin.lowering import FromPythonAST, LoweringState, Result

from . import stmts
from ._dialect import dialect


@dialect.register
class QASMUopLowering(FromPythonAST):

    def lower_Name(self, state: LoweringState, node: ast.Name) -> Result:
        name = node.id
        if isinstance(node.ctx, ast.Load):
            value = state.current_frame.get(name)
            if value is None:
                raise DialectLoweringError(f"{name} is not defined")
            return Result(value)
        elif isinstance(node.ctx, ast.Store):
            raise DialectLoweringError("unhandled store operation")
        else:  # Del
            raise DialectLoweringError("unhandled del operation")

    def lower_Assign(self, state: LoweringState, node: ast.Assign) -> Result:
        # NOTE: QASM only expects one value on right hand side
        rhs = state.visit(node.value).expect_one()
        current_frame = state.current_frame
        match node:
            case ast.Assign(targets=[ast.Name(lhs_name, ast.Store())], value=_):
                rhs.name = lhs_name
                current_frame.defs[lhs_name] = rhs
            case _:
                target_syntax = ", ".join(
                    ast.unparse(target) for target in node.targets
                )
                raise DialectLoweringError(f"unsupported target syntax {target_syntax}")
        return Result()  # python assign does not have value

    def lower_Expr(self, state: LoweringState, node: ast.Expr) -> Result:
        return state.visit(node.value)

    def lower_Constant(self, state: LoweringState, node: ast.Constant) -> Result:
        if isinstance(node.value, int):
            stmt = stmts.ConstInt(value=node.value)
            return Result(state.append_stmt(stmt))
        elif isinstance(node.value, float):
            stmt = stmts.ConstFloat(value=node.value)
            return Result(state.append_stmt(stmt))
        else:
            raise DialectLoweringError(
                f"unsupported QASM 2.0 constant type {type(node.value)}"
            )

    def lower_BinOp(self, state: LoweringState, node: ast.BinOp) -> Result:
        lhs = state.visit(node.left).expect_one()
        rhs = state.visit(node.right).expect_one()
        if isinstance(node.op, ast.Add):
            stmt = stmts.Add(lhs, rhs)
        elif isinstance(node.op, ast.Sub):
            stmt = stmts.Sub(lhs, rhs)
        elif isinstance(node.op, ast.Mult):
            stmt = stmts.Mul(lhs, rhs)
        elif isinstance(node.op, ast.Div):
            stmt = stmts.Div(lhs, rhs)
        elif isinstance(node.op, ast.Pow):
            stmt = stmts.Pow(lhs, rhs)
        else:
            raise DialectLoweringError(f"unsupported QASM 2.0 binop {node.op}")
        stmt.result.type = self.__promote_binop_type(lhs, rhs)
        return Result(state.append_stmt(stmt))

    def __promote_binop_type(self, lhs: SSAValue, rhs: SSAValue) -> types.TypeAttribute:
        if lhs.type.is_subseteq(types.Float) or rhs.type.is_subseteq(types.Float):
            return types.Float
        return types.Int

    def lower_UnaryOp(self, state: LoweringState, node: ast.UnaryOp) -> Result:
        if isinstance(node.op, ast.USub):
            value = state.visit(node.operand).expect_one()
            stmt = stmts.Neg(value)
            return Result(state.append_stmt(stmt))
        elif isinstance(node.op, ast.UAdd):
            return state.visit(node.operand)
        else:
            raise DialectLoweringError(f"unsupported QASM 2.0 unaryop {node.op}")
