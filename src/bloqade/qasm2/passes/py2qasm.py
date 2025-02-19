"""Rewrite py dialects into qasm dialects."""

from kirin import ir
from kirin.passes import Pass
from kirin.rewrite import Walk, Fixpoint
from kirin.dialects import py, math
from kirin.rewrite.abc import RewriteRule
from kirin.rewrite.result import RewriteResult
from bloqade.qasm2.dialects import core, expr


class _Py2QASM(RewriteRule):
    """Rewrite py dialects into qasm dialects."""

    UNARY_OPS = {
        py.USub: expr.Neg,
        math.sin: expr.Sin,
        math.cos: expr.Cos,
        math.tan: expr.Tan,
        math.exp: expr.Exp,
        math.sqrt: expr.Sqrt,
    }

    BINARY_OPS = {
        py.Add: expr.Add,
        py.Sub: expr.Sub,
        py.Mult: expr.Mul,
        py.Div: expr.Div,
        py.Pow: expr.Pow,
    }

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if isinstance(node, py.Constant):
            if isinstance(node.value, int):
                node.replace_by(expr.ConstInt(value=node.value))
                return RewriteResult(has_done_something=True)
            elif isinstance(node.value, float):
                node.replace_by(expr.ConstFloat(value=node.value))
                return RewriteResult(has_done_something=True)
        elif isinstance(node, py.BinOp):
            if (pystmt := self.BINARY_OPS.get(type(node))) is not None:
                node.replace_by(pystmt(node.lhs, node.rhs))
                return RewriteResult(has_done_something=True)
        elif isinstance(node, py.UnaryOp):
            if (pystmt := self.UNARY_OPS.get(type(node))) is not None:
                node.replace_by(pystmt(node.value))
                return RewriteResult(has_done_something=True)
        elif isinstance(node, py.cmp.Eq):
            node.replace_by(core.CRegEq(node.lhs, node.rhs))
            return RewriteResult(has_done_something=True)
        elif isinstance(node, py.assign.Alias):
            node.result.replace_by(node.value)
            node.delete()
            return RewriteResult(has_done_something=True)
        return RewriteResult()


class Py2QASM(Pass):

    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        return Fixpoint(Walk(_Py2QASM())).rewrite(mt.code)
