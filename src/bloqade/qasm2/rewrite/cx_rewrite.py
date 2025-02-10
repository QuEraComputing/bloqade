from dataclasses import dataclass

from kirin.ir.nodes.stmt import Statement
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.qasm2 import uop


@dataclass
class CXGateRewrite(RewriteRule):

    def rewrite_Statement(self, node: Statement) -> RewriteResult:
        if isinstance(node, uop.CX):
            return self.rewrite_cx(node)
        else:
            return RewriteResult()

    def rewrite_cx(self, node: uop.CX) -> RewriteResult:

        h_before_stmt = uop.H(qarg=node.qarg)
        cz_stmt = uop.CZ(ctrl=node.ctrl, qarg=node.qarg)
        h_after_stmt = uop.H(qarg=node.qarg)

        h_before_stmt.insert_after(node)
        cz_stmt.insert_after(h_before_stmt)
        h_after_stmt.insert_after(cz_stmt)

        node.delete()

        return RewriteResult(has_done_something=True)
