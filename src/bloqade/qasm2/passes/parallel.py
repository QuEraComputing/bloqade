from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc, walk, result
from kirin.passes.abc import Pass
from bloqade.qasm2.dialects import uop, parallel


class ParallelToUOpRule(abc.RewriteRule):

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if node.dialect == parallel.dialect:
            getattr(self, f"rewrite_{node.name}")(node)
            return result.RewriteResult(has_done_something=True)

        return result.RewriteResult()

    def rewrite_cz(self, node: ir.Statement) -> None:
        assert isinstance(node, parallel.CZ)

        for ctrl, qarg in zip(node.ctrls, node.qargs):
            new_node = uop.CZ(ctrl, qarg)
            new_node.insert_after(node)

        node.delete()

    def rewrite_u(self, node: ir.Statement) -> None:
        assert isinstance(node, parallel.UGate)

        for qarg in node.qargs:
            new_node = uop.UGate(qarg, theta=node.theta, phi=node.phi, lam=node.lam)
            new_node.insert_after(node)

        node.delete()

    def rewrite_rz(self, node: ir.Statement) -> None:
        assert isinstance(node, parallel.RZ)

        for qarg in node.qargs:
            new_node = uop.RZ(qarg, theta=node.theta)
            new_node.insert_after(node)

        node.delete()


@dataclass
class ParallelToUOp(Pass):
    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:

        rewriter = walk.Walk(ParallelToUOpRule())

        return rewriter.rewrite(mt.code)
