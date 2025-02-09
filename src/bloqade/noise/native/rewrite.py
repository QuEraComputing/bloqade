from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc, dce, walk, result, fixpoint
from kirin.passes.abc import Pass

from .stmts import PauliChannel, CZPauliChannel, AtomLossChannel
from ._dialect import dialect


class RemoveNoiseRewrite(abc.RewriteRule):
    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if isinstance(node, (AtomLossChannel, PauliChannel, CZPauliChannel)):
            node.delete()
            return result.RewriteResult(has_done_something=True)

        return result.RewriteResult()


@dataclass
class RemoveNoisePass(Pass):
    name = "remove-noise"

    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:
        delete_walk = walk.Walk(RemoveNoiseRewrite())
        dce_walk = fixpoint.Fixpoint(walk.Walk(dce.DeadCodeElimination()))

        result = delete_walk.rewrite(mt.code)

        mt.dialects = ir.DialectGroup(mt.dialects.data.symmetric_difference([dialect]))

        if result.has_done_something:
            result = dce_walk.rewrite(mt.code)

        return result
