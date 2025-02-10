from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import walk, result
from kirin.passes.abc import Pass

from ..rewrite.parallel_to_uop import ParallelToUOpRule

@dataclass
class ParallelToUOp(Pass):
    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:

        rewriter = walk.Walk(ParallelToUOpRule())

        return rewriter.rewrite(mt.code)
