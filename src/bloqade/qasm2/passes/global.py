from kirin import ir
from kirin.rewrite import abc, result

# from bloqade.qasm2.dialects import glob

# run analysis pass
# analysis_results = ...
# rewriter = walk.Walk(ParallelToUOpRule(analysis_results))


class GlobalToQASM2(abc.RewriteRule):

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        return result.RewriteResult()
