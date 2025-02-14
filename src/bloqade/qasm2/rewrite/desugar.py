from dataclasses import dataclass

from kirin import ir
from kirin.passes import Pass
from kirin.rewrite import abc, walk, result
from kirin.dialects import py
from bloqade.qasm2.dialects import core


class IndexingDesugarRule(abc.RewriteRule):
    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if isinstance(node, py.indexing.GetItem):
            if core.QRegType.is_subseteq(node.obj.type):
                node.replace_by(core.QRegGet(reg=node.obj, idx=node.index))
                return result.RewriteResult(has_done_something=True)
            elif core.CRegType.is_subseteq(node.obj.type):
                node.replace_by(core.CRegGet(reg=node.obj, idx=node.index))
                return result.RewriteResult(has_done_something=True)

        return result.RewriteResult()


@dataclass
class IndexingDesugarPass(Pass):
    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:

        return walk.Walk(IndexingDesugarRule()).rewrite(mt.code)
