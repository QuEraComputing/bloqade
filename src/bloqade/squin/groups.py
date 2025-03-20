from kirin import ir, passes
from kirin.prelude import structural_no_opt
from kirin.dialects import ilist
from bloqade.qasm2.rewrite.desugar import IndexingDesugarPass

from . import op, wire, qubit


@ir.dialect_group(structural_no_opt.union([op, qubit]))
def kernel(self):
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)
    ilist_desugar_pass = ilist.IListDesugar(self)
    indexing_desugar_pass = IndexingDesugarPass(self)

    def run_pass(method, *, fold=True, typeinfer=True):
        method.verify()
        if fold:
            fold_pass.fixpoint(method)

        if typeinfer:
            typeinfer_pass(method)
        ilist_desugar_pass(method)
        indexing_desugar_pass(method)
        if typeinfer:
            typeinfer_pass(method)  # fix types after desugaring
            method.code.typecheck()

    return run_pass


@ir.dialect_group(structural_no_opt.union([op, wire]))
def wired(self):
    def run_pass(method):
        pass

    return run_pass
