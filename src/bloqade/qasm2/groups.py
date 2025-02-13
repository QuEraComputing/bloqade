from kirin import ir, passes
from kirin.prelude import basic
from kirin.dialects import scf, func, ilist, lowering
from bloqade.qasm2.dialects import (
    uop,
    core,
    expr,
    glob,
    noise,
    inline,
    indexing,
    parallel,
)
from bloqade.qasm2.rewrite.desugar import IndexingDesugarPass


@ir.dialect_group([uop, func, expr, lowering.func, lowering.call])
def gate(self):
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)

    def run_pass(
        method: ir.Method,
        *,
        fold: bool = True,
    ):
        method.verify()
        # TODO make special Function rewrite

        if fold:
            fold_pass(method)

        typeinfer_pass(method)
        method.code.typecheck()

    return run_pass


@ir.dialect_group(
    [
        uop,
        expr,
        core,
        scf,
        indexing,
        func,
        lowering.func,
        lowering.call,
    ]
)
def main(self):
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)

    def run_pass(
        method: ir.Method,
        *,
        fold: bool = True,
    ):
        method.verify()
        # TODO make special Function rewrite

        if fold:
            fold_pass(method)

        typeinfer_pass(method)
        method.code.typecheck()

    return run_pass


@ir.dialect_group(
    basic.union(
        [
            inline,
            uop,
            glob,
            noise,
            parallel,
            core,
            lowering.func,
            lowering.call,
        ]
    )
    .discard(lowering.cf)
    .add(scf)
)
def extended(self):
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)
    ilist_desugar_pass = ilist.IListDesugar(self)
    indexing_desugar_pass = IndexingDesugarPass(self)

    def run_pass(
        mt: ir.Method,
        *,
        fold: bool = True,
    ):
        mt.verify()
        if fold:
            fold_pass(mt)

        ilist_desugar_pass(mt)
        indexing_desugar_pass(mt)
        typeinfer_pass(mt)

        mt.code.typecheck()

    return run_pass
