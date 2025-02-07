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


@ir.dialect_group(
    [uop, parallel, func, ilist, expr, glob, noise, lowering.func, lowering.call]
)
def gate(self):
    ilist_desugar = ilist.IListDesugar(self)
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)

    def run_pass(
        method: ir.Method,
        *,
        fold: bool = True,
    ):
        method.verify()
        ilist_desugar(method)
        # TODO make special Function rewrite

        if fold:
            fold_pass(method)

        typeinfer_pass(method)
        method.code.typecheck()

    return run_pass


@ir.dialect_group(
    [
        inline,
        uop,
        glob,
        noise,
        expr,
        parallel,
        core,
        scf,
        indexing,
        ilist,
        func,
        lowering.func,
        lowering.call,
    ]
)
def main(self):
    ilist_desugar = ilist.IListDesugar(self)
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)

    def run_pass(
        method: ir.Method,
        *,
        fold: bool = True,
    ):
        method.verify()
        ilist_desugar(method)
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
def kernel(self):
    def run_pass(
        method: ir.Method,
    ):
        method.verify()

    return run_pass
