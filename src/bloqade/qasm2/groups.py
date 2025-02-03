from kirin import ir, passes
from kirin.dialects import cf, func, ilist
from bloqade.qasm2.dialects import (
    uop,
    core,
    expr,
    glob,
    noise,
    inline,
    measure,
    indexing,
    parallel,
)


@ir.dialect_group([uop, parallel, func, ilist, expr, glob, noise])
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
    [inline, uop, glob, noise, expr, parallel, core, measure, indexing, cf, ilist, func]
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
