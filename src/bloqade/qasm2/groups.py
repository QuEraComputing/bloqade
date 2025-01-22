from kirin import ir, passes
from kirin.dialects import cf, func, ilist

from bloqade.qasm2.dialects import core, parallel, uop, inline, expr


@ir.dialect_group([uop, parallel, func, ilist, expr])
def gate(self):
    ilist_desugar = ilist.IListDesugar(self)
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)

    def run_pass(
        method: ir.Method,
        *,
        verify: bool = True,
        typeinfer: bool = False,
        fold: bool = True,
    ):
        if verify:
            method.verify()

        ilist_desugar(method)
        # TODO make special Function rewrite

        if fold:
            fold_pass(method)

        if typeinfer:
            typeinfer_pass(method)

    return run_pass


@ir.dialect_group([inline, uop, expr, parallel, core, cf, ilist, func])
def main(self):
    ilist_desugar = ilist.IListDesugar(self)
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)

    def run_pass(
        method: ir.Method,
        *,
        verify: bool = True,
        typeinfer: bool = False,
        fold: bool = True,
    ):
        if verify:
            method.verify()

        ilist_desugar(method)
        # TODO make special Function rewrite

        if fold:
            fold_pass(method)

        if typeinfer:
            typeinfer_pass(method)

    return run_pass
