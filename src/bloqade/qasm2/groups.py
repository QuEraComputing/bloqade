from kirin import ir, passes
from kirin.prelude import structural_no_opt
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

        if isinstance(method.code, func.Function):
            new_code = expr.GateFunction(
                sym_name=method.code.sym_name,
                signature=method.code.signature,
                body=method.code.body,
            )
            method.code = new_code
        else:
            raise ValueError(
                "Gate Method code must be a Function, cannot be lambda/closure"
            )

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
    structural_no_opt.union(
        [
            inline,
            uop,
            glob,
            noise,
            parallel,
            core,
        ]
    )
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
        typeinfer: bool = True,
    ):
        mt.verify()
        if fold:
            fold_pass.fixpoint(mt)

        if typeinfer:
            typeinfer_pass(mt)
        ilist_desugar_pass(mt)
        indexing_desugar_pass(mt)
        if typeinfer:
            typeinfer_pass(mt)  # fix types after desugaring
            mt.code.typecheck()

    return run_pass
