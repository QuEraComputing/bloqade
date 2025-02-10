from kirin import ir
from kirin.passes import TypeInfer
from kirin.dialects import func, lowering

from .passes import Simplify
from .dialects import aux, gate, noise, collapse


@ir.dialect_group([noise, gate, aux, collapse, func, lowering.func, lowering.call])
def main(self):
    typeinfer_pass = TypeInfer(self)
    simplify_pass = Simplify(self)

    def run_pass(
        mt: ir.Method,
        *,
        typeinfer: bool = False,
        simplify: bool = True,
    ) -> None:

        if typeinfer:
            typeinfer_pass(mt)

        if simplify:
            simplify_pass(mt)

    return run_pass
