from dataclasses import field, dataclass

from kirin.passes import Pass, TypeInfer
from kirin.rewrite import (
    Walk,
    Chain,
    Inline,
    Fixpoint,
    WrapConst,
    Call2Invoke,
    ConstantFold,
    CFGCompactify,
    InlineGetItem,
    InlineGetField,
    DeadCodeElimination,
    CommonSubexpressionElimination,
)
from kirin.analysis import const
from kirin.dialects import scf, ilist
from kirin.ir.method import Method
from kirin.rewrite.abc import RewriteResult
from bloqade.qasm2.dialects import expr


@dataclass
class QASM2Fold(Pass):
    """Fold pass for qasm2.extended"""

    constprop: const.Propagate = field(init=False)
    inline_gate_subroutine: bool = True

    def __post_init__(self):
        self.constprop = const.Propagate(self.dialects)
        self.typeinfer = TypeInfer(self.dialects)

    def unsafe_run(self, mt: Method) -> RewriteResult:
        result = RewriteResult()
        frame, _ = self.constprop.run_analysis(mt)
        result = Walk(WrapConst(frame)).rewrite(mt.code).join(result)
        rule = Chain(
            ConstantFold(),
            Call2Invoke(),
            InlineGetField(),
            InlineGetItem(),
            DeadCodeElimination(),
            CommonSubexpressionElimination(),
        )
        result = Fixpoint(Walk(rule)).rewrite(mt.code).join(result)

        result = (
            Walk(
                Chain(
                    scf.unroll.PickIfElse(),
                    scf.unroll.ForLoop(),
                    scf.trim.UnusedYield(),
                )
            )
            .rewrite(mt.code)
            .join(result)
        )

        # run typeinfer again after unroll etc. because we now insert
        # a lot of new nodes, which might have more precise types
        self.typeinfer(mt)
        result = (
            Walk(Chain(ilist.rewrite.ConstList2IList(), ilist.rewrite.Unroll()))
            .rewrite(mt.code)
            .join(result)
        )

        def skip_scf(node):
            return isinstance(node, (scf.For, scf.IfElse))

        result = (
            Walk(
                Inline(
                    lambda x: (
                        True
                        if self.inline_gate_subroutine
                        else not isinstance(x, expr.GateFunction)
                    )
                ),
                skip=skip_scf,
            )
            .rewrite(mt.code)
            .join(result)
        )
        result = Fixpoint(CFGCompactify()).rewrite(mt.code).join(result)
        return result
