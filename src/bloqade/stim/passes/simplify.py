from dataclasses import dataclass

from kirin import ir
from kirin.analysis import const
from kirin.analysis.const.prop import Propagate
from kirin.passes import Pass
from kirin.rewrite import Chain, Fixpoint, Walk
from kirin.rewrite.alias import InlineAlias
from kirin.rewrite.dce import DeadCodeElimination
from kirin.rewrite.getitem import InlineGetItem


@dataclass
class Simplify(Pass):

    def unsafe_run(self, mt: ir.Method) -> None:
        constprop = Propagate(self.dialects)
        results, expect = constprop.run_analysis(mt, tuple(const.JointResult.top() for _ in mt.args))
        Fixpoint(
            Walk(
                Chain(
                    [
                        InlineAlias(),
                        InlineGetItem(results),
                    ]
                )
            )
        ).rewrite(mt.code)

        # dce
        dce = DeadCodeElimination()
        Fixpoint(Walk(dce)).rewrite(mt.code)
