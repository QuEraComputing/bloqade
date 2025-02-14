from dataclasses import field, dataclass

from kirin.passes import Pass, aggressive
from kirin.rewrite import Walk, Chain
from kirin.dialects import scf
from kirin.ir.method import Method
from kirin.rewrite.abc import RewriteResult


@dataclass
class QASM2Fold(Pass):
    """Fold pass for qasm2.extended"""

    fold: aggressive.Fold = field(init=False)

    def __post_init__(self):
        self.fold = aggressive.Fold(self.dialects)

    def unsafe_run(self, mt: Method) -> RewriteResult:
        result = self.fold.unsafe_run(mt)
        return (
            Walk(Chain(scf.unroll.PickIfElse(), scf.unroll.ForLoop()))
            .rewrite(mt.code)
            .join(result)
        )
