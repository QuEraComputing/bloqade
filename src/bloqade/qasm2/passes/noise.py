from dataclasses import field, dataclass

from kirin import ir
from kirin.passes import Pass
from bloqade.noise import native
from kirin.rewrite import (
    Walk,
    Chain,
    Fixpoint,
    ConstantFold,
    DeadCodeElimination,
    CommonSubexpressionElimination,
)
from bloqade.analysis import address
from kirin.rewrite.result import RewriteResult
from bloqade.qasm2.rewrite.heuristic_noise import NoiseRewriteRule


@dataclass
class NoisePass(Pass):
    """Apply a noise model to a quantum circuit.

    NOTE: This pass is not guaranteed to be supported long-term in bloqade. We will be
    moving towards a more general approach to noise modeling in the future.

    """

    noise_model: native.MoveNoiseModelABC = field(
        default_factory=native.TwoRowZoneModel
    )
    gate_noise_params: native.GateNoiseParams = field(
        default_factory=native.GateNoiseParams
    )
    address_analysis: address.AddressAnalysis = field(init=False)

    def __post_init__(self):
        self.address_analysis = address.AddressAnalysis(self.dialects)

    def unsafe_run(self, mt: ir.Method):
        result = RewriteResult()

        frame, res = self.address_analysis.run_analysis(mt, no_raise=False)
        result = (
            Walk(
                NoiseRewriteRule(
                    address_analysis=frame.entries,
                    noise_model=self.noise_model,
                    gate_noise_params=self.gate_noise_params,
                )
            )
            .rewrite(mt.code)
            .join(result)
        )
        rule = Chain(
            ConstantFold(),
            DeadCodeElimination(),
            CommonSubexpressionElimination(),
        )
        result = Fixpoint(Walk(rule)).rewrite(mt.code).join(result)
        return result
