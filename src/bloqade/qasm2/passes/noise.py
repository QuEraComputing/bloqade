from dataclasses import field, dataclass

from kirin import ir
from bloqade.noise import native
from kirin.rewrite import cse, dce, walk, fixpoint
from bloqade.analysis import address
from kirin.passes.abc import Pass
from bloqade.qasm2.rewrite.heuristic_noise import NoiseRewriteRule


@dataclass
class NoisePass(Pass):

    noise_model: native.NoiseModelABC = field(default_factory=native.TwoRowZoneModel)

    def generate_rule(self, mt: ir.Method):
        address_analysis = address.AddressAnalysis(mt.dialects)
        frame, _ = address_analysis.run_analysis(mt)
        return walk.Walk(
            NoiseRewriteRule(
                frame.entries,
                address_analysis.qubit_ssa_value,
                noise_model=self.noise_model,
            )
        )

    def unsafe_run(self, mt: ir.Method):
        self.generate_rule(mt).rewrite(mt.code)
        fixpoint.Fixpoint(walk.Walk(cse.CommonSubexpressionElimination())).rewrite(
            mt.code
        )
        return fixpoint.Fixpoint(walk.Walk(dce.DeadCodeElimination())).rewrite(mt.code)
