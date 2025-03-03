from dataclasses import field, dataclass

from kirin import ir
from bloqade.noise import native
from kirin.rewrite import cse, dce, walk, chain, fixpoint
from bloqade.analysis import address
from kirin.passes.abc import Pass
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

    def unsafe_run(self, mt: ir.Method):
        address_analysis = address.AddressAnalysis(mt.dialects)
        frame, _ = address_analysis.run_analysis(mt)
        first_pass = walk.Walk(
            NoiseRewriteRule(
                address_analysis=frame.entries,
                qubit_ssa_value=address_analysis.qubit_ssa_value,
                num_qubits=address_analysis.qubit_count,
                noise_model=self.noise_model,
                gate_noise_params=self.gate_noise_params,
            )
        )
        second_pass = fixpoint.Fixpoint(walk.Walk(cse.CommonSubexpressionElimination()))
        third_pass = fixpoint.Fixpoint(walk.Walk(dce.DeadCodeElimination()))
        return chain.Chain(first_pass, second_pass, third_pass)
