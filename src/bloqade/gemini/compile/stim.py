import io

from kirin import ir
from bloqade.lanes.transform import LogicalPipeline, MoveToSquinLogical
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.noise_model import generate_logical_noise_model
from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.rewrite.squin2stim import RemoveReturn
from bloqade.stim.upstream.from_squin import squin_to_stim
from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC

__all__ = [
    "compile_to_stim_program",
]


def _to_physical_squin_noise_model(
    mt: ir.Method,
    noise_model: LogicalNoiseModelABC | None = None,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
) -> ir.Method:
    """Compile a logical squin kernel to a physical squin kernel with noise inserted."""
    if noise_model is None:
        noise_model = generate_logical_noise_model()
    move_mt = LogicalPipeline(
        transversal_rewrite=True, layout_heuristic=layout_heuristic
    ).emit(mt, no_raise=no_raise)
    return MoveToSquinLogical(
        arch_spec=physical.get_arch_spec(),
        noise_model=noise_model,
        add_noise=True,
        aggressive_unroll=False,
    ).emit(move_mt, no_raise=no_raise)


def compile_to_stim_program(
    mt: ir.Method,
    noise_model: LogicalNoiseModelABC | None = None,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
) -> str:
    """Compile a logical squin kernel to a Stim program string with noise inserted."""
    noise_kernel = _to_physical_squin_noise_model(
        mt, noise_model, no_raise=no_raise, layout_heuristic=layout_heuristic
    )
    RemoveReturn().rewrite(noise_kernel.code)
    noise_kernel = squin_to_stim(noise_kernel)
    buf = io.StringIO()
    emit = EmitStimMain(dialects=noise_kernel.dialects, io=buf)
    emit.initialize()
    emit.run(node=noise_kernel)
    return buf.getvalue().strip()
