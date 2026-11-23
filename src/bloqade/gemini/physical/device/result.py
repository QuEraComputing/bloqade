from bloqade.gemini.result import RetType, GeminiResult
from bloqade.gemini.utils.slm import get_slm_mapping_postprocessing
from bloqade.gemini.physical.group import kernel


class GeminiPhysicalResult(GeminiResult[RetType]):
    """Hardware-frame result for Gemini physical kernels."""

    @staticmethod
    def _program_postprocessing(kernel_json: str):
        return get_slm_mapping_postprocessing(
            kernel.decode_json(kernel_json), physical_kernel=True
        )

    @staticmethod
    def _sorted_bits(mapper, shots):
        return mapper(shots, initial=True)
