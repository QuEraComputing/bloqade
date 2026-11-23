from bloqade.gemini import logical
from bloqade.gemini.result import RetType, GeminiResult
from bloqade.gemini.logical.device import utils


class GeminiLogicalResult(GeminiResult[RetType]):
    """Hardware-frame result for Gemini logical kernels."""

    @staticmethod
    def _program_postprocessing(kernel_json: str):
        return utils.get_slm_mapping_postprocessing(
            logical.kernel.decode_json(kernel_json)
        )
