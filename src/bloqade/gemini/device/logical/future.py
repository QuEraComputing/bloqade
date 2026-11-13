from dataclasses import dataclass

from bloqade.core.device import Future

from bloqade.gemini.device.logical.result import GeminiLogicalResult


@dataclass(kw_only=True)
class GeminiLogicalFuture(Future[GeminiLogicalResult]):
    """Future for tasks submitted to the Gemini logical backend.

    Defaults `result_cls` to `GeminiLogicalResult` and `context_name` to
    "gemini-logical".
    """

    result_cls: type[GeminiLogicalResult] = GeminiLogicalResult
    context_name: str = "gemini-logical"
