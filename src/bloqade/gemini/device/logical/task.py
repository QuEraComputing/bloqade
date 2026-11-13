from dataclasses import dataclass
from importlib.metadata import version

from bloqade.core.device.task import (
    KernelBatchTask,
    SingleKernelTask,
    ParameterScanTask,
)

from bloqade.gemini.device.logical.future import GeminiLogicalFuture

_bloqade_version = version("bloqade-circuit")


@dataclass(kw_only=True)
class GeminiTaskMixin:
    """Defaults shared by Gemini logical task classes.

    Sets the program language to "squin", records the installed
    `bloqade-circuit` version as the program language version, and pins
    `future_cls` and `context_name` to the Gemini logical backend.
    """

    program_language: str = "squin"
    future_cls: type[GeminiLogicalFuture] = GeminiLogicalFuture
    context_name: str = "gemini-squin-logical-10q"

    @property
    def program_language_version(self) -> str:
        """Installed `bloqade-circuit` version recorded with each kernel.

        Returns:
            str: The version string reported by `importlib.metadata`.
        """
        return _bloqade_version


@dataclass(kw_only=True)
class GeminiSingleKernelTask(GeminiTaskMixin, SingleKernelTask[GeminiLogicalFuture]):
    """`SingleKernelTask` preconfigured for the Gemini logical backend."""


@dataclass(kw_only=True)
class GeminiParameterScanTask(GeminiTaskMixin, ParameterScanTask[GeminiLogicalFuture]):
    """`ParameterScanTask` preconfigured for the Gemini logical backend."""


@dataclass(kw_only=True)
class GeminiKernelBatchTask(GeminiTaskMixin, KernelBatchTask[GeminiLogicalFuture]):
    """`KernelBatchTask` preconfigured for the Gemini logical backend."""
