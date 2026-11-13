from bloqade.cirq_registry import register_cirq_loader as _register_cirq_loader

from bloqade.gemini.physical.group import kernel as kernel


def _get_cirq_loader():
    from bloqade.gemini.common.cirq_conversion import GeminiCirqLowerer

    return GeminiCirqLowerer


# Cirq is optional; loading this integration also installs the shared
# ``new_at`` emitter for physical kernels.
_register_cirq_loader(dialects=kernel, factory=_get_cirq_loader)

__all__ = ["_register_cirq_loader", "kernel"]
