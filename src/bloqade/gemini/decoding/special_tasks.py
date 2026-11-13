from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar
from collections.abc import Mapping

from bloqade.gemini.device import GeminiLogicalSimulatorTask

if TYPE_CHECKING:
    import tsim as tsim_backend  # type: ignore[reportMissingImports]

_TaskT = TypeVar("_TaskT", bound=GeminiLogicalSimulatorTask[Any])

_NONUNITARY_PREFIXES = (
    "M",
    "MX",
    "MY",
    "MR",
    "MRX",
    "MRY",
    "MPP",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
)


def _clear_task_tsim_artifacts(task: object) -> None:
    for attr in (
        "tsim_circuit",
        "noiseless_tsim_circuit",
        "detector_error_model",
    ):
        task.__dict__.pop(attr, None)


def _first_nonunitary_instruction_index(circuit: tsim_backend.Circuit) -> int:
    for idx in range(len(circuit)):
        if str(circuit[idx]).startswith(_NONUNITARY_PREFIXES):
            return idx
    return len(circuit)


def _apply_special_tsim_circuit_strategy(
    task_map: Mapping[str, _TaskT],
) -> dict[str, _TaskT]:
    """Prepend the inverse compiled unitary prefix to each task's circuits."""

    transformed = dict(task_map)
    for task in transformed.values():
        compiled_prefix = task.tsim_circuit[
            : _first_nonunitary_instruction_index(task.tsim_circuit)
        ]
        inverse_prefix = compiled_prefix.without_noise().inverse()
        _clear_task_tsim_artifacts(task)
        task.__dict__["tsim_circuit"] = inverse_prefix + task.tsim_circuit
        task.__dict__["noiseless_tsim_circuit"] = (
            inverse_prefix + task.noiseless_tsim_circuit
        )
    return transformed
