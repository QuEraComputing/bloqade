"""Detector error model conversion and projection helpers."""

from __future__ import annotations

from bisect import bisect_left
from typing import cast
from collections.abc import Sequence

import stim

_TargetKey = tuple[str, int]


# NOTE: We may want to move this code to bloqade-decoders as standard library functions to support subsetting a detector error model.
def _compose_independent_flip_probabilities(p_old: float, p_new: float) -> float:
    return p_old * (1.0 - p_new) + p_new * (1.0 - p_old)


def _selected_index(sorted_indices: Sequence[int], value: int) -> int | None:
    index = bisect_left(sorted_indices, value)
    if index == len(sorted_indices) or sorted_indices[index] != value:
        return None
    return index


def _target_key(
    target: stim.DemTarget,
    *,
    detector_indices: Sequence[int],
    observable_indices: Sequence[int],
) -> _TargetKey | None:
    if target.is_relative_detector_id():
        index = _selected_index(detector_indices, target.val)
        if index is None:
            return None
        return ("D", index)
    if target.is_logical_observable_id():
        index = _selected_index(observable_indices, target.val)
        if index is None:
            return None
        return ("L", index)
    return None


def _dem_target(key: _TargetKey) -> stim.DemTarget:
    kind, index = key
    if kind == "D":
        return stim.target_relative_detector_id(index)
    return stim.target_logical_observable_id(index)


def _sub_detector_error_model(
    dem: stim.DetectorErrorModel,
    detector_indices: Sequence[int],
    observable_indices: Sequence[int],
) -> stim.DetectorErrorModel:
    """Project a DEM onto selected detectors and logical observables.

    Duplicate projected error mechanisms are composed using XOR-flip
    probability semantics. This is less lossy than converting through a binary
    matrix and reconstructing a DEM.
    """

    sorted_detectors = sorted(int(index) for index in detector_indices)
    sorted_observables = sorted(int(index) for index in observable_indices)
    error_probabilities: dict[tuple[_TargetKey, ...], float] = {}

    for instruction in dem.flattened():
        if not isinstance(instruction, stim.DemInstruction):
            continue
        if instruction.type != "error":
            continue

        projected_keys: set[_TargetKey] = set()
        for target in cast(Sequence[stim.DemTarget], instruction.targets_copy()):
            if target.is_separator():
                continue
            key = _target_key(
                target,
                detector_indices=sorted_detectors,
                observable_indices=sorted_observables,
            )
            if key is None:
                continue
            if key in projected_keys:
                projected_keys.remove(key)
            else:
                projected_keys.add(key)

        if not projected_keys:
            continue

        projected_key = tuple(
            sorted(projected_keys, key=lambda key: (key[0] != "D", key[1]))
        )
        probability = float(cast(Sequence[float], instruction.args_copy())[0])
        if projected_key in error_probabilities:
            error_probabilities[projected_key] = (
                _compose_independent_flip_probabilities(
                    error_probabilities[projected_key],
                    probability,
                )
            )
        else:
            error_probabilities[projected_key] = probability

    projected_dem = stim.DetectorErrorModel()
    for projected_key, probability in error_probabilities.items():
        if probability == 0.0:
            continue
        projected_dem.append(
            "error",
            parens_arguments=[probability],
            targets=[_dem_target(key) for key in projected_key],
        )

    if sorted_detectors:
        projected_dem.append(
            "detector",
            parens_arguments=[],
            targets=[stim.target_relative_detector_id(len(sorted_detectors) - 1)],
        )
    if sorted_observables:
        projected_dem.append(
            "logical_observable",
            parens_arguments=[],
            targets=[stim.target_logical_observable_id(len(sorted_observables) - 1)],
        )

    return projected_dem
