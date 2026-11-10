from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar
from functools import cached_property
from dataclasses import replace, dataclass
from collections.abc import Callable, Sequence

import numpy as np
from bloqade.core.device import Result

from bloqade.gemini import logical

if TYPE_CHECKING:
    from bloqade.lanes.analysis.atom.analysis import PostProcessing

RetType = TypeVar("RetType")


@dataclass(kw_only=True)
class GeminiLogicalResult(Result, Generic[RetType]):
    """Result view over stored Gemini logical shots.

    Merge-oriented methods assume each selected task ID has the same subtask
    structure. Post-processing is applied to each selected subtask's flat shot
    array.

    Raw SLM result properties reconstruct their measurement mapping by compiling
    the stored logical kernel with the locally installed Bloqade Lanes compiler
    and Gemini physical architecture. The reconstructed mapping is valid only
    when those match the compiler and architecture used for remote execution.

    Attributes:
        storage (StorageBackend): Storage backend that holds shots and task
            metadata.
        shot_filter (ShotFilter): Filter used when reading shots and deriving
            subtask scope. Defaults to the DETECTED frame type.
    """

    def _program_contents_by_index(self) -> dict[int, str]:
        """Return one kernel payload per selected program index.

        Program indices are local to a task. Multiple selected task IDs may
        reuse an index only when the serialized kernel contents are identical.
        """
        programs_by_index: dict[int, tuple[str, str]] = {}
        selected_task_ids = tuple(
            sorted({subtask["task_id"] for subtask in self.full_subtasks()})
        )
        for program in self.storage.get_programs(task_ids=selected_task_ids):
            idx = program["program_index"]
            task_id = program["task_id"]
            content = program["content"]
            previous = programs_by_index.get(idx)
            if previous is not None and previous[1] != content:
                raise ValueError(
                    "Selected task IDs contain different kernels for "
                    f"program_index={idx}: {previous[0]!r} and {task_id!r}. "
                    "Narrow the result view to compatible task IDs."
                )
            programs_by_index[idx] = (task_id, content)

        return {idx: content for idx, (_, content) in programs_by_index.items()}

    def _slm_postprocessing_functions(
        self,
    ) -> dict[
        int,
        tuple[Callable[..., list[list[bool]]], PostProcessing[RetType]],
    ]:
        """Decode stored programs and build SLM-frame postprocessors.

        The compiled mapping and logical postprocessor are cached per program.
        Callers select their SLM bit convention through the mapper's
        ``invert`` keyword argument.
        """
        return self._slm_postprocessing_functions_cache

    @cached_property
    def _slm_postprocessing_functions_cache(
        self,
    ) -> dict[
        int,
        tuple[Callable[..., list[list[bool]]], PostProcessing[RetType]],
    ]:
        # Import lazily: the mapping helper imports the Lanes transform stack.
        # This result class is imported while the public Gemini package is
        # initialized, which can itself happen during Lanes analysis imports.
        from bloqade.gemini.device.logical.utils import get_slm_mapping_postprocessing

        postprocessing_functions = {}
        for idx, kernel_json in self._program_contents_by_index().items():
            kernel_mt = logical.kernel.decode_json(kernel_json)  # type: ignore[attr-defined]
            slm_to_raw, postprocessing_function = get_slm_mapping_postprocessing(
                kernel_mt,
            )
            # NOTE: what if err in postproc fn generation/errors here??
            postprocessing_functions[idx] = (slm_to_raw, postprocessing_function)

        return postprocessing_functions

    def postprocessing_functions(
        self,
    ) -> dict[int, tuple[Callable[..., list[list[bool]]], PostProcessing[RetType]]]:
        """Return the SLM-frame mapper and post-processor for each program.

        This was the compact-shot entry point, for stores whose bitstrings
        were already in physical measurement order. Nothing produces that
        format any more, so it is now an alias for
        ``_slm_postprocessing_functions`` and returns a
        ``(slm_to_raw, post_processing)`` pair per program index rather than
        a bare callable.
        """
        return self._slm_postprocessing_functions_cache

    @property
    def measurements(self) -> Sequence[Sequence[Sequence[bool]]]:
        """
        Returns a nested list of "[subtask[shot_per_subtask[measurement_val]]]",
        where `list[i]` represents the results for subtask `i`, `list[i][j]` represents the results for subtask `i` and shot `j`, and `list[i][j][k]` represents the specific measurement value for subtask `i` and shot `j`.

        True indicates a projective measurement of the |1> state or atom loss; False indicates a projective measurement of the |0> state.
        > To be specific, True means that the atom was not detected, meaning that it could be lost or in an excited state. False means the atom was detected, so it was in the ground state.
        """
        from bloqade.gemini.device.logical.utils import shot_results_for_subtasks

        ret_vals: list[list[list[bool]]] = []
        # TODO: OK to set verify=True?
        subtasks = self.subtasks(verify=True)
        postprocessing_functions = self._slm_postprocessing_functions()
        shot_results = shot_results_for_subtasks(
            self.storage, self.shot_filter, subtasks, frame_type="DETECTED"
        )

        for shot_result, subtask in zip(shot_results, subtasks):
            func = postprocessing_functions[subtask["program_index"]][0]
            ret_vals.append(func(shot_result, invert=True))

        return ret_vals

    @property
    def return_values(self) -> Sequence[Sequence[RetType]]:
        """Return canonical logical values reconstructed from raw SLM shots.

        Results are grouped as ``subtask -> shot -> kernel return value``.
        Raw shots remain available through ``shot_results``.
        """
        return [
            list(
                self._slm_postprocessing_functions()[subtask["program_index"]][
                    1
                ].emit_return(measurements)
            )
            for measurements, subtask in zip(self.measurements, self.subtasks())
        ]

    @property
    def detectors(self) -> Sequence[Sequence[Sequence[bool]]]:
        """
        Returns a nested list of "[subtask[shot_per_subtask[detector_val]]]",
        where `list[i]` represents the results for subtask `i`, `list[i][j]` represents the results for subtask `i` and shot `j`, and `list[i][j][k]` represents the specific detector value for subtask `i` and shot `j`.
        """
        return [
            list(
                self._slm_postprocessing_functions()[subtask["program_index"]][
                    1
                ].emit_detectors(measurements)
            )
            for measurements, subtask in zip(self.measurements, self.subtasks())
        ]

    @property
    def observables(self) -> Sequence[Sequence[Sequence[bool]]]:
        """
        Returns a nested list of "[subtask[shot_per_subtask[observable_val]]]",
        where `list[i]` represents the results for subtask `i`, `list[i][j]` represents the results for subtask `i` and shot `j`, and `list[i][j][k]` represents the specific observable value for subtask `i` and shot `j`.
        """
        return [
            list(
                self._slm_postprocessing_functions()[subtask["program_index"]][
                    1
                ].emit_observables(measurements)
            )
            for measurements, subtask in zip(self.measurements, self.subtasks())
        ]

    @property
    def filling_at_start(self) -> Sequence[Sequence[Sequence[bool]]]:
        """
        Returns a nested list of "[subtask[shot_per_subtask[filling_val]]]",
        where `list[i]` represents the results for subtask `i`, `list[i][j]` represents the results for subtask `i` and shot `j`, and `list[i][j][k]` represents the specific filling value for subtask `i` and shot `j`.

        True indicates that the atom was present in the sorted frame; False indicates that the atom was not present in the sorted frame.
        """
        from bloqade.gemini.device.logical.utils import (
            aligned_detected_and_sorted_shots_for_subtasks,
        )

        ret_vals: list[list[list[bool]]] = []
        subtasks = self.subtasks(verify=True)
        postprocessing_functions = self._slm_postprocessing_functions()
        _, shot_results = aligned_detected_and_sorted_shots_for_subtasks(
            self.storage,
            self.shot_filter,
            subtasks,
        )

        for shot_result, subtask in zip(shot_results, subtasks):
            func = postprocessing_functions[subtask["program_index"]][0]
            ret_vals.append(func(shot_result, invert=False))

        # TODO: can do validation later on sorted/detected frames
        return ret_vals

    def postselect_on_fully_filled(self) -> GeminiLogicalResult[RetType]:
        """Return a result view containing only fully filled shots.

        The predicate reads the ``SORTED`` SLM frame and projects it through
        the stored kernel's SLM-to-raw mapping. A shot is kept only when every
        mapped physical-measurement bit is true. The returned result preserves
        this result's output frame selection, which is normally ``DETECTED``.
        """
        program_index_by_subtask = {
            (subtask["task_id"], subtask["subtask_index"]): subtask["program_index"]
            for subtask in self.full_subtasks()
        }
        slm_postprocessing_functions = self._slm_postprocessing_functions()

        def is_fully_filled(shot) -> bool:
            program_index = program_index_by_subtask[(shot.task_id, shot.subtask_index)]
            slm_to_raw = slm_postprocessing_functions[program_index][0]
            raw_shot = slm_to_raw(
                np.asarray([shot.bitstring], dtype=bool),
                invert=False,
            )
            return bool(np.all(raw_shot))

        return self.where_shots(
            is_fully_filled,
            predicate_filter=replace(self.shot_filter, frame_type="SORTED"),
        )
