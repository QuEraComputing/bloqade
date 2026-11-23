from __future__ import annotations

from typing import Any, TypeVar
from collections.abc import Callable

import numpy as np
from kirin import ir
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.transform import LogicalPipeline, PhysicalPipeline
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.bytecode.encoding import ZoneAddress
from bloqade.lanes.analysis.atom._shot_remapping import (
    ShotRemappingErr,
    get_shot_remapping,
)

RetType = TypeVar("RetType")


class ShotRemappingException(Exception):
    """Exception if we failed to produce a shot remapping."""


def get_slm_mapping_postprocessing(
    sim_kernel: ir.Method[..., RetType],
    *,
    physical_kernel: bool = False,
) -> tuple[Callable[..., Any], atom.PostProcessing[RetType]]:
    """Create a result postprocessor for full Zone-0 SLM shots.

    Result post-processing is two steps, and this returns both::

        physical_bitstring = [frame_data[i] for i in shot_remapping]
        user_output = post_processing.emit_return([physical_bitstring])

    The returned post-processing object comes from
    :func:`~bloqade.gemini.post_processing.build_post_processing`, which
    abstract-interprets the *user's* kernel once to reconstruct its return
    value and every detector and observable annotation.

    Warning:
        This reconstructs the physical measurement mapping by compiling the
        stored source kernel with the locally installed Bloqade Lanes compiler
        and Gemini architecture. The result is valid only when they match the
        compiler and architecture used to execute the remote task. The exact
        remote physical compilation artifact is not currently stored with the
        result.
    """

    from bloqade.gemini.post_processing import build_post_processing

    arch_spec = physical.get_arch_spec()
    if physical_kernel:
        physical_move_kernel = PhysicalPipeline(arch_spec=arch_spec).emit(
            sim_kernel.similar(), no_raise=False
        )
    else:
        physical_move_kernel = LogicalPipeline(transversal_rewrite=True).emit(
            sim_kernel
        )
    zone0 = ZoneAddress(0)

    interpreter = atom.AtomInterpreter(
        physical_move_kernel.dialects,
        arch_spec=arch_spec,
    )
    positions = None
    if physical_kernel:
        positions = interpreter.get_measurement_positions(physical_move_kernel)
        shot_mapping = get_shot_remapping(positions)
    else:
        shot_mapping = interpreter.get_shot_remapping(physical_move_kernel)
    if isinstance(shot_mapping, ShotRemappingErr):
        raise ShotRemappingException(
            f"Failed to produce shot remapping, error message: {shot_mapping.diagnostic.message}, reason: {shot_mapping.diagnostic.offending_value}"
        )
    mapping = shot_mapping.mapping
    expected_zone0_sites = len(list(arch_spec.yield_zone_locations(zone0)))

    post_processing = build_post_processing(sim_kernel)

    initial_mapping = mapping
    if positions is not None:
        # A physical kernel can finish away from its load sites. SORTED is
        # recorded before transport, DETECTED at the terminal measurement.
        fills = [
            s
            for s in physical_move_kernel.callable_region.walk()
            if isinstance(s, move.Fill)
        ]
        if len(fills) != 1:
            raise ShotRemappingException("Expected one initial Fill")
        initial_mapping = []
        for record in positions.readout:
            address = fills[0].location_addresses[record.qubit_id]
            slot = arch_spec.get_zone_index(address, zone0)
            if address.zone_id != 0 or slot is None:
                raise ShotRemappingException("Initial filling is outside Zone 0")
            initial_mapping.append(slot)

    def postprocess(zone0_shots, *, invert: bool = False, initial: bool = False):
        zone0_shots = np.asarray(zone0_shots, dtype=bool)

        if zone0_shots.size == 0:
            return []
        if zone0_shots.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got {zone0_shots.shape}.")
        if zone0_shots.shape[1] != expected_zone0_sites:
            raise ValueError(
                f"Expected {expected_zone0_sites} Zone-0 columns, "
                f"got {zone0_shots.shape[1]}."
            )

        measurement_shots = zone0_shots[:, initial_mapping if initial else mapping]

        # Enable only if the stored QLAM bit convention is opposite to the
        # simulator/post-processing convention.
        if invert:
            measurement_shots = ~measurement_shots

        return measurement_shots.tolist()

    return postprocess, post_processing
