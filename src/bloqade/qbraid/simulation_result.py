# Copyright (c) 2024, QuEra Computing Inc.
# All rights reserved.

from io import StringIO
from typing import Any, Dict, Optional
from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame
from bloqade.visual.animation.runtime import qpustate as vis_qpustate

from .schema import NoiseModel


@dataclass
class QuEraSimulationResult:
    """Results of the QuEra hardware model simulation.

    Fields:
        flair_visual_version (str): The version of the Flair Visual package used to generate the simulation result.
        counts (dict[str, int]): The measurement bitstrings of the simulation.
        logs (DataFrame): Grainular logs events of what happened to each atom during the simulation.
        atom_animation_state (vis_qpustate.AnimateQPUState): Object used to play back atom trajectories and events during the simulation.
        noise_model (NoiseModel): The noise model used in the simulation.

    """

    flair_visual_version: str
    counts: dict[str, int]
    logs: DataFrame
    atom_animation_state: vis_qpustate.AnimateQPUState
    noise_model: NoiseModel

    @classmethod
    def from_json(cls, json: dict) -> "QuEraSimulationResult":
        """deserialize the object from a JSON serializable dictionary."""
        flair_visual_version = json["flair_visual_version"]
        counts = json["counts"]
        logs = pd.read_csv(StringIO(json["logs"]), index_col=0)
        atom_animation_state = vis_qpustate.AnimateQPUState.from_json(
            json["atom_animation_state"]
        )
        noise_model = NoiseModel(**json["noise_model"])

        return cls(
            flair_visual_version=flair_visual_version,
            counts=counts,
            logs=logs,
            atom_animation_state=atom_animation_state,
            noise_model=noise_model,
        )

    def to_json(self) -> Dict[str, Any]:
        """Turn the object into a JSON serializable dictionary."""
        return {
            "flair_visual_version": self.flair_visual_version,
            "counts": self.counts,
            "logs": self.logs.to_csv(),
            "atom_animation_state": self.atom_animation_state.to_json(),
            "noise_model": self.noise_model.model_dump(mode="json"),
        }

    def animate(
        self,
        dilation_rate: float = 0.05,
        fps: int = 30,
        gate_display_dilation: float = 1.0,
        save_mpeg: bool = False,
        filename: str = "vqpu_animation",
        start_block: int = 0,
        n_blocks: Optional[int] = None,
    ):
        """animate the qpu state

        Args:
            dilation_rate (float): Conversion factor from the qpu time to animation time units. when dilation_rate=1.0, 1 (us) of qpu exec time corresponds to 1 second of animation time.
            fps (int, optional): frame per second. Defaults to 30.
            gate_display_dilation (float, optional): relative dilation rate of a gate event. Defaults to 1. When setting higher value, the gate event will be displayed longer.
            save_mpeg (bool, optional): Save as mpeg. Defaults to False.
            filename (str, optional): The file name of saved mpeg file. Defaults to "vqpu_animation". When `save_mpeg` is False, this argument is ignored.
            start_block (int, optional): The start block to animate. Defaults to 0.
            n_blocks (int, optional): number of blocks to animate. Defaults to None. When None, animate all blocks after `start_block`.
        Returns:
            ani: matplotlib animation object
        """
        from bloqade.visual.animation.animate import animate_qpu_state

        ani = animate_qpu_state(
            state=self.atom_animation_state,
            dilation_rate=dilation_rate,
            fps=fps,
            gate_display_dilation=gate_display_dilation,
            start_block=start_block,
            n_blocks=n_blocks,
            save_mpeg=save_mpeg,
            filename=filename,
        )
        return ani
