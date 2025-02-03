import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from .aod import AODMoveEvent
from ..base import FieldOfView
from .atoms import AtomTrajectory
from ..gate_event import GateEvent


@dataclasses.dataclass
class QPUStateABC(ABC):

    block_durations: List[float] = dataclasses.field(default_factory=list)
    gate_events: List[Tuple[float, GateEvent]] = dataclasses.field(default_factory=list)
    qpu_fov: FieldOfView = dataclasses.field(default_factory=FieldOfView)

    def __post_init__(self):
        assert all(
            isinstance(gate, GateEvent) for _, gate in self.gate_events
        ), "All gate events must be GateEvent instances"

    @abstractmethod
    def get_slm_sites(self) -> np.ndarray: ...

    @abstractmethod
    def sample_aod_traps(self, time: float) -> List[Tuple[float, float]]: ...

    @abstractmethod
    def get_atoms_lost_info(self, time: float) -> List[str]: ...

    @abstractmethod
    def get_atoms_position(
        self, time: float, include_lost: bool = False
    ) -> List[Tuple[int, Tuple[float, float]]]: ...

    def get_gate_events_timing(self) -> List[Tuple[float, float]]:
        # return [t_start, duration]
        return [(t, gate.duration) for t, gate in self.gate_events]

    def get_gate_events(self, time: float) -> List[Tuple[float, GateEvent]]:
        out = []
        if self.gate_events:
            for t, gate in self.gate_events:
                if t > time:
                    break
                if t <= time < t + gate.duration:
                    out.append((t, gate))
        return out


@dataclasses.dataclass
class AnimateQPUState(QPUStateABC):
    atoms: List[AtomTrajectory] = dataclasses.field(default_factory=list)
    slm_zone: List[Tuple[float, float]] = dataclasses.field(default_factory=list)
    aod_moves: List[AODMoveEvent] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        assert all(
            isinstance(atom, AtomTrajectory) for atom in self.atoms
        ), "All atoms must be AtomTrajectory instances"
        assert all(
            isinstance(aod, AODMoveEvent) for aod in self.aod_moves
        ), "All AOD moves must be AODMoveEvent instances"
        assert all(
            isinstance(site, tuple) and len(site) == 2 for site in self.slm_zone
        ), "All SLM sites must be tuples of length 2"

    def get_slm_sites(self):
        return np.array(self.slm_zone)

    def sample_aod_traps(self, time: float) -> List[Tuple[float, float]]:
        out = []
        for aod_trap in self.aod_moves:
            if aod_trap.time <= time < aod_trap.time + aod_trap.duration:
                out.append(aod_trap.sample(time))

        return out

    def get_atoms_lost_info(self, time: float) -> List[str]:
        return [
            f"Lost: {atom.id} @{atom.events[-1][0]:.3f} (us)" + "\n"
            for atom in self.atoms
            if atom.is_lost(time)
        ]

    def get_atoms_position(
        self, time: float, include_lost: bool = False
    ) -> List[Tuple[int, Tuple[float, float]]]:
        return [
            (atom.id, atom.position(time))
            for atom in self.atoms
            if not atom.is_lost(time) or include_lost
        ]

    def to_json(self) -> Dict[str, Any]:
        return {
            "block_durations": self.block_durations,
            "gate_events": [(t, gate.to_json()) for t, gate in self.gate_events],
            "qpu_fov": self.qpu_fov.to_json(),
            "atoms": [atom.to_json() for atom in self.atoms],
            "slm_zone": self.slm_zone,
            "aod_moves": [aod.to_json() for aod in self.aod_moves],
        }

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> "AnimateQPUState":
        return cls(
            block_durations=json_dict["block_durations"],
            gate_events=[
                (t, GateEvent.from_json(gate)) for t, gate in json_dict["gate_events"]
            ],
            qpu_fov=FieldOfView.from_json(json_dict["qpu_fov"]),
            atoms=list(map(AtomTrajectory.from_json, json_dict["atoms"])),
            slm_zone=list(map(tuple, json_dict["slm_zone"])),
            aod_moves=list(map(AODMoveEvent.from_json, json_dict["aod_moves"])),
        )
