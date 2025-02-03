import dataclasses
from typing import List, Tuple
from functools import cached_property

import numpy as np

from .ppoly import PPoly


@dataclasses.dataclass(frozen=True)
class AtomTrajectory:
    id: int
    x: PPoly
    y: PPoly
    events: List[Tuple[float, str]]  # (time, event_type)

    def __post_init__(self):
        assert all(
            isinstance(event, tuple) and len(event) == 2 for event in self.events
        ), "All events must be (float, str) pairs"
        assert isinstance(self.id, int), "id must be an int"
        assert isinstance(self.x, PPoly), "x must be a PPoly instance"
        assert isinstance(self.y, PPoly), "y must be a PPoly instance"

    def position(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return (self.x(t), self.y(t))

    @cached_property
    def has_lost(self) -> bool:
        return any(event_type == "Lost" for _, event_type in self.events)

    def is_lost(self, time: float) -> bool:
        if not self.has_lost:
            return False

        return (
            self.events[-1][0] <= time
        )  # TODO: maybe we need approx_eq to be separate package?

    def to_json(self):
        return {
            "id": self.id,
            "x": self.x.to_json(),
            "y": self.y.to_json(),
            "events": self.events,
        }

    @classmethod
    def from_json(cls, json_dict):
        return AtomTrajectory(
            json_dict["id"],
            PPoly.from_json(json_dict["x"]),
            PPoly.from_json(json_dict["y"]),
            list(map(tuple, json_dict["events"])),
        )
