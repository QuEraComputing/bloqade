import dataclasses
from typing import Any, Dict, Tuple

from .ppoly import PPoly


@dataclasses.dataclass(frozen=True)
class AODMoveEvent:
    time: float
    duration: float
    x: PPoly
    y: PPoly

    def __post_init__(self):
        assert isinstance(self.x, PPoly), "x must be a PPoly instance"
        assert isinstance(self.y, PPoly), "y must be a PPoly instance"

    def sample(self, t: float) -> Tuple[float, float]:
        return (self.x(t - self.time), self.y(t - self.time))

    def to_json(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "duration": self.duration,
            "x": self.x.to_json(),
            "y": self.y.to_json(),
        }

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> "AODMoveEvent":
        return cls(
            json_dict["time"],
            json_dict["duration"],
            PPoly.from_json(json_dict["x"]),
            PPoly.from_json(json_dict["y"]),
        )
