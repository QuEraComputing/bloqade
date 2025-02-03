import dataclasses
from typing import Any, Dict


@dataclasses.dataclass(frozen=True)
class GateEvent:
    cls_name: str
    kwargs: Dict[str, Any]
    duration: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "cls_name": self.cls_name,
            "kwargs": self.kwargs,
            "duration": self.duration,
        }

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> "GateEvent":
        return cls(
            json_dict["cls_name"],
            json_dict["kwargs"],
            json_dict["duration"],
        )
