import dataclasses
from typing import Any, Dict
from functools import cached_property

import numpy as np
import scipy.interpolate as interp

from .utils import array_to_json, json_to_array


@dataclasses.dataclass
class PPoly:
    c: np.ndarray
    x: np.ndarray
    extrapolate: bool = dataclasses.field(default=False)

    def __post_init__(self):
        super().__setattr__("x", np.asarray(self.x))
        super().__setattr__("c", np.asarray(self.c))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PPoly):
            return False
        return (
            np.array_equal(self.x, other.x)
            and np.array_equal(self.c, other.c)
            and self.extrapolate == other.extrapolate
        )

    @cached_property
    def get_ppoly(self) -> interp.PPoly:
        return interp.PPoly(self.c, self.x, self.extrapolate)

    def __call__(self, x: float) -> float:
        return self.get_ppoly(x)

    def to_json(self) -> Dict[str, Any]:
        return {
            "c": array_to_json(self.c),
            "x": array_to_json(self.x),
            "extrapolate": self.extrapolate,
        }

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> "PPoly":
        return cls(
            c=json_to_array(json_dict["c"]),
            x=json_to_array(json_dict["x"]),
            extrapolate=json_dict["extrapolate"],
        )
