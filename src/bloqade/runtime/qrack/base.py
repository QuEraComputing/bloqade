from typing import Generic, TypeVar, Optional
from dataclasses import field, dataclass

import numpy as np
from kirin.interp import Interpreter
from typing_extensions import Self

SimRegType = TypeVar("SimRegType")


@dataclass
class Memory(Generic[SimRegType]):
    total: int
    allocated: int
    sim_reg: SimRegType


@dataclass
class PyQrackInterpreter(Interpreter, Generic[SimRegType]):
    keys = ["pyqrack", "main"]
    memory: Memory[SimRegType] = field(kw_only=True)
    rng_state: Optional[np.random.Generator] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rng_state = (
            np.random.default_rng() if self.rng_state is None else self.rng_state
        )

    def initialize(self) -> Self:
        super().initialize()
        self.memory.allocated = 0  # reset allocated qubits
        return self
