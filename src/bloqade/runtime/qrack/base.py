from typing import Generic, TypeVar
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
    rng_state: np.random.Generator = field(
        default_factory=np.random.default_rng, kw_only=True
    )

    def initialize(self) -> Self:
        super().initialize()
        self.memory.allocated = 0  # reset allocated qubits
        return self
