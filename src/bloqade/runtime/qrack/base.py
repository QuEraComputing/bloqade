import dataclasses
from typing import TYPE_CHECKING, Generic, TypeVar, Optional

import numpy as np
from kirin.interp import Interpreter

if TYPE_CHECKING:
    from pyqrack.qrack_simulator import QrackSimulator


SimRegType = TypeVar("SimRegType")


@dataclasses.dataclass
class Memory(Generic[SimRegType]):
    total: int
    allocated: int
    sim_reg: SimRegType


class PyQrackInterpreter(Interpreter):
    keys = ["pyqrack", "main"]
    memory: Memory["QrackSimulator"]

    def __init__(
        self,
        memory: Memory["QrackSimulator"],
        *args,
        rng_state: Optional[np.random.Generator] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rng_state = np.random.default_rng() if rng_state is None else rng_state
        self.memory = memory

    def eval(self, mt, args=(), kwargs={}):
        self.memory.allocated = 0  # reset allocated qubits
        return super().eval(mt=mt, args=args, kwargs=kwargs)
