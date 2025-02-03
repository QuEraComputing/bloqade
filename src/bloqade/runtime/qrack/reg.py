import enum
from typing import List, Generic, TypeVar
from dataclasses import dataclass


class QubitState(enum.Enum):
    Active = enum.auto()
    Lost = enum.auto()


@dataclass(frozen=True)
class QRegister:
    size: int

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass(frozen=True)
class QubitRef:
    ref: QRegister
    pos: int


class CRegister(list[bool]):
    def __init__(self, size: int):
        super().__init__(False for _ in range(size))


SimRegType = TypeVar("SimRegType")


@dataclass(frozen=True)
class SimQRegister(QRegister, Generic[SimRegType]):
    sim_reg: SimRegType
    addrs: tuple[int, ...]
    qubit_state: List[QubitState]

    def drop(self, pos: int):
        assert self.qubit_state[pos] is QubitState.Active, "Qubit already lost"
        self.qubit_state[pos] = QubitState.Lost


@dataclass(frozen=True)
class SimQubitRef(QubitRef, Generic[SimRegType]):
    ref: SimQRegister[SimRegType]
    pos: int

    @property
    def sim_reg(self) -> SimRegType:
        return self.ref.sim_reg

    @property
    def addr(self) -> int:
        return self.ref.addrs[self.pos]

    def is_active(self) -> bool:
        return self.ref.qubit_state[self.pos] is QubitState.Active

    def drop(self):
        self.ref.drop(self.pos)
