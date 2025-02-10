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

    def __getitem__(self, pos: int):
        return QubitRef(self, pos)


@dataclass(frozen=True)
class QubitRef:
    ref: QRegister
    pos: int


class CRegister(list[bool]):
    def __init__(self, size: int):
        super().__init__(False for _ in range(size))


@dataclass(frozen=True)
class CBitRef:
    ref: CRegister
    pos: int

    def set_value(self, value: bool):
        self.ref[self.pos] = value

    def get_value(self):
        return self.ref[self.pos]


SimRegType = TypeVar("SimRegType")


@dataclass(frozen=True)
class SimQRegister(QRegister, Generic[SimRegType]):
    sim_reg: SimRegType
    addrs: tuple[int, ...]
    qubit_state: List[QubitState]

    def drop(self, pos: int):
        assert self.qubit_state[pos] is QubitState.Active, "Qubit already lost"
        self.qubit_state[pos] = QubitState.Lost

    def __getitem__(self, pos: int):
        return SimQubitRef(self, pos)


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
