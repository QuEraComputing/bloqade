import abc
import dataclasses
from typing import TYPE_CHECKING, Tuple, Iterable
from functools import cached_property
from itertools import starmap

import numpy as np

from .reg import SimQubitRef

if TYPE_CHECKING:
    from pyqrack.qrack_simulator import QrackSimulator  # noqa: F401

QrackQubitId = SimQubitRef["QrackSimulator"]


class GateQrackRuntimeABC(abc.ABC):
    @abc.abstractmethod
    def apply(self, *qubit: QrackQubitId) -> None:
        pass

    def parallel_apply(self, qubits_ids: Tuple[Tuple[QrackQubitId, ...], ...]) -> None:
        list(starmap(self.apply, qubits_ids))


@dataclasses.dataclass(frozen=True)
class Gate(GateQrackRuntimeABC):
    method_name: str

    def apply(self, qubit: QrackQubitId) -> None:
        getattr(qubit.sim_reg, self.method_name)(qubit.addr)


@dataclasses.dataclass(frozen=True)
class Control(GateQrackRuntimeABC):
    method_name: str

    def apply(self, *targets: QrackQubitId):
        ctrl = tuple(q.addr for q in targets[:-1])
        target = targets[-1]
        getattr(target.sim_reg, self.method_name)(ctrl, target.addr)


@dataclasses.dataclass(frozen=True)
class Reset(GateQrackRuntimeABC):
    def apply(self, qubit_id: QrackQubitId):
        qubit_id.sim_reg.force_m(qubit_id.addr, False)


@dataclasses.dataclass(frozen=True)
class SRotation(GateQrackRuntimeABC):
    axis: int
    angle: float

    def apply(self, qubit_id: QrackQubitId):
        qubit_id.sim_reg.r(self.axis, self.angle, qubit_id.addr)


@dataclasses.dataclass(frozen=True)
class Rot3(GateQrackRuntimeABC):
    alpha: float
    beta: float
    gamma: float

    def apply(self, qubit_id: QrackQubitId):
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class NoisyReset(GateQrackRuntimeABC):
    p: float
    rng_state: np.random.Generator = dataclasses.field(kw_only=True)

    def apply(self, qubit_id: QrackQubitId):
        if self.rng_state.random() < self.p:
            qubit_id.sim_reg.force_m(qubit_id.addr, True)
        else:
            qubit_id.sim_reg.force_m(qubit_id.addr, False)


@dataclasses.dataclass(frozen=True)
class PPError(GateQrackRuntimeABC):
    p: float
    op: str
    rng_state: np.random.Generator = dataclasses.field(kw_only=True)

    def __post_init__(self):
        assert set(self.op) <= {"i", "x", "y", "z"}

    def apply(self, *qubit_id: QrackQubitId):
        if self.rng_state.random() >= self.p:
            return

        for err, q in zip(self.op, qubit_id):
            getattr(q.sim_reg, err, lambda x: None)(q.addr)


@dataclasses.dataclass(frozen=True)
class PauliChannelABC(GateQrackRuntimeABC):
    rng_state: np.random.Generator = dataclasses.field(kw_only=True)

    def __post_init__(self):
        assert ((len(self.probs)) ** 0.25) == 4
        assert all(p >= 0 for p in self.probs) and sum(self.probs) == 1

    @property
    @abc.abstractmethod
    def probs(self) -> Tuple[float, ...]:
        pass

    @staticmethod
    def get_str(index: int) -> Iterable[str]:
        opstr = ["i", "x", "y", "z"]
        while index > 0:
            index, r = divmod(index, 4)
            yield opstr[r]

    def select_op(self) -> int:
        return self.rng_state.choice(len(self.probs), p=self.probs)

    def apply(self, *qubit_id: QrackQubitId):
        which_errors = self.select_op()

        for q, err in zip(qubit_id, self.get_str(which_errors)):
            getattr(q.sim_reg, err, lambda x: None)(q.addr)


@dataclasses.dataclass(frozen=True)
class Depolarization(PauliChannelABC):
    p: float
    n_qubits: int

    @cached_property
    def probs(self) -> Tuple[float, ...]:
        total_errors: int = 4**self.n_qubits - 1
        p_noise = self.p / total_errors
        return (1 - self.p,) + tuple(p_noise for _ in range(total_errors))


@dataclasses.dataclass(frozen=True)
class PauliChannel(PauliChannelABC):
    """Generic Pauli channel with arbitrary number of qubits and arbitrary error probabilities.

    Probability of applying pauli errors to a qubit. The sum(p) must be less than or equal to 1.
    The position inside the tuple of probabilities maps to the error type, for single qubit errors:

    p[0]: X
    p[1]: Y
    p[2]: Z

    for multi-qubit errors, the order is lexicographic, e.g. for 2 qubits:
    p[0]: IX
    p[1]: IY
    p[2]: IZ
    p[3]: XI
    p[4]: XX
    p[5]: XY
    p[6]: XZ
    etc.

    """

    p: Tuple[float, ...]

    @cached_property
    def probs(self) -> Tuple[float, ...]:
        return (1 - sum(self.p),) + self.p


# @dataclasses.dataclass(frozen=True)
# class PauliChannel(GateQrackRuntimeABC):
#     # probability of X, Y, Z being applied to a qubit,
#     # list of floats = probability that each gate gets applied
#     ## if you have 3 values, [px, py, pz]
#     ## if you have 15 values, [pix, piy, piz, ..., pzz]
#     p: tuple[float, ...]
#     rng_state: (
#         np.random.RandomState
#     )  # some indication this is a bit outdated, should use "Generator"

#     def apply_pauli_str(self, pauli_str: str, *qubit_id: QrackQubitId):
#         for pauli_op, q in zip(pauli_str, qubit_id):
#             match pauli_op:
#                 case "X":
#                     q.sim_reg.x(q.addr)
#                 case "Y":
#                     q.sim_reg.y(q.addr)
#                 case "Z":
#                     q.sim_reg.z(q.addr)

#     def apply(
#         self, *qubit_id: QrackQubitId
#     ):  # qubit id is literally the container containing qubit IDs to run

#         probabilities_with_id = (1 - sum(self.p),) + self.p
#         for pauli_str, application_prob in zip(
#             iter.product("IXYZ", repeat=len(self.p)), probabilities_with_id
#         ):
#             if self.rng_state.binomial(n=1, p=application_prob) == 1:
#                 self.apply_pauli_str(pauli_str, *qubit_id)
#                 break
