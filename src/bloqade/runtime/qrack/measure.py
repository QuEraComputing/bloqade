import abc
import dataclasses
from typing import TYPE_CHECKING, Tuple
from operator import xor
from functools import reduce

import numpy as np

from .reg import SimQubitRef

if TYPE_CHECKING:
    from pyqrack.qrack_simulator import QrackSimulator  # noqa: F401


QrackQubitId = SimQubitRef["QrackSimulator"]


@dataclasses.dataclass(frozen=True)
class PyQrackMeasureOpBase(abc.ABC):
    basis: str

    pauli_map = {
        "x": 1,
        "y": 2,
        "z": 3,
    }

    @abc.abstractmethod
    def do_measurement(
        self, qubit_ids: QrackQubitId | Tuple[QrackQubitId, ...]
    ) -> bool:
        pass

    def do_parallel_measurement(
        self, args: Tuple[QrackQubitId | Tuple[QrackQubitId, ...], ...]
    ) -> Tuple[bool, ...]:
        return tuple(map(self.do_measurement, args))


@dataclasses.dataclass(frozen=True)
class PerfectPauli(PyQrackMeasureOpBase):
    def do_measurement(self, qubit_id: QrackQubitId) -> bool:
        return qubit_id.sim_reg.measure_pauli(self.pauli_map[self.basis], qubit_id.addr)


@dataclasses.dataclass(frozen=True)
class NoisyPauli(PerfectPauli):
    p: float
    rng_state: np.random.Generator

    def do_measurement(self, qubit_id: QrackQubitId) -> bool:
        result: bool = super().do_measurement(qubit_id)

        return result if self.rng_state.random() < self.p else not result


@dataclasses.dataclass(frozen=True)
class PerfectPP(PerfectPauli):
    def do_measurement(self, qubit_ids: Tuple[QrackQubitId, ...]) -> bool:
        measure_iter = (
            PerfectPauli(basis).do_measurement(qubit_id)
            for basis, qubit_id in zip(self.basis, qubit_ids, strict=True)
        )
        return reduce(xor, measure_iter)


@dataclasses.dataclass(frozen=True)
class NoisyPP(PerfectPP):
    p: float
    rng_state: np.random.Generator

    def do_measurement(self, qubit_ids: Tuple[QrackQubitId, ...]) -> bool:
        result: bool = super().do_measurement(qubit_ids)

        return result if self.rng_state.random() < self.p else not result
