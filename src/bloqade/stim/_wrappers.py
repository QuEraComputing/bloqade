from typing import Union

from kirin.lowering import wraps

from .dialects import aux, gate, collapse


# dialect:: gate
## 1q
@wraps(gate.X)
def x(targets: tuple[int, ...], dagger: bool = False) -> None: ...


@wraps(gate.Y)
def y(targets: tuple[int, ...], dagger: bool = False) -> None: ...


@wraps(gate.Z)
def z(targets: tuple[int, ...], dagger: bool = False) -> None: ...


@wraps(gate.Identity)
def identity(targets: tuple[int, ...], dagger: bool = False) -> None: ...


@wraps(gate.H)
def h(targets: tuple[int, ...], dagger: bool = False) -> None: ...


@wraps(gate.S)
def s(targets: tuple[int, ...], dagger: bool = False) -> None: ...


@wraps(gate.SqrtX)
def sqrt_x(targets: tuple[int, ...], dagger: bool = False) -> None: ...


@wraps(gate.SqrtY)
def sqrt_y(targets: tuple[int, ...], dagger: bool = False) -> None: ...


@wraps(gate.SqrtZ)
def sqrt_z(targets: tuple[int, ...], dagger: bool = False) -> None: ...


## clif 2q
@wraps(gate.Swap)
def swap(
    targets: tuple[int, ...], dagger: bool = False
) -> None: ...


## ctrl 2q
@wraps(gate.CX)
def cx(
    controls: tuple[int, ...], targets: tuple[int, ...], dagger: bool = False
) -> None: ...


@wraps(gate.CY)
def cy(
    controls: tuple[int, ...], targets: tuple[int, ...], dagger: bool = False
) -> None: ...


@wraps(gate.CZ)
def cz(
    controls: tuple[int, ...], targets: tuple[int, ...], dagger: bool = False
) -> None: ...


## pp
@wraps(gate.SPP)
def spp(targets: tuple[aux.PauliString, ...], dagger=False) -> None: ...


# dialect:: aux
@wraps(aux.GetRecord)
def rec(id: int) -> aux.RecordResult: ...


@wraps(aux.Detector)
def detector(
    coord: tuple[Union[int, float], ...], targets: tuple[aux.RecordResult, ...]
) -> None: ...


@wraps(aux.ObservableInclude)
def observable_include(idx: int, targets: tuple[aux.RecordResult, ...]) -> None: ...


@wraps(aux.Tick)
def tick() -> None: ...


@wraps(aux.NewPauliString)
def pauli_string(
    string: tuple[str, ...], flipped: tuple[bool, ...], targets: tuple[int, ...]
) -> aux.PauliString: ...


# dialect:: collapse
@wraps(collapse.MZ)
def mz(p: float, targets: tuple[int, ...]) -> None: ...


@wraps(collapse.MY)
def my(p: float, targets: tuple[int, ...]) -> None: ...


@wraps(collapse.MX)
def mx(p: float, targets: tuple[int, ...]) -> None: ...


@wraps(collapse.MZZ)
def mzz(p: float, targets: tuple[int, ...]) -> None: ...


@wraps(collapse.MYY)
def myy(p: float, targets: tuple[int, ...]) -> None: ...


@wraps(collapse.MXX)
def mxx(p: float, targets: tuple[int, ...]) -> None: ...


@wraps(collapse.PPMeasurement)
def mpp(p: float, targets: tuple[aux.PauliString, ...]) -> None: ...


@wraps(collapse.RZ)
def rz(targets: tuple[int, ...]) -> None: ...


@wraps(collapse.RY)
def ry(targets: tuple[int, ...]) -> None: ...


@wraps(collapse.RX)
def rx(targets: tuple[int, ...]) -> None: ...
