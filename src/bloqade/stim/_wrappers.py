from typing import Union

from kirin.lowering import wraps

from .dialects import aux, gate


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
    controls: tuple[int, ...], targets: tuple[int, ...], dagger: bool = False
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
