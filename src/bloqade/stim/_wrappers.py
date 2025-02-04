from typing import Union

from kirin.lowering import wraps

from .dialects import aux, gate


# dialect:: gate
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


# dialect:: aux
@wraps(aux.GetRecord)
def rec(id: int) -> aux.RecordResult: ...


@wraps(aux.Detector)
def detector(
    coord: tuple[Union[int, float], ...], targets: tuple[aux.RecordResult, ...]
) -> None: ...


@wraps(aux.ObservableInclude)
def obs_include(idx: int, targets: tuple[aux.RecordResult, ...]) -> None: ...


@wraps(aux.Tick)
def tick() -> None: ...


@wraps(aux.NewPauliString)
def pauli_string(
    string: tuple[str, ...], flipped: tuple[bool, ...], targets: tuple[int, ...]
) -> aux.PauliString: ...
