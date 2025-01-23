from kirin.lowering import wraps

from .types import Bit, CReg, QReg, Qubit
from .dialects import uop, core, expr, inline as inline_


@wraps(inline_.InlineQASM)
def inline(text: str) -> None: ...


@wraps(core.QRegNew)
def qreg(n_qubits: int) -> QReg: ...


@wraps(core.CRegNew)
def creg(n_bits: int) -> CReg: ...


@wraps(core.Reset)
def reset(qarg: Qubit) -> None: ...


@wraps(core.Measure)
def measure(qarg: Qubit, cbit: Bit) -> None: ...


@wraps(uop.CX)
def CX(ctrl: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.UGate)
def UGate(qarg: Qubit, theta: float, phi: float, lam: float) -> None: ...


@wraps(uop.Barrier)
def barrier(qargs: tuple[Qubit, ...]) -> None: ...


@wraps(uop.H)
def H(qarg: Qubit) -> None: ...


@wraps(uop.X)
def X(qarg: Qubit) -> None: ...


@wraps(uop.Y)
def Y(qarg: Qubit) -> None: ...


@wraps(uop.Z)
def Z(qarg: Qubit) -> None: ...


@wraps(uop.S)
def S(qarg: Qubit) -> None: ...


@wraps(uop.Sdag)
def Sdag(qarg: Qubit) -> None: ...


@wraps(uop.T)
def T(qarg: Qubit) -> None: ...


@wraps(uop.Tdag)
def Tdag(qarg: Qubit) -> None: ...


@wraps(uop.RX)
def RX(qarg: Qubit, theta: float) -> None: ...


@wraps(uop.RY)
def RY(qarg: Qubit, theta: float) -> None: ...


@wraps(uop.RZ)
def RZ(qarg: Qubit, theta: float) -> None: ...


@wraps(uop.U1)
def U1(qarg: Qubit, lam: float) -> None: ...


@wraps(uop.U2)
def U2(qarg: Qubit, phi: float, lam: float) -> None: ...


@wraps(uop.CZ)
def CZ(ctrl: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.CY)
def CY(ctrl: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.CH)
def CH(ctrl: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.CCX)
def CCX(ctrl1: Qubit, ctrl2: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.CRX)
def CRX(ctrl: Qubit, qarg: Qubit, theta: float) -> None: ...


@wraps(uop.CU1)
def CU1(ctrl: Qubit, qarg: Qubit, lam: float) -> None: ...


@wraps(uop.CU3)
def CU3(ctrl: Qubit, qarg: Qubit, theta: float, phi: float, lam: float) -> None: ...


@wraps(expr.Sin)
def sin(value: float) -> float: ...


@wraps(expr.Cos)
def cos(value: float) -> float: ...


@wraps(expr.Tan)
def tan(value: float) -> float: ...


@wraps(expr.Exp)
def exp(value: float) -> float: ...


@wraps(expr.Log)
def ln(value: float) -> float: ...


@wraps(expr.Sqrt)
def sqrt(value: float) -> float: ...
