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
def cx(ctrl: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.UGate)
def u(qarg: Qubit, theta: float, phi: float, lam: float) -> None: ...


@wraps(uop.Barrier)
def barrier(qargs: tuple[Qubit, ...]) -> None: ...


@wraps(uop.H)
def h(qarg: Qubit) -> None: ...


@wraps(uop.X)
def x(qarg: Qubit) -> None: ...


@wraps(uop.Y)
def y(qarg: Qubit) -> None: ...


@wraps(uop.Z)
def z(qarg: Qubit) -> None: ...


@wraps(uop.S)
def s(qarg: Qubit) -> None: ...


@wraps(uop.Sdag)
def sdg(qarg: Qubit) -> None: ...


@wraps(uop.T)
def t(qarg: Qubit) -> None: ...


@wraps(uop.Tdag)
def tdg(qarg: Qubit) -> None: ...


@wraps(uop.RX)
def rx(qarg: Qubit, theta: float) -> None: ...


@wraps(uop.RY)
def ry(qarg: Qubit, theta: float) -> None: ...


@wraps(uop.RZ)
def rz(qarg: Qubit, theta: float) -> None: ...


@wraps(uop.U1)
def u1(qarg: Qubit, lam: float) -> None: ...


@wraps(uop.U2)
def u2(qarg: Qubit, phi: float, lam: float) -> None: ...


@wraps(uop.CZ)
def cz(ctrl: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.CY)
def cy(ctrl: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.CH)
def ch(ctrl: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.CCX)
def ccx(ctrl1: Qubit, ctrl2: Qubit, qarg: Qubit) -> None: ...


@wraps(uop.CRX)
def crx(ctrl: Qubit, qarg: Qubit, theta: float) -> None: ...


@wraps(uop.CU1)
def cu1(ctrl: Qubit, qarg: Qubit, lam: float) -> None: ...


@wraps(uop.CU3)
def cu3(ctrl: Qubit, qarg: Qubit, theta: float, phi: float, lam: float) -> None: ...


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
