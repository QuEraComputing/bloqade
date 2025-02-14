from kirin.lowering import wraps

from .types import Bit, CReg, QReg, Qubit
from .dialects import uop, core, expr, inline as inline_


@wraps(inline_.InlineQASM)
def inline(text: str) -> None:
    """
    Inline QASM code into the current program.

    Args:
        text: The QASM code to inline.
    """
    ...


@wraps(core.QRegNew)
def qreg(n_qubits: int) -> QReg:
    """
    Create a new quantum register with `n_qubits` qubits.

    Args:
        n_qubits: The number of qubits in the register.

    Returns:
        The newly created quantum register.

    """
    ...


@wraps(core.CRegNew)
def creg(n_bits: int) -> CReg:
    """
    Create a new classical register with `n_bits` bits.

    Args:
        n_bits: The number of bits in the register.

    Returns:
        The newly created classical register.

    """
    ...


@wraps(core.Reset)
def reset(qarg: Qubit) -> None:
    """
    Reset the qubit `qarg` to the |0⟩ state.

    Args:
        qarg: The qubit to reset.

    """

    ...


@wraps(core.Measure)
def measure(qarg: Qubit, cbit: Bit) -> None:
    """
    Measure the qubit `qarg` and store the result in the classical bit `cbit`.

    Args:
        qarg: The qubit to measure.
        cbit: The classical bit to store the result in.
    """
    ...


@wraps(uop.CX)
def cx(ctrl: Qubit, qarg: Qubit) -> None:
    """
    Controlled-X (CNOT) gate.

    Args:
        ctrl: The control qubit.
        qarg: The target qubit.
    """
    ...


@wraps(uop.UGate)
def u(qarg: Qubit, theta: float, phi: float, lam: float) -> None:
    """
    U gate.

    Note:
        See https://arxiv.org/pdf/1707.03429 for definition of angles.

    Args:
        qarg: The qubit to apply the gate to.
        theta: The angle of rotation
        phi: The angle of rotation
        lam: The angle of rotation

    """
    ...


@wraps(uop.Barrier)
def barrier(qargs: tuple[Qubit, ...]) -> None:
    """
    Barrier instruction.

    Args:
        qargs: The qubits to apply the barrier to.
    """

    ...


@wraps(uop.H)
def h(qarg: Qubit) -> None:
    """
    Hadamard gate.

    Args:
        qarg: The qubit to apply the gate to.

    """
    ...


@wraps(uop.X)
def x(qarg: Qubit) -> None:
    """
    Pauli-X gate.

    Args:
        qarg: The qubit to apply the gate to.
    """

    ...


@wraps(uop.Y)
def y(qarg: Qubit) -> None:
    """
    Pauli-Y gate.

    Args:
        qarg: The qubit to apply the gate to.

    """
    ...


@wraps(uop.Z)
def z(qarg: Qubit) -> None:
    """
    Pauli-Z gate.

    Args:
        qarg: The qubit to apply the gate to.

    """
    ...


@wraps(uop.S)
def s(qarg: Qubit) -> None:
    """
    S gate.

    Args:
        qarg: The qubit to apply the gate to.
    """

    ...


@wraps(uop.Sdag)
def sdg(qarg: Qubit) -> None:
    """
    Hermitian conjugate of the S gate.

    Args:
        qarg: The qubit to apply the gate to.

    """

    ...


@wraps(uop.T)
def t(qarg: Qubit) -> None:
    """
    T gate.

    Args:
        qarg: The qubit to apply the gate to.
    """

    ...


@wraps(uop.Tdag)
def tdg(qarg: Qubit) -> None:
    """
    Hermitian conjugate of the T gate.

    Args:
        qarg: The qubit to apply the gate to.

    """

    ...


@wraps(uop.RX)
def rx(qarg: Qubit, theta: float) -> None:
    """
    Single qubit rotation about the X axis on block sphere

    Args:
        qarg: The qubit to apply the gate to.
        theta: The angle of rotation.
    """
    ...


@wraps(uop.RY)
def ry(qarg: Qubit, theta: float) -> None:
    """
    Single qubit rotation about the Y axis on block sphere

    Args:
        qarg: The qubit to apply the gate to.
        theta: The angle of rotation.

    """

    ...


@wraps(uop.RZ)
def rz(qarg: Qubit, theta: float) -> None:
    """
    Single qubit rotation about the Z axis on block sphere

    Args:
        qarg: The qubit to apply the gate to.
        theta: The angle of rotation.
    """
    ...


@wraps(uop.U1)
def u1(qarg: Qubit, lam: float) -> None: ...


@wraps(uop.U2)
def u2(qarg: Qubit, phi: float, lam: float) -> None: ...


@wraps(uop.CZ)
def cz(ctrl: Qubit, qarg: Qubit) -> None:
    """
    Controlled-Z gate.

    Args:
        ctrl: The control qubit.
        qarg: The target qubit
    """
    ...


@wraps(uop.CY)
def cy(ctrl: Qubit, qarg: Qubit) -> None:
    """
    Controlled-Y gate.

    Args:
        ctrl: The control qubit.
        qarg: The target qubit
    """

    ...


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
