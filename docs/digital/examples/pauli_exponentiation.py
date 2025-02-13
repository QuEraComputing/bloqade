import math
from typing import Any

from bloqade import qasm2
from kirin.dialects import ilist


@qasm2.extended
def zzzz_gadget(targets: ilist.IList[qasm2.types.Qubit, Any], gamma: float):
    for i in range(len(targets) - 1):
        qasm2.cx(targets[i], targets[i + 1])

    qasm2.rz(targets[-1], gamma)

    for j in range(len(targets) - 1):
        qasm2.cx(targets[-j - 1], targets[-j - 2])


@qasm2.extended
def pauli_basis_change(
    targets: ilist.IList[qasm2.types.Qubit, Any], start: str, end: str
):
    assert len(targets) == len(start)
    assert len(targets) == len(end)

    # for qubit, start_pauli, end_pauli in zip(targets, start, end):
    for i in range(len(targets)):
        qubit = targets[i]
        start_pauli = start[i]
        end_pauli = end[i]

        target = start_pauli + end_pauli
        if target == "ZX":
            qasm2.ry(qubit, math.pi / 2)
        elif target == "ZY":
            qasm2.rx(qubit, -math.pi / 2)
        # elif target == "ZZ":
        #     pass
        # elif target == "XX":
        #     pass
        elif target == "XY":
            qasm2.rz(qubit, math.pi / 2)
        elif target == "XZ":
            qasm2.ry(qubit, -math.pi / 2)
        elif target == "YX":
            qasm2.rz(qubit, -math.pi / 2)
        # elif target == "YY":
        #     pass
        elif target == "YZ":
            qasm2.rx(qubit, math.pi / 2)


@qasm2.extended
def pauli_exponential(
    targets: ilist.IList[qasm2.types.Qubit, Any], pauli: str, gamma: float
):
    assert len(targets) == len(pauli)

    pauli_basis_change(targets=targets, start="Z" * len(targets), end=pauli)
    zzzz_gadget(targets=targets, gamma=gamma)
    pauli_basis_change(targets=targets, start=pauli, end="Z" * len(targets))


if __name__ == "__main__":

    @qasm2.extended
    def main():
        register = qasm2.qreg(4)
        pauli_exponential([register[0], register[1], register[2]], "ZXY", 0.5)

    target = qasm2.emit.QASM2()
    ast = target.emit(main)
    qasm2.parse.pprint(ast)
