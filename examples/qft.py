import math

from bloqade import qasm2
from kirin.dialects import scf, lowering
from bloqade.runtime.qrack import PyQrack


@qasm2.kernel.discard(lowering.cf).add(scf)
def qft(qreg: qasm2.QReg, n: int):
    if n == 0:
        return qreg

    qasm2.h(qreg[0])
    for i in range(1, n):
        qasm2.cu1(qreg[i], qreg[0], 2 * math.pi / 2**i)
    qft(qreg, n - 1)
    return qreg


@qasm2.kernel.discard(lowering.cf).add(scf)
def main():
    return qft(qasm2.qreg(3), 3)


device = PyQrack()
qreg = device.run(main)
print(qreg)
