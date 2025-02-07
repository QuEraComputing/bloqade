import math

from bloqade import qasm2
from kirin.dialects import cf, scf, lowering
from bloqade.runtime.qrack import PyQrack
@qasm2.kernel.discard(lowering.cf).add(scf)
def qft(n: int):
    qreg = qasm2.qreg(n)
    if n == 0:
        return

    qasm2.h(qreg[0])
    for i in range(1, n):
        qasm2.cu1(qreg[i], qreg[0], 2 * math.pi / 2**i)
    qft(n - 1)


qft.print()
device = PyQrack()
device.run(qft, 3)
