import math

from bloqade import qasm2


@qasm2.kernel
def qft(n: int):
    qreg = qasm2.qreg(n)
    if n == 0:
        return

    qasm2.h(qreg[0])
    for i in range(1, n):
        qasm2.cu1(qreg[i], qreg[0], 2 * math.pi / 2**i)
    qft(n - 1)


qft.print()
