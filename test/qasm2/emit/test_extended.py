from bloqade import qasm2


@qasm2.gate(fold=False)
def my_gate(a: qasm2.Qubit, b: qasm2.Qubit):
    qasm2.cx(a, b)
    qasm2.u(a, theta=0.1, phi=0.2, lam=0.3)


@qasm2.extended
def main():
    qreg = qasm2.qreg(4)
    creg = qasm2.creg(2)
    qasm2.cx(qreg[0], qreg[1])
    qasm2.reset(qreg[0])
    qasm2.measure(qreg[0], creg[0])
    if creg[0] == 1:
        qasm2.reset(qreg[1])
    my_gate(qreg[0], qreg[1])


main.print()

target = qasm2.emit.QASM2()
ast = target.emit(main)
qasm2.parse.pprint(ast)
