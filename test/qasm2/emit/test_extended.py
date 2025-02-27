from bloqade import qasm2


@qasm2.extended
def my_gate(a: qasm2.Qubit, b: qasm2.Qubit):
    qasm2.cx(a, b)
    qasm2.u(a, theta=0.1, phi=0.2, lam=0.3)


# calling custom gate from extended
@qasm2.extended
def body2(a: qasm2.Qubit, b: qasm2.Qubit):
    my_gate(a, b)


# calling custom gate from extended from extended
@qasm2.extended
def body3(a: qasm2.Qubit, b: qasm2.Qubit):
    body2(a, b)


# calling extended from extended.
@qasm2.extended
def body1(a: qasm2.Qubit, b: qasm2.Qubit, c: qasm2.Bit):
    qasm2.cx(a, b)
    qasm2.reset(a)
    qasm2.measure(a, c)
    if c == 1:
        qasm2.reset(b)


@qasm2.extended
def main():
    qreg = qasm2.qreg(4)
    creg = qasm2.creg(2)

    # these will be inlined

    body2(qreg[0], qreg[1])
    body3(qreg[0], qreg[1])
    body1(qreg[0], qreg[1], creg[0])

    # this will stay
    my_gate(qreg[0], qreg[1])


main.print()

target = qasm2.emit.QASM2()
ast = target.emit(main)
qasm2.parse.pprint(ast)
