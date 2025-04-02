# interesting behavior where the address analysis seems to fail if the function returns multiple values,
# not sure if intended but is it ever the case we "want" to have multiple values returned???

from bloqade import qasm2
from bloqade.analysis import address


@qasm2.extended
def test_qubit_init_program():
    qreg = qasm2.qreg(10)  # 10 qubits, this gives an AddressReg
    # These become indivdiual address qubits
    q1 = qreg[0]
    q2 = qreg[1]

    y = [q1, q2] + [q1, q2]

    creg1 = qasm2.creg(1)
    creg2 = qasm2.creg(1)
    qasm2.measure(q1, creg1[0])
    qasm2.measure(q2, creg2[0])

    creg3 = qasm2.creg(1)
    q3 = qreg[2]
    qasm2.measure(q3, creg3[0])

    return y


test_qubit_init_program.print()
frame, _ = address.AddressAnalysis(test_qubit_init_program.dialects).run_analysis(
    test_qubit_init_program
)

# print(frame)
for ssa_val, addr in frame.entries.items():
    print(f"SSA Value: {ssa_val}\nAddress Type: {addr}")
