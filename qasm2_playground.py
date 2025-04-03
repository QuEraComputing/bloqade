# interesting behavior where the address analysis seems to fail if the function returns multiple values,
# not sure if intended but is it ever the case we "want" to have multiple values returned???

from bloqade import qasm2
from bloqade.analysis import address


@qasm2.extended
def complex_address_program():
    qreg = qasm2.qreg(10)  # 10 qubits, this gives an AddressReg
    # These become individual address qubits
    q1 = qreg[0]
    # q2 = qreg[1]

    # y = [q1, q2] + [q1, q2]

    # handle Alias
    q_alias = q1

    # creg1 = qasm2.creg(1)
    # creg2 = qasm2.creg(1)
    # qasm2.measure(q1, creg1[0])
    # qasm2.measure(q2, creg2[0])

    # creg3 = qasm2.creg(1)
    # q3 = qreg[2]
    # qasm2.measure(q3, creg3[0])

    return q_alias


# Take a look at tests for qasm2 analysis, should explain _why_ the lattice is structured the way it is


complex_address_program.print()
frame, _ = address.AddressAnalysis(complex_address_program.dialects).run_analysis(
    complex_address_program
)

# print(frame)
for ssa_val, addr in frame.entries.items():
    print(f"SSA Value: {ssa_val}\nAddress Type: {addr}")

"""
@qasm2.extended
def gate_behavior():
    qreg = qasm2.qreg(10)
    q = qreg[0]

    # apply gates, I just want to see what the behavior is like

    qasm2.h(q)
    qasm2.x(q)
    qasm2.y(q)
    qasm2.t(q)

    return q


gate_behavior.print()
# In the output the uops have no return values
"""
