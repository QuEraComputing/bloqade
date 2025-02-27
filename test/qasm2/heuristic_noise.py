from bloqade import qasm2
from kirin.rewrite import walk
from bloqade.analysis import address
from bloqade.qasm2.rewrite.heuristic_noise import NoiseRewriteRule


@qasm2.extended
def mt():
    q = qasm2.qreg(4)

    qasm2.parallel.cz(ctrls=[q[0], q[2]], qargs=[q[1], q[3]])

    return q


address_analysis = address.AddressAnalysis(mt.dialects)
frame, _ = address_analysis.run_analysis(mt)
print(address_analysis.qubit_ssa_value)
noise = walk.Walk(NoiseRewriteRule(frame.entries, address_analysis.qubit_ssa_value))
noise.rewrite(mt.code)


mt.print()
