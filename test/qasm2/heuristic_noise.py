from typing import List

import numpy as np
from bloqade import qasm2
from pyqrack import QrackSimulator
from bloqade.noise import native
from bloqade.pyqrack import PyQrack
from bloqade.pyqrack.reg import SimQReg
from bloqade.qasm2.passes import NoisePass


@qasm2.extended.add(native)
def mt():
    q = qasm2.qreg(4)

    ctrls = [q[0], q[2]]
    qargs = [q[3], q[1]]

    qasm2.parallel.cz(ctrls, qargs)

    return q


mt_copy = mt.similar()

NoisePass(dialects=mt.dialects)(mt)

qregs: List[SimQReg[QrackSimulator]] = PyQrack().multi_run(mt, 1000)

for qreg in qregs:
    print(qreg.sim_reg.pauli_expectation([1], [0]))

sv = np.array(qreg.sim_reg)
