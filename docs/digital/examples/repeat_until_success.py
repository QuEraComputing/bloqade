import math

from bloqade import qasm2
from kirin.dialects import ilist

@qasm2.extended
def prep_magic_state(theta: float)->qasm2.types.QReg:
    qreg = qasm2.qreg(1)
    qasm2.rz(qreg[0], theta)
    return qreg

@qasm2.extended
def star_gadget_recursive(reg, theta)->qasm2.types.QReg:
    """
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337 Fig. 7
    """
    ancilla = prep_magic_state(theta)
    qasm2.cx(ancilla[0], reg[0])
    creg = qasm2.creg(1)
    qasm2.measure(reg[0], creg[0])
    if creg[0] == 1:
        # qasm2.deallocate(reg)
        return ancilla
    else:
        qasm2.x(ancilla[0])
        return star_gadget_recursive(ancilla, 2*theta)

@qasm2.extended
def star_gadget_loop(reg:qasm2.types.QReg, theta:float,depth:int=100)->qasm2.types.QReg:
    """
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337 Fig. 7
    """
    creg = qasm2.creg(1)
    
    for ctr in range(depth):
        ancilla = prep_magic_state(theta*(2**ctr))
        qasm2.cx(ancilla[0], reg[0])
        qasm2.measure(reg[0], creg[0])
        if creg[0] == 1:
            return ancilla
        else:
            qasm2.x(ancilla[0])
            reg = ancilla
    raise RuntimeError("Did not converge")
    


star_gadget_recursive.print()