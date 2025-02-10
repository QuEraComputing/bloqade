from bloqade import qasm2

@qasm2.extended
def prep_resource_state(theta: float)->qasm2.types.QReg:
    qreg = qasm2.qreg(1)
    qasm2.rz(qreg[0], theta)
    return qreg

@qasm2.extended
def star_gadget_recursive(target:qasm2.types.Qubit, theta)->qasm2.types.QReg:
    """
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337 Fig. 7
    """
    ancilla = prep_resource_state(theta)
    qasm2.cx(ancilla[0], target)
    creg = qasm2.creg(1)
    qasm2.measure(target, creg[0])
    # qasm2.deallocate(target)
    if creg[0] == 1:
        return ancilla
    else:
        qasm2.x(ancilla[0])
        return star_gadget_recursive(ancilla[0], 2*theta)

@qasm2.extended
def star_gadget_loop(target:qasm2.types.Qubit, theta:float,depth:int=100)->qasm2.types.QReg:
    """
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.010337 Fig. 7
    """
    creg = qasm2.creg(1)
    
    for ctr in range(depth):
        ancilla = prep_resource_state(theta*(2**ctr))
        qasm2.cx(ancilla[0], target)
        qasm2.measure(target, creg[0])
        # qasm2.deallocate(reg)
        if creg[0] == 1:
            return ancilla
        else:
            qasm2.x(ancilla[0])
            target = ancilla[0]
    #raise RuntimeError("Did not converge")
    # return ancilla

