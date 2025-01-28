from typing import Any, Dict, Tuple

from kirin import ir
from pyqrack import QrackSimulator
from bloqade.runtime.qrack import Memory, PyQrackInterpreter
from bloqade.analysis.address import AnyAddress, AddressAnalysis


def pyqrack_execute(mt: ir.Method, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    """Run the given program on the device."""
    register_pass = AddressAnalysis(mt.dialects)
    _ = register_pass.eval(mt=mt, args=args, kwargs=kwargs).expect()

    if any(isinstance(a, AnyAddress) for a in register_pass.results.values()):
        raise ValueError("All addresses must be resolved.")

    num_qubits = register_pass.next_address

    memory = Memory(
        num_qubits,
        allocated=0,
        sim_reg=QrackSimulator(qubitCount=num_qubits, isTensorNetwork=False),
    )
    interpreter = PyQrackInterpreter(mt.dialects, memory=memory)

    return interpreter.eval(mt=mt, args=args, kwargs=kwargs).expect()
