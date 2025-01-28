from typing import TypeVar, ParamSpec
from dataclasses import dataclass

from kirin import ir
from pyqrack import QrackSimulator
from bloqade.analysis.address import AnyAddress, AddressAnalysis
from bloqade.runtime.qrack.base import Memory, PyQrackInterpreter

Params = ParamSpec("Params")
RetType = TypeVar("RetType")


@dataclass
class PyQrack:
    """PyQrack target runtime for Bloqade."""

    def run(
        self,
        mt: ir.Method[Params, RetType],
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> RetType:
        """Run the given kernel method on the PyQrack simulator."""
        address_analysis = AddressAnalysis(mt.dialects)
        results, ret = address_analysis.run_analysis(mt)
        if any(isinstance(a, AnyAddress) for a in results.values()):
            raise ValueError("All addresses must be resolved.")

        num_qubits = address_analysis.next_address
        memory = Memory(
            num_qubits,
            allocated=0,
            sim_reg=QrackSimulator(qubitCount=num_qubits, isTensorNetwork=False),
        )
        interpreter = PyQrackInterpreter(mt.dialects, memory=memory)
        return interpreter.run(mt, args, kwargs).expect()
