from typing import List, TypeVar, ParamSpec
from dataclasses import dataclass

from kirin import ir
from pyqrack import QrackSimulator
from kirin.passes import Fold
from bloqade.analysis.address import AnyAddress, AddressAnalysis
from bloqade.runtime.qrack.base import Memory, PyQrackInterpreter

Params = ParamSpec("Params")
RetType = TypeVar("RetType")


@dataclass
class PyQrack:
    """PyQrack target runtime for Bloqade."""

    min_qubits: int = 0
    """Minimum number of qubits required for the PyQrack simulator.
    Useful when address analysis fails to determine the number of qubits.
    """
    memory: Memory | None = None
    """Memory for the PyQrack simulator."""

    def run(
        self,
        mt: ir.Method[Params, RetType],
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> RetType:
        """Run the given kernel method on the PyQrack simulator."""
        fold = Fold(mt.dialects)
        fold(mt)
        address_analysis = AddressAnalysis(mt.dialects)
        results, ret = address_analysis.run_analysis(mt)
        if any(isinstance(a, AnyAddress) for a in results.values()):
            raise ValueError("All addresses must be resolved.")

        num_qubits = max(address_analysis.qubit_count, self.min_qubits)
        self.memory = Memory(
            num_qubits,
            allocated=0,
            sim_reg=QrackSimulator(
                qubitCount=num_qubits, isTensorNetwork=False, isOpenCL=False
            ),
        )
        interpreter = PyQrackInterpreter(mt.dialects, memory=self.memory)
        return interpreter.run(mt, args, kwargs).expect()

    def multi_run(
        self,
        mt: ir.Method[Params, RetType],
        _shots: int,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> List[RetType]:
        """Run the given kernel method on the PyQrack `_shots` times, caching analysis results."""

        address_analysis = AddressAnalysis(mt.dialects)
        results, ret = address_analysis.run_analysis(mt)
        if any(isinstance(a, AnyAddress) for a in results.values()):
            raise ValueError("All addresses must be resolved.")

        memory = Memory(
            address_analysis.next_address,
            allocated=0,
            sim_reg=QrackSimulator(
                qubitCount=address_analysis.next_address,
                isTensorNetwork=False,
                isOpenCL=False,
            ),
        )

        batched_results = []
        for _ in range(_shots):
            memory.allocated = 0
            memory.sim_reg.reset_all()
            interpreter = PyQrackInterpreter(mt.dialects, memory=memory)
            batched_results.append(interpreter.run(mt, args, kwargs).expect())

        return batched_results
