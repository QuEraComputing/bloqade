from typing import TYPE_CHECKING, Union, Optional

from kirin import ir

if TYPE_CHECKING:
    from qbraid import QbraidProvider
    from qbraid.runtime import QbraidJob

from bloqade.qasm2.emit import QASM2


class qBraid:

    def __init__(
        self,
        *,
        provider: "QbraidProvider",  # inject externally for easier mocking
        qelib1: bool = True,
        custom_gate: bool = True,
    ) -> None:
        self.qelib1 = qelib1
        self.custom_gate = custom_gate
        self.provider = provider

    def emit(
        self,
        method: ir.Method,
        shots: Optional[int] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Union["QbraidJob", list["QbraidJob"]]:

        # Convert method to QASM2 string
        qasm2_emitter = QASM2(qelib1=self.qelib1, custom_gate=self.custom_gate)
        qasm2_prog = qasm2_emitter.emit_str(method)

        # Submit the QASM2 string to the qBraid simulator
        quera_qasm_simulator = self.provider.get_device("quera_qasm_simulator")

        return quera_qasm_simulator.run(qasm2_prog, shots=shots, tags=tags)
