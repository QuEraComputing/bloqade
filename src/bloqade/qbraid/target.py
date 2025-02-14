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
        main_target: ir.DialectGroup | None = None,
        gate_target: ir.DialectGroup | None = None,
        provider: "QbraidProvider",  # inject externally for easier mocking
        qelib1: bool = True,
        custom_gate: bool = True,
    ) -> None:
        from bloqade import qasm2

        self.main_target = main_target or qasm2.main
        self.gate_target = gate_target or qasm2.gate
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
        qasm2_emitter = QASM2(
            main_target=self.main_target,
            gate_target=self.gate_target,
            qelib1=self.qelib1,
            custom_gate=self.custom_gate,
        )
        qasm2_prog = qasm2_emitter.emit_str(method)

        # Submit the QASM2 string to the qBraid simulator
        quera_qasm_simulator = self.provider.get_device("quera_qasm_simulator")

        return quera_qasm_simulator.run(qasm2_prog, shots=shots, tags=tags)
