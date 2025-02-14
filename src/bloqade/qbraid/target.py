from typing import TYPE_CHECKING, Union, Optional

from kirin import ir

if TYPE_CHECKING:
    from qbraid import QbraidProvider
    from qbraid.runtime import QbraidJob

from bloqade.qasm2.emit import QASM2


class qBraid:
    """qBraid target for Bloqade kernels.

    qBraid target that accepts a Bloqade kernel and submits the kernel to the QuEra simulator hosted on qBraid. A `QbraidJob` is obtainable
    that then lets you query the status of the submitted program on the simulator as well
    as obtain results.

    """

    def __init__(
        self,
        *,
        main_target: ir.DialectGroup | None = None,
        gate_target: ir.DialectGroup | None = None,
        provider: "QbraidProvider",  # inject externally for easier mocking
        qelib1: bool = True,
        custom_gate: bool = True,
    ) -> None:
        """Initialize the qBraid target.

        Args:
            main_target (ir.DialectGroup | None):
                The dialects that were used in the definition of the kernel. This is used to
                generate the correct header for the resulting QASM2 AST. Argument defaults to `None`.
                Internally set to the `qasm2.main` group of dialects.
            gate_target (ir.DialectGroup | None):
                The dialects involved in defining any custom gates in the kernel. Argument defaults to `None`.
                Internally set to the `qasm2.gate` group of dialects.
            provider (QbraidProvider):
                Qbraid-provided object to allow submission of the kernel to the QuEra simulator.
            qelib1 (bool):
                Include the `include "qelib1.inc"` line in the resulting QASM2 AST that's
                submitted to qBraid. Defaults to `True`.
            custom_gate (bool):
                If a custom gate was defined in the kernel, attempt to inline it
                into the main program. Otherwise, just preserve the original
                reference to the custom gate definition. Defaults to `True`.
        """

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
        """Submit the Bloqade kernel to the QuEra simulator on qBraid.

        Args:
            method (ir.Method):
                The kernel to submit to qBraid.
            shots: (Optional[int]):
                Number of times to run the kernel. Defaults to None.
            tags: (Optional[dict[str,str]]):
                A dictionary of tags to associate with the Job.

        Returns:
            Union[QbraidJob, list[QbraidJOb]]:
                An object you can query for the status of your submission as well as
                obtain simulator results from.
        """

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
