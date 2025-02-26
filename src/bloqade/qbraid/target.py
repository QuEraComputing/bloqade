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
        allow_parallel: bool = False,
        allow_global: bool = False,
        provider: "QbraidProvider",  # inject externally for easier mocking
        qelib1: bool = True,
    ) -> None:
        """Initialize the qBraid target.

        Args:
            allow_parallel (bool):
                Allow parallel gate in the resulting QASM2 AST. Defaults to `False`.
                In the case its False, and the input kernel uses parallel gates, they will get rewrite into uop gates.

            allow_global (bool):
                Allow global gate in the resulting QASM2 AST. Defaults to `False`.
                In the case its False, and the input kernel uses global gates, they will get rewrite into parallel gates.
                If both `allow_parallel` and `allow_global` are False, the input kernel will be rewritten to use uop gates.

            provider (QbraidProvider):
                Qbraid-provided object to allow submission of the kernel to the QuEra simulator.
            qelib1 (bool):
                Include the `include "qelib1.inc"` line in the resulting QASM2 AST that's
                submitted to qBraid. Defaults to `True`.
        """

        self.qelib1 = qelib1
        self.provider = provider
        self.allow_parallel = allow_parallel
        self.allow_global = allow_global

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
            Union[QbraidJob, list[QbraidJob]]:
                An object you can query for the status of your submission as well as
                obtain simulator results from.
        """

        # Convert method to QASM2 string
        qasm2_emitter = QASM2(
            allow_parallel=self.allow_parallel,
            allow_global=self.allow_global,
            qelib1=self.qelib1,
        )
        qasm2_prog = qasm2_emitter.emit_str(method)

        # Submit the QASM2 string to the qBraid simulator
        quera_qasm_simulator = self.provider.get_device("quera_qasm_simulator")

        return quera_qasm_simulator.run(qasm2_prog, shots=shots, tags=tags)
