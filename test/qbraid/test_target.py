from typing import Optional

from bloqade import qasm2
from bloqade.qbraid.target import qBraid


def test_qBraid_emit():

    @qasm2.main
    def main():
        qreg = qasm2.qreg(4)
        qasm2.cx(qreg[0], qreg[1])
        qasm2.reset(qreg[0])
        qasm2.parallel.cz(ctrls=(qreg[0], qreg[1]), qargs=(qreg[2], qreg[3]))

    class MockQBraidJob:

        def __init__(
            self, qasm: str, shots: Optional[int], tags: Optional[dict[str, str]]
        ):
            assert (
                qasm
                == 'KIRIN {cf,func,py.ilist,qasm2.core,qasm2.expr,qasm2.glob,qasm2.indexing,qasm2.inline,qasm2.noise,qasm2.parallel,qasm2.uop};\ninclude "qelib1.inc";\nqreg qreg[4];\nCX qreg[0], qreg[1];\nreset qreg[0];\nparallel.CZ {\n  qreg[0], qreg[2];\n  qreg[1], qreg[3];\n}\n'
            )
            assert shots is None
            assert tags is None

    class MockDevice:

        def run(self, qasm: str, shots: Optional[int], tags: Optional[dict[str, str]]):
            return MockQBraidJob(qasm, shots, tags)

    class MockQBraidProvider:

        def get_device(self, api_key: str):
            assert api_key == "quera_qasm_simulator"

            return MockDevice()

    mock_provider = MockQBraidProvider()
    mock_qBraid_emitter = qBraid(provider=mock_provider)
    mock_qBraid_job = mock_qBraid_emitter.emit(method=main)

    assert isinstance(mock_qBraid_job, MockQBraidJob)
