from bloqade import qasm2
from bloqade.qasm2.emit import qBraid


def test_qBraid_emit():
    @qasm2.main
    def main():
        qreg = qasm2.qreg(4)
        qasm2.CX(qreg[0], qreg[1])
        qasm2.reset(qreg[0])
        qasm2.parallel.CZ(ctrls=(qreg[0], qreg[1]), qargs=(qreg[2], qreg[3]))

    class MockQBraidJob:
        pass

    class MockDevice:

        def run(self, *args, **kwargs):
            return MockQBraidJob()

    class MockQBraidProvider:

        def get_device(self, *args):
            return MockDevice()

    qBraid_emitter = qBraid(provider=MockQBraidProvider())
    qBraid_job = qBraid_emitter.emit(method=main)

    assert isinstance(qBraid_job, MockQBraidJob)
