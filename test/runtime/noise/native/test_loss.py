from unittest.mock import Mock

from bloqade import qasm2
from bloqade.noise import native
from bloqade.runtime.qrack import Memory, PyQrackInterpreter, reg


def test_atom_loss():
    simulation = qasm2.main.add(native)

    @simulation
    def test_atom_loss():
        q = qasm2.qreg(2)
        native.atom_loss_channel(0.5, q[0])
        native.atom_loss_channel(0.8, q[1])

        return q

    rng_state = Mock()
    rng_state.uniform.return_value = 0.7

    memory = Memory(total=2, allocated=0, sim_reg=Mock())

    result: reg.SimQRegister[Mock] = (
        PyQrackInterpreter(simulation, memory=memory, rng_state=rng_state)
        .run(test_atom_loss, ())
        .expect()
    )

    assert result.qubit_state[0] is reg.QubitState.Lost
    assert result.qubit_state[1] is reg.QubitState.Active
