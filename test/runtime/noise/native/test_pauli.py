from unittest.mock import Mock, call

from bloqade import qasm2
from bloqade.noise import native
from bloqade.runtime.qrack import Memory, PyQrackInterpreter


def test_pauli_channel():
    simulation = qasm2.main.add(native)

    @simulation
    def test_atom_loss():
        q = qasm2.qreg(2)
        native.pauli_channel(0.1, 0.4, 0.3, q[0])
        native.pauli_channel(0.1, 0.4, 0.3, q[1])
        return q

    rng_state = Mock()
    rng_state.choice.side_effect = ["y", "i"]

    memory = Memory(total=2, allocated=0, sim_reg=Mock())

    PyQrackInterpreter(simulation, memory=memory, rng_state=rng_state).run(
        test_atom_loss, ()
    ).expect()

    memory.sim_reg.assert_has_calls([call.y(0)])
