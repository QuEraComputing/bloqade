import json

from bloqade.qbraid import schema


def get_noise_model():
    single_qubit_error = schema.SingleQubitError(
        survival_prob=(survival_prob := tuple(1 for _ in range(10))),
        operator_error=(pauli_error_model := schema.PauliErrorModel()),
    )
    cz_error = schema.CZError(
        survival_prob=survival_prob,
        entangled_error=pauli_error_model,
        single_error=pauli_error_model,
        storage_error=pauli_error_model,
    )
    return schema.NoiseModel(
        all_qubits=(all_qubits := tuple(range(10))),
        gate_events=[
            schema.GateEvent(
                operation=schema.LocalW(
                    participants=(0,),
                    theta=0.25,
                    phi=0,
                ),
                error=single_qubit_error,
            ),
            schema.GateEvent(
                operation=schema.CZ(participants=((0, 1),)),
                error=cz_error,
            ),
            schema.GateEvent(
                operation=schema.CZ(participants=((0, 2),)),
                error=cz_error,
            ),
            schema.GateEvent(
                operation=schema.Measurement(
                    participants=all_qubits,
                ),
                error=single_qubit_error,
            ),
        ],
    )


def test_serialization():
    circuit = get_noise_model()

    circuit_json = circuit.model_dump_json()
    deserialized_circuit = schema.NoiseModel(**json.loads(circuit_json))

    assert circuit == deserialized_circuit
