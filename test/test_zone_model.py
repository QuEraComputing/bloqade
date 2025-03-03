import math

from bloqade.noise import native


def test_zone_model_deconflict():
    noise_model = native.TwoRowZoneModel()

    result = noise_model.deconflict([0, 4, 2, 1], [3, 5, 6, 7])
    assert result == [((0, 1), (3, 7)), ((2,), (6,)), ((4,), (5,))]


def test_zone_model_assign_slots():
    noise_model = native.TwoRowZoneModel()

    result = noise_model.assign_gate_slots([0, 1, 5], [3, 7, 9])
    assert result == {1: (1, 7), 0: (0, 3), 2: (5, 9)}


def test_move_duration():
    noise_model = native.TwoRowZoneModel()

    slots = {1: (1, 7), 0: (0, 3), 2: (5, 9)}
    result = noise_model.calculate_move_duration(slots)
    assert math.isclose(result, 99.14523415238915)


def test_coeff():

    noise_model = native.TwoRowZoneModel()
    result = noise_model.parallel_cz_errors([0, 4, 2, 1], [3, 5, 6, 7], [])

    qubit_result = {}

    for p, qubits in result.items():
        for qubit in qubits:
            qubit_result[qubit] = p

    expected_p = (
        0.0012829622129098485,
        0.0012829622129098485,
        0.0012829622129098485,
        0.00038347256559846324,
    )

    expected_qubit_result = {
        0: expected_p,
        1: expected_p,
        2: expected_p,
        3: expected_p,
        4: expected_p,
        5: expected_p,
        6: expected_p,
        7: expected_p,
    }

    for q, p in qubit_result.items():
        assert math.isclose(p[0], expected_qubit_result[q][0])
        assert math.isclose(p[1], expected_qubit_result[q][1])
        assert math.isclose(p[2], expected_qubit_result[q][2])
        assert math.isclose(p[3], expected_qubit_result[q][3])


if __name__ == "__main__":
    test_zone_model_assign_slots()
    test_move_duration()
