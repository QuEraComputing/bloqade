from math import pi
from typing import List

from kirin import ir
from bloqade import qasm2
from bloqade.noise import native
from bloqade.qbraid import schema, lowering
from kirin.dialects import func


def as_int(value: int):
    return qasm2.expr.ConstInt(value=value)


def as_float(value: float):
    return qasm2.expr.ConstFloat(value=value)


def test_lowering_cz():

    survival_prob_0_value = 0.9
    survival_prob_1_value = 0.9
    survival_prob_5_value = 1.0
    survival_prob_3_value = 0.5

    px_stor_5_value = 0.1
    py_stor_5_value = 0.2
    pz_stor_5_value = 0.3

    px_sing_0_value = 0.6
    py_sing_0_value = 0.1
    pz_sing_0_value = 0.3

    px_sing_1_value = 0.3
    py_sing_1_value = 0.3
    pz_sing_1_value = 0.2

    px_sing_3_value = 0.6
    py_sing_3_value = 0.1
    pz_sing_3_value = 0.2

    px_ent_0_value = 0.0
    py_ent_0_value = 0.7
    pz_ent_0_value = 0.2

    px_ent_1_value = 0.2
    py_ent_1_value = 0.2
    pz_ent_1_value = 0.3

    operation = schema.CZ(participants=((0, 1), (3,)))
    error = schema.CZError(
        survival_prob=(
            survival_prob_0_value,
            survival_prob_1_value,
            survival_prob_5_value,
            survival_prob_3_value,
        ),
        storage_error=schema.PauliErrorModel(
            errors=((5, (px_stor_5_value, py_stor_5_value, pz_stor_5_value)),),
        ),
        entangled_error=schema.PauliErrorModel(
            errors=(
                (0, (px_ent_0_value, py_ent_0_value, pz_ent_0_value)),
                (1, (px_ent_1_value, py_ent_1_value, pz_ent_1_value)),
            ),
        ),
        single_error=schema.PauliErrorModel(
            errors=(
                (0, (px_sing_0_value, py_sing_0_value, pz_sing_0_value)),
                (1, (px_sing_1_value, py_sing_1_value, pz_sing_1_value)),
                (3, (px_sing_3_value, py_sing_3_value, pz_sing_3_value)),
            ),
        ),
    )
    gate_event = schema.GateEvent(
        operation=operation,
        error=error,
    )

    noise_model = schema.NoiseModel(
        all_qubits=(0, 1, 5, 3),
        gate_events=[gate_event],
    )

    expected: List[ir.Statement] = [
        (n_qubits := as_int(4)),
        (reg := qasm2.core.QRegNew(n_qubits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (idx1 := as_int(1)),
        (q1 := qasm2.core.QRegGet(reg.result, idx=idx1.result)),
        (idx2 := as_int(2)),
        (q5 := qasm2.core.QRegGet(reg.result, idx=idx2.result)),
        (idx3 := as_int(3)),
        (q3 := qasm2.core.QRegGet(reg.result, idx=idx3.result)),
        (survival_prob_0 := as_float(survival_prob_0_value)),
        native.AtomLossChannel(survival_prob_0.result, q0.result),
        (survival_prob_1 := as_float(survival_prob_1_value)),
        native.AtomLossChannel(survival_prob_1.result, q1.result),
        (survival_prob_5 := as_float(survival_prob_5_value)),
        native.AtomLossChannel(survival_prob_5.result, q5.result),
        (survival_prob_3 := as_float(survival_prob_3_value)),
        native.AtomLossChannel(survival_prob_3.result, q3.result),
        (px_stor_5 := as_float(px_stor_5_value)),
        (py_stor_5 := as_float(py_stor_5_value)),
        (pz_stor_5 := as_float(pz_stor_5_value)),
        native.PauliChannel(
            px_stor_5.result, py_stor_5.result, pz_stor_5.result, q5.result
        ),
        (px_sing_0 := as_float(px_sing_0_value)),
        (py_sing_0 := as_float(py_sing_0_value)),
        (pz_sing_0 := as_float(pz_sing_0_value)),
        (px_sing_1 := as_float(px_sing_1_value)),
        (py_sing_1 := as_float(py_sing_1_value)),
        (pz_sing_1 := as_float(pz_sing_1_value)),
        native.CZPauliChannel(
            px_sing_0.result,
            py_sing_0.result,
            pz_sing_0.result,
            px_sing_1.result,
            py_sing_1.result,
            pz_sing_1.result,
            q0.result,
            q1.result,
            paired=False,
        ),
        (px_ent_0 := as_float(px_ent_0_value)),
        (py_ent_0 := as_float(py_ent_0_value)),
        (pz_ent_0 := as_float(pz_ent_0_value)),
        (px_ent_1 := as_float(px_ent_1_value)),
        (py_ent_1 := as_float(py_ent_1_value)),
        (pz_ent_1 := as_float(pz_ent_1_value)),
        native.CZPauliChannel(
            px_ent_0.result,
            py_ent_0.result,
            pz_ent_0.result,
            px_ent_1.result,
            py_ent_1.result,
            pz_ent_1.result,
            q0.result,
            q1.result,
            paired=True,
        ),
        (px_sing_3 := as_float(px_sing_3_value)),
        (py_sing_3 := as_float(py_sing_3_value)),
        (pz_sing_3 := as_float(pz_sing_3_value)),
        native.PauliChannel(
            px_sing_3.result, py_sing_3.result, pz_sing_3.result, q3.result
        ),
        qasm2.uop.CZ(q0.result, q1.result),
        func.Return(reg.result),
    ]

    expected_func_stmt = func.Function(
        sym_name="test",
        signature=func.Signature(
            inputs=(),
            output=qasm2.types.QRegType,
        ),
        body=ir.Region(ir.Block(expected)),
    )

    func_stmt = lowering.Lowering().lower(sym_name="test", noise_model=noise_model).code
    try:
        assert expected_func_stmt.is_structurally_equal(func_stmt)
    except AssertionError as e:
        func_stmt.print()
        expected_func_stmt.print()
        raise e


def test_lowering_global_w():
    theta_val = 0.4
    phi_val = 0.7

    operation = schema.GlobalW(theta=theta_val, phi=phi_val)
    error = schema.SingleQubitError(
        survival_prob=(0.9,),
        operator_error=schema.PauliErrorModel(
            errors=((0, (0.1, 0.2, 0.3)),),
        ),
    )

    gate_event = schema.GateEvent(
        operation=operation,
        error=error,
    )

    noise_model = schema.NoiseModel(
        all_qubits=(0,),
        gate_events=[gate_event],
    )

    expected: List[ir.Statement] = [
        (n_qubits := as_int(1)),
        (reg := qasm2.core.QRegNew(n_qubits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (survival_prob_0 := as_float(0.9)),
        native.AtomLossChannel(survival_prob_0.result, q0.result),
        (px_sing_0 := as_float(0.1)),
        (py_sing_0 := as_float(0.2)),
        (pz_sing_0 := as_float(0.3)),
        native.PauliChannel(
            px_sing_0.result, py_sing_0.result, pz_sing_0.result, q0.result
        ),
        (theta := as_float(theta_val / 2.0)),
        (phi := as_float(phi_val - 0.5 * pi)),
        (lam := as_float(0.5 * pi - phi_val)),
        qasm2.uop.UGate(
            theta=theta.result, phi=phi.result, lam=lam.result, qarg=q0.result
        ),
        func.Return(reg.result),
    ]

    expected_func_stmt = func.Function(
        sym_name="test",
        signature=func.Signature(
            inputs=(),
            output=qasm2.types.QRegType,
        ),
        body=ir.Region(ir.Block(expected)),
    )

    func_stmt = lowering.Lowering().lower(sym_name="test", noise_model=noise_model).code
    try:
        assert expected_func_stmt.is_structurally_equal(func_stmt)
    except AssertionError as e:
        func_stmt.print()
        expected_func_stmt.print()
        raise e


def test_lowering_local_w():
    theta_val = 0.4
    phi_val = 0.7

    operation = schema.LocalW(theta=theta_val, phi=phi_val, participants=(1,))
    error = schema.SingleQubitError(
        survival_prob=(0.9, 0.4),
        operator_error=schema.PauliErrorModel(
            errors=(
                (0, (0.1, 0.2, 0.3)),
                (1, (0.5, 0.1, 0.2)),
            ),
        ),
    )

    gate_event = schema.GateEvent(
        operation=operation,
        error=error,
    )

    noise_model = schema.NoiseModel(
        all_qubits=(0, 1),
        gate_events=[gate_event],
    )

    expected: List[ir.Statement] = [
        (n_qubits := as_int(2)),
        (reg := qasm2.core.QRegNew(n_qubits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (idx1 := as_int(1)),
        (q1 := qasm2.core.QRegGet(reg.result, idx=idx1.result)),
        (survival_prob_0 := as_float(0.9)),
        native.AtomLossChannel(survival_prob_0.result, q0.result),
        (survival_prob_1 := as_float(0.4)),
        native.AtomLossChannel(survival_prob_1.result, q1.result),
        (px_sing_0 := as_float(0.1)),
        (py_sing_0 := as_float(0.2)),
        (pz_sing_0 := as_float(0.3)),
        native.PauliChannel(
            px_sing_0.result, py_sing_0.result, pz_sing_0.result, q0.result
        ),
        (px_sing_1 := as_float(0.5)),
        (py_sing_1 := as_float(0.1)),
        (pz_sing_1 := as_float(0.2)),
        native.PauliChannel(
            px_sing_1.result, py_sing_1.result, pz_sing_1.result, q1.result
        ),
        (theta := as_float(theta_val / 2.0)),
        (phi := as_float(phi_val - 0.5 * pi)),
        (lam := as_float(0.5 * pi - phi_val)),
        qasm2.uop.UGate(
            theta=theta.result, phi=phi.result, lam=lam.result, qarg=q1.result
        ),
        func.Return(reg.result),
    ]

    expected_func_stmt = func.Function(
        sym_name="test",
        signature=func.Signature(
            inputs=(),
            output=qasm2.types.QRegType,
        ),
        body=ir.Region(ir.Block(expected)),
    )

    func_stmt = lowering.Lowering().lower(sym_name="test", noise_model=noise_model).code
    try:
        assert expected_func_stmt.is_structurally_equal(func_stmt)
    except AssertionError as e:
        func_stmt.print()
        expected_func_stmt.print()
        raise e


def test_lowering_global_rz():
    phi_val = 0.7

    operation = schema.GlobalRz(phi=phi_val)
    error = schema.SingleQubitError(
        survival_prob=(0.9,),
        operator_error=schema.PauliErrorModel(
            errors=((0, (0.1, 0.2, 0.3)),),
        ),
    )

    gate_event = schema.GateEvent(
        operation=operation,
        error=error,
    )

    noise_model = schema.NoiseModel(
        all_qubits=(0,),
        gate_events=[gate_event],
    )

    expected: List[ir.Statement] = [
        (n_qubits := as_int(1)),
        (reg := qasm2.core.QRegNew(n_qubits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (survival_prob_0 := as_float(0.9)),
        native.AtomLossChannel(survival_prob_0.result, q0.result),
        (px_sing_0 := as_float(0.1)),
        (py_sing_0 := as_float(0.2)),
        (pz_sing_0 := as_float(0.3)),
        native.PauliChannel(
            px_sing_0.result, py_sing_0.result, pz_sing_0.result, q0.result
        ),
        (theta := as_float(phi_val)),
        qasm2.uop.RZ(theta=theta.result, qarg=q0.result),
        func.Return(reg.result),
    ]

    expected_func_stmt = func.Function(
        sym_name="test",
        signature=func.Signature(
            inputs=(),
            output=qasm2.types.QRegType,
        ),
        body=ir.Region(ir.Block(expected)),
    )

    func_stmt = lowering.Lowering().lower(sym_name="test", noise_model=noise_model).code
    try:
        assert expected_func_stmt.is_structurally_equal(func_stmt)
    except AssertionError as e:
        func_stmt.print()
        expected_func_stmt.print()
        raise e


def test_lowering_local_rz():
    phi_val = 0.7

    operation = schema.LocalRz(phi=phi_val, participants=(1,))
    error = schema.SingleQubitError(
        survival_prob=(0.9, 0.4),
        operator_error=schema.PauliErrorModel(
            errors=(
                (0, (0.1, 0.2, 0.3)),
                (1, (0.5, 0.1, 0.2)),
            ),
        ),
    )

    gate_event = schema.GateEvent(
        operation=operation,
        error=error,
    )

    noise_model = schema.NoiseModel(
        all_qubits=(0, 1),
        gate_events=[gate_event],
    )

    expected: List[ir.Statement] = [
        (n_qubits := as_int(2)),
        (reg := qasm2.core.QRegNew(n_qubits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (idx1 := as_int(1)),
        (q1 := qasm2.core.QRegGet(reg.result, idx=idx1.result)),
        (survival_prob_0 := as_float(0.9)),
        native.AtomLossChannel(survival_prob_0.result, q0.result),
        (survival_prob_1 := as_float(0.4)),
        native.AtomLossChannel(survival_prob_1.result, q1.result),
        (px_sing_0 := as_float(0.1)),
        (py_sing_0 := as_float(0.2)),
        (pz_sing_0 := as_float(0.3)),
        native.PauliChannel(
            px_sing_0.result, py_sing_0.result, pz_sing_0.result, q0.result
        ),
        (px_sing_1 := as_float(0.5)),
        (py_sing_1 := as_float(0.1)),
        (pz_sing_1 := as_float(0.2)),
        native.PauliChannel(
            px_sing_1.result, py_sing_1.result, pz_sing_1.result, q1.result
        ),
        (theta := as_float(phi_val)),
        qasm2.uop.RZ(theta=theta.result, qarg=q1.result),
        func.Return(reg.result),
    ]

    expected_func_stmt = func.Function(
        sym_name="test",
        signature=func.Signature(
            inputs=(),
            output=qasm2.types.QRegType,
        ),
        body=ir.Region(ir.Block(expected)),
    )

    func_stmt = lowering.Lowering().lower(sym_name="test", noise_model=noise_model).code
    try:
        assert expected_func_stmt.is_structurally_equal(func_stmt)
    except AssertionError as e:
        func_stmt.print()
        expected_func_stmt.print()
        raise e
