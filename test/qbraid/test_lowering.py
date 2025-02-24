import io
import difflib
from typing import List

from kirin import ir, types
from bloqade import qasm2
from rich.console import Console
from bloqade.noise import native
from bloqade.qbraid import schema, lowering
from kirin.dialects import func, ilist
from bloqade.qasm2.dialects import parallel


def as_int(value: int):
    return qasm2.expr.ConstInt(value=value)


def as_float(value: float):
    return qasm2.expr.ConstFloat(value=value)


def run_assert(noise_model: schema.NoiseModel, expected_stmts: List[ir.Statement]):

    block = ir.Block(stmts=expected_stmts)
    block.args.append_from(types.MethodType, name="test_self")
    region = ir.Region(block)
    expected_func_stmt = func.Function(
        sym_name="test",
        signature=func.Signature(inputs=(), output=qasm2.types.QRegType),
        body=region,
    )

    expected_mt = ir.Method(
        mod=None,
        py_func=None,
        dialects=lowering.qbraid_noise,
        sym_name="test",
        arg_names=[],
        code=expected_func_stmt,
    )

    assert lowering.qbraid_noise.run_pass
    lowering.qbraid_noise.run_pass(expected_mt)

    mt = noise_model.lower_noise_model("test")
    try:
        assert expected_mt.code.is_structurally_equal(mt.code)
    except AssertionError as e:

        gn_con = Console(record=True, file=io.StringIO())
        mt.print(console=gn_con)

        gn = gn_con.export_text()

        expected_con = Console(record=True, file=io.StringIO())
        expected_mt.print(console=expected_con)

        expected = expected_con.export_text()

        diff = difflib.Differ().compare(
            expected.splitlines(),
            gn.splitlines(),
        )

        print("\n".join(diff))

        raise e


def test_lowering_cz():

    survival_prob_0_value = 0.9
    survival_prob_1_value = 1.0

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
            survival_prob_0_value,
            survival_prob_1_value,
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
        (creg := qasm2.core.CRegNew(n_bits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx0.result)),
        (idx1 := as_int(1)),
        (q1 := qasm2.core.QRegGet(reg.result, idx=idx1.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx1.result)),
        (idx2 := as_int(2)),
        (q5 := qasm2.core.QRegGet(reg.result, idx=idx2.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx2.result)),
        (idx3 := as_int(3)),
        (q3 := qasm2.core.QRegGet(reg.result, idx=idx3.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx3.result)),
        (group_0 := ilist.New((q0.result, q5.result))),
        native.AtomLossChannel(group_0.result, prob=survival_prob_0_value),
        (group_1 := ilist.New((q1.result, q3.result))),
        native.AtomLossChannel(group_1.result, prob=survival_prob_1_value),
        (qargs := ilist.New((q5.result,))),
        native.PauliChannel(
            qargs.result, px=px_stor_5_value, py=py_stor_5_value, pz=pz_stor_5_value
        ),
        (qargs := ilist.New((q3.result,))),
        native.PauliChannel(
            qargs.result, px=px_sing_3_value, py=py_sing_3_value, pz=pz_sing_3_value
        ),
        (ctrls := ilist.New((q0.result,))),
        (qargs := ilist.New((q1.result,))),
        native.CZPauliChannel(
            ctrls.result,
            qargs.result,
            px_ctrl=px_ent_0_value,
            py_ctrl=py_ent_0_value,
            pz_ctrl=pz_ent_0_value,
            px_qarg=px_ent_1_value,
            py_qarg=py_ent_1_value,
            pz_qarg=pz_ent_1_value,
            paired=True,
        ),
        (ctrls := ilist.New((q0.result,))),
        (qargs := ilist.New((q1.result,))),
        native.CZPauliChannel(
            ctrls.result,
            qargs.result,
            px_ctrl=px_sing_0_value,
            py_ctrl=py_sing_0_value,
            pz_ctrl=pz_sing_0_value,
            px_qarg=px_sing_1_value,
            py_qarg=py_sing_1_value,
            pz_qarg=pz_sing_1_value,
            paired=False,
        ),
        (ctrls := ilist.New((q0.result,))),
        (qargs := ilist.New((q1.result,))),
        parallel.CZ(ctrls.result, qargs.result),
        func.Return(creg.result),
    ]

    run_assert(noise_model, expected)


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
        (creg := qasm2.core.CRegNew(n_bits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx0.result)),
        (qargs := ilist.New((q0.result,))),
        native.AtomLossChannel(qargs.result, prob=0.9),
        (qargs := ilist.New((q0.result,))),
        native.PauliChannel(qargs.result, px=0.1, py=0.2, pz=0.3),
        (qargs := ilist.New((q0.result,))),
        (pi_theta := qasm2.expr.ConstPI()),
        (theta_num := as_float(2 * theta_val)),
        (theta := qasm2.expr.Mul(pi_theta.result, theta_num.result)),
        (pi_phi := qasm2.expr.ConstPI()),
        (phi_num := as_float(2 * (phi_val + 0.5))),
        (phi := qasm2.expr.Mul(pi_phi.result, phi_num.result)),
        (pi_lam := qasm2.expr.ConstPI()),
        (lam_num := as_float(2 * -(0.5 + phi_val))),
        (lam := qasm2.expr.Mul(pi_lam.result, lam_num.result)),
        parallel.UGate(
            theta=theta.result, phi=phi.result, lam=lam.result, qargs=qargs.result
        ),
        func.Return(creg.result),
    ]

    run_assert(noise_model, expected)


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
        (creg := qasm2.core.CRegNew(n_bits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx0.result)),
        (idx1 := as_int(1)),
        (q1 := qasm2.core.QRegGet(reg.result, idx=idx1.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx1.result)),
        (qargs := ilist.New((q0.result,))),
        native.AtomLossChannel(qargs.result, prob=0.9),
        (qargs := ilist.New((q1.result,))),
        native.AtomLossChannel(qargs.result, prob=0.4),
        (qargs := ilist.New((q0.result,))),
        native.PauliChannel(qargs.result, px=0.1, py=0.2, pz=0.3),
        (qargs := ilist.New((q1.result,))),
        native.PauliChannel(qargs.result, px=0.5, py=0.1, pz=0.2),
        (qargs := ilist.New((q1.result,))),
        (pi_theta := qasm2.expr.ConstPI()),
        (theta_num := as_float(2 * theta_val)),
        (theta := qasm2.expr.Mul(pi_theta.result, theta_num.result)),
        (pi_phi := qasm2.expr.ConstPI()),
        (phi_num := as_float(2 * (phi_val + 0.5))),
        (phi := qasm2.expr.Mul(pi_phi.result, phi_num.result)),
        (pi_lam := qasm2.expr.ConstPI()),
        (lam_num := as_float(2 * -(0.5 + phi_val))),
        (lam := qasm2.expr.Mul(pi_lam.result, lam_num.result)),
        parallel.UGate(
            qargs=qargs.result, theta=theta.result, phi=phi.result, lam=lam.result
        ),
        func.Return(creg.result),
    ]

    run_assert(noise_model, expected)


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
        (creg := qasm2.core.CRegNew(n_bits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx0.result)),
        (qargs := ilist.New((q0.result,))),
        native.AtomLossChannel(qargs.result, prob=0.9),
        (qargs := ilist.New((q0.result,))),
        native.PauliChannel(qargs.result, px=0.1, py=0.2, pz=0.3),
        (qargs := ilist.New((q0.result,))),
        (theta_pi := qasm2.expr.ConstPI()),
        (theta_num := as_float(2 * phi_val)),
        (theta := qasm2.expr.Mul(theta_pi.result, theta_num.result)),
        parallel.RZ(theta=theta.result, qargs=qargs.result),
        func.Return(creg.result),
    ]

    run_assert(noise_model, expected)


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
        (creg := qasm2.core.CRegNew(n_bits=n_qubits.result)),
        (idx0 := as_int(0)),
        (q0 := qasm2.core.QRegGet(reg.result, idx=idx0.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx0.result)),
        (idx1 := as_int(1)),
        (q1 := qasm2.core.QRegGet(reg.result, idx=idx1.result)),
        (_ := qasm2.core.CRegGet(creg.result, idx=idx1.result)),
        (qargs := ilist.New((q0.result,))),
        native.AtomLossChannel(qargs.result, prob=0.9),
        (qargs := ilist.New((q1.result,))),
        native.AtomLossChannel(qargs.result, prob=0.4),
        (qargs := ilist.New((q0.result,))),
        native.PauliChannel(qargs.result, px=0.1, py=0.2, pz=0.3),
        (qargs := ilist.New((q1.result,))),
        native.PauliChannel(qargs.result, px=0.5, py=0.1, pz=0.2),
        (qargs := ilist.New((q1.result,))),
        (theta_pi := qasm2.expr.ConstPI()),
        (theta_num := as_float(2 * phi_val)),
        (theta := qasm2.expr.Mul(theta_pi.result, theta_num.result)),
        parallel.RZ(theta=theta.result, qargs=qargs.result),
        func.Return(creg.result),
    ]

    run_assert(noise_model, expected)
