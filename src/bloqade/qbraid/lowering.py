from math import pi
from typing import Dict, List, Tuple, Sequence
from dataclasses import field, dataclass

from kirin import ir
from bloqade import noise, qasm2
from bloqade.qbraid import schema
from kirin.dialects import func


@dataclass
class Lowering:
    qubit_list: List[ir.SSAValue] = field(init=False, default_factory=list)
    qubit_id_map: Dict[int, ir.SSAValue] = field(init=False, default_factory=dict)
    block_list: List[ir.Statement] = field(init=False, default_factory=list)

    def lower(self, sym_name: str, noise_model: schema.NoiseModel):
        """Lower the noise model to a method.

        Args:
            name (str): The name of the method to generate.

        Returns:
            Method: The generated kirin method.

        """
        self.process_noise_model(noise_model)
        region = ir.Region(blocks=[ir.Block(stmts=self.block_list)])
        func_stmt = func.Function(
            sym_name=sym_name,
            signature=func.Signature(inputs=(), output=qasm2.types.QRegType),
            body=region,
        )
        dialects = ir.DialectGroup(
            [func, qasm2.core, qasm2.uop, qasm2.expr, noise.native]
        )
        return ir.Method(
            mod=None,
            py_func=None,
            sym_name=sym_name,
            dialects=dialects,
            code=func_stmt,
            arg_names=[],
        )

    def process_noise_model(self, noise_model: schema.NoiseModel):
        num_qubits = self.lower_number(noise_model.num_qubits)

        reg = qasm2.core.QRegNew(num_qubits)
        self.block_list.append(reg)

        for idx_value, qubit in enumerate(noise_model.all_qubits):
            idx = self.lower_number(idx_value)
            qubit_stmt = qasm2.core.QRegGet(reg.result, idx)

            self.block_list.append(qubit_stmt)
            self.qubit_id_map[qubit] = qubit_stmt.result
            self.qubit_list.append(qubit_stmt.result)

        for gate_event in noise_model.gate_events:
            self.process_gate_event(gate_event)

        self.block_list.append(func.Return(reg.result))

    def process_gate_event(self, node: schema.GateEvent):
        self.lower_atom_loss(node.error.survival_prob)

        if isinstance(node.operation, schema.CZ):
            assert isinstance(node.error, schema.CZError), "Only CZError is supported"

            self.process_cz_pauli_error(node.operation.participants, node.error)
            self.lower_cz_gates(node.operation)
        else:

            error = node.error
            assert isinstance(
                error, schema.SingleQubitError
            ), "Only SingleQubitError is supported"
            self.lower_pauli_errors(error.operator_error)

            operation = node.operation
            assert isinstance(
                operation,
                (schema.GlobalW, schema.LocalW, schema.GlobalRz, schema.LocalRz),
            ), "Only W and Rz gates are supported"

            if isinstance(operation, schema.GlobalW):
                self.lower_w_gates(
                    tuple(self.qubit_id_map.keys()), operation.theta, operation.phi
                )
            elif isinstance(operation, schema.LocalW):
                self.lower_w_gates(
                    operation.participants, operation.theta, operation.phi
                )
            elif isinstance(operation, schema.GlobalRz):
                self.lower_rz_gates(tuple(self.qubit_id_map.keys()), operation.phi)
            elif isinstance(operation, schema.LocalRz):
                self.lower_rz_gates(operation.participants, operation.phi)

    def process_cz_pauli_error(
        self,
        participants: Tuple[Tuple[int] | Tuple[int, int], ...],
        node: schema.CZError[schema.PauliErrorModel],
    ):

        storage_error = node.storage_error
        single_error = node.single_error
        entangled_error = node.entangled_error

        assert isinstance(
            storage_error, schema.PauliErrorModel
        ), "Only PauliErrorModel is supported"
        assert isinstance(
            single_error, schema.PauliErrorModel
        ), "Only PauliErrorModel is supported"
        assert isinstance(
            entangled_error, schema.PauliErrorModel
        ), "Only PauliErrorModel is supported"

        self.lower_pauli_errors(storage_error)

        single_error_dict = dict(single_error.errors)
        entangled_error_dict = dict(entangled_error.errors)

        for participant in participants:
            match participant:
                case (qarg_id,):
                    qarg = self.qubit_id_map[qarg_id]
                    px, py, pz = single_error_dict[qarg_id]
                    px_val = self.lower_number(px)
                    py_val = self.lower_number(py)
                    pz_val = self.lower_number(pz)
                    self.block_list.append(
                        noise.native.PauliChannel(
                            px=px_val, py=py_val, pz=pz_val, qarg=qarg
                        )
                    )

                case (qarg1_id, qarg2_id):
                    qarg1 = self.qubit_id_map[qarg1_id]
                    qarg2 = self.qubit_id_map[qarg2_id]
                    # add single qubit errors if
                    # atom is lost during the execution of the CZ gate
                    self.lower_cz_pauli_channel(
                        False,
                        qarg1,
                        qarg2,
                        single_error_dict[qarg1_id],
                        single_error_dict[qarg2_id],
                    )

                    # add entangled errors if
                    # both qubits are active during the execution of the CZ gate
                    self.lower_cz_pauli_channel(
                        True,
                        qarg1,
                        qarg2,
                        entangled_error_dict[qarg1_id],
                        entangled_error_dict[qarg2_id],
                    )

    def lower_cz_gates(self, node: schema.CZ):
        for participant in node.participants:
            if len(participant) != 2:
                continue

            self.block_list.append(
                qasm2.uop.CZ(
                    ctrl=self.qubit_id_map[participant[0]],
                    qarg=self.qubit_id_map[participant[1]],
                )
            )

    def lower_w_gates(self, participants: Sequence[int], theta: float, phi: float):
        for participant in participants:
            self.block_list.append(
                qasm2.uop.UGate(
                    theta=self.lower_number(theta / 2),
                    phi=self.lower_number(phi - 0.5 * pi),
                    lam=self.lower_number(0.5 * pi - phi),
                    qarg=self.qubit_id_map[participant],
                )
            )

    def lower_rz_gates(self, participants: Tuple[int, ...], phi: float):
        for participant in participants:
            self.block_list.append(
                qasm2.uop.RZ(
                    theta=self.lower_number(phi),
                    qarg=self.qubit_id_map[participant],
                )
            )

    def lower_pauli_errors(self, operator_error: schema.PauliErrorModel):
        assert isinstance(
            operator_error, schema.PauliErrorModel
        ), "Only PauliErrorModel is supported"

        for qubit_num, (px, py, pz) in operator_error.errors:
            qubit = self.qubit_id_map[qubit_num]
            px_val = self.lower_number(px)
            py_val = self.lower_number(py)
            pz_val = self.lower_number(pz)
            self.block_list.append(
                noise.native.PauliChannel(px=px_val, py=py_val, pz=pz_val, qarg=qubit)
            )

    def lower_atom_loss(self, survival_probs: Tuple[float, ...]):
        for survival_prob, qubit in zip(survival_probs, self.qubit_list):
            prob = self.lower_number(survival_prob)
            self.block_list.append(noise.native.AtomLossChannel(prob=prob, qarg=qubit))

    def lower_cz_pauli_channel(
        self,
        paired: bool,
        qarg1: ir.SSAValue,
        qarg2: ir.SSAValue,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
    ):
        px1_val = self.lower_number(p1[0])
        py1_val = self.lower_number(p1[1])
        pz1_val = self.lower_number(p1[2])
        px2_val = self.lower_number(p2[0])
        py2_val = self.lower_number(p2[1])
        pz2_val = self.lower_number(p2[2])

        self.block_list.append(
            noise.native.CZPauliChannel(
                paired=paired,
                px_1=px1_val,
                py_1=py1_val,
                pz_1=pz1_val,
                px_2=px2_val,
                py_2=py2_val,
                pz_2=pz2_val,
                qarg1=qarg1,
                qarg2=qarg2,
            )
        )

    def lower_number(self, value: float | int) -> ir.SSAValue:
        if isinstance(value, int):
            stmt = qasm2.expr.ConstInt(value=value)
        else:
            stmt = qasm2.expr.ConstFloat(value=value)

        self.block_list.append(stmt)
        return stmt.result
