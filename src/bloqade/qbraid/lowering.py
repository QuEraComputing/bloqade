from typing import Dict, List, Tuple, Sequence
from dataclasses import field, dataclass

from kirin import ir, types, passes
from bloqade import noise, qasm2
from bloqade.qbraid import schema
from kirin.dialects import func, ilist
from bloqade.qasm2.dialects import parallel


@ir.dialect_group([func, qasm2.core, qasm2.uop, qasm2.expr, noise.native])
def qbraid_noise(
    self,
):
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)

    def run_pass(
        method: ir.Method,
        *,
        fold: bool = True,
    ):
        method.verify()

        if fold:
            fold_pass(method)

        typeinfer_pass(method)
        method.code.typecheck()

    return run_pass


@dataclass
class Lowering:
    qubit_list: List[ir.SSAValue] = field(init=False, default_factory=list)
    qubit_id_map: Dict[int, ir.SSAValue] = field(init=False, default_factory=dict)
    bit_id_map: Dict[int, ir.SSAValue] = field(init=False, default_factory=dict)
    block_list: List[ir.Statement] = field(init=False, default_factory=list)

    def lower(
        self,
        sym_name: str,
        noise_model: schema.NoiseModel,
        return_qreg: bool = False,
    ) -> ir.Method:
        """Lower the noise model to a method.

        Args:
            name (str): The name of the method to generate.
            return_qreg (bool): Use the quantum register as the return value.

        Returns:
            Method: The generated kirin method.

        """
        self.process_noise_model(noise_model, return_qreg)
        block = ir.Block(stmts=self.block_list)
        ret_type = qasm2.types.QRegType if return_qreg else qasm2.types.CRegType
        block.args.append_from(types.MethodType[[], ret_type], name=f"{sym_name}_self")
        region = ir.Region(block)
        func_stmt = func.Function(
            sym_name=sym_name,
            signature=func.Signature(inputs=(), output=qasm2.types.QRegType),
            body=region,
        )

        mt = ir.Method(
            mod=None,
            py_func=None,
            sym_name=sym_name,
            dialects=qbraid_noise,
            code=func_stmt,
            arg_names=[],
        )
        qbraid_noise.run_pass(mt)  # type: ignore

        return mt

    def process_noise_model(self, noise_model: schema.NoiseModel, return_qreg: bool):
        num_qubits = self.lower_number(noise_model.num_qubits)

        reg = qasm2.core.QRegNew(num_qubits)
        creg = qasm2.core.CRegNew(num_qubits)
        self.block_list.append(reg)
        self.block_list.append(creg)

        for idx_value, qubit in enumerate(noise_model.all_qubits):
            idx = self.lower_number(idx_value)
            self.block_list.append(qubit_stmt := qasm2.core.QRegGet(reg.result, idx))
            self.block_list.append(bit_stmt := qasm2.core.CRegGet(creg.result, idx))

            self.qubit_id_map[qubit] = qubit_stmt.result
            self.bit_id_map[qubit] = bit_stmt.result
            self.qubit_list.append(qubit_stmt.result)

        for gate_event in noise_model.gate_events:
            self.process_gate_event(gate_event)

        if return_qreg:
            self.block_list.append(func.Return(reg.result))
        else:
            self.block_list.append(func.Return(creg.result))

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
                (
                    schema.GlobalW,
                    schema.LocalW,
                    schema.GlobalRz,
                    schema.LocalRz,
                    schema.Measurement,
                ),
            ), f"Only W and Rz gates are supported, found {type(operation)}.__name__"

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
            elif isinstance(operation, schema.Measurement):
                self.lower_measurement(operation)

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

        single_layers = {}
        paired_layers = {}
        unpaired_layers = {}

        for participant in participants:
            if len(participant) == 1:
                p_single = single_error_dict[participant[0]]
                single_layers.setdefault(p_single, []).append(participant[0])
            elif len(participant) == 2:
                p_ctrl = entangled_error_dict[participant[0]]
                up_ctrl = single_error_dict[participant[0]]
                p_qarg = entangled_error_dict[participant[1]]
                up_qarg = single_error_dict[participant[1]]
                paired_layers.setdefault((p_ctrl, p_qarg), []).append(participant)
                unpaired_layers.setdefault((up_ctrl, up_qarg), []).append(participant)

        for (px, py, pz), qubits in single_layers.items():
            self.block_list.append(
                qargs := ilist.New(values=tuple(self.qubit_id_map[q] for q in qubits))
            )
            self.block_list.append(
                noise.native.PauliChannel(px=px, py=py, pz=pz, qargs=qargs.result)
            )

        for (p_ctrl, p_qarg), qubits in paired_layers.items():
            ctrls, qargs = list(zip(*qubits))
            self.block_list.append(
                ctrls := ilist.New(values=tuple(self.qubit_id_map[q] for q in ctrls))
            )
            self.block_list.append(
                qargs := ilist.New(values=tuple(self.qubit_id_map[q] for q in qargs))
            )
            self.block_list.append(
                noise.native.CZPauliChannel(
                    paired=True,
                    px_ctrl=p_ctrl[0],
                    py_ctrl=p_ctrl[1],
                    pz_ctrl=p_ctrl[2],
                    px_qarg=p_qarg[0],
                    py_qarg=p_qarg[1],
                    pz_qarg=p_qarg[2],
                    ctrls=ctrls.result,
                    qargs=qargs.result,
                )
            )

        for (p_ctrl, p_qarg), qubits in unpaired_layers.items():
            ctrls, qargs = list(zip(*qubits))
            self.block_list.append(
                ctrls := ilist.New(values=tuple(self.qubit_id_map[q] for q in ctrls))
            )
            self.block_list.append(
                qargs := ilist.New(values=tuple(self.qubit_id_map[q] for q in qargs))
            )
            self.block_list.append(
                noise.native.CZPauliChannel(
                    paired=False,
                    px_ctrl=p_ctrl[0],
                    py_ctrl=p_ctrl[1],
                    pz_ctrl=p_ctrl[2],
                    px_qarg=p_qarg[0],
                    py_qarg=p_qarg[1],
                    pz_qarg=p_qarg[2],
                    ctrls=ctrls.result,
                    qargs=qargs.result,
                )
            )

    def lower_cz_gates(self, node: schema.CZ):
        ctrls, qargs = list(zip(*(p for p in node.participants if len(p) == 2)))
        self.block_list.append(
            ctrls := ilist.New(values=tuple(self.qubit_id_map[q] for q in ctrls))
        )
        self.block_list.append(
            qargs := ilist.New(values=tuple(self.qubit_id_map[q] for q in qargs))
        )
        self.block_list.append(parallel.CZ(ctrls=ctrls.result, qargs=qargs.result))

    def lower_w_gates(self, participants: Sequence[int], theta: float, phi: float):
        self.block_list.append(
            qargs := ilist.New(values=tuple(self.qubit_id_map[q] for q in participants))
        )
        self.block_list.append(
            parallel.UGate(
                theta=self.lower_full_turns(theta),
                phi=self.lower_full_turns(phi + 0.5),
                lam=self.lower_full_turns(-(0.5 + phi)),
                qargs=qargs.result,
            )
        )

    def lower_rz_gates(self, participants: Tuple[int, ...], phi: float):
        self.block_list.append(
            qargs := ilist.New(values=tuple(self.qubit_id_map[q] for q in participants))
        )
        self.block_list.append(
            parallel.RZ(theta=self.lower_full_turns(phi), qargs=qargs.result)
        )

    def lower_pauli_errors(self, operator_error: schema.PauliErrorModel):
        assert isinstance(
            operator_error, schema.PauliErrorModel
        ), "Only PauliErrorModel is supported"

        layers = {}

        for qubit_num, pauli_errors in operator_error.errors:
            layers.setdefault(pauli_errors, []).append(qubit_num)

        for (px, py, pz), qubits in layers.items():
            self.block_list.append(
                qargs := ilist.New(values=tuple(self.qubit_id_map[q] for q in qubits))
            )
            self.block_list.append(
                noise.native.PauliChannel(px=px, py=py, pz=pz, qargs=qargs.result)
            )

    def lower_measurement(self, operation: schema.Measurement):
        for participant in operation.participants:
            qubit = self.qubit_id_map[participant]
            bit = self.bit_id_map[participant]
            self.block_list.append(qasm2.core.Measure(qarg=qubit, carg=bit))

    def lower_atom_loss(self, survival_probs: Tuple[float, ...]):
        layers = {}

        for qubit_num, survival_prob in zip(self.qubit_list, survival_probs):
            layers.setdefault(survival_prob, []).append(qubit_num)

        for survival_prob, qubits in layers.items():
            self.block_list.append(qargs := ilist.New(values=qubits))
            self.block_list.append(
                noise.native.AtomLossChannel(prob=survival_prob, qargs=qargs.result)
            )

    def lower_number(self, value: float | int) -> ir.SSAValue:
        if isinstance(value, int):
            stmt = qasm2.expr.ConstInt(value=value)
        else:
            stmt = qasm2.expr.ConstFloat(value=value)

        self.block_list.append(stmt)
        return stmt.result

    def lower_full_turns(self, value: float) -> ir.SSAValue:
        const_pi = qasm2.expr.ConstPI()
        self.block_list.append(const_pi)
        turns = self.lower_number(2 * value)
        mul = qasm2.expr.Mul(const_pi.result, turns)
        self.block_list.append(mul)
        return mul.result
