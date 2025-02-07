from typing import List, Tuple, Union, Generic, Literal, TypeVar

from pydantic import Field, BaseModel


class Operation(BaseModel, frozen=True, extra="forbid"):
    """Base class for operations."""

    op_type: str = Field(init=False)


class CZ(Operation):
    """A CZ gate operation.

    Fields:
        op_type (str): The type of operation (Literal["CZ"]).
        participants (Tuple[Union[Tuple[int], Tuple[int, int]], ...]): The qubit indices that are participating in the CZ gate.

    """

    op_type: Literal["CZ"] = Field(init=False, default="CZ")
    participants: Tuple[Union[Tuple[int], Tuple[int, int]], ...]


class GlobalRz(Operation):
    """GlobalRz operation.

    Fields:
        op_type (str): The type of operation (Literal["GlobalRz"]).
        phi (float): The angle of rotation.
    """

    op_type: Literal["GlobalRz"] = Field(init=False, default="GlobalRz")
    phi: float


class GlobalW(Operation):
    """GlobalW operation.

    Fields:
        op_type (str): The type of operation (Literal["GlobalW"]).
        theta (float): The angle of rotation.
        phi (float): The angle of rotation.
    """

    op_type: Literal["GlobalW"] = Field(init=False, default="GlobalW")
    theta: float
    phi: float


class LocalRz(Operation):
    """LocalRz operation.

    Fields:
        op_type (str): The type of operation (Literal["LocalRz"]).
        participants (Tuple[int, ...]): The qubit indices that are participating in the local Rz gate.
        phi (float): The angle of rotation.

    """

    op_type: Literal["LocalRz"] = Field(init=False, default="LocalRz")
    participants: Tuple[int, ...]
    phi: float


class LocalW(Operation):
    """LocalW operation.

    Fields:
        op_type (str): The type of operation (Literal["LocalW"]).
        participants (Tuple[int, ...]): The qubit indices that are participating in the local W gate.
        theta (float): The angle of rotation.
        phi (float): The angle of rotation.

    """

    op_type: Literal["LocalW"] = Field(init=False, default="LocalW")
    participants: Tuple[int, ...]
    theta: float
    phi: float


class Measurement(Operation):
    """Measurement operation.

    Fields:
        op_type (str): The type of operation (Literal["Measurement"]).
        measure_tag (str): The tag to use for the measurement.
        participants (Tuple[int, ...]): The qubit indices that are participating in the measurement.

    """

    op_type: Literal["Measurement"] = Field(init=False, default="Measurement")
    measure_tag: str = Field(default="m")
    participants: Tuple[int, ...]


OperationType = TypeVar(
    "OperationType", bound=Union[CZ, GlobalRz, GlobalW, LocalRz, LocalW, Measurement]
)


class ErrorModel(BaseModel, frozen=True, extra="forbid"):
    """Base class for error models."""

    error_model_type: str = Field(init=False)


class PauliErrorModel(ErrorModel):
    """Pauli error model.

    Fields:
        error_model_type (str): The type of error model (Literal["PauliNoise"]).
        errors (Tuple[Tuple[int, Tuple[float, float, float]], ...]): The qubit indices and the error rates for each qubit.

    """

    error_model_type: Literal["PauliNoise"] = Field(default="PauliNoise", init=False)
    errors: Tuple[Tuple[int, Tuple[float, float, float]], ...] = Field(
        default_factory=tuple
    )


ErrorModelType = TypeVar("ErrorModelType", bound=PauliErrorModel)


class ErrorOperation(BaseModel, Generic[ErrorModelType], frozen=True, extra="forbid"):
    """Base class for error operations."""

    error_type: str = Field(init=False)
    survival_prob: Tuple[float, ...]


class CZError(ErrorOperation[ErrorModelType]):
    """CZError operation.

    Fields:
        survival_prob (Tuple[float, ...]): The survival probabilities for each qubit.
        error_type (str): The type of error (Literal["CZError"]).
        storage_error (ErrorModelType): The error model for storage.
        entangled_error (ErrorModelType): The error model for entangled qubits.
        single_error (ErrorModelType): The error model for single qubits.

    """

    error_type: Literal["CZError"] = Field(default="CZError", init=False)
    storage_error: ErrorModelType
    entangled_error: ErrorModelType
    single_error: ErrorModelType


class SingleQubitError(ErrorOperation[ErrorModelType]):
    """SingleQubitError operation.

    Fields:
        survival_prob (Tuple[float, ...]): The survival probabilities for each qubit.
        error_type (str): The type of error (Literal["SingleQubitError"]).
        operator_error (ErrorModelType): The error model for the single qubit.

    """

    error_type: Literal["SingleQubitError"] = Field(
        default="SingleQubitError", init=False
    )
    operator_error: ErrorModelType


class GateEvent(BaseModel, Generic[ErrorModelType], frozen=True, extra="forbid"):
    """A gate event.

    Fields:
        error (Union[SingleQubitError[ErrorModelType], CZError[ErrorModelType]]): The error model for the gate event.
        operation (OperationType): The operation for the gate event.

    """

    error: Union[SingleQubitError[ErrorModelType], CZError[ErrorModelType]] = Field(
        union_mode="left_to_right", discriminator="error_type"
    )
    operation: OperationType = Field(
        union_mode="left_to_right", discriminator="op_type"
    )

    def __pydantic_post_init__(self):
        assert (isinstance(self.operation, CZ) and isinstance(self.error, CZError)) or (
            not isinstance(self.operation, CZ)
            and isinstance(self.error, SingleQubitError)
        ), "Operation and error must be of the same type"


class NoiseModel(BaseModel, Generic[ErrorModelType], extra="forbid"):
    """Noise model for a circuit.

    Fields:
        all_qubits (Tuple[int, ...]): The qubit indices for the noise model.
        gate_events (List[GateEvent[ErrorModelType]]): The gate events for the noise model.

    """

    all_qubits: Tuple[int, ...] = Field(default_factory=tuple)
    gate_events: List[GateEvent[ErrorModelType]] = Field(default_factory=list)

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits in the noise model."""
        return len(self.all_qubits)

    def __add__(self, other: "NoiseModel") -> "NoiseModel":
        if not isinstance(other, NoiseModel):
            raise ValueError(f"Cannot add {type(other)} to Circuit")

        if self.all_qubits != other.all_qubits:
            raise ValueError("Circuits must have the same number of qubits")

        return NoiseModel(
            all_qubits=self.all_qubits,
            gate_events=self.gate_events + other.gate_events,
        )

    def lower_noise_model(self, sym_name: str):
        """Lower the noise model to a method.

        Args:
            sym_name (str): The name of the method to generate.

        Returns:
            Method: The generated kirin method.

        """
        from bloqade.qbraid.lowering import Lowering

        return Lowering().lower(sym_name, self)

    def decompiled_circuit(self) -> str:
        """Clean the circuit of noise.

        Returns:
            str: The decompiled circuit from hardware execution.

        """
        from kirin import ir
        from bloqade.noise import native
        from bloqade.qasm2.emit import QASM2

        mt = self.lower_noise_model("method")

        native.RemoveNoisePass(mt.dialects)(mt)
        mt.dialects = ir.DialectGroup(
            mt.dialects.data.symmetric_difference([native.dialect])
        )

        return QASM2(qelib1=True).emit_str(mt)
