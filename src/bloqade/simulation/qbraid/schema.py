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

    def get_sampler(self, circuit_backend: str, *args, **kwargs):
        """Return a sampler object that can run the circuit on a given backend.

        Args:
            circuit_backend (str): Which circuit backend to use. See description
                below for more details on how to create a new backend.

        Returns:
            sampler: AtomLossCircuitSampler

        ## How to implement a new backend:

        * `join`: Combine multiple circuits into a single circuit.
        * `remove_lost_qubits`: Remove qubits that are lost from a given circuit.
        * `apply_measurement_loss_flips`: Apply a reset + X-gate to simulate the effect of atom loss on a measurement.
        * `run`: Run a circuit and return the measurement results.

        ## Methods to register with the circuit dispatcher to emit specific circuits:

        * `schema.CZ`: construct a set of CZ gates given a list of participants.
        * `schema.GlobalRz`: construct a local z rotation gate on all qubits.
        * `schema.LocalRz`: construct a global z rotation gate on list of qubits.
        * `schema.globalW`: construct a global W gate on all qubits.
        * `schema.localW`: construct a local W gate on list of qubits.
        * `schema.Measurement`: construct a measurement gate.
        * `schema.PauliErrorModel`: construct a Pauli error channel with `(px, py, pz)`.


        ### Example: Cirq backend

        Define the base class and inherit from `constructor.CircuitConstructorABC` and
        define your own `CircuitDispatcher` class.

        ```python
        from flair_visual.simulation import constructor

        class CirqCircuitDispatcher(constructor.CircuitDispatcher[cirq.Circuit]):
            pass


        @constructor.CircuitBackendRegistry.register("cirq")
        class CirqCircuitConstructor(constructor.CircuitConstructorABC[cirq.Circuit]):

            CIRCUIT_DISPATCHER = CirqCircuitDispatcher

        ```

        Currently we're using a Puali noise channel as the error model. In cirq we use a
        custom mixed unitary gate to implement this kind of error. As such we add PAULI unitaries
        as a class variable.

        ```python
        @constructor.CircuitBackendRegistry.register("cirq")
        class CirqCircuitConstructor(constructor.CircuitConstructorABC[cirq.Circuit]):

            CIRCUIT_DISPATCHER = CirqCircuitDispatcher
            PAULI = tuple(map(cirq.unitary, [cirq.I, cirq.X, cirq.Y, cirq.Z]))

        ```

        We need to define the abstract methods that satisfy the `CircuitConstructorABC` interface and
        also need to register methods with the `CircuitDispatcher` class using the `@CircuitDispatcher.register`

        ```python
        @CirqCircuitDispatcher.register(schema.CZ)
        def emit_cz(self, operation: schema.CZ) -> cirq.Circuit:
            ...

        @CirqCircuitDispatcher.register(schema.GlobalRz)
        def emit_global_rz(self, operation: schema.GlobalRz) -> cirq.Circuit:
            ...

        ...
        ```
        """
        from flair_visual.simulation.sample import AtomLossCircuitSampler
        from flair_visual.simulation.constructor import CircuitBackendRegistry

        backend_type = CircuitBackendRegistry().get(circuit_backend, *args, **kwargs)
        return AtomLossCircuitSampler(
            circuit=self,
            circuit_generator=backend_type(self.all_qubits, *args, **kwargs),
        )
