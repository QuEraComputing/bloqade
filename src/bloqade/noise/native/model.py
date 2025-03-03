import abc
import math
from typing import Dict, List, Tuple
from dataclasses import field, dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class GateNoiseParams:
    """Parameters for gate noise."""

    local_px: float = field(default=1e-3, kw_only=True)
    """The error probability for a Pauli-X error during a local single qubit gate operation."""
    local_py: float = field(default=1e-3, kw_only=True)
    """The error probability for a Pauli-Y error during a local single qubit gate operation."""
    local_pz: float = field(default=1e-3, kw_only=True)
    """The error probability for a Pauli-Z error during a local single qubit gate operation."""
    local_loss_prob: float = field(default=1e-4, kw_only=True)
    """The error probability for a loss during a local single qubit gate operation."""

    global_px: float = field(default=1e-3, kw_only=True)
    """The error probability for a Pauli-X error during a global single qubit gate operation."""
    global_py: float = field(default=1e-3, kw_only=True)
    """The error probability for a Pauli-Y error during a global single qubit gate operation."""
    global_pz: float = field(default=1e-3, kw_only=True)
    """The error probability for a Pauli-Z error during a global single qubit gate operation."""
    global_loss_prob: float = field(default=1e-3, kw_only=True)
    """The error probability for a loss during a global single qubit gate operation."""

    cz_paired_gate_px: float = field(default=1e-3, kw_only=True)
    """The error probability for a Pauli-X error during CZ gate operation when two qubits are within blockade radius."""
    cz_paired_gate_py: float = field(default=1e-3, kw_only=True)
    """The error probability for a Pauli-Y error during CZ gate operation when two qubits are within blockade radius."""
    cz_paired_gate_pz: float = field(default=1e-3, kw_only=True)
    """The error probability for a Pauli-Z error during CZ gate operation when two qubits are within blockade radius."""
    cz_gate_loss_prob: float = field(default=1e-3, kw_only=True)
    """The error probability for a loss during CZ gate operation when two qubits are within blockade radius."""

    cz_unpaired_gate_px: float = field(default=1e-3, kw_only=True)
    """The error probability for Pauli-X error during CZ gate operation when another qubit is not within blockade radius."""
    cz_unpaired_gate_py: float = field(default=1e-3, kw_only=True)
    """The error probability for Pauli-Y error during CZ gate operation when another qubit is not within blockade radius."""
    cz_unpaired_gate_pz: float = field(default=1e-3, kw_only=True)
    """The error probability for Pauli-Z error during CZ gate operation when another qubit is not within blockade radius."""
    cz_unpaired_loss_prob: float = field(default=1e-3, kw_only=True)
    """The error probability for a loss during CZ gate operation when another qubit is not within blockade radius."""


@dataclass
class NoiseModelABC(abc.ABC):
    """Abstract base class for noise models.

    This class defines the interface for a noise model. The gate noise is calculated form the parameters
    provided in this dataclass which can be updated when inheriting from this class. The move error is
    calculated by implementing the parallel_cz_errors method which takes a set of ctrl and qarg qubits
    and returns a noise model for all the qubits. The noise model is a dictionary with the keys being the
    error rates for the qubits and the values being the list of qubits that the error rate applies to.

    Once implemented the class can be used with the NoisePass to analyze a circuit and apply the noise
    model to the circuit.

    NOTE: This model is not guaranteed to be supported long-term in bloqade. We will be
    moving towards a more general approach to noise modeling in the future.

    """

    move_px_rate: float = field(default=1e-6, kw_only=True)
    """The error rate (prob/microsecond) for a Pauli-X error during a move operation."""
    move_py_rate: float = field(default=1e-6, kw_only=True)
    """The error rate e (prob/microsecond) for a Pauli-Y error during a move operation."""
    move_pz_rate: float = field(default=1e-6, kw_only=True)
    """The error rate e (prob/microsecond) for a Pauli-Z error during a move operation."""
    move_loss_rate: float = field(default=1e-6, kw_only=True)
    """The error rate e (prob/microsecond) for a loss during a move operation."""

    pick_px: float = field(default=1e-3, kw_only=True)
    """The error rate (prob per pick operation)  for a Pauli-X error during a pick operation."""
    pick_py: float = field(default=1e-3, kw_only=True)
    """The error rate (prob per pick operation) for a Pauli-Y error during a pick operation."""
    pick_pz: float = field(default=1e-3, kw_only=True)
    """The error rate (prob per pick operation) for a Pauli-Z error during a pick operation."""
    pick_loss_prob: float = field(default=1e-4, kw_only=True)
    """The error rate for a loss during a pick operation."""

    move_speed: float = field(default=5e-1, kw_only=True)
    """Maximum speed of the qubits during a move operation."""
    storage_spacing: float = field(default=4.0, kw_only=True)
    """Spacing between the qubits in the storage zone."""

    @classmethod
    @abc.abstractmethod
    def parallel_cz_errors(
        cls, ctrls: List[int], qargs: List[int], rest: List[int]
    ) -> Dict[Tuple[float, float, float, float], List[int]]:
        """Takes a set of ctrls and qargs and returns a noise model for all qubits."""
        pass

    @staticmethod
    def poisson_pauli_prob(rate: float, duration: float) -> float:
        """Calculate the number of noise events and their probabilities for a given rate and duration."""
        assert duration >= 0, "Duration must be non-negative"
        assert rate >= 0, "Rate must be non-negative"
        return 0.5 * (1 - math.exp(-2 * rate * duration))

    @classmethod
    def join_binary_probs(cls, p1: float, *args: float) -> float:
        """Merge the probabilities of an event happening if the event can only happen once.

        For example, finding the effective probability of losing an atom from multiple sources, since
        a qubit can only happen once. This is done by using the formula:

        p = p1 * (1 - p2) + p2 * (1 - p1)

        applied recursively to all the probabilities in the list.

        Args:
            p1 (float): The probability of the event happening.
            arg (float): The probabilities of the event happening from other sources.

        Returns:
            float: The effective probability of the event happening.

        """
        if len(args) == 0:
            return p1
        else:
            p2 = cls.join_binary_probs(*args)
            return p1 * (1 - p2) + p2 * (1 - p1)


@dataclass
class TwoRowZoneModel(NoiseModelABC):
    """This model assumes that the qubits are arranged in a single storage row with a row corresponding to a gate zone below it.

    The CZ gate noise is calculated using the following heuristic: The idle error is calculated by the total duration require
    to do the move and entable the qubits. Not every pair can be entangled at the same time, so we first deconflict the qargs
    assuming by finding subsets in which both the ctrl and the qarg qubits are in ascending order. This breaks the pairs into
    groups that can be moved and entangled separately. We then take each group and assign each pair to a gate zone slot. The
    slots are allocated by starting from the middle of the atoms and moving outwards making sure to keep the ctrl qubits in
    ascending order. The time to move a group is calculated by finding the maximum travel distance of the qarg and ctrl qubits
    and dviding by the move speed. The total move time is the sum of all the group move times. The error rate for all the qubits
    is then calculated by using the poisson_pauli_prob function. An additional error for the pick operation is calculated by
    joining the binary probabilities of the pick operation and the move operation.

    """

    gate_zone_y_offset: float = 20.0
    gate_spacing: float = 20.0

    def deconflict(
        self, ctrls: List[int], qargs: List[int]
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Return a list of groups of ctrl and qarg qubits that can be moved and entangled separately."""
        # sort by ctrl qubit first to guarantee that they will be in ascending order
        sorted_pairs = sorted(zip(ctrls, qargs))

        groups = []
        # group by qarg only putting it in a group if the qarg is greater than the last qarg in the group
        # thus ensuring that the qargs are in ascending order
        while len(sorted_pairs) > 0:
            ctrl, qarg = sorted_pairs.pop(0)

            found = False
            for group in groups:
                if group[-1][1] < qarg:
                    group.append((ctrl, qarg))
                    found = True
                    break
            if not found:
                groups.append([(ctrl, qarg)])

        return [tuple(zip(*group)) for group in groups]

    def assign_gate_slots(
        self, ctrls: Sequence[int], qargs: Sequence[int]
    ) -> Dict[int, Tuple[int, int]]:
        """Allocate slots for the qubits to move to. start from middle of atoms and move outwards
        making sure to keep the ctrl qubits in ascending order.

        Note that we can do this because the move strategy is to move the ctrl qubits separately
        from the qarg qubits, thus we don't have to worry about qarg qubits crossing the ctrl qubits
        and vice versa. We pick the median of all the atoms because it distributes the qubits
        as evenly as possible over the gate zone.

        """
        assert len(ctrls) == len(qargs), "Number of ctrls and qargs must be equal"
        addr_pairs = sorted(zip(ctrls, qargs), key=lambda x: x[0])
        # sort by the distance between the ctrl and qarg qubits

        ctrls, qargs = list(zip(*addr_pairs))

        n_ctrls = len(ctrls)

        ctrl_median = (
            ctrls[n_ctrls // 2]
            if n_ctrls % 2 == 1
            else (ctrls[n_ctrls // 2 - 1] + ctrls[n_ctrls // 2]) / 2
        )

        all_addr = sorted(ctrls + qargs)
        spatial_median = self.storage_spacing * (all_addr[0] + all_addr[-1]) / 2

        addr_pairs.sort(key=lambda x: abs(x[0] - ctrl_median))

        slots = {}
        med_slot = round(spatial_median / self.gate_spacing)

        left_slot = med_slot
        right_slot = med_slot
        slots[med_slot] = addr_pairs.pop(0)
        while addr_pairs:
            ctrl, qarg = addr_pairs.pop(0)

            if ctrl < ctrl_median:
                slots[left_slot := left_slot - 1] = (ctrl, qarg)
            else:
                slots[right_slot := right_slot + 1] = (ctrl, qarg)

        return slots

    def calculate_move_duration(self, slots: Dict[int, Tuple[int, int]]) -> float:
        """Calculate the time it takes to move the qubits from the ctrl to the qarg qubits."""

        qarg_x_distance = float("-inf")
        ctrl_x_distance = float("-inf")

        for slot, (ctrl, qarg) in slots.items():
            qarg_x_distance = max(
                qarg_x_distance,
                abs(qarg * self.storage_spacing - slot * self.gate_spacing),
            )
            ctrl_x_distance = max(
                ctrl_x_distance,
                abs(ctrl * self.storage_spacing - slot * self.gate_spacing),
            )

        qarg_max_distance = math.sqrt(qarg_x_distance**2 + self.gate_zone_y_offset**2)
        ctrl_max_distance = math.sqrt(
            ctrl_x_distance**2 + (self.gate_zone_y_offset - 3) ** 2
        )

        return (qarg_max_distance + ctrl_max_distance) / self.move_speed

    def parallel_cz_errors(
        self, ctrls: List[int], qargs: List[int], rest: List[int]
    ) -> Dict[Tuple[float, float, float, float], List[int]]:
        """Apply parallel gates by moving ctrl qubits to qarg qubits."""
        groups = self.deconflict(ctrls, qargs)
        slots = [self.assign_gate_slots(*group) for group in groups]

        move_duration = sum(map(self.calculate_move_duration, slots))

        px_time = self.poisson_pauli_prob(self.move_px_rate, move_duration)
        py_time = self.poisson_pauli_prob(self.move_py_rate, move_duration)
        px_time = self.poisson_pauli_prob(self.move_pz_rate, move_duration)
        p_loss_time = self.poisson_pauli_prob(self.move_loss_rate, move_duration)

        errors = {(px_time, py_time, px_time, p_loss_time): rest}

        px_moved = self.join_binary_probs(self.pick_px, px_time)
        py_moved = self.join_binary_probs(self.pick_py, py_time)
        pz_moved = self.join_binary_probs(self.pick_pz, px_time)
        p_loss_moved = self.join_binary_probs(self.pick_loss_prob, p_loss_time)

        errors[(px_moved, py_moved, pz_moved, p_loss_moved)] = sorted(ctrls + qargs)

        return errors
