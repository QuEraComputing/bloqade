import abc
import math
from typing import Dict, List, Tuple
from dataclasses import field, dataclass


@dataclass
class NoiseModelABC(abc.ABC):
    # rate = prob/time
    # numbers are just randomly chosen

    move_px_rate: float = field(default=1e-6, kw_only=True)
    move_py_rate: float = field(default=1e-6, kw_only=True)
    move_pz_rate: float = field(default=1e-6, kw_only=True)
    move_loss_rate: float = field(default=1e-6, kw_only=True)

    pick_loss_prob: float = field(default=1e-4, kw_only=True)
    pick_px: float = field(default=1e-3, kw_only=True)
    pick_py: float = field(default=1e-3, kw_only=True)
    pick_pz: float = field(default=1e-3, kw_only=True)

    local_px: float = field(default=1e-3, kw_only=True)
    local_py: float = field(default=1e-3, kw_only=True)
    local_pz: float = field(default=1e-3, kw_only=True)
    local_loss_prob: float = field(default=1e-4, kw_only=True)

    global_px: float = field(default=1e-3, kw_only=True)
    global_py: float = field(default=1e-3, kw_only=True)
    global_pz: float = field(default=1e-3, kw_only=True)
    global_loss_prob: float = field(default=1e-3, kw_only=True)

    cz_paired_gate_px: float = field(default=1e-3, kw_only=True)
    cz_paired_gate_py: float = field(default=1e-3, kw_only=True)
    cz_paired_gate_pz: float = field(default=1e-3, kw_only=True)
    cz_gate_loss_prob: float = field(default=1e-3, kw_only=True)

    cz_unpaired_gate_px: float = field(default=1e-3, kw_only=True)
    cz_unpaired_gate_py: float = field(default=1e-3, kw_only=True)
    cz_unpaired_gate_pz: float = field(default=1e-3, kw_only=True)
    cz_ungate_loss_prob: float = field(default=1e-3, kw_only=True)

    move_speed: float = field(default=1e-2, kw_only=True)
    lattice_spacing: float = field(default=4.0, kw_only=True)

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
    def join_binary_probs(cls, p1: float, *arg: float) -> float:
        if len(arg) == 0:
            return p1
        else:
            p2 = cls.join_binary_probs(*arg)
            return p1 * (1 - p2) + p2 * (1 - p1)


@dataclass
class TwoRowZoneModel(NoiseModelABC):

    gate_zone_y_offset: float = 20.0
    gate_spacing: float = 20.0

    def deconflict(
        self, ctrls: List[int], qargs: List[int]
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:

        # sort by ctrl qubit first to guarantee that they will be in ascending order
        sorted_pairs = sorted(zip(ctrls, qargs), key=lambda x: x[0])

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
        self, ctrls: List[int], qargs: List[int]
    ) -> Dict[int, Tuple[int, int]]:
        """Allocate slots for the qubits to move to."""

        position_pairs = list(zip(qargs, ctrls))
        # sort by the distance between the ctrl and qarg qubits
        sorted(position_pairs, key=lambda ele: abs(ele[0] - ele[1]))

        # greedy algorithm to find the best slot for each qubit pair
        slots = {}
        while position_pairs:
            ctrl, qarg = position_pairs.pop()

            mid = (ctrl + qarg) * self.lattice_spacing / 2
            slot = int(mid / self.gate_spacing)

            if slot not in slots:
                slots[slot] = (ctrl, qarg)
                continue

            # find the first slot that is not in slots that is close to the mid point
            for i in range(1, len(slots) + 1):
                if slot + i not in slots:
                    slots[slot + i] = (ctrl, qarg)
                    found = True
                    break
                elif slot - i not in slots:
                    slots[slot - i] = (ctrl, qarg)
                    found = True
                    break

            assert found, "No slot found"

        return slots

    def calculate_move_duration(self, ctrls: List[int], qargs: List[int]) -> float:
        """Calculate the time it takes to move the qubits from the ctrl to the qarg qubits."""

        slots = self.assign_gate_slots(ctrls, qargs)

        qarg_x_distance = float("-inf")
        ctrl_x_distance = float("-inf")

        for slot, (ctrl, qarg) in slots.items():
            qarg_x_distance = max(
                qarg_x_distance,
                abs(qarg * self.lattice_spacing - slot * self.gate_spacing),
            )
            ctrl_x_distance = max(
                ctrl_x_distance,
                abs(ctrl * self.lattice_spacing - slot * self.gate_spacing),
            )

        qarg_max_distance = math.sqrt(qarg_x_distance**2 + self.gate_zone_y_offset**2)
        ctrl_max_distance = math.sqrt(
            ctrl_x_distance**2 + (self.gate_zone_y_offset - 3) ** 2
        )

        return (qarg_max_distance + ctrl_max_distance) / self.move_speed

    def parallel_cz_errors(
        self, ctrls: List[int], qargs: List[int], rest: List[int]
    ) -> Dict[Tuple[float, float, float, float], List[int]]:
        """Apply parallel gates by moving ctrl qubits to qarg qubits.

        Deconfict the ctrl moves by finding subsets in which both the
        ctrl and the qarg qubits are in ascending order.

        """
        groups = self.deconflict(ctrls, qargs)
        move_duration = sum(map(self.calculate_move_duration, *zip(*groups)))

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
