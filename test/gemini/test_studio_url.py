"""The Gemini Studio link exporter preserves the supported logical circuit."""

import pytest
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import logical
from bloqade.gemini.logical import studio

INF = float("inf")


@logical.kernel(aggressive_unroll=True)
def bell():
    qubits = squin.qalloc(2)
    squin.h(qubits[0])
    squin.cx(qubits[0], qubits[1])
    return logical.default_post_processing(qubits)


@logical.kernel(aggressive_unroll=True)
def broadcast():
    qubits = squin.qalloc(4)
    squin.broadcast.h(qubits)
    squin.broadcast.cz(qubits[0:2], qubits[2:4])
    return logical.default_post_processing(qubits)


@logical.kernel(aggressive_unroll=True)
def pinned():
    qubits = logical.qalloc_at(ilist.IList([3, None, 1, None]))
    squin.cz(qubits[0], qubits[2])
    return logical.default_post_processing(qubits)


@logical.kernel(aggressive_unroll=True)
def pinned_off_identity():
    qubits = logical.qalloc_at(ilist.IList([5, 1]))
    squin.cz(qubits[0], qubits[1])
    return logical.default_post_processing(qubits)


@logical.kernel(aggressive_unroll=True)
def prepared():
    qubits = squin.qalloc(1)
    squin.u3(0.25, 0.5, 0.75, qubits[0])
    squin.s_adj(qubits[0])
    return logical.default_post_processing(qubits)


@logical.kernel(aggressive_unroll=True)
def unsupported_t():
    qubits = squin.qalloc(1)
    squin.t(qubits[0])
    return logical.default_post_processing(qubits)


def test_bell_link() -> None:
    assert logical.to_studio_url(bell) == (
        "https://bloqade.quera.com/studio/gemini/#circuit="
        "{version:13,qubits:2,gates:[{gate:h,column:0,targets:[0]},"
        "{gate:cx,column:3,targets:[0,1]}]}"
    )


def test_broadcast_gates_share_columns() -> None:
    url = logical.to_studio_url(broadcast)
    assert url.count("gate:h,column:0") == 4
    assert "{gate:cz,column:3,targets:[0,2]}" in url
    assert "{gate:cz,column:3,targets:[1,3]}" in url


def test_pinned_slots_and_unpinned_conflict() -> None:
    url = logical.to_studio_url(pinned)
    assert "{gate:cz,column:0,targets:[0,2]}" in url
    assert "placement:[{qubit:0,slot:3},{qubit:1,slot:0}," in url
    assert "{qubit:2,slot:1},{qubit:3,slot:2}]" in url


def test_every_pinned_slot_is_written() -> None:
    url = logical.to_studio_url(pinned_off_identity)
    assert "placement:[{qubit:0,slot:5},{qubit:1,slot:1}]" in url


def test_initial_state_and_adjoint_gate() -> None:
    url = logical.to_studio_url(prepared)
    assert "{gate:s_adj,column:0,targets:[0]}" in url
    assert "prep:[{qubit:0,theta:0.25,phi:0.5,lam:0.75}]" in url


def test_unsupported_nonclifford_does_not_disappear() -> None:
    with pytest.raises(ValueError, match="cannot represent.*T"):
        logical.to_studio_url(unsupported_t)


def test_kernel_arguments_are_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def parameterized(n: int):
        qubits = squin.qalloc(n)
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="without arguments"):
        logical.to_studio_url(parameterized)


@pytest.mark.parametrize("count", [0, 11])
def test_unsupported_qubit_counts_are_rejected(count: int) -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def bad_count():
        qubits = squin.qalloc(count)
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="supports 1–10 logical qubits"):
        logical.to_studio_url(bad_count)


def test_missing_terminal_measurement_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def unmeasured():
        qubits = squin.qalloc(1)
        squin.x(qubits[0])
        return qubits

    with pytest.raises(ValueError, match="require terminal logical measurement"):
        logical.to_studio_url(unmeasured)


def test_invalid_home_slot_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def bad_home():
        qubits = logical.qalloc_at(ilist.IList([10]))
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="logical home slots 0–9"):
        logical.to_studio_url(bad_home)


def test_preparation_after_gate_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def late_preparation():
        qubits = squin.qalloc(1)
        squin.x(qubits[0])
        squin.u3(0.25, 0.0, 0.0, qubits[0])
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="preparation before gates"):
        logical.to_studio_url(late_preparation)


def test_repeated_preparation_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def repeated_preparation():
        qubits = squin.qalloc(1)
        squin.u3(0.25, 0.0, 0.0, qubits[0])
        squin.u3(0.5, 0.0, 0.0, qubits[0])
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="repeated preparation"):
        logical.to_studio_url(repeated_preparation)


def test_nonfinite_preparation_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def nonfinite_preparation():
        qubits = squin.qalloc(1)
        squin.u3(INF, 0.0, 0.0, qubits[0])
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="angles must be finite"):
        logical.to_studio_url(nonfinite_preparation)


def test_partial_terminal_measurement_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def partial_measurement():
        qubits = squin.qalloc(2)
        logical.terminal_measure(qubits[0:1])

    with pytest.raises(ValueError, match="measurement of all qubits"):
        logical.to_studio_url(partial_measurement)


def test_permuted_terminal_measurement_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def permuted_measurement():
        qubits = squin.qalloc(2)
        return logical.terminal_measure(ilist.IList([qubits[1], qubits[0]]))

    with pytest.raises(ValueError, match="measurement in allocation order"):
        logical.to_studio_url(permuted_measurement)


def test_column_limit_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(studio, "_MAX_COLUMNS", 2)

    with pytest.raises(ValueError, match="1000-column limit"):
        logical.to_studio_url(bell)


def test_repeated_single_qubit_broadcast_operand_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def repeated_h():
        qubits = squin.qalloc(1)
        squin.broadcast.h(ilist.IList([qubits[0], qubits[0]]))
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="broadcast gate repeats a qubit"):
        logical.to_studio_url(repeated_h)


def test_repeated_adjoint_broadcast_operand_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def repeated_s():
        qubits = squin.qalloc(1)
        squin.broadcast.s(ilist.IList([qubits[0], qubits[0]]))
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="broadcast gate repeats a qubit"):
        logical.to_studio_url(repeated_s)


def test_overlapping_two_qubit_broadcast_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def overlapping_cz():
        qubits = squin.qalloc(3)
        squin.broadcast.cz(qubits[0:2], qubits[1:3])
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="must be disjoint"):
        logical.to_studio_url(overlapping_cz)


def test_unequal_two_qubit_broadcast_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def unequal_cz():
        qubits = squin.qalloc(3)
        squin.broadcast.cz(qubits[0:2], qubits[2:3])
        return logical.default_post_processing(qubits)

    with pytest.raises(ValueError, match="unequal operand lengths"):
        logical.to_studio_url(unequal_cz)


def test_gate_after_measurement_is_rejected() -> None:
    @logical.kernel(aggressive_unroll=True, verify=False)
    def late_gate():
        qubits = squin.qalloc(1)
        logical.terminal_measure(qubits)
        squin.x(qubits[0])

    with pytest.raises(ValueError, match="gates after measurement"):
        logical.to_studio_url(late_gate)
