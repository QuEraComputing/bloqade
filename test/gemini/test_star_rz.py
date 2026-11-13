import math

import pytest
import bloqade.squin as squin
from kirin.dialects import py, func, ilist
from kirin.ir.exception import ValidationError
from bloqade.lanes.dialects import move
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects.steane713 import extensions as extensions
from bloqade.lanes.rewrite.transversal import steane_star_theta
from bloqade.lanes.dialects.steane713.star import (
    VALID_STEANE_STAR_SUPPORTS,
    validate_steane_star_support,
)
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.dialects.steane713.extensions.stmts import StarRz

import bloqade.gemini as gemini


def test_star_rz_public_api_single_qubit_lowers_to_statement():
    assert gemini.logical.star_rz is gemini.logical.extensions.star_rz
    theta = math.pi / 16

    @gemini.logical.kernel
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.star_rz(theta, reg[0])
        gemini.logical.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRz)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)
    angle_owner = star_nodes[0].rotation_angle.owner
    assert isinstance(angle_owner, py.Constant)
    assert angle_owner.value.unwrap() == pytest.approx(theta / math.tau)


def test_star_rz_broadcast_lowers_to_statement():
    assert (
        gemini.logical.broadcast.star_rz is gemini.logical.extensions.broadcast.star_rz
    )
    theta = math.pi / 16

    @gemini.logical.kernel
    def kernel():
        reg = squin.qalloc(2)
        gemini.logical.broadcast.h(reg)
        gemini.logical.broadcast.star_rz(theta, reg)
        gemini.logical.terminal_measure(reg)

    star_nodes = [
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRz)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)
    angle_owner = star_nodes[0].rotation_angle.owner
    assert isinstance(angle_owner, py.Constant)
    assert angle_owner.value.unwrap() == pytest.approx(theta / math.tau)


@pytest.mark.parametrize("support", sorted(VALID_STEANE_STAR_SUPPORTS))
def test_star_rz_accepts_all_steane_weight_three_z_lines(support):
    theta_turns = (math.pi / 16) / math.tau

    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_rz(theta_turns, reg, qubit_indices=support)
        gemini.logical.terminal_measure(reg)

    star_node = next(
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRz)
    )
    assert star_node.qubit_indices == support


def test_star_rz_accepts_keyword_support():
    theta_turns = (math.pi / 16) / math.tau

    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        extensions.star_rz(theta_turns, reg, qubit_indices=(1, 2, 6))
        gemini.logical.terminal_measure(reg)

    star_node = next(
        stmt for stmt in kernel.callable_region.walk() if isinstance(stmt, StarRz)
    )
    assert star_node.qubit_indices == (1, 2, 6)


@pytest.mark.parametrize(
    "support",
    [(0, 1, 2), (4, 4, 6), (-1, 5, 6), (4, 5, 7), (4.0, 5, 6)],
)
def test_star_rz_rejects_invalid_steane_supports(support):
    with pytest.raises(ValueError, match="qubit_indices"):
        validate_steane_star_support(support)


def test_star_rz_statement_check_rejects_invalid_support():
    with pytest.raises(ValidationError, match="qubit_indices"):

        @gemini.logical.kernel(aggressive_unroll=True, no_raise=False)
        def kernel():
            reg = squin.qalloc(1)
            extensions.star_rz(
                (math.pi / 16) / math.tau,
                ilist.IList([reg[0]]),
                qubit_indices=(0, 1, 2),
            )
            gemini.logical.terminal_measure(reg)


def test_star_rz_pipeline_produces_physical_local_rz_after_transversal_rewrite():
    theta = math.pi / 16

    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.extensions.star_rz(theta, reg[0])
        gemini.logical.terminal_measure(reg)

    physical_move = LogicalPipeline(
        transversal_rewrite=True,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)

    assert not any(
        isinstance(stmt, move.StarRz) for stmt in physical_move.callable_region.walk()
    )
    local_rz_nodes = [
        stmt
        for stmt in physical_move.callable_region.walk()
        if isinstance(stmt, move.LocalRz)
    ]
    star_rz_nodes = [
        stmt
        for stmt in local_rz_nodes
        if tuple(stmt.location_addresses)
        == (
            LocationAddress(0, 4),
            LocationAddress(0, 5),
            LocationAddress(0, 6),
        )
    ]
    assert len(star_rz_nodes) == 1
    angle_owner = star_rz_nodes[0].rotation_angle.owner
    assert isinstance(angle_owner, func.Invoke)
    assert angle_owner.callee is steane_star_theta
    assert len(angle_owner.inputs) == 1
    input_owner = angle_owner.inputs[0].owner
    assert isinstance(input_owner, py.Constant)
    assert input_owner.value.unwrap() == pytest.approx(theta / math.tau)


def test_star_rz_pipeline_accepts_parameterized_theta():
    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel(theta: float):
        reg = squin.qalloc(1)
        gemini.logical.extensions.star_rz(theta, reg[0])
        gemini.logical.terminal_measure(reg)

    physical_move = LogicalPipeline(
        transversal_rewrite=True,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)

    assert not any(
        isinstance(stmt, move.StarRz) for stmt in physical_move.callable_region.walk()
    )
    assert any(
        isinstance(stmt, move.LocalRz)
        and isinstance(stmt.rotation_angle.owner, func.Invoke)
        and stmt.rotation_angle.owner.callee is steane_star_theta
        for stmt in physical_move.callable_region.walk()
    )


def test_star_rz_pipeline_preserves_support_after_prior_single_qubit_gate():
    theta = math.pi / 16
    theta_turns = theta / math.tau

    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        squin.h(reg[0])
        extensions.star_rz(theta_turns, ilist.IList([reg[0]]), qubit_indices=(0, 2, 4))
        gemini.logical.terminal_measure(reg)

    physical_move = LogicalPipeline(
        transversal_rewrite=True,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)

    local_rz_nodes = [
        stmt
        for stmt in physical_move.callable_region.walk()
        if isinstance(stmt, move.LocalRz)
        and tuple(stmt.location_addresses)
        == (
            LocationAddress(0, 0),
            LocationAddress(0, 2),
            LocationAddress(0, 4),
        )
    ]
    assert len(local_rz_nodes) == 1


def test_star_rz_without_transversal_rewrite_remains_logical_move_statement():
    @gemini.logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = squin.qalloc(1)
        gemini.logical.extensions.star_rz(math.pi / 16, reg[0])
        gemini.logical.terminal_measure(reg)

    logical_move = LogicalPipeline(
        transversal_rewrite=False,
        placement_strategy=LogicalPlacementStrategyNoHome(),
    ).emit(kernel, no_raise=False)

    star_nodes = [
        stmt
        for stmt in logical_move.callable_region.walk()
        if isinstance(stmt, move.StarRz)
    ]
    assert len(star_nodes) == 1
    assert star_nodes[0].qubit_indices == (4, 5, 6)
    assert star_nodes[0].location_addresses == (LocationAddress(0, 0, 0),)
