from unittest.mock import Mock, MagicMock, patch

import numpy as np
from kirin import ir
from matplotlib import pyplot as plt
from bloqade.lanes.dialects import move
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.analysis.atom import AtomState, AtomStateData
from bloqade.lanes.visualize.arch import ArchVisualizer
from bloqade.lanes.bytecode.encoding import LocationAddress, SiteLaneAddress

from bloqade.gemini.visualize import artist


def make_word():
    word = Mock()
    word.site_indices = [0, 1]
    word.site_position.side_effect = lambda i: (i, i)
    word.all_positions.return_value = [(0, 0), (1, 1)]
    return word


def make_arch_spec():
    arch = Mock(spec=ArchSpec)
    arch.words = (make_word(),)
    arch.zones = {0: [0]}
    arch.get_path.return_value = ((0.0, 0.0), (1.0, 1.0))
    arch.get_position.return_value = (0.0, 0.0)
    return arch


def make_mock_visualizer():
    """Return a MagicMock standing in for ArchVisualizer."""
    viz = MagicMock(spec=ArchVisualizer)
    viz.path_bounds.return_value = (0.0, 1.0, 0.0, 1.0)
    viz.x_bounds = (0.0, 1.0)
    viz.y_bounds = (0.0, 1.0)
    viz.plot.return_value = None
    return viz


def make_atom_state():
    # Use a real AtomState with minimal valid data and correct types
    # Use SiteLaneAddress for LaneAddress
    lane_addr = SiteLaneAddress(zone_id=0, word_id=0, site_id=0, bus_id=0)
    loc_addr = LocationAddress(0, 0)
    data = AtomStateData.from_fields(
        prev_lanes={0: lane_addr},
        qubit_to_locations={0: loc_addr},
        locations_to_qubit={loc_addr: 0},
    )
    return AtomState(data)


def make_atom_state_with_stationary():
    lane_addr = SiteLaneAddress(zone_id=0, word_id=0, site_id=0, bus_id=0)
    moving_loc = LocationAddress(0, 0)
    stationary_loc = LocationAddress(0, 1)
    data = AtomStateData.from_fields(
        prev_lanes={0: lane_addr},
        qubit_to_locations={0: moving_loc, 1: stationary_loc},
        locations_to_qubit={moving_loc: 0, stationary_loc: 1},
    )
    return AtomState(data)


def test_state_artist_move_renderer_and_draw_methods():
    fig, ax = plt.subplots()
    arch = make_arch_spec()
    params = artist.PlotParameters(1.0)
    sa = artist.StateArtist(ax, arch, params, 0, 10, 0, 10)
    state = make_atom_state()
    mr = sa.move_renderer(state, speed=1.0)
    assert mr is not None
    sa.draw_atoms(state)
    sa.draw_moves(state)
    plt.close(fig)


def test_state_artist_show_methods():
    fig, ax = plt.subplots()
    arch = make_arch_spec()
    params = artist.PlotParameters(1.0)
    sa = artist.StateArtist(ax, arch, params, 0, 10, 0, 10)
    # Use mocks for move.* and ir.Statement
    stmt_localr = Mock(spec=move.LocalR)
    stmt_localr.location_addresses = [Mock(word_id=0, site_id=0)]
    stmt_localrz = Mock(spec=move.LocalRz)
    stmt_localrz.location_addresses = [Mock(word_id=0, site_id=0)]
    stmt_globalr = Mock(spec=move.GlobalR)
    stmt_globalrz = Mock(spec=move.GlobalRz)
    stmt_cz = Mock(spec=move.CZ)
    stmt_cz.zone_address = Mock(zone_id=0)
    stmt_ir = Mock(spec=ir.Statement)
    sa.show_local_r(stmt_localr)
    sa.show_local_rz(stmt_localrz)
    mock_viz = make_mock_visualizer()
    with patch("bloqade.gemini.visualize.artist.ArchVisualizer", return_value=mock_viz):
        sa.show_global_r(stmt_globalr)
        sa.show_global_rz(stmt_globalrz)
        sa.show_cz(stmt_cz)
        sa.show_slm(stmt_ir, atom_marker="o")
    plt.close(fig)


def test_get_state_artist_and_drawer():
    fig, ax = plt.subplots()
    arch = make_arch_spec()
    mock_viz = make_mock_visualizer()
    with patch("bloqade.gemini.visualize.artist.ArchVisualizer", return_value=mock_viz):
        sa = artist.get_state_artist(arch, ax, atom_marker="o")
    assert isinstance(sa, artist.StateArtist)
    plt.close(fig)


def test_draw_atoms_adds_atom_number_labels():
    fig, ax = plt.subplots()
    arch = make_arch_spec()
    params = artist.PlotParameters(1.0)
    sa = artist.StateArtist(ax, arch, params, 0, 10, 0, 10)
    state = make_atom_state_with_stationary()
    arch.get_position.side_effect = lambda loc: (float(loc.word_id), float(loc.site_id))

    sa.draw_atoms(state)

    labels = sorted(text.get_text() for text in ax.texts)
    assert labels == ["0", "1"]
    plt.close(fig)


def test_move_renderer_adds_and_updates_atom_number_labels():
    fig, ax = plt.subplots()
    arch = make_arch_spec()
    params = artist.PlotParameters(1.0)
    sa = artist.StateArtist(ax, arch, params, 0, 10, 0, 10)
    state = make_atom_state_with_stationary()

    moving_loc = LocationAddress(0, 0)
    stationary_loc = LocationAddress(0, 1)
    arch.get_position.side_effect = lambda loc: {
        moving_loc: (1.0, 1.0),
        stationary_loc: (2.0, 2.0),
    }[loc]
    arch.get_path.return_value = ((0.0, 0.0), (1.0, 1.0))

    mr = sa.move_renderer(state, speed=1.0)
    assert mr is not None

    moving_renderer = next(
        renderer
        for renderer in mr.renderers
        if isinstance(renderer, artist.MovingAtomsScatter)
    )
    stationary_renderer = next(
        renderer
        for renderer in mr.renderers
        if isinstance(renderer, artist.StationaryAtomsScatter)
    )

    assert [text.get_text() for text in moving_renderer.labels] == ["0"]
    assert [text.get_text() for text in stationary_renderer.labels] == ["1"]

    moving_label = moving_renderer.labels[0]
    assert np.allclose(moving_label.get_position(), (0.0, 0.0))
    mr.update(mr.total_time)
    assert np.allclose(moving_label.get_position(), (1.0, 1.0))
    plt.close(fig)
