"""``ArchVisualizer`` on a minimal two-word architecture (moved from lanes)."""

from unittest.mock import MagicMock, patch

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.word import Word
from bloqade.lanes.bytecode._native import (
    Grid as RustGrid,
    Mode as RustMode,
    Zone as RustZone,
    SiteBus,
    WordBus,
    LocationAddress as RustLocAddr,
)

from bloqade.gemini.visualize.arch import ArchVisualizer

word = Word(sites=((0, 0), (1, 0)))

rust_zone = RustZone(
    name="test",
    grid=RustGrid.from_positions([0.0, 1.0], [0.0]),
    site_buses=[SiteBus(src=[0], dst=[1])],
    word_buses=[WordBus(src=[0], dst=[1])],
    words_with_site_buses=[0],
    sites_with_word_buses=[0],
    entangling_pairs=[(0, 1)],
)
rust_mode = RustMode(
    name="all",
    zones=[0],
    bitstring_order=[
        RustLocAddr(0, 0, 0),
        RustLocAddr(0, 0, 1),
        RustLocAddr(0, 1, 0),
        RustLocAddr(0, 1, 1),
    ],
)

arch_spec = ArchSpec.from_components(
    words=(word, word),
    zones=(rust_zone,),
    modes=[rust_mode],
)


def test_show_with_mocked_pyplot():
    """``ArchVisualizer.show`` drives matplotlib and calls ``plt.show``."""
    with (
        patch("matplotlib.pyplot.gca") as mock_gca,
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.plot") as mock_plot,
    ):
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        ArchVisualizer(arch_spec).show(
            ax=mock_ax, show_words=[0], show_intra=[0], show_inter=[0]
        )
        # Check that plot was called (either on ax or pyplot)
        assert mock_ax.plot.called or mock_plot.called
        # Check that plt.show was called
        assert mock_show.called


def test_path_bounds_x_y_bounds():
    viz = ArchVisualizer(arch_spec)
    x_min, x_max, y_min, y_max = viz.path_bounds()
    assert x_min <= x_max
    assert y_min <= y_max
    x_min2, x_max2 = viz.x_bounds
    y_min2, y_max2 = viz.y_bounds
    assert x_min2 <= x_max2
    assert y_min2 <= y_max2
