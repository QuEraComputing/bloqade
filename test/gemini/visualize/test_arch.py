"""Tests for the arch visualization helpers extracted from
``ArchSpec`` (#464 phase 1).

Covers the :class:`ArchVisualizer` class and verifies that the legacy
``ArchSpec.<method>`` shims still route through it.
"""

from __future__ import annotations

from typing import Any, cast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
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
from bloqade.lanes.bytecode.encoding import (
    MoveType,
    Direction,
    SiteLaneAddress,
    WordLaneAddress,
)

from bloqade.gemini.visualize import arch as arch_visualization
from bloqade.gemini.visualize.arch import ArchVisualizer

# ── Hand-built minimal ArchSpec fixture ──


@pytest.fixture
def small_arch_spec() -> ArchSpec:
    word = Word(sites=((0, 0), (1, 0)))
    rust_grid = RustGrid.from_positions([0.0, 1.0], [0.0])
    rust_zone = RustZone(
        name="test",
        grid=rust_grid,
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
    site_lane = SiteLaneAddress(
        word_id=0,
        site_id=0,
        bus_id=0,
        direction=Direction.FORWARD,
        zone_id=0,
    )
    return ArchSpec.from_components(
        words=(word, word),
        zones=(rust_zone,),
        modes=[rust_mode],
        paths={site_lane: ((0.0, 0.0), (0.5, 0.75), (1.0, 0.0))},
    )


# ── ArchVisualizer class ──


def test_interactive_javascript_is_loaded_from_packaged_asset() -> None:
    script = arch_visualization._arch_interactive_script()

    assert script.startswith("(function () {")
    assert "{plot_id}" in script
    assert "data-arch-visualizer-bus-selectors" in script


def test_x_bounds(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    assert viz.x_bounds == (0.0, 1.0)


def test_y_bounds(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    assert viz.y_bounds == (0.0, 0.0)


def test_path_bounds(small_arch_spec: ArchSpec) -> None:
    x_min, x_max, y_min, y_max = ArchVisualizer(small_arch_spec).path_bounds()
    assert x_min <= 0.0 <= x_max
    assert y_min <= 0.0 <= y_max


def test_bounds_are_cached(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    assert viz.x_bounds is viz.x_bounds
    assert viz.y_bounds is viz.y_bounds


def test_iter_site_bus_paths(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    paths = list(viz.iter_site_bus_paths([0], [0]))
    assert paths
    for path in paths:
        assert isinstance(path, tuple)
        assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in path)


def test_iter_word_bus_paths(small_arch_spec: ArchSpec) -> None:
    viz = ArchVisualizer(small_arch_spec)
    paths = list(viz.iter_word_bus_paths([0]))
    assert paths
    for path in paths:
        assert isinstance(path, tuple)
        assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in path)


def test_plot_returns_axes(small_arch_spec: ArchSpec) -> None:
    mock_ax = MagicMock()
    viz = ArchVisualizer(small_arch_spec)
    result = viz.plot(mock_ax, show_words=[0], show_site_bus=[0], show_word_bus=[0])
    assert result is mock_ax
    assert mock_ax.scatter.called
    assert mock_ax.plot.called


def test_plot_interactive_starts_with_bus_paths_hidden(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    figure_data = cast(Any, figure.data)

    bus_indices = figure.layout.meta["archVisualizerBusTraceIndices"]
    bus_traces = [figure_data[index] for index in bus_indices]
    assert [trace.name for trace in bus_traces] == [
        "zone 0 · site bus 0",
        "zone 0 · word bus 0",
    ]
    assert all(trace.visible == "legendonly" for trace in bus_traces)
    assert all(trace.showlegend is False for trace in bus_traces)
    assert figure.layout.dragmode == "zoom"
    assert figure.layout.legend.groupclick == "toggleitem"
    assert figure.layout.paper_bgcolor == "#ffffff"
    assert figure.layout.plot_bgcolor == "#f8fafc"
    assert figure.layout.width == 1300


def test_plot_interactive_uses_architecture_paths(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    visualizer = ArchVisualizer(small_arch_spec)
    figure = visualizer.plot_interactive(show_all_buses=True)
    figure_data = cast(Any, figure.data)
    site_trace = next(
        trace for trace in figure_data if trace.name == "zone 0 · site bus 0"
    )
    expected_path = next(visualizer.iter_site_bus_paths([0], [0]))

    assert len(expected_path) == 3
    assert list(zip(site_trace.x, site_trace.y))[: len(expected_path)] == list(
        expected_path
    )
    assert site_trace.visible is True


def test_plot_interactive_bus_hover_identifies_endpoints(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(show_all_buses=True)
    site_bus_trace = cast(Any, figure.data)[0]
    hover_data = next(value for value in site_bus_trace.customdata if value is not None)

    assert tuple(hover_data) == (
        "zone 0 · site bus 0",
        "(0, 0, 0)",
        "(0, 0, 1)",
        0,
        "site",
    )
    assert "move type: %{customdata[4]}" in site_bus_trace.hovertemplate
    assert "bus ID: %{customdata[3]}" in site_bus_trace.hovertemplate
    assert "source: %{customdata[1]}" in site_bus_trace.hovertemplate
    assert "destination: %{customdata[2]}" in site_bus_trace.hovertemplate
    assert "path point" not in site_bus_trace.hovertemplate


def test_plot_interactive_site_hover_keeps_text_compact_and_stores_lane_paths(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    assert [trace.name for trace in cast(Any, figure.data)].count("sites") == 1
    site_trace = next(
        trace for trace in cast(Any, figure.data) if trace.name == "sites"
    )
    site_data = next(
        value for value in site_trace.customdata if list(value[:3]) == [0, 0, 0]
    )

    assert list(site_data) == [0, 0, 0, 0, 0]
    assert "grid (x, y): (%{customdata[3]}, %{customdata[4]})" in (
        site_trace.hovertemplate
    )
    assert "position (x, y): (%{x:.3f}, %{y:.3f}) µm" in (site_trace.hovertemplate)
    assert "Touching lanes" not in site_trace.hovertemplate
    assert site_trace.hoverlabel.align == "left"

    meta = figure.layout.meta
    assert meta["archVisualizerSiteTraceIndex"] == 2
    assert meta["archVisualizerSiteLanePathRefs"]["0,0,0"] == [
        (0, False),
        (1, False),
    ]
    assert meta["archVisualizerSiteLanePathRefs"]["0,0,1"] == [(0, True)]
    first_path = meta["archVisualizerSiteLanePaths"][0]
    assert first_path["exactX"] == [0.0, 0.5, 1.0]
    assert first_path["exactY"] == [0.0, 0.75, 0.0]
    # The site bus stays within its column pair, so its cartoon is a line.
    assert first_path["cartoonX"] == [0.0, 1.0]
    assert first_path["cartoonY"] == [0.0, 0.0]
    assert first_path["color"].startswith("hsl(")
    assert first_path["dash"] == "dot"
    assert first_path["busName"] == "zone 0 · site bus 0"
    assert first_path["previewLabel"] == "Zone ID 0, Site bus 0"
    assert first_path["busId"] == 0
    assert first_path["moveType"] == "site"
    assert first_path["source"] == "(0, 0, 0)"
    assert first_path["destination"] == "(0, 0, 1)"


def test_plot_interactive_supports_click_previews_and_dashed_buses(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(
        site_lane_preview="click",
        bus_line_style="dashed",
    )
    bus_indices = figure.layout.meta["archVisualizerBusTraceIndices"]

    assert figure.layout.meta["archVisualizerSiteLanePreviewMode"] == "click"
    assert all(
        cast(Any, figure.data[index]).line.dash == "dash" for index in bus_indices
    )
    assert all(
        path["dash"] == "dash"
        for path in figure.layout.meta["archVisualizerSiteLanePaths"]
    )
    html = figure.to_html(full_html=False, include_plotlyjs=False)
    assert "plot.addEventListener('click', function (event)" in html
    assert "const selectedSites = new Map()" in html
    assert "toggleSelectedSite(siteCustomdata)" in html
    assert "drawnPathIndices.has(pathIndex)" in html
    assert "data-arch-visualizer-lane-path-index" in html
    assert "plot.on('plotly_animated'" in html
    assert "Undo bus or lane visibility change" in html
    assert "Redo bus or lane visibility change" in html
    assert "restorePreviewState(previewHistoryIndex - 1)" in html
    assert "restorePreviewState(previewHistoryIndex + 1)" in html
    assert "schedulePreviewHistoryRecord" not in html
    assert "previewStateMutationInProgress" not in html
    assert "commitPreviewState();" in html
    assert "plot.on('plotly_afterplot'" in html
    assert "data-arch-visualizer-lane-tooltip" in html
    assert "${lanePath.previewLabel}" in html
    assert "move type: ${lanePath.moveType}" not in html
    assert "bus ID: ${lanePath.busId}" not in html
    assert "currentPathStyle === 'cartoon'" in html
    assert "siteCustomdataFromPlotlyEvent(event)" in html
    assert "trace.meta.bloqadeTraceKind === 'atom'" in html
    assert "siteCustomdataNearPointer(pointerEvent)" in html
    assert "data-arch-visualizer-site-lane-hit-target" in html
    assert "insetPathEndpoints(pixels, 12)" in html
    assert "hitPath.setAttribute('pointer-events', 'stroke')" in html
    assert "data-bloqade-move-path-hover-target" in html
    assert "showMovePathTooltip(segment)" in html
    assert "data-arch-visualizer-lane-tooltip-connector" in html
    assert "`Atom ${segment.atomId} move path" in html
    assert "hitPath.style.cursor = 'help'" not in html
    assert "window.setTimeout(drawMovePathHoverOverlays, 0)" in html
    assert "atomHoverTargetForSite(directSitePoint.customdata)" in html
    assert "plotlyHoverLayer.style.display = 'none'" in html
    assert "showAtomTooltip(" in html
    assert "atomTarget.customdata" in html
    assert "directAtomPoint.customdata" in html
    assert "if (directAtomPoint)" in html
    assert "drawMovePathHoverOverlays();" in html


def test_plot_interactive_html_highlights_hovered_bus(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    html = figure.to_html(full_html=False, include_plotlyjs=False)

    assert "plotly_hover" in html
    assert "plotly_unhover" in html
    assert "drawHighlight(busPoint.curveNumber)" in html
    assert "data-arch-visualizer-bus-highlight" in html
    assert "data-arch-visualizer-site-lane" in html
    assert "function siteCustomdataAt(x, y)" in html
    assert ".map((item) => siteCustomdataAt(item.x, item.y))" in html
    assert "function siteCustomdataNearPointer(event)" in html
    assert "plot.addEventListener('mousemove'" in html
    assert "activateSiteLaneOverlays(siteCustomdataNearPointer(event))" in html
    assert "element.isConnected" in html
    assert "plot.on('plotly_unhover'" in html
    assert "data-arch-visualizer-site-lane-arrow" not in html
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-bus-color" in html
    assert "checkbox.type = 'checkbox'" in html
    assert "text.textContent = `ID ${control.busId} · ${control.label}`" in html
    assert "archVisualizerSiteLanePathRefs" in html
    assert "stroke-opacity', '0.42'" in html
    assert "data-arch-visualizer-path-overlays" in html
    assert "overlayLayer.appendChild(path)" in html
    assert "const hoverLayer = plot.querySelector('.hoverlayer')" not in html
    assert '"scrollZoom": true' in html
    assert '"responsive": true' in html
    assert list(figure.layout.meta["archVisualizerBusTraceIndices"]) == [0, 1]


def test_plot_interactive_notebook_representation_keeps_interactions(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    mimebundle = figure._repr_mimebundle_()

    assert set(mimebundle) == {"text/html"}
    assert "data-arch-visualizer-bus-selectors" in mimebundle["text/html"]
    assert "plotly_hover" in mimebundle["text/html"]
    assert 'src="https://cdn.plot.ly' not in mimebundle["text/html"]


def test_plot_interactive_ipython_display_keeps_interactions(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    with patch("IPython.display.display") as display:
        figure._ipython_display_()

    html = display.call_args.args[0].data
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-site-lane" in html


def test_plot_interactive_show_keeps_interactions_in_jupyter(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    zmq_shell = type("ZMQInteractiveShell", (), {})()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=zmq_shell),
        patch("IPython.display.display") as display,
    ):
        figure.show(config={"scrollZoom": False})

    html = display.call_args.args[0].data
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-site-lane" in html
    assert '"scrollZoom": false' in html


@pytest.fixture
def fresh_plotlyjs_state():
    """Run with no plotly.js emitted yet, and leave the session as found."""
    arch_visualization.reset_plotlyjs_state()
    mode = arch_visualization.PLOTLYJS_MODE
    yield
    arch_visualization.PLOTLYJS_MODE = mode
    arch_visualization.reset_plotlyjs_state()


def test_plotly_bundle_is_embedded_once_per_session(
    plotly, small_arch_spec: ArchSpec, fresh_plotlyjs_state
) -> None:
    """Only the first display carries plotly.js; later cells reference it.

    Every display is HTML rather than a Plotly MIME bundle, because the
    architecture controller needs the rendered div -- so without this, each
    cell of a notebook embedded its own ~4 MB copy of the same script.
    """
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    first = figure._repr_mimebundle_()["text/html"]
    second = figure._repr_mimebundle_()["text/html"]

    assert len(second) < len(first)
    # The figure itself is unchanged; only the script payload differs.
    assert "data-arch-visualizer-bus-selectors" in second
    # A CDN tag rather than nothing, so the cell still renders if the embedded
    # copy is missing from the page -- cleared output, or re-ordered cells.
    assert "cdn.plot.ly" in second

    # `_ipython_display_` is a separate entry point onto the same state, so it
    # must not re-embed what `_repr_mimebundle_` already sent.
    with patch("IPython.display.display") as display:
        figure._ipython_display_()

    assert len(display.call_args.args[0].data) == len(second)


def test_plotly_bundle_returns_after_reset(
    plotly, small_arch_spec: ArchSpec, fresh_plotlyjs_state
) -> None:
    """Resetting re-embeds, for a session whose bundle-carrying output is gone."""
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    first = figure._repr_mimebundle_()["text/html"]
    assert len(figure._repr_mimebundle_()["text/html"]) < len(first)

    arch_visualization.reset_plotlyjs_state()
    assert len(figure._repr_mimebundle_()["text/html"]) == len(first)


def test_plotlyjs_mode_overrides_session_dedup(
    plotly, small_arch_spec: ArchSpec, fresh_plotlyjs_state
) -> None:
    """``PLOTLYJS_MODE`` opts out, for offline use or a CDN reference."""
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    arch_visualization.PLOTLYJS_MODE = True
    embedded = [len(figure._repr_mimebundle_()["text/html"]) for _ in range(2)]
    assert embedded[0] == embedded[1], "every cell should embed under True"

    arch_visualization.PLOTLYJS_MODE = "cdn"
    cdn = figure._repr_mimebundle_()["text/html"]
    assert len(cdn) < embedded[0]
    assert "cdn.plot.ly" in cdn


def test_plot_interactive_show_recognizes_derived_jupyter_shell(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    zmq_base = type("ZMQInteractiveShell", (), {})
    hosted_shell = type("HostedKernelShell", (zmq_base,), {})()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=hosted_shell),
        patch("IPython.display.display") as display,
    ):
        figure.show()

    html = display.call_args.args[0].data
    assert "data-arch-visualizer-bus-selectors" in html


def test_plot_interactive_show_rejects_unsupported_html_options(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=None),
        pytest.raises(TypeError, match="cannot be preserved.*unsupported_option"),
    ):
        figure.show(renderer="browser", unsupported_option=True)


def test_plot_interactive_show_keeps_interactions_in_browser(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=None),
        patch("plotly.io._base_renderers.open_html_in_browser") as open_browser,
    ):
        figure.show(renderer="browser", config={"scrollZoom": False})

    html = open_browser.call_args.args[0]
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-site-lane" in html
    assert '"scrollZoom": false' in html
    assert open_browser.call_args.kwargs == {}


def test_plot_interactive_explicit_browser_keeps_interactions_in_jupyter(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    zmq_shell = type("ZMQInteractiveShell", (), {})()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=zmq_shell),
        patch("plotly.io._base_renderers.open_html_in_browser") as open_browser,
    ):
        figure.show(renderer="browser")

    html = open_browser.call_args.args[0]
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-site-lane" in html


def test_plot_interactive_browser_preserves_animation_options(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=None),
        patch("plotly.io._base_renderers.open_html_in_browser") as open_browser,
    ):
        figure.show(
            renderer="browser",
            auto_play=False,
            animation_opts={"frame": {"duration": 25}},
        )

    html = open_browser.call_args.args[0]
    assert "data-arch-visualizer-bus-selectors" in html
    assert "plotly_click" in html


def test_plot_interactive_html_preserves_caller_config(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    html = figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"scrollZoom": False, "displayModeBar": False},
    )

    assert '"scrollZoom": false' in html
    assert '"responsive": true' in html
    assert '"displayModeBar": false' in html


def test_plot_interactive_write_html_keeps_interactions(
    plotly,
    small_arch_spec: ArchSpec,
    tmp_path: Path,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    output = tmp_path / "architecture.html"

    figure.write_html(
        output,
        include_plotlyjs=True,
        config={"scrollZoom": False},
    )

    html = output.read_text(encoding="utf-8")
    assert "data-arch-visualizer-bus-selectors" in html
    assert "data-arch-visualizer-site-lane" in html
    assert '"scrollZoom": false' in html
    assert 'src="https://cdn.plot.ly' not in html


def test_plot_interactive_site_identity_toggle(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(show_site_ids=True)
    label_trace = cast(Any, figure.data)[-1]

    assert set(label_trace.text) == {
        "(0,0,0)",
        "(0,0,1)",
        "(0,1,0)",
        "(0,1,1)",
    }
    assert list(label_trace.textposition) == [
        "top left",
        "top left",
        "bottom right",
        "bottom right",
    ]
    assert label_trace.name == "sites"
    assert label_trace.mode == "markers+text"
    assert label_trace.marker.size == 9
    assert label_trace.marker.opacity is None
    assert label_trace.textfont.size == 9
    assert label_trace.cliponaxis is False
    assert figure.layout.updatemenus[0].showactive is False
    assert [button.method for button in figure.layout.updatemenus[0].buttons] == [
        "restyle",
        "restyle",
    ]
    assert figure.layout.meta["archVisualizerSiteLabels"] == list(label_trace.text)


def test_plot_interactive_controls_can_show_and_clear_all_bus_previews(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    clear_button, show_button = figure.layout.updatemenus[1].buttons
    html = figure.to_html(full_html=False, include_plotlyjs=False)

    assert clear_button.label == "Clear all buses"
    assert clear_button.method == "restyle"
    assert show_button.label == "Show all buses"
    assert show_button.method == "restyle"
    assert "setAllBusPreviews(false, true)" in html
    assert "setAllBusPreviews(true, true)" in html
    assert "target.closest('g.updatemenu-button')" in html
    assert "plot.on('plotly_buttonclicked'" not in html
    assert "syncBusCheckboxes()" in html
    assert "selectedSites.clear()" in html
    assert "clearSiteLaneOverlays()" in html
    assert "const busVisibility = busControls.map" in html
    assert "buses: [...busVisibility]" in html
    assert "checkbox.checked = busVisibility[index]" in html
    assert "busVisibility.fill(visible)" in html
    assert "{visible: visibility}" in html
    assert "let previewMutationVersion = 0" in html
    assert "const mutationVersion = ++previewMutationVersion" in html
    assert "mutationVersion !== previewMutationVersion" in html


def test_plot_interactive_controls_are_below_title(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    assert all(menu.y == pytest.approx(1.09) for menu in figure.layout.updatemenus)
    assert figure.layout.margin.t == 165


def test_plot_interactive_has_color_labelled_bus_multiselectors(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    controls = figure.layout.meta["archVisualizerBusControls"]

    assert all(menu.type != "dropdown" for menu in figure.layout.updatemenus)
    assert controls == [
        {
            "traceIndex": 0,
            "kind": "site",
            "busId": 0,
            "label": "zone 0 · site bus 0",
            "color": "hsl(0, 68%, 40%)",
        },
        {
            "traceIndex": 1,
            "kind": "word",
            "busId": 0,
            "label": "zone 0 · word bus 0",
            "color": "hsl(138, 68%, 40%)",
        },
    ]


def test_plot_interactive_can_restore_legacy_bus_legend(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(show_bus_legend=True)
    bus_traces = [
        trace for trace in cast(Any, figure.data) if "bus" in (trace.name or "")
    ]

    assert all(trace.showlegend is True for trace in bus_traces)
    assert figure.layout.legend.title.text == "Bus legend"
    assert figure.layout.legend.y == 0.62
    assert figure.layout.margin.r == 270


def test_plot_interactive_cartoon_paths_keep_column_pair_hops_straight(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(
        path_style="cartoon", show_all_buses=True
    )
    figure_data = cast(Any, figure.data)
    site_trace = next(
        trace for trace in figure_data if trace.name == "zone 0 · site bus 0"
    )
    site_path = [(x, y) for x, y in zip(site_trace.x, site_trace.y) if x is not None]

    # Sites 0 and 1 are CZ partners, so the hop between them is one column
    # pair and stays a direct line in the schematic view.
    assert ArchVisualizer(small_arch_spec)._column_pair_spacing == 1.0
    assert site_path == [(0.0, 0.0), (1.0, 0.0)]
    assert figure.layout.updatemenus[2].active == 1


def test_plot_interactive_path_toggle_preserves_bus_selection(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    exact_button, cartoon_button = figure.layout.updatemenus[2].buttons

    assert exact_button.method == "update"
    assert cartoon_button.method == "update"
    assert "visible" not in cartoon_button.args[0]
    assert exact_button.args[1]["meta.archVisualizerPathStyle"] == "exact"
    assert cartoon_button.args[1]["meta.archVisualizerPathStyle"] == "cartoon"
    assert list(cartoon_button.args[2]) == [0, 1]


def test_plot_interactive_supports_dark_theme(
    plotly,
    small_arch_spec: ArchSpec,
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive(theme="dark")

    assert figure.layout.paper_bgcolor == "#0f172a"
    assert figure.layout.plot_bgcolor == "#111827"


def test_plot_uses_plt_gca_when_ax_is_none(small_arch_spec: ArchSpec) -> None:
    with patch("matplotlib.pyplot.gca") as mock_gca:
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        result = ArchVisualizer(small_arch_spec).plot(ax=None, show_words=[0])
        assert result is mock_ax
        mock_gca.assert_called_once()


def test_show_calls_plt_show(small_arch_spec: ArchSpec) -> None:
    with (
        patch("matplotlib.pyplot.gca") as mock_gca,
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        ArchVisualizer(small_arch_spec).show(
            ax=mock_ax, show_words=[0], show_intra=[0], show_inter=[0]
        )
        assert mock_show.called


def test_archvisualizer_plot_called_directly(
    small_arch_spec: ArchSpec,
) -> None:
    mock_ax = MagicMock()
    with patch.object(ArchVisualizer, "plot", return_value=mock_ax) as mock_plot:
        result = ArchVisualizer(small_arch_spec).plot(mock_ax, show_words=[0])
        mock_plot.assert_called_once()
        assert mock_plot.call_args.args == (mock_ax,)
        assert mock_plot.call_args.kwargs["show_words"] == [0]
        assert result is mock_ax


def _column_pair_arch_spec() -> ArchSpec:
    """Four one-site words: (0, 0), (2, 0), (10, 0), and (0, 10).

    Columns 0 and 2 form a column pair; column 10 is the next pair. Word bus
    0 hops within the pair, word bus 1 crosses to the next pair, and word bus
    2 moves vertically within the first column. No entangling pairs are
    defined, so the pair spacing falls back to the smallest column gap.
    """
    rust_grid = RustGrid.from_positions([0.0, 2.0, 10.0], [0.0, 10.0])
    rust_zone = RustZone(
        name="test",
        grid=rust_grid,
        site_buses=[],
        word_buses=[
            WordBus(src=[0], dst=[1]),
            WordBus(src=[0], dst=[2]),
            WordBus(src=[0], dst=[3]),
        ],
        words_with_site_buses=[],
        sites_with_word_buses=[0],
        entangling_pairs=[],
    )
    rust_mode = RustMode(
        name="all",
        zones=[0],
        bitstring_order=[RustLocAddr(0, word_id, 0) for word_id in range(4)],
    )
    return ArchSpec.from_components(
        words=(
            Word(sites=((0, 0),)),
            Word(sites=((1, 0),)),
            Word(sites=((2, 0),)),
            Word(sites=((0, 1),)),
        ),
        zones=(rust_zone,),
        modes=[rust_mode],
    )


def test_cartoon_path_curves_lanes_that_cross_column_pairs() -> None:
    viz = ArchVisualizer(_column_pair_arch_spec())
    within_pair = WordLaneAddress(word_id=0, site_id=0, bus_id=0)
    across_pairs = WordLaneAddress(word_id=0, site_id=0, bus_id=1)
    vertical = WordLaneAddress(word_id=0, site_id=0, bus_id=2)

    assert viz._column_pair_spacing == 2.0
    # Hops within one column pair, horizontal or vertical, are direct lines.
    assert viz._cartoon_path(within_pair) == ((0.0, 0.0), (2.0, 0.0))
    assert viz._cartoon_path(vertical) == ((0.0, 0.0), (0.0, 10.0))
    # A word bus that crosses to another column pair is arched, so collinear
    # hops of different lengths stay distinguishable.
    across_path = viz._cartoon_path(across_pairs)
    # Against the constant rather than a literal: the sample count is a
    # document-size knob, not a property of the curve, and every sample is
    # paid for six times over in the exported figure.
    assert len(across_path) == arch_visualization._CARTOON_ARC_SAMPLES
    assert across_path[0] == (0.0, 0.0)
    assert across_path[-1] == (10.0, 0.0)
    assert max(abs(y) for _, y in across_path) > 1.0


# ── Pure helpers and guard clauses ──


@pytest.mark.parametrize(
    ("move_type", "zone_id", "expected"),
    [
        (MoveType.SITE, 0, "Zone ID 0, Site bus 3"),
        (MoveType.WORD, 2, "Zone ID 2, Word bus 3"),
        # An inter-zone bus belongs to no single zone, so it is named without
        # one -- the branch the debugger's move-path tooltip relies on.
        (MoveType.ZONE, None, "Zone bus 3"),
    ],
)
def test_bus_preview_label_names_every_bus_kind(
    move_type: MoveType, zone_id: int | None, expected: str
) -> None:
    assert arch_visualization.bus_preview_label(move_type, zone_id, 3) == expected


@pytest.fixture
def siteless_arch_spec() -> ArchSpec:
    """A spec whose only word has no sites, so no position is discoverable.

    Exercises the fallbacks that keep a degenerate architecture plottable
    instead of handing matplotlib or Plotly an empty range.
    """
    rust_zone = RustZone(
        name="empty",
        grid=RustGrid.from_positions([0.0], [0.0]),
        site_buses=[],
        word_buses=[],
        words_with_site_buses=[],
        sites_with_word_buses=[],
        entangling_pairs=[],
    )
    rust_mode = RustMode(name="all", zones=[0], bitstring_order=[])
    return ArchSpec.from_components(
        words=(Word(sites=()),),
        zones=(rust_zone,),
        modes=[rust_mode],
    )


def test_bounds_fall_back_when_no_site_is_discoverable(
    siteless_arch_spec: ArchSpec,
) -> None:
    viz = ArchVisualizer(siteless_arch_spec)

    assert list(viz._iter_locations()) == []
    assert viz.x_bounds == (-1.0, 1.0)
    assert viz.y_bounds == (-1.0, 1.0)
    assert viz._column_pair_spacing == 0.0


def test_plot_interactive_survives_an_architecture_with_no_sites(
    plotly, siteless_arch_spec: ArchSpec
) -> None:
    """The axis padding has nothing to pad, so it falls back to a unit range."""
    figure = ArchVisualizer(siteless_arch_spec).plot_interactive()

    assert list(figure.layout.meta["archVisualizerBusTraceIndices"]) == []
    assert tuple(figure.layout.xaxis.range) == (-1.0, 1.0)
    assert tuple(figure.layout.yaxis.range) == (-1.0, 1.0)


def test_bus_path_iterators_skip_out_of_range_selections(
    small_arch_spec: ArchSpec,
) -> None:
    """Asking for a bus the zone does not define yields nothing, not an error."""
    viz = ArchVisualizer(small_arch_spec)

    assert list(viz.iter_word_bus_paths([7])) == []
    assert list(viz.iter_site_bus_paths([0], [7])) == []
    # A word with no site buses is skipped even when its bus id is valid.
    assert list(viz.iter_site_bus_paths([1], [0])) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"path_style": "squiggly"}, "path_style must be 'exact' or 'cartoon'"),
        (
            {"site_lane_preview": "tap"},
            "site_lane_preview must be 'hover' or 'click'",
        ),
        ({"bus_line_style": "dotty"}, "bus_line_style must be 'by_type' or 'dashed'"),
        ({"theme": "sepia"}, "theme must be 'light' or 'dark'"),
    ],
)
def test_plot_interactive_rejects_unsupported_options(
    plotly, small_arch_spec: ArchSpec, kwargs: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ArchVisualizer(small_arch_spec).plot_interactive(**cast(Any, kwargs))


# ── Figure-mixin HTML and display plumbing ──


def test_is_jupyter_kernel_prefers_the_real_ipykernel_class() -> None:
    """When ipykernel is importable the check is a plain isinstance.

    The MRO-name walk below it is the fallback for hosted kernels that do not
    ship ipykernel as an importable dependency; this is the other branch.
    """
    zmqshell = pytest.importorskip("ipykernel.zmqshell")

    class FakeZMQInteractiveShell:
        pass

    with patch.object(zmqshell, "ZMQInteractiveShell", FakeZMQInteractiveShell):
        assert arch_visualization._is_jupyter_kernel(FakeZMQInteractiveShell())

    assert arch_visualization._is_jupyter_kernel(None) is False
    assert arch_visualization._is_jupyter_kernel(object()) is False


def test_interactive_post_scripts_accepts_a_string_or_a_sequence() -> None:
    """Caller scripts run after the controller, however they were passed."""
    scripts = arch_visualization._InteractiveArchFigureMixin._interactive_post_scripts

    controller_only = scripts(None)
    assert len(controller_only) == 1

    assert scripts("alert(1)") == [*controller_only, "alert(1)"]
    assert scripts(["a()", "b()"]) == [*controller_only, "a()", "b()"]


def test_repr_mimebundle_delegates_when_html_is_excluded(
    plotly, small_arch_spec: ArchSpec
) -> None:
    """Asking for anything but HTML is Plotly's business, not the controller's."""
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    assert set(figure._repr_mimebundle_(include=["application/json"])) != {"text/html"}
    assert set(figure._repr_mimebundle_(exclude=["text/html"])) != {"text/html"}


def test_show_rejects_more_than_one_renderer_argument(
    plotly, small_arch_spec: ArchSpec
) -> None:
    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    with pytest.raises(TypeError, match="at most one renderer argument"):
        figure.show("browser", "png")
    with pytest.raises(TypeError, match="at most one renderer argument"):
        figure.show("browser", renderer="browser")


def test_show_forwards_width_and_height_as_html_defaults(
    plotly, small_arch_spec: ArchSpec
) -> None:
    """``show(width=..., height=...)`` maps onto Plotly's HTML sizing options."""
    figure = ArchVisualizer(small_arch_spec).plot_interactive()
    zmq_shell = type("ZMQInteractiveShell", (), {})()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=zmq_shell),
        patch.object(type(figure), "to_html", return_value="<div/>") as to_html,
    ):
        figure.show(width=911, height=457)

    assert to_html.call_args.kwargs["default_width"] == 911
    assert to_html.call_args.kwargs["default_height"] == 457


def test_show_delegates_static_renderers_to_plotly(
    plotly, small_arch_spec: ArchSpec
) -> None:
    """A PNG has no scripts to preserve, so the override steps out of the way."""
    from plotly.basedatatypes import BaseFigure

    figure = ArchVisualizer(small_arch_spec).plot_interactive()

    with (
        patch("IPython.core.getipython.get_ipython", return_value=None),
        patch.object(BaseFigure, "show", return_value="delegated") as base_show,
    ):
        assert figure.show("png") == "delegated"

    assert base_show.call_args.args == ("png",)
