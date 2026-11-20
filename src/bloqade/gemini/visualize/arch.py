"""Visualization helpers for :class:`~bloqade.lanes.arch.spec.ArchSpec`.

These used to live as methods on ``ArchSpec`` itself. They were extracted
as part of #464 phase 1 so the core ``ArchSpec`` Python wrapper stays
focused on architectural data and validation, keeping matplotlib out of
its import surface.

The primary entry point is the :class:`ArchVisualizer` class, which
caches bounds computations and provides ``plot`` / ``show`` methods.
The ``ArchSpec`` shims (``arch_spec.plot``, ``.show``, ``.x_bounds``,
``.y_bounds``, ``.path_bounds``) create an ``ArchVisualizer`` via a
``@cached_property`` so existing call sites keep working.
"""

from __future__ import annotations

from math import hypot
from typing import TYPE_CHECKING, Any, Literal, cast
from functools import cache, cached_property
from itertools import pairwise
from collections import defaultdict
from collections.abc import Sequence
from importlib.resources import files

from bloqade.lanes.bytecode.encoding import (
    MoveType,
    Direction,
    LaneAddress,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from plotly.graph_objects import Figure  # type: ignore[reportMissingImports]
    from bloqade.lanes.arch.spec import ArchSpec


__all__ = [
    "ArchVisualizer",
]

_BusTraceData = tuple[
    list[float | None],
    list[float | None],
    list[tuple[str, str, str, int, str] | None],
]


# Slack added to the column-pair spacing when deciding whether a lane stays
# within its column pair for straight versus curved schematic paths.
_COLUMN_TOLERANCE_UM = 1e-6

_CARTOON_ARC_SAMPLES = 11
"""Points per cartoon Bézier arc, endpoints included.

Every sample is paid for six times over in the exported document -- twice in
``layout.meta`` (both path styles), and again in each path button's ``x``,
``y`` and ``customdata`` restyle payload, where the per-point ``customdata``
entry is several times the size of the coordinate itself. So the sample count
sets the size of the whole figure far more than it sets the look of one curve.

Eleven is where a quadratic Bézier stops visibly gaining at the zoom levels
these plots are read at; the previous 21 doubled the document to smooth a
curve that was already smooth.
"""

_PATH_COORD_DECIMALS = 4
"""Decimals kept on generated cartoon coordinates.

Positions are micrometres over a device tens of micrometres wide, so the
fourth decimal is far below one screen pixel at any usable zoom. Full float64
repr costs up to 17 significant digits per number for no visible return.
"""


@cache
def _arch_interactive_script() -> str:
    """Load the packaged browser interactions used by Plotly figures."""
    return (
        files("bloqade.gemini.visualize")
        .joinpath("_arch_interactive.js")
        .read_text(encoding="utf-8")
    )


def bus_preview_label(move_type: MoveType, zone_id: int | None, bus_id: int) -> str:
    """Name one bus the way both hover paths must name it.

    A single figure labels the same bus from two directions: the JS controller
    shows this string, baked in as each lane's ``previewLabel``, when a site is
    hovered, and the debugger builds a label in Python when a move path is
    hovered. They appear in the same tooltip vocabulary, so any difference in
    wording reads as two different buses. One function so they cannot drift.

    ``zone_id`` is ``None`` for an inter-zone bus, which belongs to no single
    zone; :class:`MoveType.ZONE` lanes are the same case seen from the lane
    side.
    """
    if zone_id is None:
        return f"Zone bus {bus_id}"

    return f"Zone ID {zone_id}, {move_type.name.capitalize()} bus {bus_id}"


def _is_jupyter_kernel(shell: Any) -> bool:
    """Return whether ``shell`` is a ZMQ-backed IPython kernel.

    Prefer the real class check when ipykernel is installed.  The MRO-name
    fallback keeps this helper usable with lightweight test doubles and with
    compatible hosted kernels that do not expose ipykernel as an importable
    dependency.
    """
    if shell is None:
        return False

    try:
        from ipykernel.zmqshell import ZMQInteractiveShell
    except ImportError:  # pragma: no cover - depends on optional notebook stack
        pass
    else:
        if isinstance(shell, ZMQInteractiveShell):
            return True

    return any(cls.__name__ == "ZMQInteractiveShell" for cls in type(shell).__mro__)


_plotlyjs_emitted = False
"""Whether this kernel session has already written plotly.js into an output.

These figures display as HTML rather than through Plotly's MIME bundle,
because the architecture controller has to run against the rendered div. HTML
output carries its own scripts, so every display used to embed the whole
plotly.js bundle -- ~4.3 MB per cell, repeated in full for every cell showing
a figure, and saved into the ``.ipynb`` that many times over.

The bundle only has to arrive once: it defines ``window.Plotly`` for the whole
page. So the first display in a session embeds it, and the rest fall back to a
CDN ``<script src>`` -- 252 bytes, against 4.3 MB for another copy.

Not ``include_plotlyjs=False`` for the later cells, which would be 252 bytes
smaller still. ``False`` renders nothing at all if the embedded copy is not on
the page, and it can be absent for reasons the kernel cannot see: the output
holding it was cleared before the notebook was saved, or the cells were
re-ordered. The CDN tag makes every cell independently renderable, and costs
nothing when the embedded copy *is* present -- ``window.Plotly`` is already
defined by then, so a tag that fails to load (offline, say) is harmless.
"""

PLOTLYJS_MODE: bool | str | None = None
"""How notebook displays should supply plotly.js. ``None`` dedupes per session.

Set to ``True`` to embed the full bundle in every cell, which is what a fully
air-gapped notebook wants: no cell then depends on the CDN tag resolving, even
after its outputs are cleared. Anything Plotly's ``include_plotlyjs`` accepts
works here.
"""


def _notebook_plotlyjs() -> bool | str:
    """Return the ``include_plotlyjs`` value for one notebook display."""
    global _plotlyjs_emitted

    if PLOTLYJS_MODE is not None:
        return PLOTLYJS_MODE

    if _plotlyjs_emitted:
        return "cdn"

    _plotlyjs_emitted = True
    return True


def reset_plotlyjs_state() -> None:
    """Forget that plotly.js was emitted, so the next display embeds it again.

    For tests, and for a session that needs a fresh embedded copy.
    """
    global _plotlyjs_emitted
    _plotlyjs_emitted = False


class _InteractiveArchFigureMixin:
    """Install architecture interactions in figure HTML methods and displays.

    Module-level ``plotly.io`` exports bypass these overrides.
    """

    @staticmethod
    def _interactive_post_scripts(
        post_script: str | Sequence[str] | None,
    ) -> list[str]:
        scripts = [_arch_interactive_script()]
        if isinstance(post_script, str):
            scripts.append(post_script)
        elif post_script is not None:
            scripts.extend(post_script)
        return scripts

    @staticmethod
    def _interactive_config(config: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(config or {})
        merged.setdefault("scrollZoom", True)
        merged.setdefault("responsive", True)
        return merged

    def to_html(
        self,
        *args: Any,
        post_script: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Return interactive HTML with architecture controls installed.

        Use this method (or ``write_html``) for export instead of
        ``plotly.io.to_html(self)``, which bypasses the controller. Plotly.js
        is embedded by default; ``include_plotlyjs="cdn"`` explicitly opts
        into loading a versioned bundle over the network.
        """
        scripts = self._interactive_post_scripts(post_script)
        config = self._interactive_config(kwargs.pop("config", None))
        return cast(
            str,
            cast(Any, super()).to_html(
                *args,
                post_script=scripts,
                config=config,
                **kwargs,
            ),
        )

    def write_html(
        self,
        *args: Any,
        post_script: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write a self-contained interactive HTML document.

        Plotly's base ``Figure.write_html`` delegates directly to
        :func:`plotly.io.write_html` and therefore bypasses ``to_html``.
        Forward the architecture controller explicitly so saved figures retain
        the same selectors and hover behavior as notebook displays.
        """
        from plotly.io import write_html

        write_html(
            self,
            *args,
            post_script=self._interactive_post_scripts(post_script),
            config=self._interactive_config(kwargs.pop("config", None)),
            **kwargs,
        )

    def _repr_mimebundle_(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        validate: bool = True,
        **kwargs: Any,
    ) -> dict[str, str] | Any:
        if (include is not None and "text/html" not in include) or (
            exclude is not None and "text/html" in exclude
        ):
            return cast(Any, super())._repr_mimebundle_(
                include=include,
                exclude=exclude,
                validate=validate,
                **kwargs,
            )
        return {
            "text/html": self.to_html(
                full_html=False,
                include_plotlyjs=_notebook_plotlyjs(),
                validate=validate,
            )
        }

    def _ipython_display_(self) -> None:
        """Render through HTML instead of Plotly's interaction-free MIME path."""
        try:
            from IPython.display import HTML, display
        except ImportError:  # pragma: no cover - only called by IPython
            cast(Any, super())._ipython_display_()
            return

        display(
            HTML(
                self.to_html(
                    full_html=False,
                    include_plotlyjs=_notebook_plotlyjs(),
                )
            )
        )

    def show(self, *args: Any, **kwargs: Any) -> Any:
        """Preserve custom controls in notebooks and browser windows."""
        try:
            from IPython.core.getipython import get_ipython
        except ImportError:  # pragma: no cover - IPython is an optional dependency
            shell = None
        else:
            shell = get_ipython()

        if len(args) > 1 or (args and "renderer" in kwargs):
            raise TypeError("show() accepts at most one renderer argument")

        renderer = args[0] if args else kwargs.get("renderer")
        html_option_names = {
            "animation_opts",
            "auto_play",
            "config",
            "height",
            "renderer",
            "validate",
            "width",
        }
        html_options = {
            name: kwargs[name]
            for name in ("animation_opts", "auto_play")
            if name in kwargs
        }
        if "width" in kwargs:
            html_options["default_width"] = kwargs["width"]
        if "height" in kwargs:
            html_options["default_height"] = kwargs["height"]

        static_renderers = {"png", "jpeg", "jpg", "webp", "svg", "pdf", "json"}
        is_jupyter_kernel = _is_jupyter_kernel(shell)
        notebook_requested = is_jupyter_kernel and renderer not in (
            "browser",
            *static_renderers,
        )
        browser_requested = renderer == "browser"
        browser_is_default = renderer is None and not is_jupyter_kernel
        custom_html_requested = (
            notebook_requested or browser_requested or browser_is_default
        )
        unsupported_options = set(kwargs) - html_option_names
        if custom_html_requested and unsupported_options:
            unsupported = ", ".join(sorted(unsupported_options))
            raise TypeError(
                "Interactive architecture controls cannot be preserved for "
                f"show() option(s): {unsupported}. Use figure.to_html() or "
                "figure.write_html() with the corresponding HTML option."
            )

        if notebook_requested:
            from IPython.display import HTML, display

            display(
                HTML(
                    self.to_html(
                        full_html=False,
                        include_plotlyjs=_notebook_plotlyjs(),
                        config=kwargs.get("config"),
                        validate=kwargs.get("validate", True),
                        **html_options,
                    )
                )
            )
            return None

        if browser_requested or browser_is_default:
            # Plotly's browser renderer serializes the figure dictionary
            # directly, bypassing this class's ``to_html`` override and its
            # architecture-specific post-script. Open that HTML explicitly so
            # script usage retains the bus selectors and hover overlays.
            from plotly.io._base_renderers import open_html_in_browser

            open_html_in_browser(
                self.to_html(
                    full_html=True,
                    # Always embedded, unlike the notebook paths above: this is
                    # a standalone page with no earlier output to inherit
                    # ``window.Plotly`` from.
                    include_plotlyjs=True,
                    config=kwargs.get("config"),
                    validate=kwargs.get("validate", True),
                    **html_options,
                )
            )
            return None

        return cast(Any, super()).show(*args, **kwargs)


def _location_position(
    arch_spec: ArchSpec, word_id: int, site_id: int, zone_id: int
) -> tuple[float, float] | None:
    """Return (x, y) for a site or ``None`` if the triple is invalid.

    Uses the optional-returning Rust lookup directly so callers can
    iterate all (zone, word, site) combinations without raising on
    architectures where not every word exists in every zone.
    """
    return arch_spec._inner.location_position(
        LocationAddress(word_id, site_id, zone_id)._inner
    )


class ArchVisualizer:
    """Visualization facade for an :class:`ArchSpec`.

    Construct once from an architecture spec; bounds are cached so
    repeated calls to ``plot`` or ``show`` don't recompute site
    positions.

    Example::

        viz = ArchVisualizer(arch_spec)
        viz.plot(ax, show_words=[0, 1], show_word_bus=[0])
        viz.plot_interactive()
        print(viz.x_bounds, viz.y_bounds)
    """

    def __init__(self, arch_spec: ArchSpec) -> None:
        self.arch_spec = arch_spec

    # ── Bounds (cached) ──────────────────────────────────────────

    @cached_property
    def x_bounds(self) -> tuple[float, float]:
        """``(x_min, x_max)`` across every site. Falls back to
        ``(-1.0, 1.0)`` when no sites are discoverable."""
        x_min = float("inf")
        x_max = float("-inf")
        arch = self.arch_spec
        for zone_id in range(len(arch.zones)):
            for word_id in range(len(arch.words)):
                for site_id in range(len(arch.words[word_id].site_indices)):
                    pos = _location_position(arch, word_id, site_id, zone_id)
                    if pos is not None:
                        x_min = min(x_min, pos[0])
                        x_max = max(x_max, pos[0])
        if x_min == float("inf"):
            x_min = -1.0
        if x_max == float("-inf"):
            x_max = 1.0
        return x_min, x_max

    @cached_property
    def y_bounds(self) -> tuple[float, float]:
        """``(y_min, y_max)`` across every site. Falls back to
        ``(-1.0, 1.0)`` when no sites are discoverable."""
        y_min = float("inf")
        y_max = float("-inf")
        arch = self.arch_spec
        for zone_id in range(len(arch.zones)):
            for word_id in range(len(arch.words)):
                for site_id in range(len(arch.words[word_id].site_indices)):
                    pos = _location_position(arch, word_id, site_id, zone_id)
                    if pos is not None:
                        y_min = min(y_min, pos[1])
                        y_max = max(y_max, pos[1])
        if y_min == float("inf"):
            y_min = -1.0
        if y_max == float("-inf"):
            y_max = 1.0
        return y_min, y_max

    def path_bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` covering every site **and**
        every transport-path waypoint registered on the arch."""
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        for path in self.arch_spec.paths.values():
            for x, y in path:
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
        return (x_min, x_max, y_min, y_max)

    # ── Bus-path iterators ───────────────────────────────────────

    def iter_word_bus_paths(
        self,
        show_word_bus: Sequence[int],
        zone_id: int | None = None,
    ) -> Iterator[tuple[tuple[float, float], ...]]:
        """Yield selected word-bus paths.

        When ``zone_id`` is provided, only buses in that zone are yielded.
        Otherwise buses from all zones are considered.
        """
        arch = self.arch_spec
        zone_ids = range(len(arch.zones)) if zone_id is None else (zone_id,)

        for current_zone_id in zone_ids:
            zone = arch.zones[current_zone_id]
            for lane_id in show_word_bus:
                if lane_id >= len(zone.word_buses):
                    continue
                lane = zone.word_buses[lane_id]
                for site_id in zone.sites_with_word_buses:
                    for start_word_id in lane.src:
                        lane_addr = WordLaneAddress(
                            zone_id=current_zone_id,
                            word_id=start_word_id,
                            site_id=site_id,
                            bus_id=lane_id,
                            direction=Direction.FORWARD,
                        )
                        yield arch.get_path(lane_addr)

    def iter_site_bus_paths(
        self,
        show_words: Sequence[int],
        show_site_bus: Sequence[int],
        zone_id: int | None = None,
    ) -> Iterator[tuple[tuple[float, float], ...]]:
        """Yield selected site-bus paths.

        When ``zone_id`` is provided, only buses in that zone are yielded.
        Otherwise buses from all zones are considered.
        """
        arch = self.arch_spec
        zone_ids = range(len(arch.zones)) if zone_id is None else (zone_id,)

        for current_zone_id in zone_ids:
            zone = arch.zones[current_zone_id]
            words_with_site_buses = set(zone.words_with_site_buses)
            for word_id in show_words:
                if word_id not in words_with_site_buses:
                    continue
                for lane_id in show_site_bus:
                    if lane_id >= len(zone.site_buses):
                        continue
                    lane = zone.site_buses[lane_id]
                    for i in range(len(lane.src)):
                        lane_addr = SiteLaneAddress(
                            zone_id=current_zone_id,
                            word_id=word_id,
                            site_id=lane.src[i],
                            bus_id=lane_id,
                            direction=Direction.FORWARD,
                        )
                        yield arch.get_path(lane_addr)

    def _iter_locations(
        self,
    ) -> Iterator[tuple[LocationAddress, tuple[float, float]]]:
        """Yield every valid location and its physical position."""
        arch = self.arch_spec
        for zone_id in range(len(arch.zones)):
            for word_id, word in enumerate(arch.words):
                for site_id in range(len(word.site_indices)):
                    location = LocationAddress(word_id, site_id, zone_id)
                    if (position := arch.try_get_position(location)) is not None:
                        yield location, position

    def _forward_lanes_by_bus(
        self,
    ) -> dict[tuple[MoveType, int | None, int], list[LaneAddress]]:
        """Group canonical forward lanes by their user-facing bus identity.

        Site- and word-bus IDs are local to a zone, while zone-bus IDs are
        architecture-wide. Consequently, zone buses deliberately use ``None``
        for the zone portion of the key.
        """
        buses: dict[tuple[MoveType, int | None, int], list[LaneAddress]] = defaultdict(
            list
        )
        for lane in self.arch_spec.iter_all_lanes():
            if lane.direction != Direction.FORWARD:
                continue
            zone_id = None if lane.move_type == MoveType.ZONE else lane.zone_id
            buses[(lane.move_type, zone_id, lane.bus_id)].append(lane)
        return dict(buses)

    @cached_property
    def _column_pair_spacing(self) -> float:
        """Horizontal extent of one column pair, in µm.

        Column pairs are the closely spaced site columns that sit within the
        blockade radius of each other. When the architecture defines
        entangling pairs, the spacing is the widest x separation between CZ
        partners. Otherwise, or when partners share an x position, it falls
        back to the smallest nonzero x spacing between site columns, which
        treats adjacent columns as one pair.
        """
        arch = self.arch_spec
        partner_spacings = [
            abs(arch.get_position(partner)[0] - position[0])
            for location, position in self._iter_locations()
            if (partner := arch.get_cz_partner(location)) is not None
        ]
        partner_spacing = max(partner_spacings, default=0.0)
        if partner_spacing > 0.0:
            return partner_spacing

        # No usable entangling pairs (none defined, or partners share an x
        # position): treat adjacent site columns as one pair.
        x_positions = sorted({position[0] for _, position in self._iter_locations()})
        gaps = [b - a for a, b in pairwise(x_positions) if b - a > 0]
        return min(gaps, default=0.0)

    def _cartoon_path(self, lane: LaneAddress) -> tuple[tuple[float, float], ...]:
        """Return a schematic path that emphasizes bus connectivity.

        A lane that stays within one column pair (its x change is no wider
        than the pair spacing) is shown as a direct line. Any lane that
        crosses column pairs, whatever its bus type, uses a curved arch so that collinear hops
        of different lengths do not collapse onto each other.
        Zone buses are always arched to distinguish inter-zone motion.
        """
        src, dst = self.arch_spec.get_endpoints(lane)
        start = self.arch_spec.get_position(src)
        end = self.arch_spec.get_position(dst)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = hypot(dx, dy)
        stays_in_pair = abs(dx) <= self._column_pair_spacing + _COLUMN_TOLERANCE_UM
        if distance == 0.0 or (stays_in_pair and lane.move_type != MoveType.ZONE):
            return (start, end)

        # A quadratic Bézier approximation of matplotlib's ``arc3`` path.
        # The perpendicular offset keeps the endpoint relationship legible
        # without pretending to be the device's exact transport trajectory.
        bend = -0.34 * distance
        control = (
            (start[0] + end[0]) / 2 - dy / distance * bend,
            (start[1] + end[1]) / 2 + dx / distance * bend,
        )
        path: list[tuple[float, float]] = []
        segments = _CARTOON_ARC_SAMPLES - 1
        for step in range(_CARTOON_ARC_SAMPLES):
            t = step / segments
            one_minus_t = 1.0 - t
            path.append(
                (
                    round(
                        one_minus_t**2 * start[0]
                        + 2 * one_minus_t * t * control[0]
                        + t**2 * end[0],
                        _PATH_COORD_DECIMALS,
                    ),
                    round(
                        one_minus_t**2 * start[1]
                        + 2 * one_minus_t * t * control[1]
                        + t**2 * end[1],
                        _PATH_COORD_DECIMALS,
                    ),
                )
            )
        return tuple(path)

    # ── Rendering ────────────────────────────────────────────────

    def plot(
        self,
        ax: Axes | None = None,
        show_words: Sequence[int] = (),
        show_site_bus: Sequence[int] = (),
        show_word_bus: Sequence[int] = (),
        zone_id: int | None = None,
        label_sites: bool = False,
        **scatter_kwargs,
    ) -> Axes:
        """Render the architecture onto a matplotlib axes.

        Args:
            ax: Matplotlib axes to render onto. Uses the current axes when
                omitted.
            show_words: Word IDs whose sites should be rendered.
            show_site_bus: Site-bus IDs whose paths should be rendered.
            show_word_bus: Word-bus IDs whose paths should be rendered.
            zone_id: Restrict site and bus rendering to this zone. When
                ``None``, preserve the existing default behavior.
            label_sites: Label rendered sites with their
                ``(zone_id, word_id, site_id)`` addresses.
            **scatter_kwargs: Additional keyword arguments forwarded to
                ``Axes.scatter``.

        Returns:
            The supplied axes, or the auto-resolved current axes.
        """
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        if ax is None:
            ax = plt.gca()

        arch = self.arch_spec
        zones_to_plot = range(len(arch.zones)) if zone_id is None else (zone_id,)

        for word_id in show_words:
            word = arch.words[word_id]
            positions: list[tuple[float, float]] = []
            site_labels: list[tuple[float, float, str]] = []

            for current_zone_id in zones_to_plot:
                for site_id in range(len(word.site_indices)):
                    pos = _location_position(arch, word_id, site_id, current_zone_id)
                    if pos is not None:
                        positions.append(pos)
                        if label_sites:
                            site_labels.append(
                                (
                                    pos[0],
                                    pos[1],
                                    f"({current_zone_id}, {word_id}, {site_id})",
                                )
                            )
                if positions:
                    break

            if positions:
                x_positions = [p[0] for p in positions]
                y_positions = [p[1] for p in positions]
                ax.scatter(x_positions, y_positions, **scatter_kwargs)

                if label_sites:
                    for x, y, label in site_labels:
                        ax.text(x, y, label)

        for path in self.iter_site_bus_paths(
            show_words,
            show_site_bus,
            zone_id=zone_id,
        ):
            x_vals, y_vals = zip(*path)
            ax.plot(x_vals, y_vals, linestyle="--")

        for path in self.iter_word_bus_paths(
            show_word_bus,
            zone_id=zone_id,
        ):
            x_vals, y_vals = zip(*path)
            ax.plot(x_vals, y_vals, linestyle="-")

        return ax

    def plot_interactive(
        self,
        *,
        show_site_ids: bool = False,
        show_all_buses: bool = False,
        show_bus_legend: bool = False,
        path_style: Literal["exact", "cartoon"] = "exact",
        site_lane_preview: Literal["hover", "click"] = "hover",
        bus_line_style: Literal["by_type", "dashed"] = "by_type",
        theme: Literal["light", "dark"] = "light",
        width: int = 1300,
        height: int = 800,
    ) -> Figure:
        """Build an interactive architecture and atom-path visualization.

        The returned Plotly figure supports pan, box zoom, and mode-bar zoom.
        Its HTML representation also enables trackpad pinch zoom (or
        Ctrl+wheel); a plain scroll wheel scrolls the page instead of zooming.
        A site can overlay every transport lane available from that site,
        using the lane's bus color and the currently selected exact or cartoon
        path representation.
        Hovering a bus shows the source and destination site addresses and
        temporarily draws that bus as a thicker overlay above other buses.
        Site labels can be toggled with the controls above the plot.

        The site-, word-, and zone-bus multiselectors on the right contain a
        color-labelled checkbox for every bus. Selections accumulate without a
        long permanent bus legend. Bus traces are hidden by default to keep
        large architectures readable; pass ``show_all_buses=True`` or use the
        selectors to display them. The path-view toggle switches between exact
        architecture paths from :meth:`ArchSpec.get_path` and a less cluttered
        schematic view. The selector panel also supports undo and redo. Clear
        all removes both selected bus traces and click-pinned site previews;
        show all displays every bus, clears click-pinned site previews, and
        synchronizes every selector checkbox. Exact/cartoon and label controls
        are view-only and are therefore not part of preview undo/redo history.
        Schematic paths are straight only for lanes that stay within one
        column pair; lanes that cross column pairs, and all zone-bus lanes,
        are curved to expose crossings and keep overlapping hops
        distinguishable.

        Args:
            show_site_ids: Show site identity labels initially. Site identity
                remains available on hover when labels are hidden.
            show_all_buses: Show all bus paths initially. By default, buses are
                available from the right-side selectors but hidden to avoid
                visual clutter.
            show_bus_legend: Also show Plotly's legacy bus legend below the
                right-side selectors. The selectors are the default bus
                selection interface because the legend is unwieldy for large
                architectures.
            path_style: Initial bus-path representation. ``"exact"`` preserves
                architecture-defined transport waypoints; ``"cartoon"``
                emphasizes bus connectivity with straight and curved paths.
            site_lane_preview: Show available lanes while hovering a site, or
                toggle them by clicking sites. Click mode supports multiple
                simultaneous selections and is useful when atom and move
                traces are layered over the architecture.
            bus_line_style: Use distinct dashes for each move type, or draw all
                architecture bus lanes dashed. The latter distinguishes them
                from solid executed-move paths in the Plotly debugger.
            theme: Initial color theme. The default is explicit light mode so
                the figure does not inherit a dark notebook or documentation
                background.
            width: Figure width in pixels. The roomy default helps separate the
                160 sites in the Gemini physical architecture.
            height: Figure height in pixels.

        Returns:
            A ``plotly.graph_objects.Figure``. In a notebook, return it from a
            cell or call ``figure.show(config={"scrollZoom": True})`` to also
            enable pinch zoom. Export interactive figures with the
            returned figure's ``to_html`` or ``write_html`` methods. Calling
            the module-level ``plotly.io.to_html(figure)`` or
            ``plotly.io.write_html(figure, ...)`` bypasses the custom figure
            methods and therefore cannot install the architecture interaction
            controller.

        Raises:
            ImportError: If the ``visualization`` optional dependency group is
                not installed.
            ValueError: If a style, preview mode, or theme is unsupported.
        """
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Interactive architecture plots require Plotly. Install "
                "bloqade-lanes with the 'visualization' extra."
            ) from exc

        if path_style not in ("exact", "cartoon"):
            raise ValueError("path_style must be 'exact' or 'cartoon'")
        if site_lane_preview not in ("hover", "click"):
            raise ValueError("site_lane_preview must be 'hover' or 'click'")
        if bus_line_style not in ("by_type", "dashed"):
            raise ValueError("bus_line_style must be 'by_type' or 'dashed'")
        if theme not in ("light", "dark"):
            raise ValueError("theme must be 'light' or 'dark'")

        locations = list(self._iter_locations())
        buses = self._forward_lanes_by_bus()
        theme_colors = {
            "light": {
                "paper": "#ffffff",
                "plot": "#f8fafc",
                "text": "#1e293b",
                "muted": "#64748b",
                "site": "#e2e8f0",
                "site_edge": "#475569",
                "grid": "rgba(100, 116, 139, 0.20)",
                "bus_lightness": "40%",
            },
            "dark": {
                "paper": "#0f172a",
                "plot": "#111827",
                "text": "#f8fafc",
                "muted": "#cbd5e1",
                "site": "#64748b",
                "site_edge": "#e2e8f0",
                "grid": "rgba(203, 213, 225, 0.16)",
                "bus_lightness": "62%",
            },
        }[theme]
        dash_by_type = {
            MoveType.SITE: "dot",
            MoveType.WORD: "solid",
            MoveType.ZONE: "dash",
        }

        class InteractiveArchFigure(_InteractiveArchFigureMixin, go.Figure):
            pass

        figure = InteractiveArchFigure()
        bus_trace_indices: list[int] = []
        bus_controls: list[dict[str, object]] = []
        site_lane_paths: list[dict[str, object]] = []
        site_lane_path_refs: dict[LocationAddress, list[tuple[int, bool]]] = (
            defaultdict(list)
        )
        exact_trace_data: list[_BusTraceData] = []
        cartoon_trace_data: list[_BusTraceData] = []
        for color_index, ((move_type, zone_id, bus_id), lanes) in enumerate(
            buses.items()
        ):
            # Golden-angle spacing avoids exact color repeats when an
            # architecture has more buses than a conventional fixed palette.
            bus_color = (
                f"hsl({round(color_index * 137.508) % 360}, 68%, "
                f"{theme_colors['bus_lightness']})"
            )
            kind = move_type.name.lower()
            bus_name = (
                f"zone bus {bus_id}"
                if zone_id is None
                else f"zone {zone_id} · {kind} bus {bus_id}"
            )
            exact_values: _BusTraceData = ([], [], [])
            cartoon_values: _BusTraceData = ([], [], [])
            for lane in lanes:
                src, dst = self.arch_spec.get_endpoints(lane)
                src_label = f"({src.zone_id}, {src.word_id}, {src.site_id})"
                dst_label = f"({dst.zone_id}, {dst.word_id}, {dst.site_id})"
                exact_path = self.arch_spec.get_path(lane)
                cartoon_path = self._cartoon_path(lane)
                site_lane_path_index = len(site_lane_paths)
                site_lane_paths.append(
                    {
                        "exactX": [point[0] for point in exact_path],
                        "exactY": [point[1] for point in exact_path],
                        "cartoonX": [point[0] for point in cartoon_path],
                        "cartoonY": [point[1] for point in cartoon_path],
                        "color": bus_color,
                        "busName": bus_name,
                        "previewLabel": bus_preview_label(move_type, zone_id, bus_id),
                        "busId": bus_id,
                        "moveType": kind,
                        "source": src_label,
                        "destination": dst_label,
                        "dash": (
                            "dash"
                            if bus_line_style == "dashed"
                            else dash_by_type[move_type]
                        ),
                    }
                )
                site_lane_path_refs[src].append((site_lane_path_index, False))
                site_lane_path_refs[dst].append((site_lane_path_index, True))
                lane_hover_data = (
                    bus_name,
                    src_label,
                    dst_label,
                    bus_id,
                    kind,
                )
                for values, path in (
                    (exact_values, exact_path),
                    (cartoon_values, cartoon_path),
                ):
                    values[0].extend(point[0] for point in path)
                    values[1].extend(point[1] for point in path)
                    values[2].extend(lane_hover_data for _ in path)
                    values[0].append(None)
                    values[1].append(None)
                    values[2].append(None)

            exact_trace_data.append(exact_values)
            cartoon_trace_data.append(cartoon_values)
            initial_values = exact_values if path_style == "exact" else cartoon_values

            trace_index = len(cast(Any, figure.data))
            bus_trace_indices.append(trace_index)
            bus_controls.append(
                {
                    "traceIndex": trace_index,
                    "kind": kind,
                    "busId": bus_id,
                    "label": bus_name,
                    "color": bus_color,
                }
            )
            figure.add_trace(
                go.Scattergl(
                    x=initial_values[0],
                    y=initial_values[1],
                    customdata=initial_values[2],
                    mode="lines+markers",
                    name=bus_name,
                    legendgroup=kind,
                    showlegend=show_bus_legend,
                    line={
                        "color": bus_color,
                        "dash": (
                            "dash"
                            if bus_line_style == "dashed"
                            else dash_by_type[move_type]
                        ),
                        "width": 2.25,
                    },
                    marker={"size": 3},
                    opacity=0.82,
                    visible=True if show_all_buses else "legendonly",
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "move type: %{customdata[4]}<br>"
                        "bus ID: %{customdata[3]}<br>"
                        "source: %{customdata[1]}<br>"
                        "destination: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                )
            )

        x_sites = [position[0] for _, position in locations]
        y_sites = [position[1] for _, position in locations]
        # Keep the tuple compact and put horizontally adjacent sites on opposite
        # sides of their row. Locations that share coordinates receive distinct
        # placements around the marker.
        site_labels = [
            f"({location.zone_id},{location.word_id},{location.site_id})"
            for location, _ in locations
        ]
        label_positions = ["top center"] * len(locations)
        indices_by_y: dict[float, list[int]] = defaultdict(list)
        indices_by_position: dict[tuple[float, float], list[int]] = defaultdict(list)
        for index, (_, (x, y)) in enumerate(locations):
            indices_by_y[y].append(index)
            indices_by_position[(x, y)].append(index)

        for row_indices in indices_by_y.values():
            for rank, index in enumerate(
                sorted(row_indices, key=lambda item: x_sites[item])
            ):
                label_positions[index] = (
                    "top center" if rank % 2 == 0 else "bottom center"
                )

        duplicate_positions = (
            "top left",
            "bottom right",
            "top right",
            "bottom left",
            "middle left",
            "middle right",
            "top center",
            "bottom center",
        )
        for position_indices in indices_by_position.values():
            if len(position_indices) > 1:
                for rank, index in enumerate(position_indices):
                    label_positions[index] = duplicate_positions[
                        rank % len(duplicate_positions)
                    ]

        site_customdata = [
            [
                location.zone_id,
                location.word_id,
                location.site_id,
                *self.arch_spec.words[location.word_id].sites[location.site_id],
            ]
            for location, _ in locations
        ]
        site_trace_index = len(cast(Any, figure.data))
        figure.add_trace(
            go.Scatter(
                x=x_sites,
                y=y_sites,
                customdata=site_customdata,
                text=site_labels if show_site_ids else [""] * len(site_labels),
                mode="markers+text",
                marker={
                    "color": theme_colors["site"],
                    "line": {"color": theme_colors["site_edge"], "width": 1},
                    "size": 9,
                },
                textposition=label_positions,
                textfont={
                    "color": theme_colors["text"],
                    "family": "Menlo, Consolas, monospace",
                    "size": 9,
                },
                cliponaxis=False,
                name="sites",
                showlegend=False,
                hovertemplate=(
                    "<b>(zone, word, site): (%{customdata[0]}, "
                    "%{customdata[1]}, %{customdata[2]})</b><br>"
                    "grid (x, y): (%{customdata[3]}, %{customdata[4]})<br>"
                    "position (x, y): (%{x:.3f}, %{y:.3f}) µm"
                    "<extra></extra>"
                ),
                hoverlabel={"align": "left"},
            )
        )

        def view_ranges(
            trace_data: list[_BusTraceData],
        ) -> tuple[list[float], list[float]]:
            x_values = list(x_sites)
            y_values = list(y_sites)
            for trace_x, trace_y, _ in trace_data:
                x_values.extend(value for value in trace_x if value is not None)
                y_values.extend(value for value in trace_y if value is not None)

            def padded(values: list[float]) -> list[float]:
                if not values:
                    return [-1.0, 1.0]
                lower = min(values)
                upper = max(values)
                span = upper - lower
                padding = 0.035 * span if span else 1.0
                return [lower - padding, upper + padding]

            return padded(x_values), padded(y_values)

        exact_ranges = view_ranges(exact_trace_data)
        cartoon_ranges = view_ranges(cartoon_trace_data)
        initial_ranges = exact_ranges if path_style == "exact" else cartoon_ranges

        label_buttons = [
            {
                "label": "Labels off",
                "method": "restyle",
                "args": [{"text": [[""] * len(site_labels)]}, [site_trace_index]],
            },
            {
                "label": "Labels on",
                "method": "restyle",
                "args": [{"text": [site_labels]}, [site_trace_index]],
            },
        ]
        bus_buttons = [
            {
                "label": "Clear all buses",
                "method": "restyle",
                "args": [
                    {"visible": ["legendonly"] * len(bus_trace_indices)},
                    bus_trace_indices,
                ],
            },
            {
                "label": "Show all buses",
                "method": "restyle",
                "args": [
                    {"visible": [True] * len(bus_trace_indices)},
                    bus_trace_indices,
                ],
            },
        ]

        def path_button(
            label: str,
            selected_path_style: Literal["exact", "cartoon"],
            trace_data: list[_BusTraceData],
            ranges: tuple[list[float], list[float]],
        ) -> dict[str, object]:
            return {
                "label": label,
                "method": "update",
                "args": [
                    {
                        "x": [values[0] for values in trace_data],
                        "y": [values[1] for values in trace_data],
                        "customdata": [values[2] for values in trace_data],
                    },
                    {
                        "xaxis.range": ranges[0],
                        "yaxis.range": ranges[1],
                        "meta.archVisualizerPathStyle": selected_path_style,
                    },
                    bus_trace_indices,
                ],
            }

        path_buttons = [
            path_button("Exact paths", "exact", exact_trace_data, exact_ranges),
            path_button("Cartoon paths", "cartoon", cartoon_trace_data, cartoon_ranges),
        ]
        figure.update_layout(
            template="plotly_white" if theme == "light" else "plotly_dark",
            title={
                "text": "Architecture and atom transport paths",
                "x": 0.01,
                "xanchor": "left",
            },
            annotations=[
                {
                    "text": (
                        "Use the bus multiselectors at right to show or hide "
                        f"individual paths; {site_lane_preview} a site to "
                        "preview every available lane. Export with the "
                        "figure's write_html() method to retain these controls."
                    ),
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.0,
                    "y": 1.015,
                    "xanchor": "left",
                    "font": {"color": theme_colors["muted"], "size": 12},
                }
            ],
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "right",
                    "buttons": label_buttons,
                    "showactive": False,
                    "x": 0.0,
                    "xanchor": "left",
                    "y": 1.09,
                    "yanchor": "top",
                },
                {
                    "type": "buttons",
                    "direction": "right",
                    "buttons": bus_buttons,
                    # Legend selections can diverge from either global state,
                    # so keeping one button highlighted would be misleading.
                    "showactive": False,
                    "x": 0.42,
                    "xanchor": "left",
                    "y": 1.09,
                    "yanchor": "top",
                },
                {
                    "type": "buttons",
                    "direction": "right",
                    "buttons": path_buttons,
                    "active": 0 if path_style == "exact" else 1,
                    "x": 0.72,
                    "xanchor": "left",
                    "y": 1.09,
                    "yanchor": "top",
                },
            ],
            xaxis={
                "title": "x (µm)",
                "range": initial_ranges[0],
                "showgrid": True,
                "gridcolor": theme_colors["grid"],
                "zeroline": False,
            },
            yaxis={
                "title": "y (µm)",
                "range": initial_ranges[1],
                "scaleanchor": "x",
                "scaleratio": 1,
                "showgrid": True,
                "gridcolor": theme_colors["grid"],
                "zeroline": False,
            },
            legend={
                "title": {"text": "Bus legend"},
                "groupclick": "toggleitem",
                "itemclick": "toggle",
                "itemdoubleclick": "toggleothers",
                "x": 1.01,
                "xanchor": "left",
                "y": 0.62,
                "yanchor": "top",
                "font": {"size": 11},
            },
            dragmode="zoom",
            hovermode="closest",
            width=width,
            height=height,
            margin={
                "l": 70,
                "r": 270,
                "b": 65,
                "t": 165,
            },
            font={"color": theme_colors["text"]},
            paper_bgcolor=theme_colors["paper"],
            plot_bgcolor=theme_colors["plot"],
            uirevision="arch-visualizer",
            meta={
                "archVisualizerBusTraceIndices": bus_trace_indices,
                "archVisualizerBusControls": bus_controls,
                "archVisualizerSiteTraceIndex": site_trace_index,
                "archVisualizerSiteLabels": site_labels,
                "archVisualizerSiteLanePaths": site_lane_paths,
                "archVisualizerSiteLanePreviewMode": site_lane_preview,
                "archVisualizerPathStyle": path_style,
                "archVisualizerSiteLanePathRefs": {
                    f"{location.zone_id},{location.word_id},{location.site_id}": refs
                    for location, refs in site_lane_path_refs.items()
                },
            },
        )
        return figure

    def show(
        self,
        ax: Axes | None = None,
        show_words: Sequence[int] = (),
        show_intra: Sequence[int] = (),
        show_inter: Sequence[int] = (),
        zone_id: int | None = None,
        label_sites: bool = False,
        **scatter_kwargs,
    ) -> None:
        """Render and immediately call ``plt.show()``.

        ``zone_id`` and ``label_sites`` have the same meaning as in
        :meth:`plot`.

        Convenience for interactive sessions; programmatic callers
        should prefer :meth:`plot`.
        """
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        self.plot(
            ax,
            show_words=show_words,
            show_site_bus=show_intra,
            show_word_bus=show_inter,
            zone_id=zone_id,
            label_sites=label_sites,
            **scatter_kwargs,
        )
        plt.show()
