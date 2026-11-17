# Interactive visualization and HTML export

Install `bloqade[atom-vis]` to display an architecture with bus
selectors, site previews, and transport-path tooltips:

```python
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.gemini.visualize.arch import ArchVisualizer

figure = ArchVisualizer(get_arch_spec()).plot_interactive()
figure.show()
```

## Supported HTML export

Use the returned figure's **`to_html()`** method to generate HTML with the
architecture controls installed:

```python
from pathlib import Path

html = figure.to_html(full_html=True, include_plotlyjs=True)
Path("architecture.html").write_text(html, encoding="utf-8")
```

The figure's **`write_html()`** method is also supported as a file-writing
convenience:

```python
figure.write_html("architecture.html", include_plotlyjs=True)
```

Both methods preserve caller-supplied `config` and `post_script` options.
The same export methods apply to debugger figures that include the interactive
architecture layer. In the debugger, hovering over a site previews its available
lanes, including when an atom occupies the site. Move the pointer away to clear
the preview.

Avoid the module-level `plotly.io.to_html(figure)` and
`plotly.io.write_html(figure, ...)` functions for these figures. Those functions
serialize Plotly data directly rather than invoking the custom figure methods,
so they omit the additional JavaScript that creates the bus selectors and
overlays. Converting the result to a plain `plotly.graph_objects.Figure` also
loses those custom methods.

## Offline use and JavaScript versions

Notebook and browser display embed the Plotly.js bundle from the installed
Plotly Python package. HTML export does the same by default
(`include_plotlyjs=True`). This produces a larger document but does not need to
download Plotly.js when opened.

For smaller files, `include_plotlyjs="cdn"` is an explicit opt-in requiring
network access. Plotly generates a versioned CDN URL matching its bundled
JavaScript version; it does not request an unversioned latest release.
`include_plotlyjs=False` is only appropriate when the containing page already
loads a compatible Plotly.js bundle.

In this repository, `uv.lock` pins the Plotly Python distribution, including its
bundled JavaScript. Use `uv sync --locked --extra atom-vis` to reproduce
that version. The lockfile does not govern installations made by downstream
users; those follow the version range in `pyproject.toml`.

The interaction controller uses Plotly axis-conversion and SVG internals.
Dependency upgrades still require browser checks of hovering, site previews,
bus selection, pan/zoom, and debugger playback.

## JavaScript validation

Run the same syntax check used by CI:

```bash
just check-visualization-js
```

This requires Node.js; CI uses Node 22. A pre-commit hook also checks edits to
the controller. The `{plot_id}` placeholder is inside a quoted JavaScript
string, so the source can be checked before Plotly substitutes the figure ID.
The command also executes controller regression tests with a minimal DOM/Plotly
event fixture, covering lane previews over empty sites, occupied sites, and
atoms. These tests do not replace browser integration checks of Plotly's real
DOM internals.
