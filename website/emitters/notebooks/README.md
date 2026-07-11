# bloqade-docs-emit-notebooks

Executes the Bloqade **tutorials** (jupytext *percent*-format `.py` files under
`docs/**`) and emits Starlight-compatible **MDX** with the outputs inlined. This
is Phase C3 of the docs-site build. It is a self-contained `uv` project and only
writes into an output dir + a public-assets dir you point it at; it never edits
`astro.config.mjs`, `src/components/**`, `src/styles/**`, or `website/scripts/`.

The MDX is component-free markdown (fenced code blocks, Starlight asides,
KaTeX-ready `$`-math, `<img>` tags), so generated pages need **no** import lines
and render through the same expressive-code + `remark-math`/`rehype-katex`
pipeline the rest of the site uses.

## Install / run (uv project)

```bash
cd website/emitters/notebooks
uv sync

# Static render (no execution) -> scratch _out/:
uv run bloqade-docs-emit-notebooks ../../../docs/digital/examples/qasm2/ghz.py

# Executed render into the site, images into public/tutorials/:
EXECUTE_NOTEBOOKS=1 uv run bloqade-docs-emit-notebooks \
  ../../../docs/digital/examples/qasm2/ghz.py \
  --out       ../../src/content/docs/guides/tutorials \
  --public-dir ../../public/tutorials \
  --asset-base /tutorials
```

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `input` (positional) | — | Path to the jupytext percent `.py` tutorial. |
| `--out` | `emitters/notebooks/_out` | Dir for the generated `<name>.mdx`. Point a real build at `src/content/docs/guides/tutorials`. |
| `--public-dir` | `<out>/public/tutorials` | Filesystem dir images are written to (a `<name>/` subdir is created). Point at `website/public/tutorials`. |
| `--asset-base` | `/tutorials` | URL prefix for emitted `<img src>` attributes; the site serves `public/tutorials/*` at `/tutorials/*`. |
| `--name` | slugified filename stem | Tutorial slug; names the `.mdx`, the `public/<name>/` asset dir, and the asset URL path. |
| `--repo-root` | git top-level | Kernel cwd + by-basename asset fallback search root. |
| `--cache-dir` | `emitters/notebooks/.cache` | Executed-notebook cache (keyed by source hash). |
| `--kernel-python` | this venv's interpreter | Python used to launch the execution kernel. |
| `--kernel-name` | `python3` | Kernel spec name. |
| `--timeout` | `300` | Per-cell execution timeout (seconds). |
| `--execute` / `--no-execute` | (env) | Force execution on/off, overriding `EXECUTE_NOTEBOOKS`. |
| `--no-cache` | off | Ignore the cache and always re-execute. |
| `--allow-errors` | off | Capture cell errors as outputs instead of falling back to a static render. |
| `-v/--verbose` | off | Info-level logging. |

## Pipeline

```
.py (jupytext percent)
   │  jupytext.reads(fmt="py:percent") -> nbformat NotebookNode
   ▼
[execution gate]  EXECUTE_NOTEBOOKS in {1,true,yes,on} ?
   ├── no  → render code + markdown only (no outputs)
   └── yes → cache hit?  ── yes → load .cache/<hash>.ipynb
              └── no → nbclient executes in a bloqade kernel, then cache it
   ▼
render → MDX (frontmatter + prose + fenced code + rendered outputs)
   ▼   images → public/<name>/…            .mdx → <out>/<name>.mdx
```

Source files:

* `pipeline.py` — read/parse, the execution gate, caching, `nbclient`
  execution, and orchestration (`Config`, `run`).
* `render.py` — notebook → MDX; markdown/admonition/image conversion, code
  fences, and the output→MDX mapping (`RenderContext`, `render_notebook`).
* `escape.py` — MDX-safety escaping, ANSI stripping, fenced-block emission.
* `__main__.py` — the CLI.

## Execution gating

Execution mirrors the current mkdocs nightly (`.github/workflows/doc-nightly.yml`
sets `EXECUTE_NOTEBOOKS`): it runs **only** when `EXECUTE_NOTEBOOKS` is one of
`1`, `true`, `yes`, `on` (case-insensitive). `--execute`/`--no-execute` override
the env for local use. When disabled, the tutorial is rendered from source with
**no** outputs and a small "static render" note.

`nbclient` runs the notebook in a kernel launched from `--kernel-python`
(defaults to this venv's own interpreter, which has `bloqade` installed via the
path dependency in `pyproject.toml`). A throwaway kernelspec is written into a
temp dir and exposed via `JUPYTER_PATH`, so execution never depends on a
globally-registered `python3` kernelspec and always uses the interpreter we
control. The kernel's cwd is the repo root, matching how the mkdocs build runs.

**Failure handling.** If the kernel can't start or a cell raises (with the
default `--allow-errors` off), the error is logged and the pipeline **falls back
to a static (non-executed) render** rather than failing the build. Pass
`--allow-errors` to instead capture the traceback as a rendered output.

## Output caching

Executed notebooks are cached under `.cache/` (gitignored), keyed by
`sha256(CACHE_SCHEMA + "\0" + source_bytes)` — i.e. the exact bytes of the input
`.py` plus a schema tag (`CACHE_SCHEMA` in `pipeline.py`, bump it to invalidate
all caches when the executor changes). The cache file is the fully-executed
notebook, `.cache/<hash>.ipynb`.

* Execution enabled + `.cache/<hash>.ipynb` exists → the executed notebook is
  loaded and the kernel is **skipped**.
* Hash changed / no cache entry → execute, then write the cache.
* `--no-cache` forces re-execution (and refreshes the entry).

Because the key is the *source* hash, editing a tutorial re-runs it while an
unchanged tutorial is free on every build. Rendering is always re-done (cheap),
so tweaks to the emitter don't require re-execution.

## Output → MDX mapping

Jupyter `In[ ]`/`Out[ ]` prompts and execution counts are never emitted (the
site hides them). ANSI colour codes (from rich/pretty-printers) are stripped
from all text. Consecutive `stream` outputs of the same name are **coalesced**
into one block (a single `print` from a pretty-printer can otherwise arrive as
hundreds of fragments).

| Output | Rendered as |
| --- | --- |
| `stream` stdout | fenced ```` ```text ```` block |
| `stream` stderr | fenced ```` ```text title="stderr" ```` block |
| `error` | fenced ```` ```text title="Traceback" ```` (ANSI-stripped traceback) |
| `image/png` | decoded to `public/<name>/cell<c>_out<o>.png`, referenced with a self-closed `<img src="/tutorials/<name>/…" />` |
| `image/svg+xml` | written to `public/<name>/cell<c>_out<o>.svg`, referenced with `<img>` |
| `text/latex` | a `$$ … $$` KaTeX math block |
| `text/plain` | fenced ```` ```text ```` (preferred over `text/html` when both exist) |
| `text/html` | escaped fenced ```` ```html ```` **source** block (see limitations) |
| `text/markdown` | rendered as markdown prose |

Images are **not** routed through `astro:assets`/Sharp (not installed); they are
plain files under `public/` referenced by absolute URL, so they are copied
verbatim into the build.

### Markdown cells

* The first markdown `# H1` becomes the frontmatter `title` and is stripped from
  the body (Starlight renders the title as the page `<h1>`).
* mkdocs-material admonitions
  (`<div class="admonition note"><p class="admonition-title">…</p>…</div>`) are
  converted to **Starlight asides** (`:::note[Title] … :::`); the class maps to
  `note`/`tip`/`caution`/`danger`.
* `<img>` tags are self-closed and their `src` rewritten to
  `/tutorials/<name>/assets/<file>` after the referenced file is copied into
  `public/`. Relative refs are resolved against the `.py` dir first, then by
  basename under the repo (the legacy mkdocs sources use stale `../../` paths).
* `<div>`/`<picture>`/`<figure>` wrappers pass through as valid JSX.
* All remaining free text is MDX-escaped: `<`, `{`, `}` → character references,
  **except** inside fenced code, inline code spans, and `$`-math, which are
  preserved verbatim so KaTeX braces (`2^{n}`) and code survive. `>` and `&` are
  left alone. This guarantees the MDX build never breaks on tutorial prose.

## Dependencies

Declared in `pyproject.toml`:

* `jupytext` — parse the percent `.py` into an `nbformat` notebook.
* `nbclient` — execute the notebook against a kernel.
* `nbformat` — notebook read/write (cache) and output model.
* `ipykernel` — the in-venv kernel the notebooks execute in.
* `bloqade` — path dependency on the repo root (`../../..`, editable), so this
  venv's interpreter is a complete kernel with `bloqade` importable. Execution
  is the only thing that imports it; a static render works without it.

## Verification (this checkout)

`docs/digital/examples/qasm2/ghz.py` was **executed** end-to-end via the repo's
`bloqade` and rendered to `src/content/docs/guides/tutorials/ghz.mdx` (assets in
`public/tutorials/ghz/`). It exercises H1→title, admonitions→asides, KaTeX math,
`<img>` asset copying, python code fences, and a large coalesced stdout block
(the `qasm2.parse.pprint` output). `astro build` renders it (asides, code
blocks, math, images) with no errors. The `image/png`/`image/svg+xml`/`error`/
`text/html` branches were verified with a synthetic notebook.

## Known limitations

* **Interactive widgets** (`ipywidgets`, `application/vnd.jupyter.widget-*`):
  not rendered — no `text/plain` fallback is shown for a pure-widget output.
* **Interactive plots** (Plotly/Bokeh via `text/html` + JS): emitted as escaped
  HTML *source* in a fenced block, not as a live widget. Static images
  (`image/png`/`svg`) render fully. Rich HTML tables likewise show their
  `text/plain` form (preferred) or HTML source, not a rendered table — this is a
  deliberate build-safety choice (arbitrary library HTML is not guaranteed to be
  valid JSX/MDX).
* **Long-running cells**: bounded by `--timeout` (per cell). A timeout raises and
  triggers the static-render fallback (unless `--allow-errors`).
* **Cross-references**: docstring/markdown cross-refs are not rewritten to
  `<ApiXref>` (that is Phase C1's job).
