# bloqade-docs-emit-python

Emits the Bloqade **Python** API reference as Starlight-compatible **MDX**,
driven by [`griffe`](https://mkdocstrings.github.io/griffe/). One MDX page is
produced per module, using the frozen API-component contract (see
`website/src/components/api/`). Generated pages carry **no import lines** — the
components are auto-imported globally (`astro-auto-import`, see
`website/astro.config.mjs`).

This is Phase B2 of the docs-site build. It writes only into its own output
directory; it never edits `astro.config.mjs`, `src/components/**`, or any other
site file.

## Install / run (uv project)

```bash
cd website/emitters/python
uv sync

uv run python -m bloqade_docs_emit_python \
  --package bloqade \
  --src /path/to/bloqade-circuit/src/bloqade \
  --repo QuEraComputing/bloqade-circuit \
  --ref v0.14.1 \
  --version dev \
  --mount api/python \
  --out ./_out \
  --docstring-style google
```

### Flags

| Flag | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--package` | yes | — | Top-level import package (e.g. `bloqade`). |
| `--src` | yes | — | Path to the package root dir (the folder named like `--package`, e.g. `.../src/bloqade`). |
| `--repo` | yes | — | Source repo slug for GitHub source links (`owner/name`). |
| `--ref` | yes | — | Git ref (tag/branch/sha) used in `blob/<ref>` source URLs. |
| `--version` | yes | — | API doc version; written to frontmatter `apiVersion`. |
| `--mount` | no | `api/python` | Site mount path used to build inventory URLs (independent of `--out`). |
| `--out` | no | `emitters/python/_out` | Output dir for MDX + `inventory.python.json`. Point a later phase at `src/content/docs/api/python/vX`. |
| `--repo-root` | no | git top-level of `--src` | Root used to compute file relpaths in source URLs. |
| `--docstring-style` | no | `google` | `google` \| `numpy` \| `sphinx` — how griffe parses docstrings. |
| `-v/--verbose` | no | off | Show griffe + emitter warnings (silenced by default). |

griffe search paths are set to the **parent** of `--src` so submodules resolve
(`search_paths=[dirname(src)]`, `loader.load("<package>")`).

## Output layout

```
<out>/
  bloqade/
    index.mdx                # bloqade (package __init__ -> index)
    task.mdx                 # bloqade.task
    squin/
      index.mdx              # bloqade.squin
      kernel.mdx             # bloqade.squin.kernel
      ...
  inventory.python.json
```

* `__init__.py` (package) -> `index.mdx`; a plain module `foo.py` -> `foo.mdx`.
* Private modules (`_name`) and anything matching the ported `skip_keywords`
  (below) are skipped. Private members (`_name`, incl. dunders) are omitted;
  `__init__` is used only to build the class constructor signature.

## What each page contains

* Frontmatter: `title`, `description` (module summary, if any), `language`,
  `fqName`, `apiVersion`, `sourceRepo`, `sourceRef`, `sourceUrl`.
* `<ApiModule>` wrapper (summary = first docstring paragraph).
* `<ApiClass>` per class: `bases`, `sourceUrl`, `<Signature>` (constructor),
  constructor `<Params>`, an `<Params title="Attributes">` table, then nested
  `<ApiFn kind="method">` / `<ApiFn kind="property">`.
* `<ApiFn kind="function">` per module-level function.
* `<Signature lang="python" code={`...`} />`, `<Params>`, `<Returns>`,
  `<Raises>`, and a `<Source>` link, mapped from the parsed docstring.
* A module-level `<Params title="Attributes">` table for public module
  attributes / type aliases / TypeVars.

Docstring section mapping (griffe -> component):

| griffe section | rendered as |
| --- | --- |
| text (summary / long description) | prose (summary also -> `summary=` / frontmatter `description`) |
| parameters | `<Params>` (merged with the real signature: name/type/default from the signature, description from the docstring) |
| other parameters | `<Params title="Other Parameters">` |
| attributes | `<Params title="Attributes">` |
| returns | `<Returns>` (also falls back to the signature return annotation) |
| raises | `<Raises>` |
| yields / receives / admonitions / examples / deprecated | prose blocks in the body |

## Docstring-style assumption

Defaults to **Google-style** docstrings (`--docstring-style google`), which is
what the Bloqade packages use. `numpy` and `sphinx` are accepted but untested
against the Bloqade corpus. griffe's own parser warnings (missing annotations,
params not in signature, etc.) are silenced unless `-v` is passed; they never
affect output.

**Style selection.** The docstring style is chosen **per emitter run** via the
`--docstring-style` flag and applied to griffe's parser for the whole package;
there is no per-object metadata or auto-detection. This only affects how griffe
splits a docstring into *sections* (params/returns/see-also/...). Cross-reference
detection (below) runs on the resulting **prose text**, so inline Sphinx roles
and Markdown autorefs are recognized regardless of the selected style; only the
NumPy `See Also` **section** depends on the parser (it is surfaced by griffe as
a `see-also` admonition when parsed as `numpy`).

## Docstring cross-references

Cross-references in docstring prose are resolved and, when they point at a
symbol this emitter documents, rewritten to the site's `<ApiXref>` component
(`<ApiXref to="<fqName>" origin="docstring" label="<display>" />`). See
`xref.py`.

**Syntaxes handled** (all detected on prose, so they work under any
`--docstring-style`):

* **reStructuredText / Sphinx object roles** — ``:class:`` , ``:func:`` ,
  ``:meth:`` , ``:obj:`` , ``:attr:`` , ``:exc:`` , ``:mod:`` , ``:data:`` ,
  ``:const:`` (plus ``:py:*:`` prefixed and ``:function:`` / ``:method:``
  aliases), including the ``~Target`` short-display form and the explicit-title
  ``Text <target>`` form. Roles inside ``.. seealso::`` blocks are picked up as
  ordinary inline roles.
* **Google style** — sections are Google-formatted, but prose cross-refs use
  either Sphinx roles (Napoleon) or Markdown autorefs; both are handled.
* **NumPy / SciPy style** — RST roles in prose *plus* the NumPy `See Also`
  section (comma-separated, optionally role-prefixed object names, each with an
  optional `: description`).
* **Markdown / mkdocstrings autorefs** — `[label][target]`, `[target][]`,
  ``[`target`][]`` (backtick target), the bare ``[`Target`]`` form, and the
  mkdocstrings **backtick autoref** (a bare `` `pkg.mod.Name` `` code span that
  is a *dotted* qualified identifier). Plain `[text](url)` links are left to the
  escape pipeline (external kept; broken intra-doc relative links degraded).

**Resolution** uses griffe's own `Object.resolve` on the leading name segment
(which walks the object's members, its module namespace, and griffe-tracked
imports/aliases) and re-attaches the dotted tail; the raw target is kept as a
fallback so absolute references still resolve.

**Safe-emit rule (keeps `XREF_STRICT` green).** A reference becomes an
`<ApiXref>` **only** when its resolved fully-qualified name is present in the
inventory this emitter builds (checked after the full inventory is known — pages
are buffered with placeholder tokens and finalized in `run()`). Everything else
— external packages (`kirin`, `numpy`, ...), undocumented/private names, and
anything unresolvable — degrades to inline **code/text**, so it can never become
an unresolved `<ApiXref>` anchor. Detection never runs inside code spans or
fenced blocks, and output flows through the same MDX-escape pipeline.

**Deliberately skipped:** cross-refs inside `<Params>`/`<Returns>`/`<Raises>`
description cells (rendered as single-line `set:html` text, which cannot host a
component) and bare *single-word* code spans (too ambiguous to linkify safely;
use a dotted name or an explicit `[`Target`]` autoref).

## MDX-safety strategy

Docstrings can contain `{ } < >`, backticks, and JSX-hostile text. A docstring
must **never** break the Astro build, so every string is escaped for its exact
output context (see `escape.py`):

* **Prose** (summaries, long descriptions, examples): `<`, `{`, `}` are replaced
  with character references (`&lt;`, `&#123;`, `&#125;`) — MDX renders these as
  literal characters and does not treat them as element/expression delimiters.
  `>` and `&` are left alone (harmless in MDX flow). Inline code spans and
  fenced code blocks are detected and passed through verbatim so entities do
  not leak into code. An **unterminated code fence is force-closed** at the end
  of a prose block so it can never swallow the components that follow it.
* **JSX attribute values** (`name="..."`, `summary="..."`, `sourceUrl="..."`):
  `&`, `"`, `<`, `>` -> HTML entities; whitespace collapsed to one line.
* **JS string literals** in `items={[{ name: '...' }]}`: `\` and `'` escaped.
* **`description` fields** (injected by the components via `set:html`):
  HTML-escaped first (so any markup shows literally), then JS-escaped.
* **Signature template literals** (`code={`...`}`): `\`, `` ` `` and `${`
  escaped.
* **YAML frontmatter**: double-quoted scalars with `\`/`"` escaped.

Every per-object render is additionally wrapped in try/except: a single bad
docstring degrades to raw prose or a minimal block instead of failing the run.

Verified: all 240 pages generated from `bloqade-circuit` compile with
`astro build` (including a real docstring with an unterminated code fence).

## inventory.python.json

A JSON array of `{ fqName, kind, url }` for every emitted symbol
(`kind` in `module | class | function | method | property | attribute`),
consumed by the Phase C cross-reference resolver.

```json
[
  { "fqName": "bloqade.task", "kind": "module",
    "url": "/api/python/bloqade/task/#bloqade.task" },
  { "fqName": "bloqade.task.BatchFuture.result", "kind": "method",
    "url": "/api/python/bloqade/task/#bloqade.task.BatchFuture.result" }
]
```

**URL derivation rule.** Let `mount` = `--mount` with surrounding slashes
stripped. Each module `M` is served at `/{mount}/{M.fqName.replace('.', '/')}/`
(packages emit `index.mdx`, plain modules `<name>.mdx`; both resolve to that
trailing-slash directory URL). Every symbol `S` documented on module `M` gets:

```
url = /{mount}/{M.fqName.replace('.', '/')}/#{S.fqName}
```

The module's own entry uses `S = M`. The `#{fqName}` anchor is exactly the
`id={fqName}` the ApiModule/ApiClass/ApiFn headings render, so an `<ApiXref
to="fqName">` resolves to this URL.

## Ported skip rules

`SKIP_KEYWORDS` in `emitter.py` is copied **verbatim** from
`docs/scripts/gen_ref_nav.py` (the mkdocstrings nav generator): a module is
skipped if its repo-relative path contains any keyword. The `__init__ -> index`
mapping and the `skip private _* modules/members` behavior are ported too.

## Known parity gaps vs mkdocstrings

* **Namespace-package subtrees**: modules living under an *implicit* namespace
  package (a directory with **no** `__init__.py`) that is nested inside a
  regular package are not discovered — griffe's loader does not descend into
  them, and even an explicit `load("a.b.c")` raises `KeyError`. In
  `bloqade-circuit` this affects exactly the 6 files under
  `.../analysis/validation/nocloning/` (the `validation/` dir has no
  `__init__.py`). mkdocstrings hits the same class of problem — see the
  `squin/cirq/emit` "missing `__init__.py`" note in `gen_ref_nav.py`. Fix
  upstream by adding `__init__.py`.
* **Overloads**: only the implementation signature is rendered;
  `@typing.overload` variants are not listed.
* **Inherited members**: not shown (matches mkdocstrings' default
  `inherited_members: false`); only members defined directly on a class appear.
* **Re-exports / aliases**: imported names (griffe `Alias`) are intentionally
  skipped, so a re-exporting `__init__.py` documents only what it defines, not
  what it re-imports. This also avoids `AliasResolutionError` on unresolved
  imports.
* **Rich param/return/attribute descriptions**: `description` cells are rendered
  as single-line, HTML-escaped plain text (they flow through the components'
  `set:html`), so embedded markdown/reST in a description is shown literally
  rather than formatted. Body prose keeps full markdown.
* **Cross-reference linking**: docstring cross-refs ARE now rewritten to
  `<ApiXref origin="docstring">` when they resolve to a documented symbol (see
  "Docstring cross-references" above); external / undocumented / unresolvable
  refs degrade to inline code/text rather than becoming unresolved anchors.
* **Type aliases / TypeVars**: surfaced as module/class "Attributes" rows
  (name + value), not as dedicated typedef entries.
