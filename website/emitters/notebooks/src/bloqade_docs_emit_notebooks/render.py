"""Render an (optionally executed) notebook to Starlight-compatible MDX.

The mapping, cell by cell:

* **markdown cell** -> prose MDX. mkdocs-material admonition ``<div>`` blocks are
  converted to Starlight asides; ``<img>`` tags are self-closed and their
  relative ``src`` rewritten to an absolute ``/tutorials/<name>/...`` URL after
  the referenced asset is copied into ``public/``. All remaining free text is
  MDX-escaped (see ``escape.py``); code spans and ``$``-math survive verbatim.
* **code cell** -> a fenced ```python block (input prompt stripped).
* **outputs** (only present when the notebook was executed) -> rendered blocks:
  streams / ``text/plain`` / tracebacks as fenced ``text``; ``image/png`` and
  ``image/svg+xml`` written to ``public/`` and referenced with a self-closed
  ``<img>``; ``text/html`` emitted as an escaped fenced ``html`` source block;
  ``text/latex`` as a ``$$`` math block. ``Out[ ]``/``In[ ]`` prompts are never
  emitted.
"""

from __future__ import annotations

import base64
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .escape import escape_prose, fenced, strip_ansi, yaml_dq

# nbconvert-style cell tags we honour.
_REMOVE_CELL = {"remove_cell", "remove-cell"}
_REMOVE_INPUT = {"remove_input", "remove-input", "hide_input", "hide-input"}
_REMOVE_OUTPUT = {"remove_output", "remove-output"}

# mkdocs-material admonition class -> Starlight aside type.
_ADMONITION_TYPES = {
    "note": "note",
    "info": "note",
    "abstract": "note",
    "summary": "note",
    "example": "note",
    "quote": "note",
    "cite": "note",
    "tip": "tip",
    "hint": "tip",
    "success": "tip",
    "check": "tip",
    "done": "tip",
    "question": "tip",
    "help": "tip",
    "faq": "tip",
    "warning": "caution",
    "caution": "caution",
    "attention": "caution",
    "important": "caution",
    "danger": "danger",
    "error": "danger",
    "bug": "danger",
    "failure": "danger",
    "fail": "danger",
    "missing": "danger",
    "deprecated": "danger",
}

_ADMONITION_RE = re.compile(
    r'<div\s+class="admonition\s+([^"]*?)"\s*>(.*?)</div>',
    re.DOTALL | re.IGNORECASE,
)
_TITLE_RE = re.compile(
    r'<p\s+class="admonition-title"\s*>(.*?)</p>', re.DOTALL | re.IGNORECASE
)
_P_TAG_RE = re.compile(r"</?p\b[^>]*>", re.IGNORECASE)
_IMG_RE = re.compile(r"<img\b([^>]*?)/?>", re.IGNORECASE)
# Markdown image syntax: ![alt](src "optional title"). The legacy sources mix
# HTML <img> tags and markdown images; we resolve+copy both so neither 404s.
_MD_IMG_RE = re.compile(r'!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+"[^"]*")?\s*\)')
# Markdown link syntax: [text](target). ``(?<!\!)`` skips images (handled
# above). Used to rewrite legacy mkdocs cross-reference links (see below).
_MD_LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^)\s]+)\)")
# Legacy mkdocs API-reference path: ``(../)*reference/<pkg>/src/bloqade/<path>``
# with an optional ``#<Name>`` fragment. The mkdocs relative depth and the
# ``reference/…/src/`` scheme are both wrong for the Astro site, which serves
# the Python API at ``/api/latest/python/bloqade/<path>/`` with fully-qualified
# anchor ids (``#bloqade.<dotted.path>.<Name>``).
_REFERENCE_LINK_RE = re.compile(
    r"^(?:\.\./)*reference/[^/]+/src/bloqade/([^#)]+?)/?(?:#([^)]+))?$"
)
# Guide pages that were renamed/relocated in the mkdocs -> Astro migration.
# The sources link to the OLD single-segment relative names; map them to the
# Astro URLs that exist today.
_GUIDE_LINK_RENAMES = {
    "cirq_interop": "/guides/digital/cirq-interop/",
    "dialects_and_kernels": "/guides/digital/dialects/",
}
_RELATIVE_GUIDE_RE = re.compile(r"^(?:\.\./)*([A-Za-z0-9_]+)/?$")
_SRC_RE = re.compile(r'\bsrc\s*=\s*"([^"]*)"', re.IGNORECASE)
_ALT_RE = re.compile(r'\balt\s*=\s*"([^"]*)"', re.IGNORECASE)
# Wrapper tags that are valid JSX as-is and should pass through un-escaped.
_WRAPPER_RE = re.compile(
    r"</?(?:div|picture|center|figure|figcaption)\b[^>]*>", re.IGNORECASE
)
_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$")

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif"}


@dataclass
class RenderContext:
    """Everything the renderer needs to place assets and build URLs."""

    name: str  # tutorial slug
    py_dir: Path  # dir of the source .py (base for relative <img> refs)
    asset_dir: Path  # filesystem dir images are written to (public/<name>)
    asset_base: str  # URL prefix, e.g. "/tutorials/<name>"
    repo_root: Path  # for a broader by-basename asset fallback search
    _protected: list[str] = field(default_factory=list)
    _seen_refs: dict[str, str] = field(default_factory=dict)

    # -- placeholder machinery (keep generated MDX out of the prose escaper) --
    def stash(self, text: str) -> str:
        token = f"\x00NB{len(self._protected)}\x00"
        self._protected.append(text)
        return token

    def unstash(self, text: str) -> str:
        for i, val in enumerate(self._protected):
            text = text.replace(f"\x00NB{i}\x00", val)
        return text

    # -- asset I/O ---------------------------------------------------------
    def _ensure_dir(self) -> None:
        self.asset_dir.mkdir(parents=True, exist_ok=True)

    def write_output_image(self, payload: bytes, cell_idx: int, out_idx: int, ext: str) -> str:
        self._ensure_dir()
        fname = f"cell{cell_idx}_out{out_idx}{ext}"
        (self.asset_dir / fname).write_bytes(payload)
        return f"{self.asset_base}/{fname}"

    def resolve_ref(self, rel_src: str) -> str | None:
        """Resolve a relative markdown/HTML image ref to a copied public URL.

        Tries the path relative to the source .py first, then falls back to a
        by-basename search under the repo (handles the stale ``../../`` paths in
        the legacy mkdocs sources). Returns None if nothing is found.
        """
        if rel_src in self._seen_refs:
            return self._seen_refs[rel_src]
        if re.match(r"^(?:[a-z]+:)?//|^/|^data:", rel_src, re.IGNORECASE):
            return None  # absolute / remote / data URI: leave as-is
        candidate = (self.py_dir / rel_src).resolve()
        found: Path | None = candidate if candidate.is_file() else None
        if found is None:
            basename = Path(rel_src).name
            # Bounded search: the source dir and up to three ancestors.
            base = self.py_dir
            for _ in range(4):
                for hit in base.rglob(basename):
                    if hit.is_file():
                        found = hit
                        break
                if found is not None or base == self.repo_root or base == base.parent:
                    break
                base = base.parent
        if found is None:
            return None
        self._ensure_dir()
        assets = self.asset_dir / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        dest = assets / found.name
        shutil.copyfile(found, dest)
        url = f"{self.asset_base}/assets/{found.name}"
        self._seen_refs[rel_src] = url
        return url


# --------------------------------------------------------------------------- #
# Markdown cell rendering
# --------------------------------------------------------------------------- #


def _aside_type(raw_class: str) -> str:
    for word in raw_class.lower().split():
        if word in _ADMONITION_TYPES:
            return _ADMONITION_TYPES[word]
    return "note"


def _convert_admonition(m: re.Match[str], ctx: RenderContext) -> str:
    aside_type = _aside_type(m.group(1))
    inner = m.group(2)
    title = ""
    tm = _TITLE_RE.search(inner)
    if tm:
        title = re.sub(r"\s+", " ", tm.group(1)).strip()
        inner = inner[: tm.start()] + inner[tm.end() :]
    body = _P_TAG_RE.sub("\n", inner).strip()
    body = escape_prose(body)
    head = f":::{aside_type}"
    if title:
        head += f"[{title}]"
    aside = f"{head}\n{body}\n:::"
    return ctx.stash(aside)


def _convert_img(m: re.Match[str], ctx: RenderContext) -> str:
    attrs = m.group(1)
    sm = _SRC_RE.search(attrs)
    if not sm:
        return ctx.stash("")  # img with no src: drop
    src = sm.group(1)
    url = ctx.resolve_ref(src)
    if url is None:
        # Keep the original ref but self-close so MDX still parses; the image
        # may 404 at runtime, but the build stays green.
        url = src
    am = _ALT_RE.search(attrs)
    alt = am.group(1) if am else Path(src).stem
    return ctx.stash(f'<img src="{url}" alt="{alt}" />')


def _convert_md_img(m: re.Match[str], ctx: RenderContext) -> str:
    alt = m.group(1)
    src = m.group(2)
    url = ctx.resolve_ref(src)
    if url is None:
        # remote/absolute/missing: keep the original markdown image verbatim
        # (stashed so the prose escaper leaves it alone).
        return ctx.stash(m.group(0))
    return ctx.stash(f'<img src="{url}" alt="{alt or Path(src).stem}" />')


def _rewrite_link_target(target: str) -> str:
    """Map a legacy mkdocs cross-reference link to its Astro URL.

    Only known-broken patterns are rewritten (API ``reference/…`` paths and the
    two renamed guide pages); every other target is returned unchanged, so this
    can never break a link that already resolves.
    """
    t = target.strip()
    m = _REFERENCE_LINK_RE.match(t)
    if m:
        path = m.group(1).strip("/")
        url = f"/api/latest/python/bloqade/{path}/"
        frag = m.group(2)
        if frag:
            # Astro API anchors are the fully-qualified object id, e.g.
            # ``bloqade.qasm2.dialects.noise.model.MoveNoiseModelABC``.
            dotted = "bloqade." + path.replace("/", ".")
            url += f"#{dotted}.{frag}"
        return url
    g = _RELATIVE_GUIDE_RE.match(t)
    if g and g.group(1) in _GUIDE_LINK_RENAMES:
        return _GUIDE_LINK_RENAMES[g.group(1)]
    return target


def _rewrite_md_link(m: re.Match[str]) -> str:
    return f"[{m.group(1)}]({_rewrite_link_target(m.group(2))})"


def render_markdown(text: str, ctx: RenderContext) -> str:
    # 1. admonition <div> blocks -> Starlight asides (stashed verbatim).
    text = _ADMONITION_RE.sub(lambda m: _convert_admonition(m, ctx), text)
    # 2a. markdown images ![alt](src) -> self-closed, asset-copied <img> (stashed).
    text = _MD_IMG_RE.sub(lambda m: _convert_md_img(m, ctx), text)
    # 2b. <img> tags -> self-closed, asset-copied, absolute-src (stashed).
    text = _IMG_RE.sub(lambda m: _convert_img(m, ctx), text)
    # 2c. rewrite legacy mkdocs cross-reference links (API reference paths + a
    #     couple of renamed guide pages) to their Astro URLs, so generated
    #     tutorials don't ship dangling links. Only known-broken patterns are
    #     touched; the link text (which may hold a code span) is left verbatim.
    text = _MD_LINK_RE.sub(_rewrite_md_link, text)
    # 3. keep known-safe wrapper tags (div/picture/...) verbatim as JSX.
    text = _WRAPPER_RE.sub(lambda m: ctx.stash(m.group(0)), text)
    # 4. escape everything else, then restore the stashed regions.
    text = escape_prose(text)
    return ctx.unstash(text)


# --------------------------------------------------------------------------- #
# Output rendering
# --------------------------------------------------------------------------- #


def _as_text(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def _render_data(data: dict, cell_idx: int, out_idx: int, ctx: RenderContext) -> str:
    if "image/svg+xml" in data:
        svg = _as_text(data["image/svg+xml"]).encode("utf-8")
        url = ctx.write_output_image(svg, cell_idx, out_idx, ".svg")
        return f'<img src="{url}" alt="output" />'
    if "image/png" in data:
        raw = data["image/png"]
        payload = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
        url = ctx.write_output_image(payload, cell_idx, out_idx, ".png")
        return f'<img src="{url}" alt="output" />'
    if "text/latex" in data:
        latex = _as_text(data["text/latex"]).strip()
        if "$" in latex:
            return latex
        return f"$$\n{latex}\n$$"
    if "text/plain" in data:
        text = strip_ansi(_as_text(data["text/plain"]))
        return fenced(text, "text") if text.strip() else ""
    if "text/html" in data:
        html = _as_text(data["text/html"])
        return fenced(html, "html", 'title="HTML output"') if html.strip() else ""
    if "text/markdown" in data:
        return render_markdown(_as_text(data["text/markdown"]), ctx)
    return ""


def _coalesce_streams(outputs: list) -> list:
    """Merge consecutive ``stream`` outputs of the same name into one.

    Rich/pretty-printers emit stdout token-by-token, so a single ``print`` can
    become hundreds of stream fragments; nbclient does not coalesce them. We do,
    so the whole run renders as one fenced block instead of one-per-token.
    """
    merged: list = []
    for out in outputs or []:
        if (
            out.get("output_type") == "stream"
            and merged
            and merged[-1].get("output_type") == "stream"
            and merged[-1].get("name") == out.get("name")
        ):
            merged[-1]["text"] = _as_text(merged[-1].get("text", "")) + _as_text(
                out.get("text", "")
            )
        else:
            merged.append(dict(out))
    return merged


def render_outputs(cell, cell_idx: int, ctx: RenderContext) -> list[str]:
    parts: list[str] = []
    for oi, out in enumerate(_coalesce_streams(cell.get("outputs", []) or [])):
        otype = out.get("output_type")
        if otype == "stream":
            text = strip_ansi(_as_text(out.get("text", "")))
            if not text.strip():
                continue
            meta = 'title="stderr"' if out.get("name") == "stderr" else ""
            parts.append(fenced(text, "text", meta))
        elif otype in ("execute_result", "display_data"):
            block = _render_data(out.get("data", {}) or {}, cell_idx, oi, ctx)
            if block.strip():
                parts.append(block)
        elif otype == "error":
            tb = strip_ansi("\n".join(out.get("traceback", []) or []))
            if not tb.strip():
                tb = f"{out.get('ename', '')}: {out.get('evalue', '')}"
            parts.append(fenced(tb, "text", 'title="Traceback"'))
    return parts


# --------------------------------------------------------------------------- #
# Notebook -> MDX
# --------------------------------------------------------------------------- #


def _extract_title(nb, fallback: str) -> str:
    """Return the first markdown H1 as the page title and strip it from the body."""
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        lines = cell.source.split("\n")
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            m = _H1_RE.match(line)
            if m:
                del lines[i]
                # drop a now-leading blank line for tidiness
                if i < len(lines) and not lines[i].strip():
                    del lines[i]
                cell.source = "\n".join(lines)
                return m.group(1).strip()
            break  # first non-blank line wasn't an H1 -> no title here
    # ``fallback`` is the slug, which may be namespaced (e.g. ``qasm2/qaoa``);
    # only humanize the final path segment for the title.
    leaf = fallback.rsplit("/", 1)[-1]
    return leaf.replace("_", " ").replace("-", " ").strip().title()


def _humanize(slug: str) -> str:
    return slug.replace("_", " ").replace("-", " ").strip().title()


def render_notebook(nb, ctx: RenderContext, executed: bool) -> str:
    title = _extract_title(nb, ctx.name)
    body: list[str] = ["{/* Generated by bloqade-docs-emit-notebooks. Do not edit by hand. */}"]
    if not executed:
        body.append(
            ":::note[Static render]\n"
            "Outputs are not shown for this tutorial (notebook execution was "
            "disabled at build time).\n:::"
        )
    for ci, cell in enumerate(nb.cells):
        tags = set(cell.get("metadata", {}).get("tags", []) or [])
        if tags & _REMOVE_CELL:
            continue
        if cell.cell_type == "markdown":
            md = render_markdown(cell.source, ctx)
            if md.strip():
                body.append(md)
        elif cell.cell_type == "code":
            src = cell.source.strip("\n")
            if src.strip() and not (tags & _REMOVE_INPUT):
                body.append(fenced(src, "python"))
            if executed and not (tags & _REMOVE_OUTPUT):
                body.extend(render_outputs(cell, ci, ctx))
    front = f"---\ntitle: {yaml_dq(title)}\n---"
    return front + "\n\n" + "\n\n".join(p for p in body if p.strip()) + "\n"
