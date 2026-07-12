#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
r"""Bloqade docs — Rust API emitter (Phase B3).

Turns **rustdoc JSON** into **MDX** pages that use the frozen Bloqade API
component contract (ApiModule / ApiClass / ApiFn / Signature / Params / Returns
/ Raises / Source / ApiXref). One MDX page is emitted per Rust *module*; every
item defined in that module is documented on the module's page. An inventory of
every documented symbol is written to ``<out>/inventory.rust.json``.

This is a standalone, dependency-free script (stdlib only). Run it with ``uv``
(``uv run website/emitters/rust/emit_rust.py ...``) or any Python >= 3.11.

===========================================================================
TOOLCHAIN + COMMAND (how the input JSON is produced)
===========================================================================
rustdoc JSON is an unstable nightly-only format. Generate it with a PINNED
nightly from the crate's workspace root, e.g. for ``bloqade-lanes``::

    cd /path/to/bloqade-lanes
    cargo +nightly rustdoc -p bloqade-lanes-search -- \
        -Z unstable-options --output-format json

The JSON lands at ``target/doc/<lib_name>.json`` (``lib_name`` is the crate's
library name with hyphens turned into underscores, e.g.
``target/doc/bloqade_lanes_search.json``).

Toolchains observed on this machine (2026-07):
  * ``+nightly``            -> cargo 1.97.0-nightly (2026-04-09), rustc
                              1.97.0-nightly (2026-04-17) -> format_version 57
  * ``+nightly-2025-03-13`` -> cargo 1.87.0-nightly (fallback if the primary
                              nightly disappears / changes format_version)

This emitter PINS ``format_version`` and refuses to run against a different
one unless ``--allow-format-version`` is passed, so a toolchain bump can never
silently emit garbage.

You may also let this script run cargo for you (convenience wrapper) with
``--run-cargo --workspace <DIR> --crate <NAME> [--toolchain nightly]``; it then
locates the freshest ``target/doc/*.json`` and parses it.

===========================================================================
RUST ITEM -> COMPONENT MAPPING (stays within the frozen vocabulary)
===========================================================================
  crate / module ................ <ApiModule>            (one page each)
  struct / enum / trait /
    type alias ................... <ApiClass>             (see kind note)
  free function ................. <ApiFn kind="function">
  method / trait method ......... <ApiFn kind="method">
  struct fields ................. <Params title="Fields">
  enum variants ................. <Params title="Variants">
  trait assoc types/consts ...... <Params title="Associated items">
  signatures .................... <Signature lang="rust" code={`...`} />
  doc comments (///, //!) ....... escaped MDX prose (see MDX-SAFETY)
  implemented (non-blanket,
    non-auto) traits ............ ApiClass `bases={[...]}`
  cross references .............. <ApiXref to="fqName" version="<V>" />
                                    (submodule links; version keeps resolution
                                     inside the referring page's own version)
  source file/line .............. <Source href=.../> + sourceUrl frontmatter

KIND-BADGE NOTE: the frozen <ApiClass> always renders the literal word
"class" in its heading (we may not edit components). To convey the true Rust
item kind (struct/enum/trait/type alias) we prefix the ApiClass `summary` with
"Rust <kind>." — e.g. summary="Rust enum. Outcome status of a solve attempt."
This keeps us inside the frozen prop set while still surfacing the kind.

===========================================================================
fqName + URL derivation
===========================================================================
fqName is the canonical Rust path using ``::`` separators and the crate's Rust
identifier (underscored), e.g. ``bloqade_lanes_search::frontier::Frontier``.
Method/field/variant fqNames extend the parent's path
(``...::Config::new``, ``...::ConfigError::duplicate_qubit_id``).

Page files mirror the module tree under ``<out>/``:
  * crate root module  -> ``<crate>/index.mdx``
  * submodule a::b     -> ``<crate>/a/b.mdx``

A page's URL is ``/<mount>/<page-path>/`` (mount defaults to ``api/rust``),
with the trailing ``index`` dropped:
  * ``<crate>/index.mdx``    -> ``/api/rust/<crate>/``
  * ``<crate>/frontier.mdx`` -> ``/api/rust/<crate>/frontier/``

Every documented symbol's inventory URL is ``<page-url>#<fqName>`` — the anchor
equals the fqName because the components set ``id={fqName}`` on their headings
(ApiModule/ApiClass/ApiFn), and <ApiXref to="fqName"> links to ``#fqName``.
Fields / variants / associated items render inside a <Params> table (no heading
of their own), so their inventory URL anchors to their PARENT item's fqName.

inventory.rust.json shape: a JSON list of ``{fqName, kind, url}`` objects, e.g.
  {"fqName": "bloqade_lanes_search::frontier::Frontier",
   "kind": "trait",
   "url": "/api/rust/bloqade_lanes_search/frontier/#bloqade_lanes_search::frontier::Frontier"}

===========================================================================
MDX-SAFETY (a doc comment must never break the Astro build)
===========================================================================
Different sinks need different escaping:
  * PROSE (doc comments rendered as MDX body text): backslash-escape every
    MDX/JSX-significant punctuation char — ``\ { } < > `` and the backtick —
    so a comment can never open a JS expression (`{`), a JSX tag (`<`), or an
    unterminated code span. Doc comments are therefore rendered as literal
    plain text (markdown / intra-doc links are intentionally NOT interpreted;
    build safety beats rich rendering — richer rendering can come later).
  * <Signature code={`...`}> template literal: escape ``\``, backtick and
    ``${`` so the JS template literal parses verbatim.
  * <Params>/<Returns>/<Raises> descriptions (rendered via `set:html`):
    HTML-escape ``& < >`` then JS-string-escape for a single-quoted literal.
  * plain double-quoted JSX attributes (name/fqName/kind/summary/sourceUrl):
    HTML-escape ``& < > "``.
  * YAML frontmatter scalars: double-quote and escape ``\`` and ``"``.

===========================================================================
FALLBACK PLAN (documented; NOT implemented here)
===========================================================================
rustdoc JSON is nightly-only and its ``format_version`` bumps often. If it ever
becomes too unstable for CI, mount plain ``cargo doc`` HTML under the SAME
``/api/rust/**`` URLs instead of this JSON->MDX path:

  1. ``cargo doc --no-deps -p <crate>`` (stable toolchain) -> ``target/doc/``.
  2. Copy ``target/doc/<lib_name>/`` into ``website/public/api/rust/<crate>/``.
     Files in ``public/`` are served verbatim by Astro at the same base path,
     so ``/api/rust/<crate>/index.html`` resolves to rustdoc's own HTML — no
     Starlight shell, but stable and zero-maintenance.
  3. Keep the same mount + directory names so cross-links/inventory URLs still
     resolve; emit ``inventory.rust.json`` from ``target/doc/*/all.html`` (or
     the search index ``search-index.js``) so search/xref keep working.
  4. A single Starlight redirect (``/api/rust`` -> the crate index) preserves
     the sidebar entry.
This is a strictly-worse presentation (loses the shared components/theme) but
is a safe escape hatch; the PRIMARY, preferred path is this JSON->MDX emitter.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import subprocess
import sys
from pathlib import Path

# The rustdoc JSON format this emitter understands. Bump deliberately after
# reviewing schema changes; see --allow-format-version to override at runtime.
EXPECTED_FORMAT_VERSION = 57

# Auto/marker traits and common blanket-impl traits we never surface as `bases`.
# (Synthetic + blanket impls are already filtered structurally; this is a belt.)
_NOISE_TRAITS = {
    "Send", "Sync", "Unpin", "Freeze", "UnsafeUnpin",
    "UnwindSafe", "RefUnwindSafe", "Sized",
}


# --------------------------------------------------------------------------- #
# Escaping helpers (see MDX-SAFETY docstring above).
# --------------------------------------------------------------------------- #
def esc_prose(s: str | None) -> str:
    """Backslash-escape MDX/JSX-significant punctuation for body prose."""
    if not s:
        return ""
    out: list[str] = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch in "`<>{}":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def esc_template(s: str | None) -> str:
    """Escape a string for use inside a JS template literal (`...`)."""
    if not s:
        return ""
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def esc_attr(s: str | None) -> str:
    """HTML-escape a value for a double-quoted JSX string attribute."""
    if not s:
        return ""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def esc_desc_js(s: str | None) -> str:
    """HTML-escape then JS-single-quote-escape a `set:html` description."""
    if not s:
        return ""
    s = html.escape(s, quote=False)  # & < >  -> entities
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    return s.replace("\n", " ").replace("\r", " ")


def yaml_str(s: str | None) -> str:
    """Double-quote + escape a YAML scalar."""
    if s is None:
        s = ""
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")
    return '"' + s + '"'


def first_paragraph(docs: str | None) -> str:
    """First paragraph of a doc comment, collapsed to a single line."""
    if not docs:
        return ""
    para = docs.strip().split("\n\n", 1)[0]
    return " ".join(line.strip() for line in para.splitlines() if line.strip())


def rest_paragraphs(docs: str | None) -> str:
    """Everything after the first paragraph of a doc comment (trimmed)."""
    if not docs:
        return ""
    parts = docs.strip().split("\n\n", 1)
    return parts[1].strip() if len(parts) == 2 else ""


# --------------------------------------------------------------------------- #
# Type / signature rendering (rustdoc `Type` + `GenericArgs` -> Rust source).
# Every branch degrades gracefully to "_" so a novel node can never crash.
# --------------------------------------------------------------------------- #
def render_const(c: object) -> str:
    if isinstance(c, dict):
        return str(c.get("expr") or c.get("value") or "_")
    return "_"


def render_bounds(bounds: list | None) -> str:
    parts: list[str] = []
    for b in bounds or []:
        if not isinstance(b, dict):
            continue
        if "trait_bound" in b:
            parts.append(render_path(b["trait_bound"].get("trait", {})))
        elif "outlives" in b:
            parts.append(str(b["outlives"]))
    return " + ".join(p for p in parts if p)


def render_args(args: object) -> str:
    if not args or not isinstance(args, dict):
        return ""
    if "angle_bracketed" in args:
        ab = args["angle_bracketed"]
        parts: list[str] = []
        for a in ab.get("args", []):
            if isinstance(a, str):
                parts.append("_")
            elif isinstance(a, dict):
                if "lifetime" in a:
                    parts.append(str(a["lifetime"]))
                elif "type" in a:
                    parts.append(render_type(a["type"]))
                elif "const" in a:
                    parts.append(render_const(a["const"]))
        return "<" + ", ".join(parts) + ">" if parts else ""
    if "parenthesized" in args:
        pz = args["parenthesized"]
        ins = ", ".join(render_type(t) for t in pz.get("inputs", []))
        out = pz.get("output")
        s = "(" + ins + ")"
        if out is not None:
            s += " -> " + render_type(out)
        return s
    return ""


def render_path(p: object) -> str:
    if not isinstance(p, dict):
        return "_"
    name = str(p.get("path", "_")).split("::")[-1]
    return name + render_args(p.get("args"))


def render_type(t: object) -> str:
    if t is None:
        return "()"
    if isinstance(t, str):
        return t
    if not isinstance(t, dict):
        return "_"
    key = next(iter(t), None)
    if key is None:
        return "_"
    v = t[key]
    if key == "resolved_path":
        name = str(v.get("path", "_")).split("::")[-1]
        return name + render_args(v.get("args"))
    if key == "generic":
        return v if isinstance(v, str) else "_"
    if key == "primitive":
        return v if isinstance(v, str) else "_"
    if key == "borrowed_ref":
        s = "&"
        if v.get("lifetime"):
            s += str(v["lifetime"]) + " "
        if v.get("is_mutable"):
            s += "mut "
        return s + render_type(v.get("type"))
    if key == "slice":
        return "[" + render_type(v) + "]"
    if key == "array":
        return "[" + render_type(v.get("type")) + "; " + str(v.get("len", "")) + "]"
    if key == "tuple":
        return "(" + ", ".join(render_type(x) for x in v) + ")"
    if key == "impl_trait":
        return "impl " + render_bounds(v)
    if key == "dyn_trait":
        parts = [render_path(pt["trait"]) for pt in v.get("traits", []) if isinstance(pt, dict) and "trait" in pt]
        if v.get("lifetime"):
            parts.append(str(v["lifetime"]))
        return "dyn " + " + ".join(parts)
    if key == "raw_pointer":
        return ("*mut " if v.get("is_mutable") else "*const ") + render_type(v.get("type"))
    if key == "qualified_path":
        self_t = render_type(v.get("self_type"))
        nm = str(v.get("name", ""))
        tr = v.get("trait")
        if tr:
            return "<" + self_t + " as " + render_path(tr) + ">::" + nm
        return self_t + "::" + nm
    if key == "function_pointer":
        sig = v.get("sig", {}) if isinstance(v, dict) else {}
        ins = ", ".join(render_type(t2) for _n, t2 in sig.get("inputs", []))
        out = sig.get("output")
        s = "fn(" + ins + ")"
        if out is not None:
            s += " -> " + render_type(out)
        return s
    return "_"


def render_generics(generics: object) -> str:
    if not isinstance(generics, dict):
        return ""
    out: list[str] = []
    for p in generics.get("params", []):
        if not isinstance(p, dict):
            continue
        k = p.get("kind", {})
        name = p.get("name", "")
        if "lifetime" in k:
            out.append(name)
        elif "type" in k:
            tp = k["type"]
            if tp.get("is_synthetic"):
                continue  # `impl Trait` in argument position; shown inline
            bounds = render_bounds(tp.get("bounds"))
            out.append(name + (": " + bounds if bounds else ""))
        elif "const" in k:
            out.append("const " + name + ": " + render_type(k["const"].get("type")))
    return "<" + ", ".join(out) + ">" if out else ""


def _vis_prefix(item: dict) -> str:
    return "pub " if item.get("visibility") == "public" else ""


def render_self(pt: object) -> str:
    if isinstance(pt, dict) and "borrowed_ref" in pt:
        b = pt["borrowed_ref"]
        s = "&"
        if b.get("lifetime"):
            s += str(b["lifetime"]) + " "
        if b.get("is_mutable"):
            s += "mut "
        return s + "self"
    return "self"


def render_fn_sig(item: dict, name: str) -> str:
    f = item["inner"]["function"]
    header = f.get("header", {}) or {}
    prefix = _vis_prefix(item)
    if header.get("is_const"):
        prefix += "const "
    if header.get("is_async"):
        prefix += "async "
    if header.get("is_unsafe"):
        prefix += "unsafe "
    abi = header.get("abi")
    if abi and abi != "Rust":
        prefix += 'extern "%s" ' % (abi if isinstance(abi, str) else "C")
    gen = render_generics(f.get("generics", {}))
    sig = f.get("sig", {})
    inputs: list[str] = []
    for pname, pt in sig.get("inputs", []):
        inputs.append(render_self(pt) if pname == "self" else pname + ": " + render_type(pt))
    out = sig.get("output")
    ret = "" if out is None else " -> " + render_type(out)
    return "%sfn %s%s(%s)%s" % (prefix, name, gen, ", ".join(inputs), ret)


def render_struct_sig(item: dict, name: str) -> str:
    s = item["inner"]["struct"]
    return "%sstruct %s%s" % (_vis_prefix(item), name, render_generics(s.get("generics", {})))


def render_enum_sig(item: dict, name: str) -> str:
    e = item["inner"]["enum"]
    return "%senum %s%s" % (_vis_prefix(item), name, render_generics(e.get("generics", {})))


def render_trait_sig(item: dict, name: str) -> str:
    tr = item["inner"]["trait"]
    pre = _vis_prefix(item)
    if tr.get("is_unsafe"):
        pre += "unsafe "
    gen = render_generics(tr.get("generics", {}))
    bounds = render_bounds(tr.get("bounds"))
    b = ": " + bounds if bounds else ""
    return "%strait %s%s%s" % (pre, name, gen, b)


def render_type_alias_sig(item: dict, name: str) -> str:
    ta = item["inner"]["type_alias"]
    return "%stype %s%s = %s" % (
        _vis_prefix(item),
        name,
        render_generics(ta.get("generics", {})),
        render_type(ta.get("type")),
    )


def render_constant_sig(item: dict, name: str) -> str:
    c = item["inner"].get("constant", {})
    ty = render_type(c.get("type"))
    return "%sconst %s: %s" % (_vis_prefix(item), name, ty)


# --------------------------------------------------------------------------- #
# Emitter
# --------------------------------------------------------------------------- #
class RustEmitter:
    def __init__(
        self,
        doc: dict,
        *,
        mount: str,
        repo: str | None,
        ref: str | None,
        version: str | None = None,
    ):
        self.index: dict[str, dict] = doc["index"]
        self.paths: dict[str, dict] = doc["paths"]
        self.root_id = str(doc["root"])
        self.crate_version = doc.get("crate_version")
        self.mount = mount.strip("/")
        self.repo = repo
        self.ref = ref
        # The API doc version label (e.g. "dev" / "0.35"). Explicit --version
        # wins; otherwise it is the <V> segment of a `api/<V>/rust` mount. This
        # is the SAME label the pages live under, so xrefs resolve within their
        # own version's inventory map instead of the version-agnostic fallback.
        self.version = version or self._version_from_mount(self.mount)
        self.crate = self.index[self.root_id]["name"]
        self.inventory: list[dict] = []

    @staticmethod
    def _version_from_mount(mount: str) -> str | None:
        """Extract the version label from a `api/<version>/<lang>` mount."""
        segs = mount.strip("/").split("/")
        if len(segs) == 3 and segs[0] == "api":
            return segs[1]
        return None

    # -- lookups -----------------------------------------------------------
    def get(self, item_id) -> dict | None:
        return self.index.get(str(item_id))

    @staticmethod
    def kind_of(item: dict) -> str:
        inner = item.get("inner")
        if isinstance(inner, dict) and inner:
            return next(iter(inner))
        return "unknown"

    def fq_from_paths(self, item_id) -> str | None:
        p = self.paths.get(str(item_id))
        if p and p.get("path"):
            return "::".join(p["path"])
        return None

    # -- source URL --------------------------------------------------------
    def source_url(self, item: dict) -> str | None:
        if not self.repo or not self.ref:
            return None
        span = item.get("span")
        if not span or not span.get("filename"):
            return None
        line = span.get("begin", [None])[0]
        anchor = "#L%d" % line if line else ""
        return "https://github.com/%s/blob/%s/%s%s" % (
            self.repo, self.ref, span["filename"], anchor,
        )

    # -- module discovery --------------------------------------------------
    def modules(self) -> list[str]:
        """All module ids reachable from the crate root (root first)."""
        found: list[str] = []
        seen: set[str] = set()

        def walk(mid: str):
            if mid in seen:
                return
            seen.add(mid)
            found.append(mid)
            item = self.get(mid)
            if not item:
                return
            for cid in item["inner"]["module"].get("items", []):
                child = self.get(cid)
                if child and self.kind_of(child) == "module":
                    walk(str(cid))

        walk(self.root_id)
        return found

    # -- page path / url ---------------------------------------------------
    def module_relpath(self, module_fq: str) -> str:
        """`crate/index.mdx`, `crate/a/b.mdx` from an fqName."""
        segs = module_fq.split("::")
        if len(segs) == 1:
            return "%s/index.mdx" % segs[0]
        return "%s.mdx" % "/".join(segs)

    def page_url(self, module_fq: str) -> str:
        rel = self.module_relpath(module_fq)[:-4]  # drop .mdx
        if rel.endswith("/index"):
            rel = rel[: -len("index")]
        rel = rel.rstrip("/")
        return "/%s/%s/" % (self.mount, rel) if rel else "/%s/" % self.mount

    def add_inventory(self, fq: str, kind: str, page_url: str, anchor_fq: str):
        self.inventory.append(
            {"fqName": fq, "kind": kind, "url": "%s#%s" % (page_url, anchor_fq)}
        )

    # -- impl analysis (methods + implemented traits) ----------------------
    def analyze_impls(self, impl_ids: list) -> tuple[list[dict], list[str]]:
        """Return (inherent+trait methods, implemented-trait display names)."""
        methods: list[dict] = []
        traits: list[str] = []
        seen_methods: set[str] = set()
        for iid in impl_ids or []:
            imp = self.get(iid)
            if not imp or self.kind_of(imp) != "impl":
                continue
            inner = imp["inner"]["impl"]
            if inner.get("is_synthetic") or inner.get("blanket_impl") is not None:
                continue  # auto trait (Send/Sync/...) or foreign blanket impl
            if inner.get("is_negative"):
                continue
            tr = inner.get("trait")
            if tr:
                name = render_path(tr)
                bare = str(tr.get("path", "")).split("::")[-1]
                if bare not in _NOISE_TRAITS and name not in traits:
                    traits.append(name)
                # We list the trait but do not re-document its methods on the
                # type page (they belong to the trait's own entry); only
                # inherent-impl methods are documented as the type's methods.
                continue
            for mid in inner.get("items", []):
                m = self.get(mid)
                if m and self.kind_of(m) == "function" and m.get("name") not in seen_methods:
                    seen_methods.add(m["name"])
                    methods.append(m)
        methods.sort(key=lambda m: m.get("name") or "")
        return methods, traits

    # -- MDX rendering of a single callable --------------------------------
    def render_callable(self, item: dict, fq: str, kind: str, page_url: str) -> str:
        name = item.get("name") or fq.split("::")[-1]
        sig = render_fn_sig(item, name)
        src = self.source_url(item)
        self.add_inventory(fq, "method" if kind == "method" else "function", page_url, fq)
        lines = [
            '<ApiFn name="%s" fqName="%s" kind="%s"%s>' % (
                esc_attr(name), esc_attr(fq), kind,
                ' sourceUrl="%s"' % esc_attr(src) if src else "",
            ),
            "",
            "<Signature lang=\"rust\" code={`%s`} />" % esc_template(sig),
            "",
        ]
        # Parameters table (skip a bare `self` receiver).
        params = self.render_params_table(item)
        if params:
            lines += [params, ""]
        ret = item["inner"]["function"].get("sig", {}).get("output")
        if ret is not None:
            lines += ['<Returns type="%s" />' % esc_attr(render_type(ret)), ""]
        docs = item.get("docs")
        if docs:
            lines += [esc_prose(docs), ""]
        if src:
            lines += ['<Source href="%s" />' % esc_attr(src), ""]
        lines.append("</ApiFn>")
        return "\n".join(lines)

    def render_params_table(self, fn_item: dict) -> str:
        sig = fn_item["inner"]["function"].get("sig", {})
        rows: list[str] = []
        for pname, pt in sig.get("inputs", []):
            if pname == "self":
                continue
            rows.append(
                "    { name: '%s', type: '%s' }," % (esc_desc_js(pname), esc_desc_js(render_type(pt)))
            )
        if not rows:
            return ""
        return "<Params\n  items={[\n%s\n  ]}\n/>" % "\n".join(rows)

    # -- MDX rendering of a type (struct/enum/trait/type alias) ------------
    def render_type_item(self, item: dict, fq: str, page_url: str) -> str:
        kind = self.kind_of(item)
        name = item.get("name") or fq.split("::")[-1]
        docs = item.get("docs")
        src = self.source_url(item)

        if kind == "struct":
            sig = render_struct_sig(item, name)
            inner = item["inner"]["struct"]
            field_ids = self._struct_field_ids(inner)
            methods, traits = self.analyze_impls(inner.get("impls", []))
            member_title, members = "Fields", self._fields_rows(field_ids, fq, page_url)
        elif kind == "enum":
            sig = render_enum_sig(item, name)
            inner = item["inner"]["enum"]
            methods, traits = self.analyze_impls(inner.get("impls", []))
            member_title, members = "Variants", self._variants_rows(inner.get("variants", []), fq, page_url)
        elif kind == "trait":
            sig = render_trait_sig(item, name)
            inner = item["inner"]["trait"]
            traits = []
            methods = [m for m in (self.get(i) for i in inner.get("items", []))
                       if m and self.kind_of(m) == "function"]
            methods.sort(key=lambda m: m.get("name") or "")
            member_title, members = "Associated items", self._assoc_rows(inner.get("items", []), fq, page_url)
        elif kind == "type_alias":
            sig = render_type_alias_sig(item, name)
            methods, traits, member_title, members = [], [], "", []
        else:
            return ""  # unsupported type-like kind

        self.add_inventory(fq, kind, page_url, fq)
        summary = ("Rust %s." % kind.replace("_", " ")) + (
            " " + first_paragraph(docs) if first_paragraph(docs) else ""
        )
        attrs = 'name="%s" fqName="%s" summary="%s"' % (
            esc_attr(name), esc_attr(fq), esc_attr(summary),
        )
        if traits:
            attrs += " bases={[%s]}" % ", ".join("'%s'" % esc_desc_js(t) for t in traits)
        if src:
            attrs += ' sourceUrl="%s"' % esc_attr(src)

        lines = ["<ApiClass %s>" % attrs, ""]
        lines += ["<Signature lang=\"rust\" code={`%s`} />" % esc_template(sig), ""]
        if members:
            lines += [
                "<Params\n  title=\"%s\"\n  items={[\n%s\n  ]}\n/>" % (member_title, "\n".join(members)),
                "",
            ]
        body = rest_paragraphs(docs)
        if body:
            lines += [esc_prose(body), ""]
        if src:
            lines += ['<Source href="%s" />' % esc_attr(src), ""]
        # Methods (inherent impl or trait-required) as nested ApiFn entries.
        for m in methods:
            mname = m.get("name")
            if not mname:
                continue
            mfq = "%s::%s" % (fq, mname)
            lines += [self.render_callable(m, mfq, "method", page_url), ""]
        lines.append("</ApiClass>")
        return "\n".join(lines)

    def _struct_field_ids(self, struct_inner: dict) -> list:
        kind = struct_inner.get("kind", {})
        if isinstance(kind, dict) and "plain" in kind:
            return kind["plain"].get("fields", [])
        if isinstance(kind, dict) and "tuple" in kind:
            return [f for f in kind["tuple"] if f is not None]
        return []

    def _fields_rows(self, field_ids: list, parent_fq: str, page_url: str) -> list[str]:
        rows: list[str] = []
        for fid in field_ids:
            f = self.get(fid)
            if not f or self.kind_of(f) != "struct_field":
                continue
            fname = f.get("name") or "_"
            ftype = render_type(f["inner"]["struct_field"])
            desc = first_paragraph(f.get("docs"))
            self.add_inventory("%s::%s" % (parent_fq, fname), "field", page_url, parent_fq)
            rows.append(
                "    { name: '%s', type: '%s', description: '%s' }," % (
                    esc_desc_js(fname), esc_desc_js(ftype), esc_desc_js(desc),
                )
            )
        return rows

    def _variants_rows(self, variant_ids: list, parent_fq: str, page_url: str) -> list[str]:
        rows: list[str] = []
        for vid in variant_ids:
            v = self.get(vid)
            if not v or self.kind_of(v) != "variant":
                continue
            vname = v.get("name") or "_"
            vk = v["inner"]["variant"].get("kind")
            payload = ""
            if isinstance(vk, dict):
                if "tuple" in vk:
                    parts = [render_type(self.get(i)["inner"]["struct_field"])
                             for i in vk["tuple"] if i is not None and self.get(i)]
                    payload = "(" + ", ".join(parts) + ")"
                elif "struct" in vk:
                    payload = "{ ... }"
            desc = first_paragraph(v.get("docs"))
            self.add_inventory("%s::%s" % (parent_fq, vname), "variant", page_url, parent_fq)
            rows.append(
                "    { name: '%s', type: '%s', description: '%s' }," % (
                    esc_desc_js(vname), esc_desc_js(payload), esc_desc_js(desc),
                )
            )
        return rows

    def _assoc_rows(self, item_ids: list, parent_fq: str, page_url: str) -> list[str]:
        rows: list[str] = []
        for iid in item_ids:
            it = self.get(iid)
            if not it:
                continue
            k = self.kind_of(it)
            if k == "assoc_type":
                ty = "type " + (it.get("name") or "_")
                bounds = render_bounds(it["inner"]["assoc_type"].get("bounds"))
                ty += (": " + bounds) if bounds else ""
                label = "associated type"
            elif k == "assoc_const":
                ty = render_type(it["inner"]["assoc_const"].get("type"))
                label = "associated const"
            else:
                continue
            name = it.get("name") or "_"
            desc = "%s. %s" % (label, first_paragraph(it.get("docs"))) if it.get("docs") else label
            self.add_inventory("%s::%s" % (parent_fq, name), k, page_url, parent_fq)
            rows.append(
                "    { name: '%s', type: '%s', description: '%s' }," % (
                    esc_desc_js(name), esc_desc_js(ty), esc_desc_js(desc),
                )
            )
        return rows

    # -- page assembly -----------------------------------------------------
    def render_module_page(self, module_id: str) -> tuple[str, str]:
        item = self.get(module_id)
        module_fq = self.fq_from_paths(module_id) or item["name"]
        page_url = self.page_url(module_fq)
        self.add_inventory(module_fq, "module", page_url, module_fq)

        child_ids = item["inner"]["module"].get("items", [])
        buckets: dict[str, list[tuple[str, dict]]] = {
            "struct": [], "enum": [], "trait": [], "type_alias": [],
            "constant": [], "function": [],
        }
        submodules: list[tuple[str, str]] = []
        for cid in child_ids:
            child = self.get(cid)
            if not child:
                continue
            k = self.kind_of(child)
            if k == "module":
                cfq = self.fq_from_paths(cid) or ("%s::%s" % (module_fq, child["name"]))
                submodules.append((child["name"], cfq))
            elif k in buckets:
                cfq = self.fq_from_paths(cid) or ("%s::%s" % (module_fq, child["name"]))
                buckets[k].append((cfq, child))

        for b in buckets.values():
            b.sort(key=lambda t: t[0])
        submodules.sort()

        # frontmatter
        docs = item.get("docs")
        title = module_fq if len(module_fq.split("::")) > 1 else "%s (crate)" % module_fq
        fm = [
            "---",
            "title: %s" % yaml_str(title),
            "description: %s" % yaml_str(first_paragraph(docs) or ("Rust API for %s" % module_fq)),
            "language: rust",
            "fqName: %s" % yaml_str(module_fq),
        ]
        api_version = self.version or self.crate_version
        if api_version:
            fm.append("apiVersion: %s" % yaml_str(api_version))
        if self.repo:
            fm.append("sourceRepo: %s" % yaml_str(self.repo))
        if self.ref:
            fm.append("sourceRef: %s" % yaml_str(self.ref))
        src = self.source_url(item)
        if src:
            fm.append("sourceUrl: %s" % yaml_str(src))
        fm.append("---")

        body = [
            "{/* GENERATED by website/emitters/rust/emit_rust.py — do not edit by hand. */}",
            "",
            '<ApiModule name="%s" fqName="%s"%s>' % (
                esc_attr(module_fq.split("::")[-1]),
                esc_attr(module_fq),
                ' summary="%s"' % esc_attr(first_paragraph(docs)) if first_paragraph(docs) else "",
            ),
            "",
        ]
        rest = rest_paragraphs(docs)
        if rest:
            body += [esc_prose(rest), ""]
        if submodules:
            body += ["**Modules**", ""]
            # Pass the referring page's version so the xref resolves within the
            # SAME version's inventory map (byVersion[<V>]) rather than falling
            # back to the version-agnostic map — which can hold a stale/other
            # version's (or versionless) URL.
            ver_attr = ' version="%s"' % esc_attr(self.version) if self.version else ""
            for sname, sfq in submodules:
                body += [
                    '- <ApiXref to="%s" label="%s"%s />'
                    % (esc_attr(sfq), esc_attr(sname), ver_attr)
                ]
            body += [""]

        for k in ("struct", "enum", "trait", "type_alias", "constant"):
            for cfq, child in buckets[k]:
                rendered = self.render_type_item(child, cfq, page_url) if k != "constant" else None
                if k == "constant":
                    rendered = self._render_constant(child, cfq, page_url)
                if rendered:
                    body += [rendered, ""]
        for cfq, child in buckets["function"]:
            body += [self.render_callable(child, cfq, "function", page_url), ""]

        body.append("</ApiModule>")
        return self.module_relpath(module_fq), "\n".join(fm) + "\n\n" + "\n".join(body) + "\n"

    def _render_constant(self, item: dict, fq: str, page_url: str) -> str:
        name = item.get("name") or fq.split("::")[-1]
        self.add_inventory(fq, "constant", page_url, fq)
        sig = render_constant_sig(item, name)
        docs = item.get("docs")
        lines = [
            '<ApiFn name="%s" fqName="%s" kind="property">' % (esc_attr(name), esc_attr(fq)),
            "",
            "<Signature lang=\"rust\" code={`%s`} />" % esc_template(sig),
            "",
        ]
        if docs:
            lines += [esc_prose(docs), ""]
        lines.append("</ApiFn>")
        return "\n".join(lines)

    # -- top level ---------------------------------------------------------
    def emit(self, out_dir: Path) -> dict:
        pages = 0
        for mid in self.modules():
            rel, content = self.render_module_page(mid)
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            pages += 1
        inv_path = out_dir / "inventory.rust.json"
        inv_path.write_text(json.dumps(self.inventory, indent=2) + "\n", encoding="utf-8")
        return {"pages": pages, "symbols": len(self.inventory), "inventory": str(inv_path)}


# --------------------------------------------------------------------------- #
# cargo runner + CLI
# --------------------------------------------------------------------------- #
def run_cargo(workspace: Path, crate: str, toolchain: str) -> Path:
    cmd = [
        "cargo", "+%s" % toolchain, "rustdoc", "-p", crate,
        "--", "-Z", "unstable-options", "--output-format", "json",
    ]
    print("[rust-emitter] $ %s (cwd=%s)" % (" ".join(cmd), workspace), file=sys.stderr)
    proc = subprocess.run(cmd, cwd=workspace)
    if proc.returncode != 0:
        raise SystemExit("[rust-emitter] cargo rustdoc failed (exit %d)" % proc.returncode)
    doc_dir = workspace / "target" / "doc"
    candidates = sorted(doc_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit("[rust-emitter] no JSON found in %s" % doc_dir)
    print("[rust-emitter] using %s" % candidates[0], file=sys.stderr)
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="rustdoc JSON -> MDX (Bloqade API contract).")
    ap.add_argument("--json", type=Path, help="Path to rustdoc JSON (target/doc/<crate>.json).")
    ap.add_argument("--out", type=Path, default=here / "_out", help="Output directory (default: ./_out).")
    ap.add_argument("--mount", default="api/rust", help="Docs mount for URLs (default: api/rust).")
    ap.add_argument(
        "--version", dest="version", default=None,
        help="API doc version label (e.g. 'dev' or '0.35'). "
             "Defaults to the <V> segment of a 'api/<V>/rust' mount.",
    )
    ap.add_argument("--repo", default="QuEraComputing/bloqade-lanes", help="owner/name for source links.")
    ap.add_argument("--ref", dest="ref", default="main", help="git ref for source links (default: main).")
    ap.add_argument("--no-source", action="store_true", help="Do not emit sourceUrl/Source links.")
    ap.add_argument("--allow-format-version", action="store_true",
                    help="Proceed even if format_version != %d." % EXPECTED_FORMAT_VERSION)
    # cargo runner
    ap.add_argument("--run-cargo", action="store_true", help="Run cargo rustdoc to produce the JSON.")
    ap.add_argument("--workspace", type=Path, help="Cargo workspace dir (with --run-cargo).")
    ap.add_argument("--crate", help="Crate name to document (with --run-cargo).")
    ap.add_argument("--toolchain", default="nightly", help="Nightly toolchain (default: nightly).")
    args = ap.parse_args(argv)

    if args.run_cargo:
        if not args.workspace or not args.crate:
            ap.error("--run-cargo requires --workspace and --crate")
        json_path = run_cargo(args.workspace, args.crate, args.toolchain)
    elif args.json:
        json_path = args.json
    else:
        ap.error("provide --json PATH or --run-cargo --workspace DIR --crate NAME")

    if not json_path.is_file():
        raise SystemExit("[rust-emitter] JSON not found: %s" % json_path)
    doc = json.loads(json_path.read_text(encoding="utf-8"))

    fv = doc.get("format_version")
    if fv != EXPECTED_FORMAT_VERSION:
        msg = ("[rust-emitter] rustdoc format_version=%s but this emitter targets %d. "
               "Regenerate with the pinned nightly, or pass --allow-format-version "
               "after reviewing schema changes." % (fv, EXPECTED_FORMAT_VERSION))
        if not args.allow_format_version:
            raise SystemExit(msg)
        print("[rust-emitter][warn] " + msg, file=sys.stderr)

    repo = None if args.no_source else args.repo
    ref = None if args.no_source else args.ref
    emitter = RustEmitter(doc, mount=args.mount, repo=repo, ref=ref, version=args.version)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = emitter.emit(args.out)
    print("[rust-emitter] crate=%s format_version=%s -> %d pages, %d symbols"
          % (emitter.crate, fv, stats["pages"], stats["symbols"]), file=sys.stderr)
    print("[rust-emitter] output: %s" % args.out, file=sys.stderr)
    print("[rust-emitter] inventory: %s" % stats["inventory"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
