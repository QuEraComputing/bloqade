"""Docstring cross-reference detection, resolution, and safe emission.

Docstrings across the Bloqade corpus (and downstream repos) mix four
cross-reference conventions, each with its own syntax. This module detects them
in *parsed prose*, resolves each target to a fully-qualified name using griffe's
own scope resolution, and defers the final markup decision until the emitter's
full inventory is known.

Conventions handled (all detected on prose text, so they work regardless of the
``--docstring-style`` griffe uses to parse the docstring):

* **reStructuredText / Sphinx object roles** — ``:class:`` , ``:func:`` ,
  ``:meth:`` , ``:obj:`` , ``:attr:`` , ``:exc:`` , ``:mod:`` , ``:data:`` ,
  ``:const:`` (plus the ``:py:*:`` prefixed variants and ``:function:`` /
  ``:method:`` aliases), including the ``~Target`` short-display form and the
  explicit-title ``Text <target>`` form. Roles inside ``.. seealso::`` blocks are
  ordinary inline roles and are picked up the same way.
* **Google style** — Google sections are parsed by griffe; the prose inside them
  uses either Sphinx roles (Napoleon) or Markdown autorefs. Both are handled
  because detection runs on the resulting prose text.
* **NumPy / SciPy style** — RST roles in prose *plus* the NumPy ``See Also``
  section. griffe (numpy parser) surfaces ``See Also`` as an admonition whose
  ``value.kind == "see-also"``; :func:`render_see_also` linkifies the bare
  (optionally role-prefixed / comma-separated) object names it lists.
* **Markdown / mkdocstrings autorefs** — ``[label][target]`` , ``[target][]`` ,
  ``[`target`][]`` (backtick target), and the bare ``[`Target`]`` form. Plain
  ``[text](url)`` links are left to the escape pipeline (external kept, broken
  intra-doc relative links degraded to text).

Safe-emit rule: a detected reference becomes ``<ApiXref origin="docstring">``
ONLY when its resolved fully-qualified name is a symbol this emitter documents
(present in the inventory it builds). Everything else — external packages such
as ``numpy``/``kirin`` and anything unresolvable — degrades to inline code/text,
so it can never become an unresolved ``<ApiXref>`` anchor (which would fail
``XREF_STRICT=1``).
"""

from __future__ import annotations

import re
from typing import Any

from . import escape

# --------------------------------------------------------------------------- #
# Placeholder plumbing
# --------------------------------------------------------------------------- #
#
# Detection runs BEFORE MDX escaping (it must consume the backticks that belong
# to a role target), but the final markup decision (ApiXref vs. text) needs the
# COMPLETE documented-symbol set, which only exists after every page has been
# rendered. So each detected reference is replaced by an opaque placeholder that
# survives escaping unharmed; the emitter substitutes the real markup once the
# inventory is complete (see ``XrefCollector.resolve``).
#
# NUL is used as the sentinel: it can never appear in Python source (and thus in
# a docstring), it is not an MDX-significant character, and it is left untouched
# by every escape helper.

_SENTINEL = "\x00"
_PLACEHOLDER_RE = re.compile(r"\x00XREF(\d+)\x00")


class XrefCollector:
    """Registers detected references and later renders them to final markup."""

    def __init__(self) -> None:
        self._refs: list[tuple[tuple[str, ...], str]] = []

    def add(self, candidates: list[str], label: str) -> str:
        """Register a reference and return its placeholder token.

        ``candidates`` is an ordered list of fully-qualified names to try (most
        specific first); ``label`` is the display text.
        """
        idx = len(self._refs)
        self._refs.append((tuple(candidates), label))
        return f"{_SENTINEL}XREF{idx}{_SENTINEL}"

    def resolve(self, text: str, documented: set[str]) -> str:
        """Replace every placeholder in ``text`` with its final MDX markup.

        A reference resolves to ``<ApiXref origin="docstring">`` iff one of its
        candidates is in ``documented`` (the set of fully-qualified names this
        emitter records in its inventory). Otherwise it degrades to inline
        code/text so it never becomes an unresolved cross-reference.
        """

        def _repl(match: "re.Match[str]") -> str:
            candidates, label = self._refs[int(match.group(1))]
            for cand in candidates:
                if cand in documented:
                    return (
                        f'<ApiXref to="{escape.attr(cand)}" '
                        f'origin="docstring" label="{escape.attr(label)}" />'
                    )
            return _code_fallback(label)

        return _PLACEHOLDER_RE.sub(_repl, text)


def _code_fallback(label: str) -> str:
    """Render an unresolved/external reference as safe inline code or text.

    Identifier-like labels become an inline code span; labels with whitespace
    (or a stray backtick) become plain text with MDX-hostile characters
    neutralized the same way :func:`escape.prose` does.
    """
    label = label.strip()
    if not label:
        return ""
    if "`" in label or any(ch.isspace() for ch in label):
        return _escape_text(label)
    return f"`{label}`"


def _escape_text(text: str) -> str:
    """Neutralize `<`, `{`, `}` (matching escape.prose) for post-escape insertion."""
    return text.replace("<", "&lt;").replace("{", "&#123;").replace("}", "&#125;")


# --------------------------------------------------------------------------- #
# Target resolution (griffe scope)
# --------------------------------------------------------------------------- #

# Cross-reference role names (the last component of a possibly ``py:``-prefixed
# role). Field roles like ``:param:`` / ``:returns:`` / ``:math:`` are NOT here,
# so they are never treated as object cross-references.
_ROLE_NAMES = frozenset(
    {
        "class",
        "func",
        "function",
        "meth",
        "method",
        "obj",
        "attr",
        "exc",
        "mod",
        "data",
        "const",
        "property",
    }
)


def resolve_candidates(obj: Any, target: str) -> list[str]:
    """Resolve a docstring target to an ordered list of candidate fq-names.

    Uses griffe's own :meth:`Object.resolve` on the leading name segment (which
    walks the object's members, its module namespace, and griffe-tracked
    imports/aliases), then re-attaches the dotted tail. The raw target is always
    included as a fallback so absolute references (whose leading package is not
    in scope) still work. Membership against the documented set decides which
    candidate — if any — actually becomes a link.
    """
    target = (target or "").strip()
    if target.startswith("~"):
        target = target[1:].strip()
    if not target:
        return []

    candidates: list[str] = []
    head, _, tail = target.partition(".")
    try:
        base = obj.resolve(head)
    except Exception:
        base = None
    if base:
        resolved = f"{base}.{tail}" if tail else base
        candidates.append(resolved)
    if target not in candidates:
        candidates.append(target)
    return candidates


def make_ref(collector: XrefCollector, obj: Any, target: str, label: str) -> str:
    """Resolve ``target`` in ``obj``'s scope and register a placeholder."""
    candidates = resolve_candidates(obj, target)
    label = (label or target).strip()
    return collector.add(candidates, label)


# --------------------------------------------------------------------------- #
# Inline detection (roles + markdown autorefs)
# --------------------------------------------------------------------------- #

# ``:role:`content``` or ``:py:role:`content``` (content parsed separately).
_ROLE_RE = re.compile(r":(?P<role>[A-Za-z]+(?::[A-Za-z]+)?):`(?P<content>[^`\n]+)`")
# Explicit autoref ``[label][target]`` (target may be empty).
_AUTOREF_RE = re.compile(r"\[(?P<label>[^\[\]]+)\]\[(?P<target>[^\[\]]*)\]")
# Bare backtick autoref ``[`Target`]`` not followed by another ``[`` or ``(``.
_BARE_RE = re.compile(r"\[`(?P<target>[^`\]]+)`\](?![\[(])")
# Explicit-title role content: ``Text <target>``.
_TITLE_RE = re.compile(r"^(?P<label>.*?)\s*<(?P<target>[^<>]+)>$")
# A dotted qualified identifier (``pkg.mod.Name``): the ONLY bare code span
# treated as an mkdocstrings backtick autoref. Requiring a dot avoids linkifying
# every single-word code span; the inventory-membership check then guarantees a
# non-reference code span is restored verbatim (it degrades back to code).
_DOTTED_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")

_FENCE_OPEN_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")


def _known_role(role: str) -> bool:
    return role.split(":")[-1] in _ROLE_NAMES


def _strip_ticks(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        return text[1:-1].strip()
    return text


def _parse_role_content(content: str) -> tuple[str, str]:
    """Parse a role's backtick content into ``(target, label)``.

    Handles the ``~Target`` short-display form and the explicit-title
    ``Text <target>`` form; otherwise the target is shown as written.
    """
    content = content.strip()
    m = _TITLE_RE.match(content)
    if m:
        target = m.group("target").strip()
        label = m.group("label").strip()
        return target, (label or target)
    if content.startswith("~"):
        target = content[1:].strip()
        return target, target.rsplit(".", 1)[-1]
    return content, content


def _parse_autoref(label_raw: str, target_raw: str) -> tuple[str, str]:
    """Parse a markdown autoref into ``(target, label)``."""
    label = _strip_ticks(label_raw)
    target = target_raw.strip()
    if not target:
        target = label_raw.strip()
    target = _strip_ticks(target)
    display = label
    if target.startswith("~"):
        bare = target[1:].strip()
        target = bare
        if display == label_raw.strip():
            display = bare.rsplit(".", 1)[-1]
    return target, (display or target)


def _find_code_span_end(line: str, start: int) -> int:
    """Return the index just past a backtick code span starting at ``start``.

    Mirrors ``escape._escape_inline``: an unterminated run yields just past the
    opening backticks so scanning continues.
    """
    n = len(line)
    j = start
    while j < n and line[j] == "`":
        j += 1
    run = j - start
    k = j
    while k < n:
        if line[k] == "`":
            m = k
            while m < n and line[m] == "`":
                m += 1
            if (m - k) == run:
                return m
            k = m
            continue
        k += 1
    return j  # unterminated: emit the opening backticks literally


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _process_inline(line: str, obj: Any, collector: XrefCollector) -> str:
    """Detect roles/autorefs on a single (non-fenced) prose line.

    Genuine inline code spans are passed through verbatim so references are
    never fabricated from code; a role's own backticks are consumed as part of
    the role.
    """
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]

        # Sphinx/RST object role (must sit at a word boundary).
        if ch == ":" and (i == 0 or not _is_word_char(line[i - 1])):
            m = _ROLE_RE.match(line, i)
            if m and _known_role(m.group("role")):
                target, label = _parse_role_content(m.group("content"))
                out.append(make_ref(collector, obj, target, label))
                i = m.end()
                continue

        # Inline code span.
        if ch == "`":
            end = _find_code_span_end(line, i)
            span = line[i:end]
            # mkdocstrings backtick autoref: a single-backtick span wrapping a
            # dotted qualified identifier is treated as a bare reference. Its
            # internal syntax is NOT inspected (no roles/brackets scanned inside
            # a code span); only the whole span is considered. Anything that is
            # not a documented symbol degrades right back to this code span.
            if (
                len(span) >= 3
                and span[1] != "`"
                and span[-1] == "`"
                and _DOTTED_ID_RE.match(span[1:-1])
            ):
                inner = span[1:-1]
                out.append(make_ref(collector, obj, inner, inner))
                i = end
                continue
            out.append(span)
            i = end
            continue

        # Markdown autorefs.
        if ch == "[":
            m = _AUTOREF_RE.match(line, i)
            if m:
                target, label = _parse_autoref(m.group("label"), m.group("target"))
                if target:
                    out.append(make_ref(collector, obj, target, label))
                    i = m.end()
                    continue
            m = _BARE_RE.match(line, i)
            if m:
                raw = m.group("target").strip()
                if raw.startswith("~"):
                    bare = raw[1:].strip()
                    out.append(make_ref(collector, obj, bare, bare.rsplit(".", 1)[-1]))
                else:
                    out.append(make_ref(collector, obj, raw, raw))
                i = m.end()
                continue

        out.append(ch)
        i += 1
    return "".join(out)


def linkify(text: str, obj: Any, collector: XrefCollector) -> str:
    """Replace docstring cross-references in ``text`` with placeholder tokens.

    Fenced code blocks are passed through untouched; inline detection runs on
    the remaining lines. The returned text is safe to feed to
    :func:`escape.prose`; the emitter later swaps the placeholders for final
    markup once the inventory is complete.
    """
    if not text or _SENTINEL in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in lines:
        if in_fence:
            out.append(line)
            stripped = line.strip()
            if stripped and stripped[0] == fence_char:
                run = len(stripped) - len(stripped.lstrip(fence_char))
                if run >= fence_len:
                    in_fence = False
            continue
        m = _FENCE_OPEN_RE.match(line)
        if m:
            marker = m.group(2)
            in_fence = True
            fence_char = marker[0]
            fence_len = len(marker)
            out.append(line)
            continue
        out.append(_process_inline(line, obj, collector))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# NumPy "See Also" section
# --------------------------------------------------------------------------- #

_SEEALSO_NAME_RE = re.compile(
    r"^(?::(?P<role>[A-Za-z]+(?::[A-Za-z]+)?):)?`?(?P<target>[^`]+?)`?$"
)


def _parse_seealso_name(token: str) -> tuple[str, str]:
    """Parse one NumPy See-Also object token into ``(target, label)``.

    Accepts a bare dotted name, a backtick-wrapped name, or a role-prefixed name
    (``:func:`x```), plus the ``~short`` display form.
    """
    token = token.strip()
    m = _SEEALSO_NAME_RE.match(token)
    target = m.group("target").strip() if m else token
    if target.startswith("~"):
        bare = target[1:].strip()
        return bare, bare.rsplit(".", 1)[-1]
    return target, target


def render_see_also(
    contents: str,
    obj: Any,
    collector: XrefCollector,
) -> list[str]:
    """Render a NumPy ``See Also`` admonition body as a linkified list.

    Each source line is ``names [: description]`` where ``names`` may be a
    comma-separated list of (optionally role-prefixed) object names. Names are
    linkified via the collector; the description (if any) is escaped prose.
    """
    lines = ["**See Also**", ""]
    for raw in (contents or "").split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if " : " in raw:
            names_part, desc = raw.split(" : ", 1)
        elif raw.endswith(" :"):
            names_part, desc = raw[:-2], ""
        else:
            names_part, desc = raw, ""
        rendered: list[str] = []
        for name in names_part.split(","):
            name = name.strip()
            if not name:
                continue
            target, label = _parse_seealso_name(name)
            rendered.append(make_ref(collector, obj, target, label))
        if not rendered:
            continue
        line = ", ".join(rendered)
        desc = desc.strip()
        if desc:
            line += " — " + escape.prose(desc)
        lines.append(line)
        lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Per-object linker (used by render.py)
# --------------------------------------------------------------------------- #


class Linker:
    """A prose linkifier bound to a single griffe object + shared collector.

    ``linker(text)`` runs inline cross-reference detection over prose text;
    ``linker.see_also(contents)`` renders a NumPy See-Also body. Both resolve
    targets against the bound object's griffe scope and register placeholders on
    the shared collector.
    """

    __slots__ = ("_obj", "_collector")

    def __init__(self, obj: Any, collector: XrefCollector) -> None:
        self._obj = obj
        self._collector = collector

    def __call__(self, text: str) -> str:
        return linkify(text, self._obj, self._collector)

    def see_also(self, contents: str) -> list[str]:
        return render_see_also(contents, self._obj, self._collector)
