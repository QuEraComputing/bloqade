"""MDX-safety escaping helpers.

Generated docstrings can contain characters that are hostile to the MDX
parser (``{`` opens a JS expression, ``<`` opens a JSX element, backticks open
code spans, etc.). The MDX build MUST NEVER break on a docstring, so every
string that reaches the ``.mdx`` output flows through one of the helpers here.

There are four distinct escaping *contexts*, each with different rules:

1. ``yaml_dq``  - a YAML double-quoted scalar (frontmatter values).
2. ``attr``     - a JSX/HTML double-quoted attribute value
                  (``name="..."``, ``summary="..."``, ...).
3. ``js_str``   - the *content* of a single-quoted JS string literal
                  (the ``items={[{ name: '...' }]}`` arrays).
4. ``template`` - the *content* of a backtick template literal
                  (``<Signature code={`...`} />``).
5. ``prose``    - free-form MDX flow content (summaries, long descriptions).

Plus ``html_text`` which renders arbitrary text as *literal* HTML text, used
for the ``description`` fields that the components inject via ``set:html``.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Attribute / string-literal contexts
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")


def _collapse_ws(text: str) -> str:
    """Collapse all runs of whitespace (incl. newlines) to a single space."""
    return _WS_RE.sub(" ", text).strip()


def yaml_dq(value: str) -> str:
    """Return ``value`` as a YAML double-quoted scalar (including the quotes).

    Only backslash and double-quote need escaping inside a YAML double-quoted
    scalar; newlines are collapsed so the scalar stays on one line.
    """
    value = _collapse_ws(value)
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def attr(value: str) -> str:
    """Escape ``value`` for use inside a JSX double-quoted attribute value.

    Returns the escaped *content* (no surrounding quotes). Braces are literal
    inside a quoted JSX attribute, so only the HTML-significant characters and
    the closing quote need escaping.
    """
    value = _collapse_ws(value)
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def js_str(value: str) -> str:
    """Escape ``value`` for the content of a single-quoted JS string literal.

    Returns the escaped *content* (no surrounding quotes).
    """
    value = _collapse_ws(value)
    return value.replace("\\", "\\\\").replace("'", "\\'")


def html_text(value: str) -> str:
    """Escape ``value`` so it renders as *literal* text through ``set:html``.

    The ``description`` fields on Params/Returns/Raises are injected by the
    Astro components with ``set:html={...}`` (i.e. treated as raw HTML). We
    HTML-escape here so any ``<``/``&`` in a docstring shows verbatim instead
    of being interpreted as markup. The result is later wrapped by ``js_str``.
    """
    value = _collapse_ws(value)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def desc_literal(value: str) -> str:
    """Full pipeline for a `description` field: HTML-escape then JS-escape.

    Produces the escaped *content* of a single-quoted JS string whose value,
    when handed to ``set:html``, renders as the literal source text.
    """
    return js_str(html_text(value))


def template(value: str) -> str:
    """Escape ``value`` for the content of a backtick template literal.

    Returns the escaped *content* (no surrounding backticks). Order matters:
    escape backslashes first, then backticks, then the ``${`` interpolation
    opener.
    """
    value = value.replace("\\", "\\\\")
    value = value.replace("`", "\\`")
    value = value.replace("${", "\\${")
    return value


# --------------------------------------------------------------------------- #
# Prose (MDX flow content)
# --------------------------------------------------------------------------- #
#
# Strategy: escape the MDX-hostile characters `<`, `{`, `}` using HTML/numeric
# character references, which MDX passes through as literal characters and does
# NOT treat as expression/element delimiters. To keep intentional code spans
# and fenced code blocks rendering as code (where entity references would show
# up verbatim), we tokenize the text and only escape the NON-code segments.
#
# `>` and `&` are deliberately left alone in prose: neither breaks an MDX build,
# and leaving them preserves markdown blockquotes, entities and comparisons.

_FENCE_OPEN_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")


def _escape_specials(segment: str) -> str:
    """Escape `<`, `{`, `}` in a non-code prose segment."""
    return (
        segment.replace("<", "&lt;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
    )


def _escape_inline(line: str) -> str:
    """Escape a single prose line, leaving inline code spans verbatim."""
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == "`":
            # Measure the opening backtick run.
            j = i
            while j < n and line[j] == "`":
                j += 1
            run = j - i
            # Find a closing run of exactly the same length.
            k = j
            close = -1
            while k < n:
                if line[k] == "`":
                    m = k
                    while m < n and line[m] == "`":
                        m += 1
                    if (m - k) == run:
                        close = k
                        break
                    k = m
                    continue
                k += 1
            if close != -1:
                # Verbatim code span (backticks protect it from MDX + markdown).
                out.append(line[i : close + run])
                i = close + run
                continue
            # Unterminated run: emit the backticks literally, keep scanning.
            out.append(line[i:j])
            i = j
            continue
        # Ordinary text up to the next backtick.
        j = i
        while j < n and line[j] != "`":
            j += 1
        out.append(_escape_specials(line[i:j]))
        i = j
    return "".join(out)


def prose(text: str) -> str:
    """Escape free-form docstring text so it is safe as MDX flow content.

    Fenced code blocks and inline code spans are preserved verbatim; only the
    surrounding prose has `<`, `{`, `}` neutralized.
    """
    if not text:
        return ""
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
            out.append(line)  # the opening fence line itself is safe verbatim
            continue
        out.append(_escape_inline(line))
    if in_fence:
        # A docstring with an unterminated code fence must never swallow the
        # MDX that follows it: close the dangling fence so the block is
        # self-contained.
        out.append(fence_char * fence_len)
    return "\n".join(out)


def truncate(value: str, limit: int = 120) -> str:
    """Collapse whitespace and truncate a value for a table cell."""
    value = _collapse_ws(value)
    if len(value) > limit:
        return value[: limit - 1].rstrip() + "…"
    return value
