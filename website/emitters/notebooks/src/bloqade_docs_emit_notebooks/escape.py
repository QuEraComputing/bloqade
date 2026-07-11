"""MDX-safety escaping helpers for notebook-derived content.

Notebook markdown/outputs can contain characters that are hostile to the MDX
parser (``<`` opens a JSX element, ``{`` opens a JS expression). The MDX build
MUST NEVER break on tutorial content, so every string that reaches the ``.mdx``
output flows through one of the helpers here.

The two hard constraints for markdown *prose*:

* ``<`` and ``{``/``}`` must be neutralized in free text, because MDX would try
  to parse them as JSX/expressions.
* ...but NOT inside fenced code blocks, inline code spans, or ``$``-math, where
  MDX treats the content as literal and KaTeX needs braces verbatim
  (``2^{n}``). Those regions are detected and passed through unchanged.

``>`` and ``&`` are deliberately left alone: neither breaks an MDX build, and
leaving them preserves markdown blockquotes and HTML entities.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# YAML frontmatter
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")


def yaml_dq(value: str) -> str:
    """Return ``value`` as a YAML double-quoted scalar (including the quotes)."""
    value = _WS_RE.sub(" ", value).strip()
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


# --------------------------------------------------------------------------- #
# ANSI stripping (rich/pretty-printer output is full of colour codes)
# --------------------------------------------------------------------------- #

# CSI sequences (colours, cursor moves) + OSC sequences (hyperlinks).
_ANSI_RE = re.compile(
    r"""
    \x1B\[[0-?]*[ -/]*[@-~]     # CSI ... final byte
    | \x1B\][^\x07\x1B]*(?:\x07|\x1B\\)  # OSC ... BEL / ST
    | \x1B[@-Z\\-_]             # two-char escapes
    """,
    re.VERBOSE,
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colour codes, hyperlinks) from text."""
    return _ANSI_RE.sub("", text)


# --------------------------------------------------------------------------- #
# Prose (MDX flow content)
# --------------------------------------------------------------------------- #

_FENCE_OPEN_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")

# Protected inline spans inside a prose line/segment: inline code (backtick
# runs), block math ($$...$$) and inline math ($...$). Order matters: match the
# longer $$ before $. `re.DOTALL` lets $$...$$ span lines within a segment.
_PROTECT_RE = re.compile(
    r"(?P<code>`+)(?P<code_body>.+?)(?P=code)"  # `code` / ``code`` spans
    r"|(?P<dmath>\$\$.+?\$\$)"  # $$ display math $$
    r"|(?P<imath>\$(?!\s)[^$\n]*?(?<!\s)\$)",  # $ inline math $
    re.DOTALL,
)


def _escape_specials(segment: str) -> str:
    """Escape MDX-hostile ``<`` ``{`` ``}`` in a plain (non-code/math) segment."""
    return (
        segment.replace("<", "&lt;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
    )


def escape_prose_inline(text: str) -> str:
    """Escape a prose region, leaving code spans and ``$``-math verbatim."""
    out: list[str] = []
    pos = 0
    for m in _PROTECT_RE.finditer(text):
        out.append(_escape_specials(text[pos : m.start()]))
        out.append(m.group(0))  # protected verbatim
        pos = m.end()
    out.append(_escape_specials(text[pos:]))
    return "".join(out)


def escape_prose(text: str) -> str:
    """Escape free-form markdown so it is safe as MDX flow content.

    Fenced code blocks are passed through verbatim; every other line is
    inline-escaped (code spans and ``$``-math within it stay verbatim).
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
            out.append(line)  # the opening fence line is safe verbatim
            continue
        out.append(escape_prose_inline(line))
    if in_fence:
        # Never let an unterminated fence swallow the MDX that follows.
        out.append(fence_char * fence_len)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Fenced code emission (code cells + text outputs)
# --------------------------------------------------------------------------- #


def fenced(body: str, lang: str = "", meta: str = "") -> str:
    """Wrap ``body`` in a fenced code block using a backtick run long enough to
    not collide with any run inside ``body`` (minimum three).
    """
    longest = 0
    for run in re.findall(r"`+", body):
        longest = max(longest, len(run))
    fence = "`" * max(3, longest + 1)
    info = lang
    if meta:
        info = f"{lang} {meta}".strip()
    body = body.rstrip("\n")
    return f"{fence}{info}\n{body}\n{fence}"
