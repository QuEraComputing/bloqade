"""MDX-safety tests for ``bloqade_docs_emit_python.escape``.

These pin the CURRENT behavior of the escaping helpers. A docstring must never
be able to break the Astro/MDX build, so each helper is exercised with the
adversarial inputs it was built to survive (JS-expression openers ``{``, JSX
tag openers ``<``, backticks, ``${`` interpolation openers, and an unterminated
code fence).
"""

from __future__ import annotations

from bloqade_docs_emit_python import escape


# --------------------------------------------------------------------------- #
# prose: `<`, `{`, `}` become character references in flow content
# --------------------------------------------------------------------------- #
def test_prose_neutralizes_jsx_and_expression_openers():
    out = escape.prose("Use <T> and {x} and a bare } here")
    # `<` -> &lt;, `{` -> &#123;, `}` -> &#125;
    assert "&lt;T" in out
    assert "&#123;x&#125;" in out
    assert out.count("&#125;") == 2  # both the closing of {x} and the bare }
    # `>` is deliberately left alone in prose (harmless in MDX flow).
    assert "&gt;" not in out
    assert ">" in out


def test_prose_leaves_ampersand_and_gt_untouched():
    out = escape.prose("a > b && c")
    assert out == "a > b && c"


def test_prose_preserves_inline_code_span_verbatim():
    # Hostile chars INSIDE an inline code span must stay literal (backticks
    # protect them from both MDX and markdown).
    out = escape.prose("call `f<T>({x})` now")
    assert "`f<T>({x})`" in out
    assert "&lt;" not in out
    assert "&#123;" not in out
    # Surrounding prose is still escaped.
    out2 = escape.prose("before <A> `raw<B>` after {c}")
    assert "&lt;A" in out2
    assert "`raw<B>`" in out2
    assert "&#123;c&#125;" in out2


def test_prose_unterminated_inline_backtick_run_is_literal():
    # A single unmatched backtick is emitted literally and scanning continues,
    # so specials after it are still escaped.
    out = escape.prose("weird ` then <x>")
    assert "`" in out
    assert "&lt;x" in out


def test_prose_preserves_fenced_code_block_verbatim():
    text = "intro <a>\n```python\nx = {1: 2}\ny < z\n```\ntrailing <b>"
    out = escape.prose(text)
    lines = out.split("\n")
    # Fence body is verbatim.
    assert "x = {1: 2}" in lines
    assert "y < z" in lines
    # Prose around the fence is escaped.
    assert "intro &lt;a" in out
    assert "trailing &lt;b" in out


def test_prose_unterminated_fence_is_force_closed():
    # THE adversarial case: a docstring whose code fence is never closed must
    # not swallow the MDX that follows it. The emitter force-closes it.
    text = "intro\n```python\ncode line\nno closing fence here"
    out = escape.prose(text)
    lines = out.split("\n")
    assert lines[-1] == "```"  # dangling fence force-closed at the end
    # The still-open body was passed through verbatim (not escaped).
    assert "code line" in lines


def test_prose_tilde_fence_force_closed_with_matching_marker():
    text = "~~~\nunterminated tilde block"
    out = escape.prose(text)
    assert out.split("\n")[-1] == "~~~"


def test_prose_closing_fence_must_be_at_least_as_long():
    # A shorter run inside a longer fence does not close it.
    text = "````\n```\nstill inside\n````"
    out = escape.prose(text)
    lines = out.split("\n")
    # The inner ``` did not close the ```` fence; the real closing ```` did,
    # so the block is self-contained and NOT force-closed again.
    assert lines.count("````") == 2
    assert "still inside" in lines


def test_prose_empty_string():
    assert escape.prose("") == ""


# --------------------------------------------------------------------------- #
# prose: intra-doc relative `.md` / bare-relative links are stripped to text
# --------------------------------------------------------------------------- #
def test_prose_strips_relative_md_link():
    # A mkdocs-style relative link would otherwise render as a dead
    # <a href="factory.md">; we drop the href and keep only the text.
    out = escape.prose("See [factory](factory.md) for more.")
    assert "factory" in out
    assert ".md" not in out
    assert "](" not in out  # no residual markdown-link syntax


def test_prose_strips_parent_relative_md_link_with_fragment():
    out = escape.prose("See [caps](../reference/hardware-capabilities.md#limits).")
    assert "caps" in out
    assert ".md" not in out
    assert "](" not in out


def test_prose_strips_bare_relative_path_link():
    # A relative link with no scheme and no extension is still intra-doc.
    out = escape.prose("Read the [guide](getting-started/intro).")
    assert "guide" in out
    assert "getting-started/intro" not in out


def test_prose_keeps_external_link():
    out = escape.prose("Visit [the site](https://example.com/docs.md).")
    # External links (with a scheme) are preserved verbatim, .md target and all.
    assert "[the site](https://example.com/docs.md)" in out


def test_prose_keeps_absolute_and_fragment_links():
    out = escape.prose("[api](/api/dev/python/bloqade/) and [here](#section).")
    assert "[api](/api/dev/python/bloqade/)" in out
    assert "[here](#section)" in out


def test_prose_leaves_md_link_inside_code_span():
    # Backtick-protected content is verbatim; the ".md" inside a code span is
    # literal code, not a link, so it must survive untouched.
    out = escape.prose("call `open(x.md)` and [a](b.md)")
    assert "`open(x.md)`" in out  # code span preserved verbatim
    assert "a" in out
    assert "b.md" not in out  # the real link target was stripped


def test_prose_leaves_image_syntax_alone():
    # Image syntax (`![alt](path)`) is not an <a href>; don't mangle it.
    out = escape.prose("![diagram](diagram.png)")
    assert "![diagram](diagram.png)" in out


# --------------------------------------------------------------------------- #
# template: backtick template-literal context
# --------------------------------------------------------------------------- #
def test_template_escapes_backtick_and_interpolation():
    out = escape.template("a `code` and ${expr}")
    assert out == "a \\`code\\` and \\${expr}"


def test_template_escapes_backslash_first():
    # Order matters: backslash is escaped before backtick/`${`.
    out = escape.template("path\\to `x` ${y}")
    assert out == "path\\\\to \\`x\\` \\${y}"


def test_template_bare_dollar_not_escaped():
    # Only the `${` opener is escaped, not a lone `$`.
    assert escape.template("cost is $5") == "cost is $5"


# --------------------------------------------------------------------------- #
# attr: JSX double-quoted attribute value
# --------------------------------------------------------------------------- #
def test_attr_escapes_html_significant_chars():
    out = escape.attr('a & b < c > d " e')
    assert out == "a &amp; b &lt; c &gt; d &quot; e"


def test_attr_ampersand_escaped_before_others():
    # A pre-existing entity-looking sequence must not double-escape wrongly:
    # `<` -> `&lt;` uses an `&` that must already be encoded.
    out = escape.attr("<tag>")
    assert out == "&lt;tag&gt;"


def test_attr_collapses_whitespace():
    out = escape.attr("multi\n   line\ttext")
    assert out == "multi line text"


def test_attr_braces_left_literal():
    # Braces are literal inside a quoted JSX attribute, so attr() leaves them.
    assert escape.attr("{x}") == "{x}"


# --------------------------------------------------------------------------- #
# js_str: single-quoted JS string literal content
# --------------------------------------------------------------------------- #
def test_js_str_escapes_backslash_and_single_quote():
    out = escape.js_str("it's a \\ backslash")
    assert out == "it\\'s a \\\\ backslash"


def test_js_str_backslash_escaped_before_quote():
    out = escape.js_str("\\'")
    assert out == "\\\\\\'"


# --------------------------------------------------------------------------- #
# html_text / desc_literal: `set:html` description pipeline
# --------------------------------------------------------------------------- #
def test_html_text_escapes_markup():
    out = escape.html_text("a & b < c > d")
    assert out == "a &amp; b &lt; c &gt; d"


def test_desc_literal_is_html_then_js_escaped():
    # HTML-escape first so markup shows literally, then JS-escape for a
    # single-quoted literal.
    out = escape.desc_literal("x < y & it's <b>")
    # html_text: "x &lt; y &amp; it's &lt;b&gt;"
    # js_str:    escape the single quote
    assert out == "x &lt; y &amp; it\\'s &lt;b&gt;"


# --------------------------------------------------------------------------- #
# yaml_dq: YAML double-quoted scalar
# --------------------------------------------------------------------------- #
def test_yaml_dq_wraps_and_escapes():
    assert escape.yaml_dq('say "hi"') == '"say \\"hi\\""'
    assert escape.yaml_dq("back\\slash") == '"back\\\\slash"'


def test_yaml_dq_collapses_newlines():
    assert escape.yaml_dq("line1\nline2") == '"line1 line2"'


# --------------------------------------------------------------------------- #
# truncate
# --------------------------------------------------------------------------- #
def test_truncate_collapses_and_limits():
    long = "word " * 50
    out = escape.truncate(long, limit=20)
    assert len(out) <= 20
    assert out.endswith("…")  # ellipsis


def test_truncate_short_unchanged():
    assert escape.truncate("hello world") == "hello world"
