"""MDX-safety tests for the Rust emitter's escaping helpers.

A Rust doc comment must never break the Astro/MDX build. Unlike the Python
emitter's prose (which uses character references and preserves code spans), the
Rust emitter renders doc comments as *literal plain text*: it backslash-escapes
every MDX/JSX-significant punctuation char, so markdown/intra-doc links are
intentionally NOT interpreted (build safety over rich rendering).
"""

from __future__ import annotations

from emit_rust import (
    esc_attr,
    esc_desc_js,
    esc_prose,
    esc_template,
    first_paragraph,
    rest_paragraphs,
    yaml_str,
)


# --------------------------------------------------------------------------- #
# esc_prose: backslash-escape \ { } < > `
# --------------------------------------------------------------------------- #
def test_esc_prose_escapes_all_significant_punctuation():
    assert esc_prose("a<b>{c}`d`") == "a\\<b\\>\\{c\\}\\`d\\`"


def test_esc_prose_escapes_backslash():
    assert esc_prose("a\\b") == "a\\\\b"


def test_esc_prose_escapes_gt_unlike_python_prose():
    # The Rust emitter DOES escape `>` (the Python one leaves it alone).
    assert esc_prose("x > y") == "x \\> y"


def test_esc_prose_none_and_empty():
    assert esc_prose(None) == ""
    assert esc_prose("") == ""


def test_esc_prose_neutralizes_jsx_expression_opener():
    # `{` can never open a JS expression once escaped.
    assert "\\{" in esc_prose("value is {x}")
    assert "\\<" in esc_prose("a <Component/>")


# --------------------------------------------------------------------------- #
# esc_template: backtick template-literal context
# --------------------------------------------------------------------------- #
def test_esc_template_escapes_backtick_and_interpolation():
    assert esc_template("a `b` ${c}") == "a \\`b\\` \\${c}"


def test_esc_template_backslash_first():
    assert esc_template("p\\q `r`") == "p\\\\q \\`r\\`"


def test_esc_template_none():
    assert esc_template(None) == ""


# --------------------------------------------------------------------------- #
# esc_attr: double-quoted JSX attribute
# --------------------------------------------------------------------------- #
def test_esc_attr_html_escapes_and_flattens_newlines():
    assert esc_attr('a & b < c > d "e"') == "a &amp; b &lt; c &gt; d &quot;e&quot;"
    assert esc_attr("line1\nline2\rline3") == "line1 line2 line3"


def test_esc_attr_none():
    assert esc_attr(None) == ""


# --------------------------------------------------------------------------- #
# esc_desc_js: `set:html` description -> HTML-escape then JS-single-quote-escape
# --------------------------------------------------------------------------- #
def test_esc_desc_js_html_then_js_escape():
    # & < > become entities; single quote is backslash-escaped; newline -> space.
    assert esc_desc_js("a & b < c > 'd'\n") == "a &amp; b &lt; c &gt; \\'d\\' "


def test_esc_desc_js_backslash_escaped():
    assert esc_desc_js("a\\b") == "a\\\\b"


def test_esc_desc_js_none():
    assert esc_desc_js(None) == ""


# --------------------------------------------------------------------------- #
# yaml_str: YAML double-quoted scalar
# --------------------------------------------------------------------------- #
def test_yaml_str_wraps_and_escapes():
    assert yaml_str('he said "hi"') == '"he said \\"hi\\""'
    assert yaml_str("back\\slash") == '"back\\\\slash"'


def test_yaml_str_collapses_newlines():
    assert yaml_str("line1\nline2") == '"line1 line2"'


def test_yaml_str_none_is_empty_quotes():
    assert yaml_str(None) == '""'


# --------------------------------------------------------------------------- #
# first_paragraph / rest_paragraphs
# --------------------------------------------------------------------------- #
def test_first_paragraph_collapses_lines():
    docs = "Line one.\nLine two.\n\nSecond para.\n\nThird."
    assert first_paragraph(docs) == "Line one. Line two."


def test_rest_paragraphs():
    docs = "Line one.\nLine two.\n\nSecond para.\n\nThird."
    assert rest_paragraphs(docs) == "Second para.\n\nThird."


def test_first_and_rest_on_single_paragraph():
    assert first_paragraph("Only one.") == "Only one."
    assert rest_paragraphs("Only one.") == ""


def test_first_paragraph_none():
    assert first_paragraph(None) == ""
    assert rest_paragraphs(None) == ""
