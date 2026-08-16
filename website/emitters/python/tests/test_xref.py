"""Docstring cross-reference tests for the Python emitter.

Two layers:

1. Unit tests for ``xref.py`` in isolation (resolution + detection + the
   safe-emit collector), using a tiny fake griffe object so they need no real
   package.
2. End-to-end tests that build a synthetic package whose documented objects
   reference each other via EACH of the four docstring conventions (RST/Sphinx
   roles, Google-with-roles, Google-with-markdown, NumPy roles + See Also,
   Markdown autorefs) and assert the emitted MDX turns resolvable references
   into ``<ApiXref ... origin="docstring">`` and degrades external/undocumented
   references to inline code/text — never an unresolved cross-reference.
"""

from __future__ import annotations

import json
import re
import textwrap

from bloqade_docs_emit_python import xref
from bloqade_docs_emit_python.emitter import Config, Emitter


# --------------------------------------------------------------------------- #
# Unit: resolution
# --------------------------------------------------------------------------- #
class FakeObj:
    """Stand-in griffe object exposing just ``resolve`` (scope lookup)."""

    def __init__(self, scope: dict[str, str]) -> None:
        self._scope = scope

    def resolve(self, name: str) -> str:
        if name in self._scope:
            return self._scope[name]
        raise KeyError(name)  # mimics griffe NameResolutionError


def test_resolve_bare_name_uses_scope():
    obj = FakeObj({"Target": "pkg.core.Target"})
    assert xref.resolve_candidates(obj, "Target")[0] == "pkg.core.Target"


def test_resolve_dotted_name_reattaches_tail():
    obj = FakeObj({"Target": "pkg.core.Target"})
    # Widget.area -> resolve head "Target" then append ".area".
    assert xref.resolve_candidates(obj, "Target.area")[0] == "pkg.core.Target.area"


def test_resolve_absolute_falls_back_to_raw():
    # Leading package not in scope (raises) -> raw target is a candidate.
    obj = FakeObj({})
    assert "pkg.core.Target" in xref.resolve_candidates(obj, "pkg.core.Target")


def test_resolve_strips_tilde_short_form():
    obj = FakeObj({})
    assert xref.resolve_candidates(obj, "~pkg.core.Target")[0] == "pkg.core.Target"


def test_resolve_empty_target():
    assert xref.resolve_candidates(FakeObj({}), "") == []


# --------------------------------------------------------------------------- #
# Unit: role/autoref content parsing
# --------------------------------------------------------------------------- #
def test_parse_role_plain():
    assert xref._parse_role_content("pkg.core.Target") == (
        "pkg.core.Target",
        "pkg.core.Target",
    )


def test_parse_role_tilde_short_display():
    assert xref._parse_role_content("~pkg.core.Target") == ("pkg.core.Target", "Target")


def test_parse_role_explicit_title():
    assert xref._parse_role_content("the target <pkg.core.Target>") == (
        "pkg.core.Target",
        "the target",
    )


def test_parse_autoref_explicit():
    assert xref._parse_autoref("the target", "pkg.core.Target") == (
        "pkg.core.Target",
        "the target",
    )


def test_parse_autoref_empty_target_uses_label():
    assert xref._parse_autoref("pkg.core.greet", "") == (
        "pkg.core.greet",
        "pkg.core.greet",
    )


def test_parse_autoref_backtick_target():
    assert xref._parse_autoref("`Target`", "") == ("Target", "Target")


# --------------------------------------------------------------------------- #
# Unit: collector safe-emit rule
# --------------------------------------------------------------------------- #
def test_collector_resolvable_becomes_apixref():
    col = xref.XrefCollector()
    ph = col.add(["pkg.core.Target"], "Target")
    out = col.resolve(f"see {ph} now", documented={"pkg.core.Target"})
    assert '<ApiXref to="pkg.core.Target" origin="docstring" label="Target" />' in out


def test_collector_external_degrades_to_code():
    col = xref.XrefCollector()
    ph = col.add(["numpy.ndarray"], "numpy.ndarray")
    out = col.resolve(f"see {ph} now", documented={"pkg.core.Target"})
    assert "<ApiXref" not in out
    assert "`numpy.ndarray`" in out


def test_collector_label_with_spaces_degrades_to_text_not_code():
    col = xref.XrefCollector()
    ph = col.add(["ext.Thing"], "the thing")
    out = col.resolve(ph, documented=set())
    assert "`" not in out
    assert "the thing" in out


def test_collector_prefers_first_documented_candidate():
    col = xref.XrefCollector()
    ph = col.add(["pkg.core.Widget.area", "Widget.area"], "Widget.area")
    out = col.resolve(ph, documented={"pkg.core.Widget.area"})
    assert 'to="pkg.core.Widget.area"' in out


# --------------------------------------------------------------------------- #
# Unit: linkify respects code spans and fenced blocks
# --------------------------------------------------------------------------- #
def test_linkify_skips_inline_code_span():
    obj = FakeObj({"Target": "pkg.core.Target"})
    col = xref.XrefCollector()
    out = xref.linkify("code `:class:\\`Target\\`` stays", obj, col)
    # No placeholder was created for the reference inside the code span.
    assert "\x00" not in out


def test_linkify_skips_fenced_block():
    obj = FakeObj({"Target": "pkg.core.Target"})
    col = xref.XrefCollector()
    text = "intro\n```\n:class:`Target`\n```\ntrailing"
    out = xref.linkify(text, obj, col)
    # The role inside the fence must be left verbatim (no placeholder).
    assert ":class:`Target`" in out


def test_linkify_bare_dotted_code_span_is_autoref():
    # mkdocstrings backtick autoref: a single-backtick span wrapping a DOTTED
    # identifier is treated as a bare reference.
    obj = FakeObj({"pkg": "pkg"})
    col = xref.XrefCollector()
    out = xref.linkify("use `pkg.core.Target` now", obj, col)
    assert "\x00XREF0\x00" in out
    resolved = col.resolve(out, documented={"pkg.core.Target"})
    assert 'to="pkg.core.Target"' in resolved


def test_linkify_bare_dotted_code_span_undocumented_restored_as_code():
    obj = FakeObj({})
    col = xref.XrefCollector()
    out = xref.linkify("use `ext.other.Thing` now", obj, col)
    resolved = col.resolve(out, documented=set())
    # Non-reference code span is restored verbatim (degrades back to code).
    assert "`ext.other.Thing`" in resolved
    assert "<ApiXref" not in resolved


def test_linkify_single_word_code_span_not_autoref():
    # A single-word (no dot) code span is left as code, never linkified.
    obj = FakeObj({"times": "pkg.core.times"})
    col = xref.XrefCollector()
    out = xref.linkify("the `times` value", obj, col)
    assert "\x00" not in out
    assert "`times`" in out


def test_linkify_non_identifier_code_span_left_verbatim():
    # Array-indexing-like code (`arr[-1]`) must not be mistaken for a reference.
    obj = FakeObj({})
    col = xref.XrefCollector()
    out = xref.linkify("value `arr[-1].x` here", obj, col)
    assert "\x00" not in out
    assert "`arr[-1].x`" in out


def test_linkify_role_outside_code_is_detected():
    obj = FakeObj({"Target": "pkg.core.Target"})
    col = xref.XrefCollector()
    out = xref.linkify("see :class:`Target`", obj, col)
    assert "\x00XREF0\x00" in out
    resolved = col.resolve(out, documented={"pkg.core.Target"})
    assert 'to="pkg.core.Target"' in resolved


# --------------------------------------------------------------------------- #
# End-to-end helpers
# --------------------------------------------------------------------------- #
def _emit(tmp_path, core_src: str, style: str = "google"):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""Pkg."""\n', encoding="utf-8")
    (pkg / "core.py").write_text(core_src, encoding="utf-8")
    out = tmp_path / "out"
    cfg = Config(
        package="pkg",
        src=str(pkg),
        repo="owner/repo",
        ref="main",
        version="dev",
        mount="api/python",
        out=str(out),
        repo_root=str(tmp_path),
        docstring_style=style,
    )
    Emitter(cfg).run()
    inv = {e["fqName"] for e in json.loads((out / "inventory.python.json").read_text())}
    mdx = (out / "pkg" / "core.mdx").read_text()
    return mdx, inv


def _xref_targets(mdx: str) -> list[str]:
    return re.findall(r'<ApiXref to="([^"]+)" origin="docstring"', mdx)


def _assert_no_unresolved(mdx: str, inv: set[str]) -> None:
    # THE safety invariant: no NUL placeholder leaked, and every emitted
    # docstring xref targets a symbol we documented (so it can never render as
    # api-xref--unresolved and fail XREF_STRICT).
    assert "\x00" not in mdx
    for target in _xref_targets(mdx):
        assert target in inv, f"emitted xref to undocumented {target!r}"


# Shared class defined once; each convention module references it.
_TARGET = '''\
class Target:
    """The target class."""

    def area(self) -> int:
        """Compute the area."""
        return 1
'''


def _module(func_src: str) -> str:
    """Compose a valid ``core.py`` = module docstring + Target class + a func.

    ``func_src`` is dedented so tests can indent the function body for
    readability without corrupting the emitted Python source.
    """
    return (
        '"""Core."""\n\n'
        + _TARGET
        + "\n\n"
        + textwrap.dedent(func_src).strip("\n")
        + "\n"
    )


# --------------------------------------------------------------------------- #
# Convention 1: reStructuredText / Sphinx roles
# --------------------------------------------------------------------------- #
def test_rst_roles_each_variant(tmp_path):
    src = _module('''
        def use() -> None:
            """Use things.

            Roles :class:`Target`, :meth:`Target.area`, :func:`pkg.core.use`,
            :obj:`pkg.core.Target`. Short :class:`~pkg.core.Target`; title
            :class:`the target <pkg.core.Target>`. External :class:`numpy.ndarray`.

            .. seealso::

               :class:`Target` for details.
            """
        ''')
    mdx, inv = _emit(tmp_path, src, style="sphinx")
    _assert_no_unresolved(mdx, inv)
    # Resolvable roles -> ApiXref (relative, dotted, and absolute forms).
    assert '<ApiXref to="pkg.core.Target" origin="docstring" label="Target" />' in mdx
    assert (
        '<ApiXref to="pkg.core.Target.area" origin="docstring" label="Target.area" />'
        in mdx
    )
    assert (
        '<ApiXref to="pkg.core.use" origin="docstring" label="pkg.core.use" />' in mdx
    )
    # ~short display + explicit-title display forms.
    assert (
        '<ApiXref to="pkg.core.Target" origin="docstring" label="Target" />' in mdx
    )  # ~form
    assert (
        '<ApiXref to="pkg.core.Target" origin="docstring" label="the target" />' in mdx
    )
    # `.. seealso::` roles are linkified inline too.
    assert mdx.count('<ApiXref to="pkg.core.Target"') >= 3
    # External role -> inline code, no xref.
    assert "`numpy.ndarray`" in mdx
    assert "numpy.ndarray" not in " ".join(_xref_targets(mdx))


# --------------------------------------------------------------------------- #
# Convention 2a: Google style with Sphinx roles
# --------------------------------------------------------------------------- #
def test_google_style_with_roles(tmp_path):
    src = _module('''
        def use() -> None:
            """Use things.

            See :class:`Target` and the external :func:`kirin.ir.Method`.
            """
        ''')
    mdx, inv = _emit(tmp_path, src, style="google")
    _assert_no_unresolved(mdx, inv)
    assert '<ApiXref to="pkg.core.Target" origin="docstring" label="Target" />' in mdx
    assert "`kirin.ir.Method`" in mdx  # external -> code
    assert "kirin.ir.Method" not in " ".join(_xref_targets(mdx))


# --------------------------------------------------------------------------- #
# Convention 2b: Google style with Markdown autorefs
# --------------------------------------------------------------------------- #
def test_google_style_with_markdown_autorefs(tmp_path):
    src = _module('''
        def use() -> None:
            """Use things.

            Markdown: [the target][pkg.core.Target], [pkg.core.use][],
            [`Target`][], bare [`Target`]. External [nd][numpy.ndarray].
            """
        ''')
    mdx, inv = _emit(tmp_path, src, style="google")
    _assert_no_unresolved(mdx, inv)
    assert (
        '<ApiXref to="pkg.core.Target" origin="docstring" label="the target" />' in mdx
    )
    assert (
        '<ApiXref to="pkg.core.use" origin="docstring" label="pkg.core.use" />' in mdx
    )
    assert '<ApiXref to="pkg.core.Target" origin="docstring" label="Target" />' in mdx
    # External autoref -> plain text label (has no whitespace here -> code).
    assert "`nd`" in mdx
    assert "numpy.ndarray" not in " ".join(_xref_targets(mdx))


# --------------------------------------------------------------------------- #
# Convention 3: NumPy style — roles in prose AND a See Also section
# --------------------------------------------------------------------------- #
def test_numpy_roles_and_see_also(tmp_path):
    src = _module('''
        def use(x):
            """Use things.

            Reference :func:`pkg.core.use` inline.

            See Also
            --------
            pkg.core.Target : The target class.
            pkg.core.use
            numpy.dot : External, stays text.
            """
            return x
        ''')
    mdx, inv = _emit(tmp_path, src, style="numpy")
    _assert_no_unresolved(mdx, inv)
    # Inline role.
    assert (
        '<ApiXref to="pkg.core.use" origin="docstring" label="pkg.core.use" />' in mdx
    )
    # See Also section header + linkified documented names.
    assert "**See Also**" in mdx
    assert (
        '<ApiXref to="pkg.core.Target" origin="docstring" label="pkg.core.Target" />'
        in mdx
    )
    # External See-Also entry -> inline code, not an xref.
    assert "`numpy.dot`" in mdx
    assert "numpy.dot" not in " ".join(_xref_targets(mdx))


# --------------------------------------------------------------------------- #
# Convention 4: Markdown / mkdocstrings autorefs (default google parser)
# --------------------------------------------------------------------------- #
def test_markdown_autorefs_and_plain_links(tmp_path):
    src = _module('''
        def use() -> None:
            """Use things.

            Autoref [Target][pkg.core.Target]. Undocumented [gone][pkg.core.Nope].
            Plain external link [site](https://example.com).
            """
        ''')
    mdx, inv = _emit(tmp_path, src, style="google")
    _assert_no_unresolved(mdx, inv)
    assert '<ApiXref to="pkg.core.Target" origin="docstring" label="Target" />' in mdx
    # Undocumented target (resolvable syntactically, but not in inventory) -> text.
    assert "`gone`" in mdx
    assert "pkg.core.Nope" not in mdx
    # Regular markdown links are left to the escape pipeline (external kept).
    assert "[site](https://example.com)" in mdx


# --------------------------------------------------------------------------- #
# Cross-module relative resolution + undocumented sibling
# --------------------------------------------------------------------------- #
def test_bare_backtick_autoref_e2e(tmp_path):
    src = _module('''
        def use() -> None:
            """Uses `pkg.core.Target` and the external `numpy.ndarray`."""
        ''')
    mdx, inv = _emit(tmp_path, src, style="google")
    _assert_no_unresolved(mdx, inv)
    # Documented dotted code span -> ApiXref (mkdocstrings backtick autoref).
    assert (
        '<ApiXref to="pkg.core.Target" origin="docstring" label="pkg.core.Target" />'
        in mdx
    )
    # External dotted code span stays inline code.
    assert "`numpy.ndarray`" in mdx
    assert "numpy.ndarray" not in " ".join(_xref_targets(mdx))


def test_reference_to_undocumented_private_is_text(tmp_path):
    # `_Hidden` is private -> never documented -> reference must degrade to text.
    src = textwrap.dedent('''\
        """Core."""

        class _Hidden:
            """Private."""

        class Target:
            """Target."""

        def use() -> None:
            """See :class:`Target` and :class:`_Hidden`."""
        ''')
    mdx, inv = _emit(tmp_path, src, style="google")
    _assert_no_unresolved(mdx, inv)
    assert '<ApiXref to="pkg.core.Target" origin="docstring" label="Target" />' in mdx
    # _Hidden is not documented; it must not become an xref.
    assert "pkg.core._Hidden" not in " ".join(_xref_targets(mdx))
    assert "`_Hidden`" in mdx


def test_mdx_safety_hostile_chars_around_refs(tmp_path):
    # A reference next to MDX-hostile characters must stay safe.
    src = textwrap.dedent('''\
        """Core."""

        class Target:
            """Target."""

        def use() -> None:
            """Compare {a} < b then see :class:`Target`."""
        ''')
    mdx, inv = _emit(tmp_path, src, style="google")
    _assert_no_unresolved(mdx, inv)
    assert "&#123;a&#125; &lt; b" in mdx
    assert '<ApiXref to="pkg.core.Target" origin="docstring" label="Target" />' in mdx
