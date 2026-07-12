"""Intra-doc link rendering tests for the Rust emitter.

rustdoc gives each item a ``links`` map (``{"<dest text>": <Id>}``) of the
intra-doc links it already RESOLVED. The emitter turns those into
``<ApiXref origin="docstring" ...>`` when the target is an item this crate
documents, and leaves external/std targets (and non-links) as plain prose.

These tests exercise both the ``render_prose`` unit (with a hand-set
``id_to_fq``) and the full ``emit`` pipeline (which builds ``id_to_fq`` itself
in a pre-pass) on a hand-crafted rustdoc-JSON document. No cargo/nightly.
"""

from __future__ import annotations

import json

import pytest

from emit_rust import RustEmitter, esc_prose


# --------------------------------------------------------------------------- #
# render_prose unit tests (id_to_fq set by hand; no emission needed)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def em(mini_doc):
    # A versionless emitter; we hand-populate id_to_fq for unit-level control.
    e = RustEmitter(mini_doc, mount="api/rust", repo=None, ref=None)
    e.id_to_fq = {"1": "mini_crate::Bar", "2": "mini_crate::Baz"}
    return e


def test_shortcut_code_link_in_crate_becomes_xref(em):
    out = em.render_prose("See [`Bar`] please.", {"`Bar`": 1})
    assert '<ApiXref to="mini_crate::Bar" origin="docstring" label="Bar" />' in out
    assert "See " in out and " please." in out


def test_inline_link_in_crate_uses_display_text_as_label(em):
    out = em.render_prose("Use [the baz](Baz) here.", {"Baz": 2})
    assert '<ApiXref to="mini_crate::Baz" origin="docstring" label="the baz" />' in out


def test_reference_link_in_crate(em):
    out = em.render_prose("Use [a bar][Bar] here.", {"Bar": 1})
    assert '<ApiXref to="mini_crate::Bar" origin="docstring" label="a bar" />' in out


def test_bare_shortcut_link_in_crate(em):
    # `[Baz]` (no backticks) still resolves against a backticked map key.
    out = em.render_prose("Ref [Baz].", {"`Baz`": 2})
    assert '<ApiXref to="mini_crate::Baz" origin="docstring" label="Baz" />' in out


def test_external_link_degrades_to_plain_text(em):
    # id 99 is present in the links map but absent from id_to_fq -> external.
    out = em.render_prose("Uses [`ArchSpec`] elsewhere.", {"`ArchSpec`": 99})
    assert "<ApiXref" not in out
    assert "ArchSpec" in out  # display text survives as plain prose
    assert "99" not in out


def test_version_is_carried_on_xref(mini_doc):
    e = RustEmitter(mini_doc, mount="api/dev/rust", repo=None, ref=None)
    e.id_to_fq = {"1": "mini_crate::Bar"}
    out = e.render_prose("See [`Bar`].", {"`Bar`": 1})
    assert (
        '<ApiXref to="mini_crate::Bar" origin="docstring" label="Bar" version="dev" />'
        in out
    )


def test_no_links_is_byte_identical_to_esc_prose(em):
    # An item with no links map: prose is unchanged (escaped exactly as before),
    # including bracketed text that is NOT a rustdoc-resolved link.
    text = "Plain prose with [not a link], {braces}, <tags> and `code`."
    assert em.render_prose(text, None) == esc_prose(text)
    assert em.render_prose(text, {}) == esc_prose(text)
    assert "<ApiXref" not in em.render_prose(text, None)


def test_bracketed_text_not_in_links_map_stays_literal(em):
    # `[Bar]` is NOT in this item's links map -> ordinary prose, kept verbatim.
    out = em.render_prose("An array index arr[Bar] stays literal.", {"Other": 1})
    assert "<ApiXref" not in out
    assert out == esc_prose("An array index arr[Bar] stays literal.")


def test_link_inside_inline_code_span_is_not_interpreted(em):
    # `[`Bar`]` inside a code span must not be turned into an xref.
    out = em.render_prose("Literal `let x = [Bar];` code.", {"Bar": 1})
    assert "<ApiXref" not in out
    assert out == esc_prose("Literal `let x = [Bar];` code.")


def test_link_inside_fenced_code_block_is_not_interpreted(em):
    text = "Example:\n\n```rust\nlet b = [`Bar`];\n```\n\nDone."
    out = em.render_prose(text, {"`Bar`": 1})
    assert "<ApiXref" not in out
    # The fence is escaped verbatim; the `[`Bar`]` inside stays literal text.
    assert "\\`\\`\\`rust" in out


def test_real_link_after_code_span_still_resolves(em):
    out = em.render_prose("Code `x` then [`Bar`].", {"`Bar`": 1})
    assert '<ApiXref to="mini_crate::Bar" origin="docstring" label="Bar" />' in out
    assert out.startswith("Code \\`x\\` then ")


# --------------------------------------------------------------------------- #
# End-to-end emit(): id_to_fq is built by the emitter's own pre-pass
# --------------------------------------------------------------------------- #
def _doc_with_links() -> dict:
    """Crate `mini` with structs `Bar`/`Baz` and a free fn `demo` whose docs
    hold three intra-doc links: two in-crate (`Bar`, `Baz`) and one external
    (`Ext`, resolved by rustdoc to an id that lives only in `paths`)."""
    fv = __import__("emit_rust").EXPECTED_FORMAT_VERSION
    empty_fn = {
        "sig": {"inputs": [], "output": None, "is_c_variadic": False},
        "generics": {"params": [], "where_predicates": []},
        "header": {"is_const": False, "is_unsafe": False, "is_async": False, "abi": "Rust"},
        "has_body": True,
    }
    return {
        "root": "0",
        "crate_version": "0.1.0",
        "format_version": fv,
        "includes_private": False,
        "index": {
            "0": {
                "id": 0,
                "name": "mini",
                "visibility": "public",
                "docs": "Crate root.",
                "inner": {"module": {"is_crate": True, "items": ["1", "2", "3", "4"], "is_stripped": False}},
            },
            "1": {
                "id": 1,
                "name": "Bar",
                "visibility": "public",
                "docs": "A bar.",
                "inner": {"struct": {"kind": {"unit": None}, "generics": {"params": [], "where_predicates": []}, "impls": []}},
            },
            "2": {
                "id": 2,
                "name": "Baz",
                "visibility": "public",
                "docs": "A baz.",
                "inner": {"struct": {"kind": {"unit": None}, "generics": {"params": [], "where_predicates": []}, "impls": []}},
            },
            "3": {
                "id": 3,
                "name": "demo",
                "visibility": "public",
                # Two in-crate links + one external link, all in one paragraph.
                "docs": "Uses [`Bar`] and [the baz](Baz), plus external [`Ext`].",
                "links": {"`Bar`": 1, "Baz": 2, "`Ext`": 99},
                "inner": {"function": empty_fn},
            },
            "4": {
                "id": 4,
                "name": "plain",
                "visibility": "public",
                # No links map at all: plain prose, must be unchanged.
                "docs": "Plain prose, no links, has [brackets] and {braces}.",
                "inner": {"function": empty_fn},
            },
        },
        "paths": {
            "0": {"crate_id": 0, "path": ["mini"], "kind": "module"},
            "1": {"crate_id": 0, "path": ["mini", "Bar"], "kind": "struct"},
            "2": {"crate_id": 0, "path": ["mini", "Baz"], "kind": "struct"},
            "3": {"crate_id": 0, "path": ["mini", "demo"], "kind": "function"},
            "4": {"crate_id": 0, "path": ["mini", "plain"], "kind": "function"},
            # id 99 lives ONLY in paths (external crate) and never in `index`:
            # rustdoc resolved the link, but this crate does not document it.
            "99": {"crate_id": 7, "path": ["ext_crate", "Ext"], "kind": "struct"},
        },
    }


@pytest.fixture()
def emitted_links(tmp_path):
    doc = _doc_with_links()
    em = RustEmitter(doc, mount="api/dev/rust", repo=None, ref=None)
    stats = em.emit(tmp_path)
    return {
        "stats": stats,
        "mdx": (tmp_path / "mini" / "index.mdx").read_text(),
        "id_to_fq": em.id_to_fq,
        "inventory": json.loads((tmp_path / "inventory.rust.json").read_text()),
    }


def test_prepass_builds_id_to_fq_for_documented_items(emitted_links):
    id_to_fq = emitted_links["id_to_fq"]
    assert id_to_fq["1"] == "mini::Bar"
    assert id_to_fq["2"] == "mini::Baz"
    assert id_to_fq["3"] == "mini::demo"
    # The external id (99) is NOT documented, so it never enters id_to_fq.
    assert "99" not in id_to_fq


def test_emit_in_crate_links_become_xrefs(emitted_links):
    mdx = emitted_links["mdx"]
    assert '<ApiXref to="mini::Bar" origin="docstring" label="Bar" version="dev" />' in mdx
    assert '<ApiXref to="mini::Baz" origin="docstring" label="the baz" version="dev" />' in mdx


def test_emit_external_link_stays_plain_text(emitted_links):
    mdx = emitted_links["mdx"]
    # No xref to the external target, and its external path never appears.
    assert "ext_crate::Ext" not in mdx
    assert 'to="ext_crate' not in mdx
    # Exactly the two in-crate links became docstring xrefs.
    assert mdx.count('origin="docstring"') == 2
    # The external link's display text survives as plain prose.
    assert "Ext" in mdx


def test_emit_plain_prose_unchanged(emitted_links):
    mdx = emitted_links["mdx"]
    # The `plain` fn's docs contain bracketed text but no links -> no xref, and
    # the bracketed text is preserved verbatim (escaped like any prose).
    assert "has [brackets] and \\{braces\\}." in mdx


def test_inventory_still_correct_after_prepass(emitted_links):
    # The pre-pass must not double-count: inventory has each symbol once.
    fqs = [e["fqName"] for e in emitted_links["inventory"]]
    assert len(fqs) == len(set(fqs))
    assert set(fqs) == {"mini", "mini::Bar", "mini::Baz", "mini::demo", "mini::plain"}
    assert emitted_links["stats"]["symbols"] == 5
