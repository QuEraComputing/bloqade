"""End-to-end emitter + inventory-URL tests for the Rust emitter.

Builds a ``RustEmitter`` from the hand-crafted ``mini_doc`` fixture (no cargo,
no nightly) and asserts the emitted MDX uses the frozen component contract and
that ``inventory.rust.json`` derives ``{fqName, kind, url}`` correctly.
"""

from __future__ import annotations

import json

import pytest

from emit_rust import RustEmitter


# --------------------------------------------------------------------------- #
# URL derivation units:  module_relpath + page_url
# --------------------------------------------------------------------------- #
def test_module_relpath(mini_doc):
    em = RustEmitter(mini_doc, mount="api/rust", repo=None, ref=None)
    assert em.module_relpath("mini_crate") == "mini_crate/index.mdx"
    assert em.module_relpath("mini_crate::a::b") == "mini_crate/a/b.mdx"


def test_page_url(mini_doc):
    em = RustEmitter(mini_doc, mount="api/rust", repo=None, ref=None)
    # Crate root drops the trailing `index`.
    assert em.page_url("mini_crate") == "/api/rust/mini_crate/"
    assert em.page_url("mini_crate::frontier") == "/api/rust/mini_crate/frontier/"


def test_mount_slashes_stripped(mini_doc):
    em = RustEmitter(mini_doc, mount="/api/rust/", repo=None, ref=None)
    assert em.mount == "api/rust"
    assert em.page_url("mini_crate") == "/api/rust/mini_crate/"


# --------------------------------------------------------------------------- #
# Version-aware mount: every emitted URL carries the /api/<V>/rust/ segment
# --------------------------------------------------------------------------- #
def test_version_derived_from_mount(mini_doc):
    # `api/<V>/rust` -> version <V>; a versionless `api/rust` -> None.
    assert RustEmitter(mini_doc, mount="api/dev/rust", repo=None, ref=None).version == "dev"
    assert RustEmitter(mini_doc, mount="api/0.35/rust", repo=None, ref=None).version == "0.35"
    assert RustEmitter(mini_doc, mount="api/rust", repo=None, ref=None).version is None


def test_explicit_version_flag_wins(mini_doc):
    em = RustEmitter(mini_doc, mount="api/dev/rust", repo=None, ref=None, version="1.2")
    assert em.version == "1.2"


def test_version_aware_page_and_inventory_urls(tmp_path, mini_doc):
    em = RustEmitter(mini_doc, mount="api/dev/rust", repo="owner/repo", ref="main")
    assert em.page_url("mini_crate") == "/api/dev/rust/mini_crate/"
    em.emit(tmp_path)
    inv = json.loads((tmp_path / "inventory.rust.json").read_text())
    # Every recorded URL includes the version segment; none are versionless.
    for entry in inv:
        assert entry["url"].startswith("/api/dev/rust/mini_crate/#")
    mdx = (tmp_path / "mini_crate" / "index.mdx").read_text()
    # No bare `/api/rust/` (versionless) string anywhere in the output.
    assert "/api/rust/" not in json.dumps(inv)
    assert "/api/rust/" not in mdx
    # Frontmatter apiVersion tracks the mount version (not crate_version).
    assert 'apiVersion: "dev"' in mdx


# --------------------------------------------------------------------------- #
# Submodule cross-references carry the referring page's version
# --------------------------------------------------------------------------- #
def _doc_with_submodule() -> dict:
    """A crate root whose only child is a submodule `sub` (for xref nav)."""
    return {
        "root": "0",
        "crate_version": "0.1.0",
        "format_version": __import__("emit_rust").EXPECTED_FORMAT_VERSION,
        "includes_private": False,
        "index": {
            "0": {
                "id": 0,
                "name": "mini_crate",
                "visibility": "public",
                "docs": "Crate root.",
                "inner": {"module": {"is_crate": True, "items": ["1"], "is_stripped": False}},
            },
            "1": {
                "id": 1,
                "name": "sub",
                "visibility": "public",
                "docs": "A submodule.",
                "inner": {"module": {"is_crate": False, "items": [], "is_stripped": False}},
            },
        },
        "paths": {
            "0": {"crate_id": 0, "path": ["mini_crate"], "kind": "module"},
            "1": {"crate_id": 0, "path": ["mini_crate", "sub"], "kind": "module"},
        },
    }


def test_submodule_xref_carries_version(tmp_path):
    doc = _doc_with_submodule()
    em = RustEmitter(doc, mount="api/dev/rust", repo=None, ref=None)
    em.emit(tmp_path)
    mdx = (tmp_path / "mini_crate" / "index.mdx").read_text()
    assert (
        '<ApiXref to="mini_crate::sub" label="sub" version="dev" />' in mdx
    )


def test_submodule_xref_omits_version_when_unknown(tmp_path):
    doc = _doc_with_submodule()
    em = RustEmitter(doc, mount="api/rust", repo=None, ref=None)  # versionless
    em.emit(tmp_path)
    mdx = (tmp_path / "mini_crate" / "index.mdx").read_text()
    assert '<ApiXref to="mini_crate::sub" label="sub" />' in mdx


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
@pytest.fixture()
def emitted(tmp_path, mini_doc):
    em = RustEmitter(mini_doc, mount="api/rust", repo="owner/repo", ref="main")
    stats = em.emit(tmp_path)
    return {
        "stats": stats,
        "inventory": json.loads((tmp_path / "inventory.rust.json").read_text()),
        "mdx": (tmp_path / "mini_crate" / "index.mdx").read_text(),
        "out": tmp_path,
    }


def test_stats(emitted):
    assert emitted["stats"]["pages"] == 1
    # module + field + struct + method + free function
    assert emitted["stats"]["symbols"] == 5


def test_inventory_module_entry(emitted):
    by_fq = {e["fqName"]: e for e in emitted["inventory"]}
    assert by_fq["mini_crate"] == {
        "fqName": "mini_crate",
        "kind": "module",
        "url": "/api/rust/mini_crate/#mini_crate",
    }


def test_inventory_struct_and_field(emitted):
    by_fq = {e["fqName"]: e for e in emitted["inventory"]}
    assert by_fq["mini_crate::Widget"] == {
        "fqName": "mini_crate::Widget",
        "kind": "struct",
        "url": "/api/rust/mini_crate/#mini_crate::Widget",
    }
    # A field has no heading of its own; its anchor is the PARENT struct fqName.
    assert by_fq["mini_crate::Widget::size"] == {
        "fqName": "mini_crate::Widget::size",
        "kind": "field",
        "url": "/api/rust/mini_crate/#mini_crate::Widget",
    }


def test_inventory_method_and_function(emitted):
    by_fq = {e["fqName"]: e for e in emitted["inventory"]}
    assert by_fq["mini_crate::Widget::area"] == {
        "fqName": "mini_crate::Widget::area",
        "kind": "method",
        "url": "/api/rust/mini_crate/#mini_crate::Widget::area",
    }
    assert by_fq["mini_crate::make_widget"] == {
        "fqName": "mini_crate::make_widget",
        "kind": "function",
        "url": "/api/rust/mini_crate/#mini_crate::make_widget",
    }


def test_inventory_url_derivation_rule(emitted):
    # Every URL is <page-url>#<anchor>, and the page-url is the module's.
    for entry in emitted["inventory"]:
        assert entry["url"].startswith("/api/rust/mini_crate/#")


# --------------------------------------------------------------------------- #
# MDX shape (frozen component contract)
# --------------------------------------------------------------------------- #
def test_frontmatter(emitted):
    mdx = emitted["mdx"]
    assert mdx.startswith("---\n")
    assert 'title: "mini_crate (crate)"' in mdx
    assert "language: rust" in mdx
    assert 'fqName: "mini_crate"' in mdx
    assert 'apiVersion: "0.1.0"' in mdx
    assert 'sourceRepo: "owner/repo"' in mdx
    assert 'sourceRef: "main"' in mdx
    assert 'description: "Mini crate root."' in mdx


def test_module_component(emitted):
    mdx = emitted["mdx"]
    assert '<ApiModule name="mini_crate" fqName="mini_crate"' in mdx
    assert 'summary="Mini crate root."' in mdx
    assert "</ApiModule>" in mdx
    # Second paragraph of crate docs is rendered as escaped prose.
    assert "Second paragraph of crate docs." in mdx


def test_struct_component(emitted):
    mdx = emitted["mdx"]
    assert '<ApiClass name="Widget" fqName="mini_crate::Widget"' in mdx
    # The frozen ApiClass renders "class"; the true Rust kind is prefixed on summary.
    assert 'summary="Rust struct. A widget."' in mdx
    assert "<Signature lang=\"rust\" code={`pub struct Widget`} />" in mdx
    # Fields render as a Params table titled "Fields".
    assert 'title="Fields"' in mdx
    assert "name: 'size'" in mdx
    assert "type: 'usize'" in mdx


def test_method_component(emitted):
    mdx = emitted["mdx"]
    assert '<ApiFn name="area" fqName="mini_crate::Widget::area" kind="method"' in mdx
    assert "<Signature lang=\"rust\" code={`pub fn area(&self) -> usize`} />" in mdx
    assert '<Returns type="usize" />' in mdx


def test_function_component(emitted):
    mdx = emitted["mdx"]
    assert '<ApiFn name="make_widget" fqName="mini_crate::make_widget" kind="function"' in mdx
    assert "<Signature lang=\"rust\" code={`pub fn make_widget(size: usize) -> Widget`} />" in mdx
    # Free-function param table.
    assert "name: 'size'" in mdx


def test_source_links(emitted):
    mdx = emitted["mdx"]
    # Widget's span begins at line 10 in src/lib.rs.
    assert (
        "https://github.com/owner/repo/blob/main/src/lib.rs#L10" in mdx
    )
    assert "<Source href=" in mdx


def test_no_source_flag_omits_links(tmp_path, mini_doc):
    em = RustEmitter(mini_doc, mount="api/rust", repo=None, ref=None)
    em.emit(tmp_path)
    mdx = (tmp_path / "mini_crate" / "index.mdx").read_text()
    assert "<Source" not in mdx
    assert "sourceUrl" not in mdx
