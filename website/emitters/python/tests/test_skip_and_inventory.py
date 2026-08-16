"""Skip-rule + inventory-URL tests for the Python emitter.

Uses lightweight stand-in module objects (``SimpleNamespace``) so these unit
tests do not need griffe. They pin the ported ``gen_ref_nav`` behavior:
skip_keywords, private ``_*`` modules/members, and ``__init__`` -> ``index``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bloqade_docs_emit_python.emitter import SKIP_KEYWORDS, Config, Emitter


def make_config(**overrides) -> Config:
    base = dict(
        package="bloqade",
        src="/repo/src/bloqade",
        repo="owner/repo",
        ref="main",
        version="dev",
        mount="api/python",
        out="/tmp/out",
        repo_root="/repo",
        docstring_style="google",
    )
    base.update(overrides)
    return Config(**base)


def fake_module(name, path, filepath=None):
    return SimpleNamespace(name=name, path=path, filepath=filepath)


# --------------------------------------------------------------------------- #
# should_skip_module
# --------------------------------------------------------------------------- #
def test_private_module_skipped(tmp_path):
    em = Emitter(make_config(repo_root=str(tmp_path)))
    mod = fake_module(
        "_internal", "bloqade._internal", tmp_path / "bloqade" / "_internal.py"
    )
    skip, reason = em.should_skip_module(mod)
    assert skip is True
    assert "private module" in reason


def test_dunder_init_named_module_not_private(tmp_path):
    # The private rule explicitly exempts a module literally named "__init__".
    em = Emitter(make_config(repo_root=str(tmp_path)))
    mod = fake_module(
        "__init__", "bloqade.pkg", tmp_path / "bloqade" / "pkg" / "__init__.py"
    )
    skip, _ = em.should_skip_module(mod)
    assert skip is False


def test_public_module_not_skipped(tmp_path):
    em = Emitter(make_config(repo_root=str(tmp_path)))
    fp = tmp_path / "bloqade" / "task.py"
    mod = fake_module("task", "bloqade.task", fp)
    skip, reason = em.should_skip_module(mod)
    assert skip is False
    assert reason == ""


@pytest.mark.parametrize(
    "relpath, keyword",
    [
        ("bloqade/codegen/foo.py", "codegen/"),
        ("bloqade/qasm2/tests/test_x.py", "tests/"),
        ("bloqade/visual/visualization.py", "visualization"),
        ("bloqade/builder/base.py", "builder/base"),
        ("bloqade/squin/cirq/emit/x.py", "squin/cirq/emit/"),
        ("bloqade/scripts/gen.py", "scripts/"),
    ],
)
def test_skip_keyword_matches(tmp_path, relpath, keyword):
    em = Emitter(make_config(repo_root=str(tmp_path)))
    fp = tmp_path.joinpath(*relpath.split("/"))
    mod = fake_module(fp.stem, "bloqade." + fp.stem, fp)
    skip, reason = em.should_skip_module(mod)
    assert skip is True, f"expected {relpath} to be skipped via {keyword!r}"
    assert keyword in reason


def test_every_skip_keyword_triggers(tmp_path):
    # Each keyword in the ported list should cause a skip when present in the
    # repo-relative path.
    em = Emitter(make_config(repo_root=str(tmp_path)))
    for kw in SKIP_KEYWORDS:
        rel = (
            "pkg/" + kw + "/leaf.py"
            if not kw.endswith("/")
            else "pkg/" + kw + "leaf.py"
        )
        fp = tmp_path / rel
        mod = fake_module("leaf", "pkg.leaf", fp)
        skip, reason = em.should_skip_module(mod)
        assert skip is True, f"keyword {kw!r} did not trigger a skip"


# --------------------------------------------------------------------------- #
# is_private (member selection)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, expected",
    [
        ("_hidden", True),
        ("__dunder__", True),
        ("__init__", True),
        ("public", False),
        ("PublicClass", False),
    ],
)
def test_is_private(name, expected):
    assert Emitter.is_private(name) is expected


# --------------------------------------------------------------------------- #
# _is_package + page_path  (__init__ -> index)
# --------------------------------------------------------------------------- #
def test_page_path_package_is_index(tmp_path):
    em = Emitter(make_config(repo_root=str(tmp_path)))
    mod = fake_module(
        "squin", "bloqade.squin", tmp_path / "bloqade" / "squin" / "__init__.py"
    )
    assert em.page_path(mod) == Path("bloqade", "squin", "index.mdx")


def test_page_path_plain_module(tmp_path):
    em = Emitter(make_config(repo_root=str(tmp_path)))
    mod = fake_module("task", "bloqade.task", tmp_path / "bloqade" / "task.py")
    assert em.page_path(mod) == Path("bloqade", "task.mdx")


def test_is_package_namespace_list_filepath():
    em = Emitter(make_config())
    # griffe represents a namespace package's filepath as a list of dirs.
    mod = fake_module("ns", "bloqade.ns", ["/repo/a/bloqade/ns", "/repo/b/bloqade/ns"])
    assert em._is_package(mod) is True
    assert em.page_path(mod) == Path("bloqade", "ns", "index.mdx")


# --------------------------------------------------------------------------- #
# Inventory URL derivation:  url = /<mount>/<module-path>/#<fqName>
# --------------------------------------------------------------------------- #
def test_page_url_basic():
    em = Emitter(make_config())
    assert em.page_url("bloqade.task") == "/api/python/bloqade/task/"
    assert em.page_url("bloqade.squin.kernel") == "/api/python/bloqade/squin/kernel/"


def test_symbol_url_appends_fqname_anchor():
    em = Emitter(make_config())
    assert (
        em.symbol_url("bloqade.task", "bloqade.task.BatchFuture.result")
        == "/api/python/bloqade/task/#bloqade.task.BatchFuture.result"
    )


def test_mount_slashes_are_stripped():
    em = Emitter(make_config(mount="/api/python/"))
    assert em.mount == "api/python"
    assert em.page_url("bloqade.task") == "/api/python/bloqade/task/"


def test_record_appends_inventory_and_counts():
    em = Emitter(make_config())
    em._record("bloqade.task.foo", "function", "bloqade.task")
    assert em.inventory == [
        {
            "fqName": "bloqade.task.foo",
            "kind": "function",
            "url": "/api/python/bloqade/task/#bloqade.task.foo",
        }
    ]
    assert em.stats.symbols == 1
