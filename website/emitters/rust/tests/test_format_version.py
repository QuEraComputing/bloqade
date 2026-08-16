"""Tests for the rustdoc ``format_version`` guard in ``emit_rust.main``.

The emitter PINS ``EXPECTED_FORMAT_VERSION`` and refuses to run against a
different one unless ``--allow-format-version`` is passed, so a nightly
toolchain bump can never silently emit garbage.
"""

from __future__ import annotations

import json

import pytest

import emit_rust


def _write(tmp_path, doc) -> str:
    p = tmp_path / "doc.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def test_correct_format_version_passes(tmp_path, mini_doc):
    assert mini_doc["format_version"] == emit_rust.EXPECTED_FORMAT_VERSION
    json_path = _write(tmp_path, mini_doc)
    out = tmp_path / "out"
    rc = emit_rust.main(["--json", json_path, "--out", str(out), "--no-source"])
    assert rc == 0
    assert (out / "mini_crate" / "index.mdx").is_file()
    assert (out / "inventory.rust.json").is_file()


def test_wrong_format_version_errors(tmp_path, mini_doc):
    mini_doc["format_version"] = emit_rust.EXPECTED_FORMAT_VERSION + 999
    json_path = _write(tmp_path, mini_doc)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as excinfo:
        emit_rust.main(["--json", json_path, "--out", str(out), "--no-source"])
    # Exits with a helpful non-empty message (not a clean exit).
    assert excinfo.value.code != 0
    assert not (out / "mini_crate" / "index.mdx").exists()


def test_wrong_format_version_allowed_with_flag(tmp_path, mini_doc, capsys):
    mini_doc["format_version"] = emit_rust.EXPECTED_FORMAT_VERSION + 999
    json_path = _write(tmp_path, mini_doc)
    out = tmp_path / "out"
    rc = emit_rust.main(
        [
            "--json",
            json_path,
            "--out",
            str(out),
            "--no-source",
            "--allow-format-version",
        ]
    )
    assert rc == 0
    assert (out / "mini_crate" / "index.mdx").is_file()
    # A warning is surfaced on stderr.
    assert "format_version" in capsys.readouterr().err


def test_missing_format_version_field_errors(tmp_path, mini_doc):
    del mini_doc["format_version"]
    json_path = _write(tmp_path, mini_doc)
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        emit_rust.main(["--json", json_path, "--out", str(out), "--no-source"])
