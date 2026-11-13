"""Tests for the daily direct-dependency updater."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/bump_dependencies.py"
SPEC = importlib.util.spec_from_file_location("bump_dependencies", SCRIPT)
assert SPEC and SPEC.loader
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)


def releases(*versions, yanked=()):
    return {version: [{"yanked": version in yanked}] for version in versions}


def project(*dependencies, version="0.35.0-dev"):
    entries = ",\n".join(f'    "{dependency}"' for dependency in dependencies)
    return f'[project]\nname = "bloqade"\nversion = "{version}"\ndependencies = [\n{entries}\n]\n'


def test_patch_only_does_not_bump_bloqade():
    original = project("bloqade-analog~=0.16.3")
    updated, changes, minor = updater.update_text(
        original,
        lambda _: releases("0.16.4", "0.16.8", "0.17.0rc1", "1.0.0"),
    )
    assert '"bloqade-analog~=0.16.8"' in updated
    assert 'version = "0.35.0-dev"' in updated
    assert changes == [("bloqade-analog", "0.16.3", "0.16.8")]
    assert not minor


def test_multiple_updates_bump_bloqade_once():
    original = project(
        "bloqade-circuit[stim]~=0.14.1",
        "bloqade-analog~=0.16.3",
        "ppvm @ git+https://github.com/QuEraComputing/ppvm.git@main#subdirectory=ppvm-python",
    )
    available = {
        "bloqade-circuit": releases("0.14.4", "0.15.2", "0.16.0", yanked=("0.16.0",)),
        "bloqade-analog": releases("0.16.8"),
    }
    updated, changes, minor = updater.update_text(original, available.__getitem__)
    assert '"bloqade-circuit[stim]~=0.15.2"' in updated
    assert '"bloqade-analog~=0.16.8"' in updated
    assert "ppvm @ git+" in updated
    assert 'version = "0.36.0-dev"' in updated
    assert len(changes) == 2
    assert minor


def test_no_change_is_idempotent():
    original = project("bloqade-analog~=0.16.8")
    updated, changes, minor = updater.update_text(
        original, lambda _: releases("0.16.8", "0.17.0.dev1", "1.0.0")
    )
    assert updated == original
    assert not changes
    assert not minor


def test_unsupported_constraint_fails_loudly():
    original = project("bloqade-analog>=0.16.3")
    try:
        updater.update_text(original, lambda _: {})
    except ValueError as error:
        assert "Unsupported direct dependency constraint" in str(error)
    else:
        raise AssertionError("Unsupported requirements must not be silently skipped")
