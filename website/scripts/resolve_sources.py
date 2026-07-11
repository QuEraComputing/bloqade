#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Resolve the Bloqade docs API sources manifest.

This is a standalone, dependency-free script (stdlib + ``tomllib`` only). Run it
with ``uv run`` (which will auto-provision a Python >= 3.11 for ``tomllib``) or
with any Python >= 3.11 directly::

    uv run website/scripts/resolve_sources.py
    python website/scripts/resolve_sources.py

What it does
------------
Reads ``docs.sources.toml`` and turns each declared source into a *resolved*
entry, printing the whole set to stdout as a JSON list. For each source:

* ``ref = "pin-from:<pkg>"`` -> look up the installed version of the pypi package
  ``<pkg>`` and compute the git tag ``v<version>``. If the package is not
  installed, the entry is still emitted with ``ref: null`` and a warning is
  written to stderr (this never crashes).
* ``ref = "<literal>"`` (anything not starting with ``pin-from:``) -> used as-is.
* ``path = "<...>"`` -> the source is local; no version lookup is done.

This ports, verbatim in intent, two pieces of the existing MkDocs tooling:

* ``.github/pull_bloqade_submodules/action.yml`` — the version pinning
  (``uv pip show <pkg> | awk '/^Version: / {print "v"$2}'``) and the
  ``git clone`` of each repo at that tag (see ``--clone``).
* ``docs/scripts/gen_ref_nav.py`` — the local-dev sibling-directory convention
  (``../bloqade-circuit`` etc.), honored here behind ``--prefer-local``.

Flags
-----
``--manifest PATH``   Path to docs.sources.toml (default: alongside this repo's
                      website/ dir).
``--clone DIR``       For every repo source, ``git clone --depth 1 --branch
                      <ref>`` into ``DIR/<name>``. Idempotent: existing clones
                      are left alone. Local sources are a no-op. This is what CI
                      calls.
``--prefer-local``    Before resolving a repo source, look for a sibling
                      checkout (``<repo-parent>/<name>``, the gen_ref_nav.py
                      convention). If found, emit it as a local source instead
                      of a pinned repo.

Resolved manifest shape (one object per source)::

    {
      "name": str,
      "language": "python" | "rust",
      "kind": "repo" | "local",
      "repo": str | null,          # "owner/name" for repo sources
      "ref": str | null,           # "v1.2.3" for repo sources (null if unresolved)
      "path": str | null,          # filesystem path for local sources
      "package_root": str | null,  # importable package path within the source
      "crate": str | null,         # cargo crate name (rust sources)
      "mount": str                 # mount point in the docs tree, e.g. "api/python"
    }
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

# Repo layout: this file lives at <repo>/website/scripts/resolve_sources.py.
SCRIPT_DIR = Path(__file__).resolve().parent
WEBSITE_DIR = SCRIPT_DIR.parent
REPO_ROOT = WEBSITE_DIR.parent
DEFAULT_MANIFEST = WEBSITE_DIR / "docs.sources.toml"

PIN_PREFIX = "pin-from:"


def eprint(*args: object) -> None:
    """Print to stderr (progress + warnings; stdout stays pure JSON)."""
    print(*args, file=sys.stderr)


def pinned_version_tag(package: str) -> str | None:
    """Return the git tag ``v<version>`` for an installed pypi package.

    Ports ``uv pip show <pkg> | awk '/^Version: / {print "v"$2}'`` from
    ``.github/pull_bloqade_submodules/action.yml``. Returns ``None`` (and warns)
    if uv is missing or the package is not installed, so callers never crash.
    """
    if shutil.which("uv") is None:
        eprint(f"[warn] `uv` not found on PATH; cannot pin '{package}'.")
        return None
    try:
        proc = subprocess.run(
            ["uv", "pip", "show", package],
            capture_output=True,
            text=True,
        )
    except OSError as exc:  # pragma: no cover - defensive
        eprint(f"[warn] failed to run `uv pip show {package}`: {exc}")
        return None

    if proc.returncode != 0:
        eprint(
            f"[warn] `uv pip show {package}` failed (package not installed?); "
            f"leaving ref unresolved (ref=null)."
        )
        return None

    for line in proc.stdout.splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
            if version:
                return f"v{version}"

    eprint(f"[warn] no Version found for '{package}'; leaving ref unresolved.")
    return None


def sibling_checkout(name: str) -> Path | None:
    """Return a local sibling checkout for ``name`` if one exists.

    Mirrors the gen_ref_nav.py convention where bloqade-* repos are checked out
    next to each other, e.g. ``../bloqade-circuit`` relative to the repo root.
    """
    candidate = (REPO_ROOT.parent / name).resolve()
    if candidate.is_dir():
        return candidate
    return None


def resolve_source(source: dict, prefer_local: bool) -> dict:
    """Resolve a single raw TOML source into a resolved manifest entry."""
    name = source["name"]
    language = source["language"]
    mount = source["mount"]
    package_root = source.get("package_root")
    crate = source.get("crate")

    entry: dict = {
        "name": name,
        "language": language,
        "kind": None,
        # `repo` is populated from the manifest for EVERY source (including
        # local ones) so downstream tools can still build GitHub source links
        # for a locally-emitted package. `ref` stays null for local sources.
        "repo": source.get("repo"),
        "ref": None,
        "path": None,
        "package_root": package_root,
        "crate": crate,
        "mount": mount,
    }

    # 1) Explicit local path always wins.
    if source.get("path"):
        entry["kind"] = "local"
        entry["path"] = source["path"]
        return entry

    # 2) --prefer-local: use a sibling checkout if present.
    if prefer_local:
        sibling = sibling_checkout(name)
        if sibling is not None:
            eprint(f"[local] {name}: using sibling checkout {sibling}")
            entry["kind"] = "local"
            entry["path"] = str(sibling)
            return entry
        eprint(f"[local] {name}: no sibling checkout found, falling back to repo.")

    # 3) Otherwise it's a remote repo pinned to a ref.
    entry["kind"] = "repo"

    ref = source.get("ref")
    if ref is None:
        eprint(f"[warn] {name}: no ref/path given; ref stays null.")
    elif ref.startswith(PIN_PREFIX):
        package = ref[len(PIN_PREFIX):].strip()
        entry["ref"] = pinned_version_tag(package)
    else:
        # Literal tag/branch/sha.
        entry["ref"] = ref

    return entry


def clone_repo(entry: dict, clone_dir: Path) -> None:
    """git clone a repo source into ``clone_dir/<name>`` (idempotent).

    Ports the ``git clone``/checkout step of pull_bloqade_submodules/action.yml
    (``actions/checkout`` at the pinned ref) to a plain shallow clone.
    """
    name = entry["name"]
    if entry["kind"] != "repo":
        eprint(f"[clone] {name}: local source, nothing to clone.")
        return

    repo = entry.get("repo")
    ref = entry.get("ref")
    if not repo:
        eprint(f"[clone] {name}: no repo configured, skipping.")
        return
    if not ref:
        eprint(f"[clone] {name}: ref unresolved (null), skipping clone.")
        return

    target = clone_dir / name
    if (target / ".git").is_dir():
        eprint(f"[clone] {name}: already cloned at {target}, skipping.")
        return
    if target.exists():
        eprint(f"[clone] {name}: {target} exists but is not a git repo, skipping.")
        return

    clone_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    cmd = ["git", "clone", "--depth", "1", "--branch", ref, url, str(target)]
    eprint(f"[clone] {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        eprint(f"[warn] {name}: git clone failed (ref '{ref}' may not exist).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to docs.sources.toml (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--clone",
        type=Path,
        metavar="DIR",
        default=None,
        help="Clone every repo source into DIR/<name> (idempotent).",
    )
    parser.add_argument(
        "--prefer-local",
        action="store_true",
        help="Prefer sibling checkouts (../<name>) over pinned repos.",
    )
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        eprint(f"[error] manifest not found: {args.manifest}")
        return 1

    with args.manifest.open("rb") as fh:
        data = tomllib.load(fh)

    sources = data.get("source", [])
    if not sources:
        eprint(f"[warn] no [[source]] entries in {args.manifest}.")

    resolved = [resolve_source(src, args.prefer_local) for src in sources]

    if args.clone is not None:
        for entry in resolved:
            clone_repo(entry, args.clone)

    json.dump(resolved, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
