#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""build_versions.py — the Bloqade API docs versioning orchestrator (Phase C2).

The Bloqade docs have a SINGLE distribution axis: the ``bloqade`` meta-version.
Only the ``/api`` reference is versioned; landing / blog / guides are evergreen.
For each requested version (a label like ``dev`` or ``0.35``) this script:

  1. resolves the API sources (``scripts/resolve_sources.py``) — sibling
     checkouts with ``--prefer-local``, or ``git clone`` of the pinned tags with
     ``--clone-dir`` (the CI path);
  2. runs the python + rust emitters with a **version-aware mount**
     (``--mount api/<V>/<lang>``) into ``src/content/docs/api/<V>/<lang>``;
  3. drops each merged inventory into ``.api-inventory/inventory.<lang>.<V>.json``
     (the version is parsed back out of the filename by
     ``scripts/build_inventory.mjs``);
  4. writes a small per-version, per-language ``index.mdx`` landing;
  5. regenerates ``src/generated/versions.json`` (the switcher manifest) by
     scanning which ``api/<V>`` trees actually exist, and tagging ``latest`` /
     ``dev``;
  6. optionally prunes stale releases (``--keep N``, ports
     ``.github/scripts/select_old_doc_versions.py``).

It is idempotent: re-running rebuilds the requested versions in place and leaves
every other version untouched. ``versions.json`` always reflects the versions
present on disk, so building versions one-at-a-time (as CI does) accumulates.

--------------------------------------------------------------------------------
HOW CI (Phase D2) CALLS IT
--------------------------------------------------------------------------------
* On a push to ``main`` (the ``dev`` build)::

      python website/scripts/build_versions.py --versions dev --prefer-local

  (In CI ``--prefer-local`` resolves the just-checked-out sibling repos; locally
  it uses ``/Users/<you>/Code/python/bloqade-*``. Without siblings you can pass
  ``--clone-dir website/sources`` to clone ``main`` of each repo instead.)

* On a release tag ``vX.Y.Z`` (a release build)::

      python website/scripts/build_versions.py --versions X.Y \
          --clone-dir website/sources --keep 3

  ``--clone-dir`` clones each source at its pinned tag (resolved from the
  installed ``bloqade-*`` versions via ``resolve_sources.py``); ``--keep 3``
  prunes all but the newest three releases (``dev`` is always kept).

VERIFICATION SHORTCUT (used to prove the mechanism without network clones):
build two versions from the SAME local siblings and subset the second::

      python website/scripts/build_versions.py --versions dev,0.35 \
          --prefer-local \
          --rust-json /path/to/target/doc/bloqade_lanes_search.json \
          --limit 0.35=bloqade-analog

Real releases clone their pinned tags in CI — the same-source shortcut only
affects which bytes the emitters read, not the layout / switcher / inventories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Repo layout: this file lives at <repo>/website/scripts/build_versions.py.
SCRIPT_DIR = Path(__file__).resolve().parent
WEBSITE_DIR = SCRIPT_DIR.parent
REPO_ROOT = WEBSITE_DIR.parent

DEFAULT_MANIFEST = WEBSITE_DIR / "docs.sources.toml"
CONTENT_ROOT = WEBSITE_DIR / "src" / "content" / "docs"
API_DIR = CONTENT_ROOT / "api"
INVENTORY_DIR = WEBSITE_DIR / ".api-inventory"
GENERATED_DIR = WEBSITE_DIR / "src" / "generated"
VERSIONS_JSON = GENERATED_DIR / "versions.json"

PY_EMITTER_DIR = WEBSITE_DIR / "emitters" / "python"
RUST_EMITTER = WEBSITE_DIR / "emitters" / "rust" / "emit_rust.py"
RESOLVE_SOURCES = SCRIPT_DIR / "resolve_sources.py"
BUILD_INVENTORY = SCRIPT_DIR / "build_inventory.mjs"

# The top-level python namespace package everything mounts under. Each source
# contributes a slice of the `bloqade` namespace (bloqade.squin, bloqade.analog,
# ...), so every python emit uses `--package bloqade` and points `--src` at that
# repo's `bloqade` directory.
TOP_PACKAGE = "bloqade"

RELEASE_RE = re.compile(r"^v?\d+\.\d+")


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Version helpers (ported from .github/scripts/select_old_doc_versions.py)
# --------------------------------------------------------------------------- #
def is_release(version: str) -> bool:
    """A doc version that looks like a release (``0.35``, ``v1.2.0``)."""
    return RELEASE_RE.match(version) is not None


def sem_key(version: str) -> list[int]:
    """Numeric sort key: ``0.35`` -> ``[0, 35]``. Ties break lexically upstream."""
    return [int(n) for n in re.findall(r"\d+", version)]


def select_versions_to_keep(all_versions: list[str], keep: int = 3) -> list[str]:
    """Return the versions to KEEP: the newest ``keep`` releases + every
    non-release (``dev`` and friends are always kept).

    Ports ``.github/scripts/select_old_doc_versions.py`` (which printed the
    versions to *delete*, i.e. ``releases[keep:]``). Here we return the
    complementary keep-set so the orchestrator can prune the rest.
    """
    releases = sorted(
        (v for v in all_versions if is_release(v)),
        key=sem_key,
        reverse=True,
    )
    non_releases = [v for v in all_versions if not is_release(v)]
    return releases[:keep] + non_releases


def newest_release(labels: list[str]) -> str | None:
    releases = sorted((v for v in labels if is_release(v)), key=sem_key, reverse=True)
    return releases[0] if releases else None


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #
def resolve_sources(manifest: Path, prefer_local: bool, clone_dir: Path | None) -> list[dict]:
    """Run resolve_sources.py and return the resolved manifest as a list."""
    cmd = [sys.executable, str(RESOLVE_SOURCES), "--manifest", str(manifest)]
    if prefer_local:
        cmd.append("--prefer-local")
    if clone_dir is not None:
        cmd += ["--clone", str(clone_dir)]
    eprint(f"[resolve] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"[error] resolve_sources.py failed (exit {proc.returncode}).")
    return json.loads(proc.stdout)


def source_root(entry: dict, clone_dir: Path | None) -> Path | None:
    """Filesystem root of a resolved source (local path or cloned repo)."""
    if entry.get("kind") == "local" and entry.get("path"):
        return Path(entry["path"]).resolve()
    if entry.get("kind") == "repo":
        if clone_dir is not None:
            return (clone_dir / entry["name"]).resolve()
    return None


def python_src_and_package(root: Path, package_root: str | None) -> tuple[Path, str]:
    """Compute the emitter's ``--src`` dir + ``--package`` name for a python
    source. Every python source is documented as part of the shared ``bloqade``
    namespace, so ``--src`` is that repo's ``bloqade`` directory and
    ``--package`` is ``bloqade`` (griffe discovers the repo's namespace slice).
    """
    pr = (package_root or "").strip("/")
    parts = [p for p in pr.split("/") if p]
    if TOP_PACKAGE in parts:
        idx = parts.index(TOP_PACKAGE)
        return root.joinpath(*parts[: idx + 1]), TOP_PACKAGE
    if parts:
        return root.joinpath(*parts), parts[-1]
    return root, root.name


# --------------------------------------------------------------------------- #
# Emitters
# --------------------------------------------------------------------------- #
def run_python_emitter(entry: dict, root: Path, version: str, out_dir: Path, dry: bool) -> bool:
    src, package = python_src_and_package(root, entry.get("package_root"))
    if not src.is_dir():
        eprint(f"[warn] {entry['name']}: python src not found: {src} (skipping).")
        return False
    cmd = [
        "uv", "run", "python", "-m", "bloqade_docs_emit_python",
        "--package", package,
        "--src", str(src),
        "--repo", entry.get("repo") or "QuEraComputing/bloqade",
        "--ref", entry.get("ref") or "main",
        "--version", version,
        "--mount", f"api/{version}/python",
        "--out", str(out_dir),
    ]
    eprint(f"[python] {entry['name']} ({version}): {' '.join(cmd)}")
    if dry:
        return True
    subprocess.run(cmd, cwd=PY_EMITTER_DIR, check=True)
    return True


def run_rust_emitter(
    entry: dict,
    root: Path | None,
    version: str,
    out_dir: Path,
    toolchain: str,
    rust_json: Path | None,
    dry: bool,
) -> bool:
    crate = entry.get("crate")
    cmd = ["uv", "run", str(RUST_EMITTER)]
    if rust_json is not None:
        cmd += ["--json", str(rust_json)]
    else:
        if root is None:
            eprint(f"[warn] {entry['name']}: no rust workspace resolved (skipping).")
            return False
        if not crate:
            eprint(f"[warn] {entry['name']}: rust source has no `crate` (skipping).")
            return False
        cmd += ["--run-cargo", "--workspace", str(root), "--crate", crate, "--toolchain", toolchain]
    cmd += [
        "--repo", entry.get("repo") or "QuEraComputing/bloqade-lanes",
        "--ref", entry.get("ref") or "main",
        "--mount", f"api/{version}/rust",
        "--out", str(out_dir),
    ]
    eprint(f"[rust] {entry['name']} ({version}): {' '.join(cmd)}")
    if dry:
        return True
    subprocess.run(cmd, check=True)
    return True


# --------------------------------------------------------------------------- #
# Inventory merge + per-lang index pages
# --------------------------------------------------------------------------- #
def merge_inventory(out_dir: Path, lang: str, version: str, extra: list[dict]) -> None:
    """Read a freshly-emitted ``inventory.<lang>.json`` from ``out_dir`` and
    append its entries to ``extra`` (in-place). The stray per-source inventory
    file left in the content tree is removed (only MDX belongs there)."""
    inv_file = out_dir / f"inventory.{lang}.json"
    if inv_file.is_file():
        try:
            extra.extend(json.loads(inv_file.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover - defensive
            eprint(f"[warn] could not read {inv_file}: {exc}")
        inv_file.unlink()


def write_index_page(lang_dir: Path, version: str, lang: str) -> None:
    """Write ``api/<V>/<lang>/index.mdx`` — a tiny landing linking to the
    top-level modules/crates emitted under it (so ``/api/<V>/<lang>/`` resolves
    and the ``/api/latest/<lang>`` sidebar link has a real target)."""
    if not lang_dir.is_dir():
        return
    tops: list[tuple[str, str]] = []  # (label, href)
    for child in sorted(lang_dir.iterdir()):
        if child.name.startswith((".", "_")) or child.name == "index.mdx":
            continue
        if child.is_dir():
            tops.append((child.name, f"/api/{version}/{lang}/{child.name}/"))
        elif child.suffix == ".mdx":
            tops.append((child.stem, f"/api/{version}/{lang}/{child.stem}/"))
    title = "Python API" if lang == "python" else "Rust API"
    lines = [
        "---",
        f'title: "{title} ({version})"',
        f'description: "Generated {lang} API reference for bloqade {version}."',
        f"language: {lang}",
        f'apiVersion: "{version}"',
        "---",
        "",
        "{/* GENERATED by website/scripts/build_versions.py — do not edit by hand. */}",
        "",
        f"Generated {lang} API reference for the `{version}` distribution.",
        "",
    ]
    if tops:
        lines.append("## Packages" if lang == "python" else "## Crates")
        lines.append("")
        for label, href in tops:
            lines.append(f"- [`{label}`]({href})")
        lines.append("")
    (lang_dir / "index.mdx").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# --limit parsing:  "NAME[,NAME]"  (all versions)  or  "VER=NAME[,NAME]"
# --------------------------------------------------------------------------- #
def parse_limits(values: list[str]) -> tuple[set[str] | None, dict[str, set[str]]]:
    glob: set[str] | None = None
    per: dict[str, set[str]] = {}
    for raw in values or []:
        if "=" in raw:
            ver, names = raw.split("=", 1)
            per.setdefault(ver.strip(), set()).update(
                n.strip() for n in names.split(",") if n.strip()
            )
        else:
            glob = (glob or set())
            glob.update(n.strip() for n in raw.split(",") if n.strip())
    return glob, per


def allowed_names(version: str, glob: set[str] | None, per: dict[str, set[str]]) -> set[str] | None:
    if version in per:
        return per[version]
    return glob  # None => all


# --------------------------------------------------------------------------- #
# Per-version build
# --------------------------------------------------------------------------- #
def build_version(
    version: str,
    sources: list[dict],
    *,
    clone_dir: Path | None,
    langs: set[str],
    limit_names: set[str] | None,
    rust_json: Path | None,
    toolchain: str,
    clean: bool,
    dry: bool,
) -> None:
    version_dir = API_DIR / version
    if clean and version_dir.exists() and not dry:
        eprint(f"[clean] removing {version_dir}")
        shutil.rmtree(version_dir)

    py_out = version_dir / "python"
    rust_out = version_dir / "rust"

    py_inventory: list[dict] = []
    rust_inventory: list[dict] = []
    emitted_py = emitted_rust = False

    for entry in sources:
        name = entry.get("name")
        if limit_names is not None and name not in limit_names:
            continue
        lang = entry.get("language")
        if lang not in langs:
            continue
        root = source_root(entry, clone_dir)
        if lang == "python":
            if root is None:
                eprint(f"[warn] {name}: python source unresolved (kind={entry.get('kind')}).")
                continue
            if run_python_emitter(entry, root, version, py_out, dry) and not dry:
                merge_inventory(py_out, "python", version, py_inventory)
                emitted_py = True
        elif lang == "rust":
            if run_rust_emitter(entry, root, version, rust_out, toolchain, rust_json, dry):
                if not dry:
                    merge_inventory(rust_out, "rust", version, rust_inventory)
                emitted_rust = True

    if dry:
        return

    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    if emitted_py:
        write_index_page(py_out, version, "python")
        _write_inventory(py_inventory, "python", version)
    if emitted_rust:
        write_index_page(rust_out, version, "rust")
        _write_inventory(rust_inventory, "rust", version)


def _write_inventory(entries: list[dict], lang: str, version: str) -> None:
    dest = INVENTORY_DIR / f"inventory.{lang}.{version}.json"
    dest.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    eprint(f"[inventory] wrote {dest} ({len(entries)} symbols)")


# --------------------------------------------------------------------------- #
# versions.json manifest + pruning
# --------------------------------------------------------------------------- #
def discover_versions() -> list[str]:
    """Every ``api/<V>`` directory currently on disk (a directory is a version;
    the evergreen ``index.mdx`` / ``compatibility.mdx`` are files, not dirs)."""
    if not API_DIR.is_dir():
        return []
    return sorted(p.name for p in API_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))


def prune_versions(keep: int) -> list[str]:
    present = discover_versions()
    keep_set = set(select_versions_to_keep(present, keep=keep))
    pruned: list[str] = []
    for v in present:
        if v in keep_set:
            continue
        vdir = API_DIR / v
        if vdir.exists():
            eprint(f"[prune] removing {vdir} (keep={keep})")
            shutil.rmtree(vdir)
        for lang in ("python", "rust"):
            inv = INVENTORY_DIR / f"inventory.{lang}.{v}.json"
            if inv.exists():
                inv.unlink()
        pruned.append(v)
    return pruned


def write_versions_manifest(latest_override: str | None) -> dict:
    present = discover_versions()
    if not present:
        # committed-safe default so the site still builds with no api content.
        present = ["dev"]
    has_dev = "dev" in present
    latest = latest_override or newest_release(present) or ("dev" if has_dev else present[0])

    # Order: releases newest-first, then any non-release labels (dev last).
    releases = sorted((v for v in present if is_release(v)), key=sem_key, reverse=True)
    others = [v for v in present if not is_release(v)]
    ordered = releases + others

    versions = [
        {
            "label": v,
            "path": f"/api/{v}",
            "latest": v == latest,
            "dev": v == "dev",
            "release": is_release(v),
        }
        for v in ordered
    ]
    manifest = {
        "latest": latest,
        "dev": "dev" if has_dev else None,
        "versions": versions,
    }
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    eprint(f"[versions] wrote {VERSIONS_JSON}: {[v['label'] for v in versions]} (latest={latest})")
    return manifest


def _ensure_pagefind_false(path: Path) -> None:
    """Insert ``pagefind: false`` into an MDX file's YAML frontmatter
    (idempotent). Excludes the page from the Pagefind search index WITHOUT
    affecting the page itself, its links, or its cross-references."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return
    try:
        end = lines.index("---", 1)
    except ValueError:
        return
    if any(l.strip().startswith("pagefind:") for l in lines[1:end]):
        return  # already scoped
    path.write_text("\n".join(["---", "pagefind: false", *lines[1:]]), encoding="utf-8")


def apply_search_scope(latest: str) -> None:
    """Scope site search to the LATEST API version. Adds ``pagefind: false`` to
    every generated MDX under non-latest ``api/<V>/`` trees, so the site-wide
    search returns only the latest API (plus the evergreen guides / blog /
    reference, which live OUTSIDE ``api/<V>/`` and are untouched). Older versions
    stay fully browsable via the header version selector."""
    scoped = 0
    for v in discover_versions():
        if v == latest:
            continue
        for mdx in (API_DIR / v).rglob("*.mdx"):
            _ensure_pagefind_false(mdx)
            scoped += 1
    eprint(f"[search] scoped {scoped} non-latest API page(s) out of search (latest={latest})")


def run_build_inventory() -> None:
    """Best-effort: regenerate the merged xref inventory via node (the astro
    build also does this at config:setup, so this is optional convenience)."""
    if shutil.which("node") is None or not BUILD_INVENTORY.is_file():
        return
    try:
        subprocess.run(["node", str(BUILD_INVENTORY)], check=True)
    except Exception as exc:  # pragma: no cover - defensive
        eprint(f"[warn] build_inventory.mjs failed: {exc}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--versions", default="dev", help="Comma list of version labels (default: dev).")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="docs.sources.toml.")
    ap.add_argument("--prefer-local", action="store_true", help="Emit from sibling checkouts.")
    ap.add_argument("--clone-dir", type=Path, default=None, help="Clone repo sources here (CI).")
    ap.add_argument(
        "--limit", action="append", default=[],
        help="Restrict sources. 'NAME[,NAME]' (all versions) or 'VER=NAME[,NAME]'.",
    )
    ap.add_argument(
        "--langs", default="python,rust",
        help="Languages to emit (comma list; default: python,rust).",
    )
    ap.add_argument("--rust-json", type=Path, default=None, help="Reuse an existing rustdoc JSON.")
    ap.add_argument(
        # Default to the shared RUSTDOC_NIGHTLY constant (mise.toml [env]) so the
        # pinned nightly date lives in ONE place; fall back to plain "nightly"
        # when the env var is unset (e.g. an ad-hoc local run outside mise).
        "--rust-toolchain",
        default=os.environ.get("RUSTDOC_NIGHTLY", "nightly"),
        help="rustdoc-JSON nightly toolchain (default: $RUSTDOC_NIGHTLY or 'nightly').",
    )
    ap.add_argument("--clean", action="store_true", help="Wipe api/<V> before emitting each version.")
    ap.add_argument("--latest", default=None, help="Force which label the 'latest' alias points at.")
    ap.add_argument("--keep", type=int, default=None, help="Prune to the newest N releases (dev kept).")
    ap.add_argument("--dry-run", action="store_true", help="Print emitter commands; do not run them.")
    args = ap.parse_args(argv)

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    if not versions:
        ap.error("--versions must list at least one label")
    langs = {l.strip() for l in args.langs.split(",") if l.strip()}
    glob_limit, per_limit = parse_limits(args.limit)

    if not args.manifest.is_file():
        ap.error(f"manifest not found: {args.manifest}")

    sources = resolve_sources(args.manifest, args.prefer_local, args.clone_dir)
    eprint(f"[resolve] {len(sources)} source(s): "
           + ", ".join(f"{s['name']}({s['language']})" for s in sources))

    for version in versions:
        eprint(f"\n=== building api version '{version}' ===")
        build_version(
            version,
            sources,
            clone_dir=args.clone_dir,
            langs=langs,
            limit_names=allowed_names(version, glob_limit, per_limit),
            rust_json=args.rust_json,
            toolchain=args.rust_toolchain,
            clean=args.clean,
            dry=args.dry_run,
        )

    if args.dry_run:
        eprint("\n[dry-run] no files written; skipping versions.json + inventory merge.")
        return 0

    if args.keep is not None:
        prune_versions(args.keep)

    manifest = write_versions_manifest(args.latest)
    apply_search_scope(manifest["latest"])
    run_build_inventory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
