"""Cache-first *batch* build of every Bloqade tutorial.

This is the entry point behind ``mise run docs:notebooks``. Where ``__main__``
emits **one** tutorial, this drives the whole set from an explicit manifest so a
docs build never re-executes an unchanged notebook.

Source of truth
---------------

The jupytext ``percent`` sources still live in the legacy ``docs/digital/**``
tree (tracked in git). We read them **in place** via the ``MANIFEST`` below
rather than copying them into ``website/`` for two reasons:

* The sources reference sibling image assets by relative/basename paths
  (e.g. ``docs/digital/tutorials/figures/*.svg``); the renderer resolves those
  with a bounded by-basename search from the source's directory. Reading in
  place keeps that resolution working with zero asset duplication or drift.
* It avoids maintaining a second copy of every tutorial.

The manifest — not the on-disk filename — defines each tutorial's **group** and
**slug**, so it pins the exact URLs the site links to (some sources are renamed:
``deutsch_squin.py`` -> ``squin/deutsch``, ``circuits_with_bloqade.py`` ->
``tutorials/circuits``, and the ``gemini_logical`` directory maps to the
``gemini`` group). The generated pages land at
``website/src/content/docs/guides/tutorials/<group>/<slug>.mdx`` with images
under ``website/public/tutorials/<group>/<slug>/``.

When ``docs/`` is eventually removed, either (a) point ``SOURCE_ROOT`` at a new
tracked location and move the sources + their referenced figures there, or
(b) copy the sources into ``website/notebooks/**``. Either way only this
manifest changes; the slugs/groups (and therefore every link) stay put.

Execution & the cache
---------------------

``execute`` per entry is an opt-out for notebooks whose backend cannot run in
the docs environment yet (``tsim`` / ``gemini`` fail to execute today, so they
ship as static renders — see ``guides/tutorials/index.mdx``). Effective
execution is ``execution_enabled()`` (the ``EXECUTE_NOTEBOOKS`` gate) AND the
per-entry flag. When the gate is off, every tutorial renders statically.

When the gate is on, each executed notebook is content-addressed cached (see
``pipeline.py``): the second build of an unchanged tutorial is a cache hit with
**no** kernel execution, and editing one tutorial re-executes only that one.
Flipping an ``execute`` flag here edits this file, which is part of the cache
fingerprint, so it correctly re-executes the affected set.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .pipeline import (
    Config,
    Result,
    cache_enabled_default,
    default_cache_dir,
    default_python,
    detect_repo_root,
    execution_enabled,
)


@dataclass(frozen=True)
class NotebookSpec:
    """One tutorial: its source (relative to ``SOURCE_ROOT``), its
    ``<group>/<slug>`` name (which fixes the page URL), and whether it should
    be executed (vs. shipped as a static render)."""

    source: str
    name: str
    execute: bool


# Legacy jupytext source tree, relative to the repo root.
SOURCE_ROOT = Path("docs/digital")

# The complete tutorial set. Order is cosmetic (only affects the build log).
MANIFEST: list[NotebookSpec] = [
    # --- General tutorials -------------------------------------------------
    NotebookSpec("tutorials/circuits_with_bloqade.py", "tutorials/circuits", True),
    NotebookSpec("tutorials/auto_parallelism.py", "tutorials/auto_parallelism", True),
    # --- Squin -------------------------------------------------------------
    NotebookSpec("examples/squin/deutsch_squin.py", "squin/deutsch", True),
    NotebookSpec("examples/squin/ghz.py", "squin/ghz", True),
    # --- QASM2 -------------------------------------------------------------
    NotebookSpec("examples/qasm2/qft.py", "qasm2/qft", True),
    NotebookSpec("examples/qasm2/ghz.py", "qasm2/ghz", True),
    NotebookSpec(
        "examples/qasm2/pauli_exponentiation.py",
        "qasm2/pauli_exponentiation",
        True,
    ),
    NotebookSpec(
        "examples/qasm2/repeat_until_success.py",
        "qasm2/repeat_until_success",
        True,
    ),
    NotebookSpec("examples/qasm2/qaoa.py", "qasm2/qaoa", True),
    # --- tsim (static: the tsim backend cannot execute in docs yet) --------
    NotebookSpec(
        "examples/tsim/magic_state_distillation.py",
        "tsim/magic_state_distillation",
        False,
    ),
    # --- Gemini logical (static: depends on the tsim backend) --------------
    NotebookSpec(
        "examples/gemini_logical/simulator_device_demo.py",
        "gemini/simulator_device_demo",
        False,
    ),
    # --- Interop -----------------------------------------------------------
    NotebookSpec("examples/interop/noisy_ghz.py", "interop/noisy_ghz", True),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bloqade-docs-build-notebooks",
        description="Cache-first batch build of every Bloqade tutorial to MDX.",
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="Repo root (default: git top-level of the cwd).",
    )
    p.add_argument(
        "--source-root",
        default=None,
        help="Jupytext source tree (default: <repo-root>/docs/digital).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output dir for the generated .mdx "
        "(default: <repo-root>/website/src/content/docs/guides/tutorials).",
    )
    p.add_argument(
        "--public-dir",
        default=None,
        help="Dir images are written to "
        "(default: <repo-root>/website/public/tutorials).",
    )
    p.add_argument(
        "--asset-base",
        default="/tutorials",
        help="URL prefix for emitted <img src> attributes (default: /tutorials).",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Content-addressed cache dir "
        "(default: $BLOQADE_NOTEBOOK_CACHE_DIR or "
        "website/emitters/notebooks/.notebook-cache).",
    )
    p.add_argument("--kernel-python", default=default_python())
    p.add_argument("--kernel-name", default="python3")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Ignore the cache and re-execute every executable notebook.",
    )
    p.add_argument(
        "--allow-errors",
        action="store_true",
        help="Capture cell errors as outputs instead of a static fallback.",
    )
    return p


def _config_for(
    spec: NotebookSpec,
    args,
    repo_root: Path,
    source_root: Path,
    out: Path,
    public_dir: Path,
    cache_dir: Path,
) -> Config:
    # Effective execution: the global EXECUTE_NOTEBOOKS gate AND the per-entry
    # opt-out. When the gate is off, everything renders statically.
    execute = execution_enabled() and spec.execute
    return Config(
        input=(source_root / spec.source).resolve(),
        out=out,
        public_dir=public_dir,
        asset_base=args.asset_base,
        name=spec.name,
        repo_root=repo_root,
        cache_dir=cache_dir,
        kernel_python=args.kernel_python,
        kernel_name=args.kernel_name,
        timeout=args.timeout,
        execute=execute,
        use_cache=args.use_cache and cache_enabled_default(),
        allow_errors=args.allow_errors,
    )


def main(argv: list[str] | None = None) -> int:
    from .pipeline import run  # deferred: importing run pulls in render, fine

    args = build_parser().parse_args(argv)

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else detect_repo_root(Path.cwd())
    )
    source_root = (
        Path(args.source_root).resolve()
        if args.source_root
        else (repo_root / SOURCE_ROOT).resolve()
    )
    out = (
        Path(args.out).resolve()
        if args.out
        else (repo_root / "website/src/content/docs/guides/tutorials").resolve()
    )
    public_dir = (
        Path(args.public_dir).resolve()
        if args.public_dir
        else (repo_root / "website/public/tutorials").resolve()
    )
    cache_dir = (
        Path(args.cache_dir).resolve() if args.cache_dir else default_cache_dir()
    )

    gate = execution_enabled()
    sys.stderr.write(
        f"[notebooks] building {len(MANIFEST)} tutorial(s); "
        f"EXECUTE_NOTEBOOKS={'on' if gate else 'off'}; "
        f"cache={'on' if cache_enabled_default() and args.use_cache else 'off'} "
        f"@ {cache_dir}\n"
    )

    results: list[tuple[NotebookSpec, Result | None, str]] = []
    exit_code = 0
    for spec in MANIFEST:
        cfg = _config_for(
            spec, args, repo_root, source_root, out, public_dir, cache_dir
        )
        if not cfg.input.is_file():
            sys.stderr.write(f"[notebooks] MISSING  {spec.name} ({cfg.input})\n")
            results.append((spec, None, "missing"))
            exit_code = 1
            continue
        result = run(cfg)
        results.append((spec, result, result.status))

    # Summary: one line per tutorial + counts, so a build log makes the
    # cache behaviour (hits vs executions) obvious at a glance.
    hits = sum(1 for _, _, s in results if s == "cache-hit")
    execs = sum(1 for _, _, s in results if s == "executed")
    static = sum(1 for _, _, s in results if s in ("static-disabled", "static-failed"))
    sys.stderr.write(
        f"[notebooks] done: {len(MANIFEST)} tutorial(s) — "
        f"{hits} cache hit, {execs} executed, {static} static\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
