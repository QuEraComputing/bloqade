"""Parse -> (optionally execute, with a content-addressed cache) -> render a
jupytext tutorial into Starlight-compatible MDX.

Execution is gated on the ``EXECUTE_NOTEBOOKS`` env flag (mirroring the mkdocs
nightly). When enabled, the notebook is run with ``nbclient`` in a kernel
launched from ``--kernel-python`` (defaults to this venv's interpreter, which
has ``bloqade`` installed).

Content-addressed cache
-----------------------

This mirrors the design of ppvm's ``docs/scripts/build-notebooks.py``. Executed
notebooks are cached under ``.notebook-cache/`` (or
``$BLOQADE_NOTEBOOK_CACHE_DIR``) so that an unchanged tutorial is **never**
re-executed, while a docs-only / CSS-only / prose-only change re-executes
*nothing*.

The cache key for one notebook is::

    sha256( CACHE_SCHEMA_VERSION + shared_fingerprint + notebook_source_bytes )

where ``shared_fingerprint`` is::

    sha256( CACHE_SCHEMA_VERSION + this emitter's own *.py sources
            + website/emitters/notebooks/uv.lock + the repo-root uv.lock )

Consequences (identical to ppvm's model):

* Editing **one** tutorial's ``.py`` changes only that notebook's key, so only
  that notebook re-executes; every other notebook stays a cache hit.
* Editing prose / CSS / the Astro site changes nothing the fingerprint hashes,
  so **no** notebook re-executes.
* Editing the emitter itself (this file, ``render.py``, ``escape.py``, the
  driver, …) or bumping a dependency (either ``uv.lock``) changes the
  ``shared_fingerprint`` and therefore **every** notebook's key, so the whole
  set re-executes. That is intentional: a rendering / sanitiser change must be
  reflected in every page, and a dependency bump may change numerical output.
* ``CACHE_SCHEMA_VERSION`` is a manual override to force a global invalidation
  when the cached artefact layout changes incompatibly.

We deliberately do **not** hash the ``bloqade`` package source tree (only its
pinned versions via the lockfiles). A pure-source change that doesn't touch a
lockfile won't invalidate cached outputs; ``pytest`` is the safety net for
those, and bumping ``CACHE_SCHEMA_VERSION`` (or ``BLOQADE_NOTEBOOK_CACHE=0``)
forces a rebuild when needed.

What is cached: the fully-executed ``.ipynb`` (as ``<key>.ipynb``). On a cache
hit we load it and re-run the **render** path with **no kernel** — which both
writes the ``.mdx`` and re-extracts the output images into ``public/`` — so the
whole page is reproduced offline and deterministically. Because ``render.py``
is part of the fingerprint, a cache hit can only happen when the renderer is
unchanged, so the reproduced output is byte-identical to what was cached.

We never cache a **static fallback** (a run where execution was requested but
failed, e.g. a missing backend): that would poison the cache with a "fake
executed" entry and shadow a real execution once the backend lands. Such a run
re-attempts on every build until it succeeds. A notebook that is *intentionally*
static (execution disabled) never touches the cache at all.

Environment variables
---------------------

* ``BLOQADE_NOTEBOOK_CACHE_DIR`` — override the cache directory (CI points this
  at a stable location restored via ``actions/cache``).
* ``BLOQADE_NOTEBOOK_CACHE=0`` — force re-execution regardless of what's on
  disk (useful when investigating numerical drift).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jupytext
import nbformat

from .render import RenderContext, render_notebook

log = logging.getLogger("bloqade-docs-emit-notebooks")

# Bump this to force-invalidate every cached notebook output (e.g. when the
# cached artefact layout changes incompatibly). Routine emitter edits do NOT
# need a bump — the emitter's own source is part of ``shared_fingerprint``.
CACHE_SCHEMA_VERSION = "2"

_TRUTHY = {"1", "true", "yes", "on"}

# Layout anchors, resolved relative to THIS file so the fingerprint is
# independent of the current working directory:
#   .../website/emitters/notebooks/src/bloqade_docs_emit_notebooks/pipeline.py
_HERE = Path(__file__).resolve()
_PKG_DIR = _HERE.parent  # .../src/bloqade_docs_emit_notebooks
_EMITTER_ROOT = _HERE.parents[2]  # .../website/emitters/notebooks
# notebooks -> emitters -> website -> <repo root>
_REPO_ROOT = _EMITTER_ROOT.parents[2]


def execution_enabled() -> bool:
    """Mirror the mkdocs gate: EXECUTE_NOTEBOOKS in {1,true,yes,on}."""
    return os.environ.get("EXECUTE_NOTEBOOKS", "").strip().lower() in _TRUTHY


def cache_enabled_default() -> bool:
    """Cache is on unless ``BLOQADE_NOTEBOOK_CACHE`` is explicitly ``0``."""
    return os.environ.get("BLOQADE_NOTEBOOK_CACHE", "1").strip() != "0"


def default_cache_dir() -> Path:
    """Content-addressed cache location.

    ``$BLOQADE_NOTEBOOK_CACHE_DIR`` wins (CI points this at an ``actions/cache``
    location); otherwise ``website/emitters/notebooks/.notebook-cache``.
    """
    override = os.environ.get("BLOQADE_NOTEBOOK_CACHE_DIR")
    if override:
        return Path(override).resolve()
    return (_EMITTER_ROOT / ".notebook-cache").resolve()


def slugify(stem: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower()
    return slug or "tutorial"


@dataclass
class Config:
    input: Path
    out: Path
    public_dir: Path
    asset_base: str
    name: str
    repo_root: Path
    cache_dir: Path
    kernel_python: str
    kernel_name: str
    timeout: int
    execute: bool
    use_cache: bool
    allow_errors: bool


# --------------------------------------------------------------------------- #
# Content-addressed cache fingerprint
# --------------------------------------------------------------------------- #


def _shared_fingerprint_files() -> list[Path]:
    """Files whose contents influence *every* notebook's output.

    Hashed once into ``shared_fingerprint`` and combined with each notebook's
    own source bytes to form its cache key.

    Includes every ``.py`` of this emitter (this file, ``render.py``,
    ``escape.py``, ``__main__.py``, the driver, …) so a rendering / sanitiser /
    output-mapping change invalidates cached outputs automatically. Includes the
    two lockfiles so a dependency bump (or a ``bloqade`` version change recorded
    there) invalidates them too. It deliberately does NOT hash the ``bloqade``
    package source tree — that would blow up the fingerprint on cosmetic edits.
    """
    files: list[Path] = sorted(_PKG_DIR.glob("*.py"))
    for lock in (_EMITTER_ROOT / "uv.lock", _REPO_ROOT / "uv.lock"):
        if lock.is_file():
            files.append(lock)
    return files


@lru_cache(maxsize=1)
def shared_fingerprint() -> bytes:
    h = hashlib.sha256()
    h.update(b"schema=")
    h.update(CACHE_SCHEMA_VERSION.encode("utf-8"))
    h.update(b"\0")
    for f in _shared_fingerprint_files():
        try:
            label = f.relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            label = f.name
        h.update(label.encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.digest()


def notebook_cache_key(source_bytes: bytes) -> str:
    """``sha256(shared_fingerprint + notebook_source_bytes)`` as hex."""
    h = hashlib.sha256()
    h.update(shared_fingerprint())
    h.update(b"\0")
    h.update(source_bytes)
    return h.hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.ipynb"


def _log(msg: str) -> None:
    """Emit a build-log line to stderr (always visible, unlike ``log.info``)."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def execute_notebook(nb, cfg: Config):
    """Execute ``nb`` in place using a kernel launched from cfg.kernel_python.

    A throwaway kernelspec is written pointing ``argv`` at ``cfg.kernel_python``
    so the executing interpreter is exactly the one we control (no dependence on
    a globally-registered ``python3`` kernelspec).
    """
    from nbclient import NotebookClient

    with tempfile.TemporaryDirectory(prefix="bloqade-kspec-") as tmp:
        spec_dir = Path(tmp) / "kernels" / cfg.kernel_name
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        cfg.kernel_python,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "bloqade",
                    "language": "python",
                }
            )
        )
        old_jp = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = tmp + (os.pathsep + old_jp if old_jp else "")
        try:
            client = NotebookClient(
                nb,
                timeout=cfg.timeout,
                kernel_name=cfg.kernel_name,
                allow_errors=cfg.allow_errors,
                # Run with the repo root as cwd so tutorials resolve imports and
                # relative data files the same way the mkdocs build does.
                resources={"metadata": {"path": str(cfg.repo_root)}},
            )
            client.execute()
        finally:
            if old_jp is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old_jp
    return nb


@dataclass
class Result:
    mdx_path: Path
    executed: bool
    from_cache: bool
    title_slug: str
    # One of: "cache-hit", "executed", "static-disabled", "static-failed".
    status: str


def run(cfg: Config) -> Result:
    source_bytes = cfg.input.read_bytes()
    source = source_bytes.decode("utf-8")
    key = notebook_cache_key(source_bytes)

    # jupytext parses the percent .py into an nbformat NotebookNode.
    nb = jupytext.reads(source, fmt="py:percent")
    # Normalise to a known nbformat version so outputs round-trip cleanly.
    nb = nbformat.from_dict(nb)

    executed = False
    from_cache = False
    status = "static-disabled"

    if cfg.execute:
        cache_file = _cache_path(cfg.cache_dir, key)
        if cfg.use_cache and cache_file.is_file():
            # Cache hit: reproduce the page from the cached executed notebook
            # with NO kernel execution.
            _log(f"[notebooks] cache hit  {cfg.name} ({key[:12]})")
            nb = nbformat.read(cache_file, as_version=4)
            executed = True
            from_cache = True
            status = "cache-hit"
        else:
            _log(f"[notebooks] executing {cfg.name} ... ({key[:12]})")
            try:
                execute_notebook(nb, cfg)
                executed = True
                status = "executed"
                # Populate the cache with the fully-executed notebook.
                cfg.cache_dir.mkdir(parents=True, exist_ok=True)
                nbformat.write(nb, cache_file)
                _log(f"[notebooks] cached     {cfg.name} -> {cache_file.name}")
            except Exception as exc:  # noqa: BLE001 - degrade, never crash build
                # Static fallback. Deliberately NOT cached: a deps-missing /
                # transient failure must re-attempt on the next build so a
                # fixed environment populates real outputs.
                _log(
                    f"[notebooks] FAILED    {cfg.name}: {type(exc).__name__}: "
                    f"{str(exc).splitlines()[0][:120]} -> static fallback "
                    "(not cached)"
                )
                # Drop any partial outputs left by a mid-notebook failure.
                for cell in nb.cells:
                    if cell.get("cell_type") == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None
                executed = False
                status = "static-failed"
    else:
        _log(f"[notebooks] static    {cfg.name} (execution disabled)")

    ctx = RenderContext(
        name=cfg.name,
        py_dir=cfg.input.resolve().parent,
        asset_dir=cfg.public_dir / cfg.name,
        asset_base=f"{cfg.asset_base.rstrip('/')}/{cfg.name}",
        repo_root=cfg.repo_root,
    )
    mdx = render_notebook(nb, ctx, executed=executed)

    cfg.out.mkdir(parents=True, exist_ok=True)
    mdx_path = cfg.out / f"{cfg.name}.mdx"
    # ``name`` may be namespaced with slashes (e.g. ``qasm2/qft``) to place the
    # page in a sidebar subgroup, so ensure the intermediate dirs exist.
    mdx_path.parent.mkdir(parents=True, exist_ok=True)
    mdx_path.write_text(mdx, encoding="utf-8")

    return Result(
        mdx_path=mdx_path,
        executed=executed,
        from_cache=from_cache,
        title_slug=cfg.name,
        status=status,
    )


def detect_repo_root(start: Path) -> Path:
    """git top-level of ``start``, else the layout-derived repo root."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        )
        return Path(out.decode().strip())
    except Exception:  # noqa: BLE001
        return _REPO_ROOT


def default_python() -> str:
    return sys.executable
