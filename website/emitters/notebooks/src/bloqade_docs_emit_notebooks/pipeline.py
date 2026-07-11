"""Parse -> (optionally execute, with caching) -> render a jupytext tutorial.

Execution is gated on the ``EXECUTE_NOTEBOOKS`` env flag (mirroring the current
mkdocs nightly). When enabled, the notebook is run with ``nbclient`` in a kernel
launched from ``--kernel-python`` (defaults to this venv's interpreter, which
has ``bloqade`` installed). Executed notebooks are cached by the source file's
content hash so an unchanged tutorial is never re-run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jupytext
import nbformat

from .render import RenderContext, render_notebook

log = logging.getLogger("bloqade-docs-emit-notebooks")

# Bump when the executor changes in a way that should invalidate caches.
CACHE_SCHEMA = "1"

_TRUTHY = {"1", "true", "yes", "on"}


def execution_enabled() -> bool:
    """Mirror the current mkdocs gate: EXECUTE_NOTEBOOKS in {1,true,yes,on}."""
    return os.environ.get("EXECUTE_NOTEBOOKS", "").strip().lower() in _TRUTHY


def slugify(stem: str) -> str:
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


def source_hash(source: str) -> str:
    h = hashlib.sha256()
    h.update(CACHE_SCHEMA.encode())
    h.update(b"\0")
    h.update(source.encode("utf-8"))
    return h.hexdigest()


def _cache_path(cache_dir: Path, digest: str) -> Path:
    return cache_dir / f"{digest}.ipynb"


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
        os.environ["JUPYTER_PATH"] = (
            tmp + (os.pathsep + old_jp if old_jp else "")
        )
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


def run(cfg: Config) -> Result:
    source = cfg.input.read_text(encoding="utf-8")
    digest = source_hash(source)

    # jupytext parses the percent .py into an nbformat NotebookNode.
    nb = jupytext.reads(source, fmt="py:percent")
    # Normalise to a known nbformat version so outputs round-trip cleanly.
    nb = nbformat.from_dict(nb)

    executed = False
    from_cache = False

    if cfg.execute:
        cache_file = _cache_path(cfg.cache_dir, digest)
        if cfg.use_cache and cache_file.is_file():
            log.info("cache hit for %s (%s)", cfg.input.name, digest[:12])
            nb = nbformat.read(cache_file, as_version=4)
            executed = True
            from_cache = True
        else:
            log.info("executing %s ...", cfg.input.name)
            try:
                execute_notebook(nb, cfg)
                executed = True
                cfg.cache_dir.mkdir(parents=True, exist_ok=True)
                nbformat.write(nb, cache_file)
                log.info("cached executed notebook -> %s", cache_file)
            except Exception as exc:  # noqa: BLE001 - degrade, never crash the build
                log.warning(
                    "execution FAILED for %s (%s); falling back to static "
                    "(non-executed) render",
                    cfg.input.name,
                    exc,
                )
                # Drop any partial outputs left by a mid-notebook failure.
                for cell in nb.cells:
                    if cell.get("cell_type") == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None
                executed = False
    else:
        log.info("execution disabled (EXECUTE_NOTEBOOKS not set); static render")

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
    mdx_path.write_text(mdx, encoding="utf-8")

    return Result(
        mdx_path=mdx_path,
        executed=executed,
        from_cache=from_cache,
        title_slug=cfg.name,
    )


def detect_repo_root(start: Path) -> Path:
    """git top-level of ``start``, else the repo root three levels above this file."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        )
        return Path(out.decode().strip())
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[4]


def default_python() -> str:
    return sys.executable
