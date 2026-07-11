"""CLI for the Bloqade notebook/tutorial MDX emitter.

Examples:

    # Static render (no execution) into the scratch _out dir:
    uv run bloqade-docs-emit-notebooks \
        ../../../docs/digital/examples/qasm2/ghz.py

    # Executed render into the site, images into public/tutorials/:
    EXECUTE_NOTEBOOKS=1 uv run bloqade-docs-emit-notebooks \
        ../../../docs/digital/examples/qasm2/ghz.py \
        --out ../../src/content/docs/guides/tutorials \
        --public-dir ../../public/tutorials \
        --asset-base /tutorials
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import (
    Config,
    default_python,
    detect_repo_root,
    execution_enabled,
    run,
    slugify,
)


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parents[2]  # emitters/notebooks
    parser = argparse.ArgumentParser(
        prog="bloqade-docs-emit-notebooks",
        description="Execute a jupytext percent .py tutorial and emit MDX with outputs.",
    )
    parser.add_argument("input", help="Path to the jupytext percent-format .py tutorial.")
    parser.add_argument(
        "--out",
        default=str(here / "_out"),
        help="Output dir for the generated .mdx (default: emitters/notebooks/_out).",
    )
    parser.add_argument(
        "--public-dir",
        default=None,
        help="Dir images are written to (default: <out>/public/tutorials). "
        "Point at website/public/tutorials for a real site build.",
    )
    parser.add_argument(
        "--asset-base",
        default="/tutorials",
        help="URL prefix for emitted image src attributes (default: /tutorials).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Tutorial slug (default: slugified filename stem). Names the .mdx, "
        "the public/<name>/ asset dir, and the asset URL path.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root; kernel cwd + asset fallback search (default: git top-level).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(here / ".cache"),
        help="Executed-notebook cache dir, keyed by source hash "
        "(default: emitters/notebooks/.cache).",
    )
    parser.add_argument(
        "--kernel-python",
        default=default_python(),
        help="Python executable used to launch the execution kernel "
        "(default: this interpreter, which has bloqade).",
    )
    parser.add_argument(
        "--kernel-name", default="python3", help="Kernel name (default: python3)."
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Per-cell execution timeout, seconds."
    )
    execgroup = parser.add_mutually_exclusive_group()
    execgroup.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        default=None,
        help="Force execution (overrides EXECUTE_NOTEBOOKS).",
    )
    execgroup.add_argument(
        "--no-execute",
        dest="execute",
        action="store_false",
        help="Force static render (overrides EXECUTE_NOTEBOOKS).",
    )
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Ignore the cache and always re-execute.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Capture cell errors as outputs instead of falling back to static.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    name = args.name or slugify(input_path.stem)
    out = Path(args.out).resolve()
    public_dir = (
        Path(args.public_dir).resolve()
        if args.public_dir
        else out / "public" / "tutorials"
    )
    repo_root = Path(args.repo_root).resolve() if args.repo_root else detect_repo_root(
        input_path.parent
    )
    execute = execution_enabled() if args.execute is None else args.execute

    cfg = Config(
        input=input_path,
        out=out,
        public_dir=public_dir,
        asset_base=args.asset_base,
        name=name,
        repo_root=repo_root,
        cache_dir=Path(args.cache_dir).resolve(),
        kernel_python=args.kernel_python,
        kernel_name=args.kernel_name,
        timeout=args.timeout,
        execute=execute,
        use_cache=args.use_cache,
        allow_errors=args.allow_errors,
    )

    result = run(cfg)

    print(f"[emit-notebooks] input   : {cfg.input}", file=sys.stderr)
    print(f"[emit-notebooks] name    : {cfg.name}", file=sys.stderr)
    print(f"[emit-notebooks] execute : {cfg.execute}", file=sys.stderr)
    print(
        f"[emit-notebooks] executed: {result.executed}"
        f" (from_cache={result.from_cache})",
        file=sys.stderr,
    )
    print(f"[emit-notebooks] mdx     : {result.mdx_path}", file=sys.stderr)
    if cfg.execute and result.executed:
        print(f"[emit-notebooks] assets  : {cfg.public_dir / cfg.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
