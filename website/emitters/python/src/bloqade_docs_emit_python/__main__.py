"""CLI for the Bloqade Python API MDX emitter.

Example:

    uv run python -m bloqade_docs_emit_python \
        --package bloqade \
        --src /path/to/bloqade-circuit/src/bloqade \
        --repo QuEraComputing/bloqade-circuit \
        --ref v0.14.1 \
        --version dev \
        --mount api/python \
        --out ./_out \
        --docstring-style google
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .emitter import Config, Emitter, detect_repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bloqade-docs-emit-python",
        description="Emit Bloqade Python API reference as Starlight-compatible MDX.",
    )
    parser.add_argument(
        "--package", required=True, help="Top-level import package, e.g. 'bloqade'."
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to the package root (the dir named like --package), e.g. .../src/bloqade.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Source repo slug, e.g. 'QuEraComputing/bloqade-circuit'.",
    )
    parser.add_argument(
        "--ref", required=True, help="Git ref (tag/branch/sha) for source links."
    )
    parser.add_argument(
        "--version", required=True, help="API doc version, e.g. 'dev' or '0.14'."
    )
    parser.add_argument(
        "--mount",
        default="api/python",
        help="Site mount path used to build inventory URLs (default: api/python).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "_out"),
        help="Output directory for generated MDX + inventory (default: emitters/python/_out).",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repo root for computing source-file relpaths (default: git top-level of --src).",
    )
    parser.add_argument(
        "--docstring-style",
        default="google",
        choices=["google", "numpy", "sphinx"],
        help="Docstring style griffe should parse (default: google).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show griffe + emitter warnings."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        # griffe is chatty about missing annotations; silence it unless -v.
        logging.getLogger("griffe").setLevel(logging.CRITICAL)

    repo_root = args.repo_root or detect_repo_root(args.src)

    cfg = Config(
        package=args.package,
        src=args.src,
        repo=args.repo,
        ref=args.ref,
        version=args.version,
        mount=args.mount,
        out=args.out,
        repo_root=repo_root,
        docstring_style=args.docstring_style,
    )

    emitter = Emitter(cfg)
    stats = emitter.run()

    print(f"[bloqade-docs-emit-python] package : {cfg.package}", file=sys.stderr)
    print(f"[bloqade-docs-emit-python] src     : {cfg.src}", file=sys.stderr)
    print(f"[bloqade-docs-emit-python] repoRoot: {repo_root}", file=sys.stderr)
    print(f"[bloqade-docs-emit-python] out     : {cfg.out}", file=sys.stderr)
    print(
        f"[bloqade-docs-emit-python] modules emitted={stats.modules_emitted} "
        f"skipped={stats.modules_skipped} symbols={stats.symbols}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
