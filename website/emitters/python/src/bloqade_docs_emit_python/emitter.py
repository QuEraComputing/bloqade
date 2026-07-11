"""Walk a griffe module tree and emit one MDX page per module + an inventory."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from griffe import GriffeLoader, Kind, Parser

from . import escape, render

logger = logging.getLogger("bloqade_docs_emit_python")


# Ported verbatim from docs/scripts/gen_ref_nav.py (mkdocstrings navigation
# generator). These substrings hide internal / deprecated / non-doc paths.
SKIP_KEYWORDS = [
    ".venv",  ## skip virtual environment
    "julia",  ## [KHW] skip for now since we didn't have julia codegen rdy
    "builder/base",  ## hiding from user
    "builder/terminate",  ## hiding from user
    "ir/tree_print",  ## hiding from user
    "ir/visitor",  ## hiding from user
    "codegen/",  ## hiding from user
    "builder/factory",  ## hiding from user
    "builder_old",  ## deprecated from user
    "task_old",  ## deprecated from user
    "visualization",  ## hiding from user
    "submission/capabilities",  ## hiding from user
    "submission/quera_api_client",
    "test/",
    "tests/",
    "test_utils",
    "docs/",
    "debug/",
    "squin/cirq/emit/",  # missing __init__.py upstream; no docs anyway
    "demo/",
    "scripts/",
    "examples/",
    "crates/",
    "benchmarks/",
]


@dataclass
class Config:
    package: str
    src: str
    repo: str
    ref: str
    version: str
    mount: str
    out: str
    repo_root: str
    docstring_style: str


@dataclass
class Stats:
    modules_emitted: int = 0
    modules_skipped: int = 0
    skipped_paths: list[str] = field(default_factory=list)
    symbols: int = 0


class Emitter:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.mount = config.mount.strip("/")
        self.repo_root = Path(config.repo_root).resolve()
        self.inventory: list[dict] = []
        self.stats = Stats()

    # ------------------------------------------------------------------ #
    # URL / path helpers
    # ------------------------------------------------------------------ #

    def _relpath(self, filepath: Any) -> str | None:
        if filepath is None:
            return None
        if isinstance(filepath, (list, tuple)):
            filepath = filepath[0] if filepath else None
        if filepath is None:
            return None
        try:
            rel = os.path.relpath(Path(filepath).resolve(), self.repo_root)
        except Exception:  # pragma: no cover - defensive
            return None
        return rel.replace(os.sep, "/")

    def source_url(self, obj: Any, with_line: bool = True) -> str | None:
        rel = self._relpath(getattr(obj, "filepath", None))
        if rel is None:
            return None
        url = f"https://github.com/{self.cfg.repo}/blob/{self.cfg.ref}/{rel}"
        lineno = getattr(obj, "lineno", None)
        if with_line and lineno:
            url += f"#L{lineno}"
        return url

    def page_url(self, module_fq: str) -> str:
        slug = module_fq.replace(".", "/")
        return f"/{self.mount}/{slug}/"

    def symbol_url(self, module_fq: str, fq_name: str) -> str:
        return f"{self.page_url(module_fq)}#{fq_name}"

    def page_path(self, module: Any) -> Path:
        parts = module.path.split(".")
        if self._is_package(module):
            return Path(*parts, "index.mdx")
        return Path(*parts[:-1], parts[-1] + ".mdx")

    @staticmethod
    def _is_package(module: Any) -> bool:
        fp = getattr(module, "filepath", None)
        if isinstance(fp, (list, tuple)):
            return True
        if fp is not None and Path(fp).name == "__init__.py":
            return True
        return False

    # ------------------------------------------------------------------ #
    # Skip rules (ported from gen_ref_nav.py)
    # ------------------------------------------------------------------ #

    def should_skip_module(self, module: Any) -> tuple[bool, str]:
        name = module.name
        # Private module (matches gen_ref_nav's parts[-1].startswith("_")).
        if name.startswith("_") and name != "__init__":
            return True, f"{module.path} (private module)"
        rel = self._relpath(getattr(module, "filepath", None)) or module.path
        for kw in SKIP_KEYWORDS:
            if kw in rel:
                return True, f"{rel} (matches '{kw}')"
        return False, ""

    @staticmethod
    def is_private(name: str) -> bool:
        return name.startswith("_")

    # ------------------------------------------------------------------ #
    # Loading + walking
    # ------------------------------------------------------------------ #

    def load(self) -> Any:
        search_path = str(Path(self.cfg.src).resolve().parent)
        parser = Parser(self.cfg.docstring_style)
        loader = GriffeLoader(search_paths=[search_path], docstring_parser=parser)
        return loader.load(self.cfg.package)

    def iter_modules(self, module: Any) -> Iterator[Any]:
        """Yield emittable modules depth-first, honoring skip rules."""
        skip, reason = self.should_skip_module(module)
        if skip:
            self.stats.modules_skipped += 1
            self.stats.skipped_paths.append(reason)
            return
        yield module
        for member in module.members.values():
            if getattr(member, "is_alias", False):
                continue
            try:
                is_mod = member.kind == Kind.MODULE
            except Exception:
                continue
            if is_mod:
                yield from self.iter_modules(member)

    # ------------------------------------------------------------------ #
    # Emission
    # ------------------------------------------------------------------ #

    def run(self) -> Stats:
        root = self.load()
        out_dir = Path(self.cfg.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for module in self.iter_modules(root):
            try:
                self.emit_module(module, out_dir)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("failed to emit %s: %s", module.path, exc)
        self._write_inventory(out_dir)
        return self.stats

    def _write_inventory(self, out_dir: Path) -> None:
        path = out_dir / "inventory.python.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.inventory, fh, indent=2)
            fh.write("\n")

    def _record(self, fq_name: str, kind: str, module_fq: str) -> None:
        self.inventory.append(
            {"fqName": fq_name, "kind": kind, "url": self.symbol_url(module_fq, fq_name)}
        )
        self.stats.symbols += 1

    def emit_module(self, module: Any, out_dir: Path) -> None:
        module_fq = module.path
        doc = render.analyze_docstring(module)
        lines: list[str] = []
        lines.extend(self._frontmatter(module, doc))
        lines.append(
            "{/* AUTO-GENERATED by bloqade-docs-emit-python. Do not edit by hand. */}"
        )
        lines.append("")

        lines.append("<ApiModule")
        lines.append(f'  name="{escape.attr(module.name)}"')
        lines.append(f'  fqName="{escape.attr(module_fq)}"')
        if doc.summary:
            lines.append(f'  summary="{escape.attr(doc.summary)}"')
        lines.append(">")
        lines.append("")
        self.inventory.append(
            {"fqName": module_fq, "kind": "module", "url": self.symbol_url(module_fq, module_fq)}
        )

        lines.extend(render.prose_block(doc.body))

        attr_members: list[Any] = []
        for member in module.members.values():
            if getattr(member, "is_alias", False):
                continue
            try:
                kind = member.kind
            except Exception:
                continue
            if self.is_private(member.name):
                continue
            if kind == Kind.CLASS:
                lines.extend(self.emit_class(member, module_fq))
            elif kind == Kind.FUNCTION:
                lines.extend(self.emit_callable(member, module_fq, "function"))
            elif kind == Kind.ATTRIBUTE:
                attr_members.append(member)

        if attr_members:
            empty_doc = render.ParsedDoc()
            table = render.attributes_table(empty_doc, attr_members)
            if table:
                for member in attr_members:
                    self._record(member.path, "attribute", module_fq)
                lines.extend(table)

        lines.append("</ApiModule>")
        lines.append("")

        page = out_dir / self.page_path(module)
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("\n".join(lines), encoding="utf-8")
        self.stats.modules_emitted += 1

    def _frontmatter(self, module: Any, doc: render.ParsedDoc) -> list[str]:
        cfg = self.cfg
        lines = ["---", f"title: {escape.yaml_dq(module.path)}"]
        if doc.summary:
            lines.append(f"description: {escape.yaml_dq(doc.summary)}")
        lines.append("language: python")
        lines.append(f"fqName: {escape.yaml_dq(module.path)}")
        lines.append(f"apiVersion: {escape.yaml_dq(cfg.version)}")
        lines.append(f"sourceRepo: {escape.yaml_dq(cfg.repo)}")
        lines.append(f"sourceRef: {escape.yaml_dq(cfg.ref)}")
        src = self.source_url(module, with_line=False)
        if src:
            lines.append(f"sourceUrl: {escape.yaml_dq(src)}")
        lines.append("---")
        lines.append("")
        return lines

    def emit_class(self, cls: Any, module_fq: str) -> list[str]:
        fq = cls.path
        doc = render.analyze_docstring(cls)
        source_url = self.source_url(cls)
        self._record(fq, "class", module_fq)

        bases = []
        for base in getattr(cls, "bases", []) or []:
            b = render._annotation_str(base)
            if b:
                bases.append(b)

        lines: list[str] = ["<ApiClass"]
        lines.append(f'  name="{escape.attr(cls.name)}"')
        lines.append(f'  fqName="{escape.attr(fq)}"')
        if bases:
            arr = ", ".join(f"'{escape.js_str(b)}'" for b in bases)
            lines.append(f"  bases={{[{arr}]}}")
        if doc.summary:
            lines.append(f'  summary="{escape.attr(doc.summary)}"')
        if source_url:
            lines.append(f'  sourceUrl="{escape.attr(source_url)}"')
        lines.append(">")
        lines.append("")

        lines.extend(render.signature_block(render.class_signature(cls)))
        lines.extend(render.prose_block(doc.body))

        # Constructor parameters (merge __init__ signature + class/__init__ docs).
        init = cls.members.get("__init__")
        if init is not None and not getattr(init, "is_alias", False):
            init_doc = render.analyze_docstring(init)
            param_doc = init_doc if init_doc.params else doc
            rows = render._params_from_signature(init, param_doc, drop_self=True)
            lines.extend(render.params_table(rows))

        # Attributes (docstring section preferred, else attribute members).
        attr_members = [
            m
            for m in cls.members.values()
            if not getattr(m, "is_alias", False)
            and getattr(m, "kind", None) == Kind.ATTRIBUTE
            and not self.is_private(m.name)
            and "property" not in getattr(m, "labels", set())
        ]
        table = render.attributes_table(doc, attr_members)
        if table:
            for member in attr_members:
                self._record(member.path, "attribute", module_fq)
            lines.extend(table)

        # Methods and properties.
        for member in cls.members.values():
            if getattr(member, "is_alias", False):
                continue
            try:
                kind = member.kind
            except Exception:
                continue
            name = member.name
            if self.is_private(name):
                continue
            labels = getattr(member, "labels", set())
            if kind == Kind.FUNCTION:
                lines.extend(self.emit_callable(member, module_fq, "method"))
            elif kind == Kind.ATTRIBUTE and "property" in labels:
                lines.extend(self.emit_property(member, module_fq))

        lines.append("</ApiClass>")
        lines.append("")
        return lines

    def emit_callable(self, func: Any, module_fq: str, kind: str) -> list[str]:
        fq = func.path
        doc = render.analyze_docstring(func)
        source_url = self.source_url(func)
        self._record(fq, kind, module_fq)

        lines: list[str] = ["<ApiFn"]
        lines.append(f'  name="{escape.attr(func.name)}"')
        lines.append(f'  fqName="{escape.attr(fq)}"')
        lines.append(f'  kind="{kind}"')
        if source_url:
            lines.append(f'  sourceUrl="{escape.attr(source_url)}"')
        lines.append(">")
        lines.append("")

        is_method = kind in ("method", "property")
        try:
            sig = render.function_signature(func, is_method=is_method)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("signature failed for %s: %s", fq, exc)
            sig = f"def {func.name}(...)"
        lines.extend(render.signature_block(sig))

        if doc.summary or doc.body:
            combined = (doc.summary + ("\n\n" + doc.body if doc.body else "")).strip()
            lines.extend(render.prose_block(combined))

        rows = render._params_from_signature(func, doc, drop_self=is_method)
        lines.extend(render.params_table(rows))
        if doc.other_params:
            other_rows = [
                {
                    "name": getattr(i, "name", ""),
                    "type": render._annotation_str(getattr(i, "annotation", None)),
                    "default": None,
                    "description": (getattr(i, "description", "") or "").strip(),
                }
                for i in doc.other_params
            ]
            lines.extend(render.params_table(other_rows, title="Other Parameters"))

        lines.extend(render.returns_block(doc, func))
        lines.extend(render.raises_block(doc))

        for chunk in doc.extra_prose:
            lines.append(chunk)
        if doc.extra_prose:
            lines.append("")

        if source_url:
            lines.append(f'<Source href="{escape.attr(source_url)}" />')
            lines.append("")

        lines.append("</ApiFn>")
        lines.append("")
        return lines

    def emit_property(self, prop: Any, module_fq: str) -> list[str]:
        fq = prop.path
        doc = render.analyze_docstring(prop)
        source_url = self.source_url(prop)
        self._record(fq, "property", module_fq)

        lines: list[str] = ["<ApiFn"]
        lines.append(f'  name="{escape.attr(prop.name)}"')
        lines.append(f'  fqName="{escape.attr(fq)}"')
        lines.append('  kind="property"')
        if source_url:
            lines.append(f'  sourceUrl="{escape.attr(source_url)}"')
        lines.append(">")
        lines.append("")

        annotation = render._annotation_str(getattr(prop, "annotation", None))
        sig = f"{prop.name}: {annotation}" if annotation else prop.name
        lines.extend(render.signature_block(sig))

        if doc.summary or doc.body:
            combined = (doc.summary + ("\n\n" + doc.body if doc.body else "")).strip()
            lines.extend(render.prose_block(combined))

        lines.extend(render.returns_block(doc, prop))

        if source_url:
            lines.append(f'<Source href="{escape.attr(source_url)}" />')
            lines.append("")

        lines.append("</ApiFn>")
        lines.append("")
        return lines


def detect_repo_root(src: str) -> str:
    src_path = Path(src).resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(src_path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        top = out.stdout.strip()
        if top:
            return top
    except Exception:
        pass
    # Fallback: assume a src-layout package at <root>/<something>/<package>.
    return str(src_path.parent.parent)
