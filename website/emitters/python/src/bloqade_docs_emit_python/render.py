"""Render griffe objects into MDX using the frozen component contract.

Every function here returns a list of MDX line-strings. The functions never
raise on bad docstrings: docstring analysis is wrapped in try/except and falls
back to the raw docstring text so a malformed docstring can never break the
Astro build.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from griffe import (
    DocstringSectionKind,
    Kind,
    ParameterKind,
)

from . import escape

logger = logging.getLogger("bloqade_docs_emit_python")


# --------------------------------------------------------------------------- #
# Signature rendering
# --------------------------------------------------------------------------- #

_VAR_POSITIONAL = ParameterKind.var_positional
_VAR_KEYWORD = ParameterKind.var_keyword
_KEYWORD_ONLY = ParameterKind.keyword_only
_POSITIONAL_ONLY = ParameterKind.positional_only


def _annotation_str(annotation: Any) -> str | None:
    if annotation is None:
        return None
    try:
        text = str(annotation)
    except Exception:  # pragma: no cover - defensive
        return None
    text = text.strip()
    return text or None


def _default_str(default: Any) -> str | None:
    if default is None:
        return None
    try:
        text = str(default)
    except Exception:  # pragma: no cover - defensive
        return None
    text = text.strip()
    return text or None


def _format_one(param: Any, prefix: str = "") -> str:
    piece = prefix + param.name
    annotation = _annotation_str(param.annotation)
    default = _default_str(param.default)
    if annotation:
        piece += f": {annotation}"
    if default is not None:
        piece += f" = {default}" if annotation else f"={default}"
    return piece


def format_parameters(parameters: Any, drop_first_self: bool = False) -> str:
    """Render a griffe ``Parameters`` collection to a signature string."""
    params = list(parameters)
    if (
        drop_first_self
        and params
        and params[0].name in ("self", "cls")
        and params[0].annotation is None
    ):
        params = params[1:]

    last_pos_only = -1
    for idx, param in enumerate(params):
        if param.kind == _POSITIONAL_ONLY:
            last_pos_only = idx

    rendered: list[str] = []
    kw_only_sep_pending = True
    for idx, param in enumerate(params):
        kind = param.kind
        if kind == _VAR_POSITIONAL:
            rendered.append(_format_one(param, prefix="*"))
            kw_only_sep_pending = False
        elif kind == _VAR_KEYWORD:
            rendered.append(_format_one(param, prefix="**"))
        elif kind == _KEYWORD_ONLY:
            if kw_only_sep_pending:
                rendered.append("*")
                kw_only_sep_pending = False
            rendered.append(_format_one(param))
        else:
            rendered.append(_format_one(param))
        if idx == last_pos_only and last_pos_only != -1:
            rendered.append("/")
    return ", ".join(rendered)


def function_signature(func: Any, is_method: bool) -> str:
    prefix = "async def " if "async" in getattr(func, "labels", set()) else "def "
    params = format_parameters(func.parameters, drop_first_self=is_method)
    sig = f"{prefix}{func.name}({params})"
    returns = _annotation_str(getattr(func, "returns", None))
    if returns:
        sig += f" -> {returns}"
    return sig


def class_signature(cls: Any) -> str:
    init = cls.members.get("__init__")
    params_str = ""
    if init is not None and not getattr(init, "is_alias", False):
        try:
            params_str = format_parameters(init.parameters, drop_first_self=True)
        except Exception:  # pragma: no cover - defensive
            params_str = ""
    if params_str:
        return f"class {cls.name}({params_str})"
    return f"class {cls.name}"


def signature_block(code: str) -> list[str]:
    return [f"<Signature lang=\"python\" code={{`{escape.template(code)}`}} />", ""]


# --------------------------------------------------------------------------- #
# Docstring analysis
# --------------------------------------------------------------------------- #


@dataclass
class ParsedDoc:
    summary: str = ""
    body: str = ""
    params: list[Any] = field(default_factory=list)
    other_params: list[Any] = field(default_factory=list)
    returns: list[Any] = field(default_factory=list)
    raises: list[Any] = field(default_factory=list)
    attributes: list[Any] = field(default_factory=list)
    extra_prose: list[str] = field(default_factory=list)


def _split_summary(text: str) -> tuple[str, str]:
    """Split a text block into (first-paragraph summary, remaining body)."""
    text = text.strip("\n")
    if not text:
        return "", ""
    parts = text.split("\n\n", 1)
    summary = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""
    return summary, body


def analyze_docstring(obj: Any) -> ParsedDoc:
    """Parse a griffe object's docstring into a structured ``ParsedDoc``.

    Robust by construction: any failure falls back to the raw docstring value
    rendered as prose.
    """
    doc = ParsedDoc()
    docstring = getattr(obj, "docstring", None)
    if docstring is None:
        return doc

    try:
        sections = docstring.parsed
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("docstring parse failed for %s: %s", obj.path, exc)
        summary, body = _split_summary(docstring.value or "")
        doc.summary, doc.body = summary, body
        return doc

    text_chunks: list[str] = []
    for section in sections:
        try:
            kind = section.kind
            if kind == DocstringSectionKind.text:
                text_chunks.append(section.value or "")
            elif kind == DocstringSectionKind.parameters:
                doc.params = list(section.value)
            elif kind == DocstringSectionKind.other_parameters:
                doc.other_params = list(section.value)
            elif kind == DocstringSectionKind.returns:
                doc.returns = list(section.value)
            elif kind == DocstringSectionKind.raises:
                doc.raises = list(section.value)
            elif kind == DocstringSectionKind.attributes:
                doc.attributes = list(section.value)
            elif kind == DocstringSectionKind.yields:
                doc.extra_prose.extend(_render_yields(section))
            elif kind == DocstringSectionKind.receives:
                doc.extra_prose.extend(_render_named("Receives", section))
            elif kind == DocstringSectionKind.admonition:
                doc.extra_prose.extend(_render_admonition(section))
            elif kind == DocstringSectionKind.examples:
                doc.extra_prose.extend(_render_examples(section))
            elif kind == DocstringSectionKind.deprecated:
                doc.extra_prose.extend(_render_admonition(section, "Deprecated"))
            else:
                # Unknown/less-common section: best-effort prose fallback.
                doc.extra_prose.extend(_render_generic(section))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("section render failed for %s: %s", getattr(obj, "path", "?"), exc)
            continue

    combined = "\n\n".join(chunk for chunk in text_chunks if chunk.strip())
    doc.summary, doc.body = _split_summary(combined)
    return doc


def _section_title(section: Any, default: str) -> str:
    title = getattr(section, "title", None)
    return title if isinstance(title, str) and title else default


def _render_admonition(section: Any, default_title: str = "Note") -> list[str]:
    value = section.value
    title = _section_title(section, default_title)
    contents = getattr(value, "contents", None)
    if contents is None:
        contents = str(value)
    lines = [f"**{escape.prose(title)}**", ""]
    for para in str(contents).split("\n\n"):
        para = para.strip()
        if para:
            lines.append(escape.prose(para))
            lines.append("")
    return lines


def _render_examples(section: Any) -> list[str]:
    lines = ["**Examples**", ""]
    value = section.value
    # Google examples: list of (kind, text) tuples where kind == "examples" is code.
    try:
        for item in value:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                kind, text = item
                kind_str = str(getattr(kind, "value", kind))
                if kind_str == "examples":
                    lines.append("```python")
                    lines.extend(str(text).rstrip("\n").split("\n"))
                    lines.append("```")
                    lines.append("")
                else:
                    lines.append(escape.prose(str(text)))
                    lines.append("")
            else:
                lines.append(escape.prose(str(item)))
                lines.append("")
    except TypeError:
        lines.append(escape.prose(str(value)))
        lines.append("")
    return lines


def _render_yields(section: Any) -> list[str]:
    lines = ["**Yields**", ""]
    for item in section.value:
        annotation = _annotation_str(getattr(item, "annotation", None))
        desc = (getattr(item, "description", "") or "").strip()
        bits = []
        if annotation:
            bits.append(f"`{annotation}`")
        if desc:
            bits.append(escape.prose(desc))
        if bits:
            lines.append(" — ".join(bits) if annotation else bits[-1])
            lines.append("")
    return lines


def _render_named(title: str, section: Any) -> list[str]:
    lines = [f"**{title}**", ""]
    for item in section.value:
        annotation = _annotation_str(getattr(item, "annotation", None))
        desc = (getattr(item, "description", "") or "").strip()
        bits = []
        if annotation:
            bits.append(f"`{annotation}`")
        if desc:
            bits.append(escape.prose(desc))
        if bits:
            lines.append(" — ".join(bits))
            lines.append("")
    return lines


def _render_generic(section: Any) -> list[str]:
    value = getattr(section, "value", None)
    title = _section_title(section, "")
    lines: list[str] = []
    if title:
        lines.extend([f"**{escape.prose(title)}**", ""])
    if isinstance(value, str):
        lines.append(escape.prose(value))
        lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Prose + table blocks
# --------------------------------------------------------------------------- #


def prose_block(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [escape.prose(text), ""]


def _clean_return_desc(desc: str, annotation: str | None) -> str:
    desc = (desc or "").strip()
    if annotation and desc.startswith(annotation):
        rest = desc[len(annotation):].lstrip()
        if rest.startswith(":"):
            rest = rest[1:].lstrip()
        desc = rest
    return desc


def _params_from_signature(callable_obj: Any, doc: ParsedDoc, drop_self: bool) -> list[dict]:
    """Merge signature parameters with docstring parameter descriptions."""
    doc_by_name = {}
    for item in doc.params:
        name = getattr(item, "name", None)
        if name:
            doc_by_name[name] = item

    params = list(callable_obj.parameters)
    if drop_self and params and params[0].name in ("self", "cls") and params[0].annotation is None:
        params = params[1:]

    rows: list[dict] = []
    for param in params:
        name = param.name
        if param.kind == _VAR_POSITIONAL:
            display = f"*{name}"
        elif param.kind == _VAR_KEYWORD:
            display = f"**{name}"
        else:
            display = name
        annotation = _annotation_str(param.annotation)
        default = _default_str(param.default)
        desc = ""
        doc_item = doc_by_name.get(name)
        if doc_item is not None:
            if annotation is None:
                annotation = _annotation_str(getattr(doc_item, "annotation", None))
            desc = (getattr(doc_item, "description", "") or "").strip()
        rows.append(
            {"name": display, "type": annotation, "default": default, "description": desc}
        )
    return rows


def params_table(rows: list[dict], title: str | None = None) -> list[str]:
    if not rows:
        return []
    lines = ["<Params"]
    if title:
        lines[0] = f'<Params title="{escape.attr(title)}"'
    lines.append("  items={[")
    for row in rows:
        lines.append("    {")
        lines.append(f"      name: '{escape.js_str(row['name'])}',")
        if row.get("type"):
            lines.append(f"      type: '{escape.js_str(escape.truncate(row['type'], 200))}',")
        if row.get("default") is not None:
            lines.append(
                f"      default: '{escape.js_str(escape.truncate(str(row['default']), 120))}',"
            )
        if row.get("description"):
            lines.append(f"      description: '{escape.desc_literal(row['description'])}',")
        lines.append("    },")
    lines.append("  ]}")
    lines.append("/>")
    lines.append("")
    return lines


def returns_block(doc: ParsedDoc, callable_obj: Any) -> list[str]:
    type_str: str | None = None
    desc_str = ""
    if doc.returns:
        anns = [_annotation_str(getattr(i, "annotation", None)) for i in doc.returns]
        anns = [a for a in anns if a]
        descs = []
        for item in doc.returns:
            ann = _annotation_str(getattr(item, "annotation", None))
            d = _clean_return_desc(getattr(item, "description", "") or "", ann)
            name = getattr(item, "name", "") or ""
            if d:
                descs.append(f"{name}: {d}" if name else d)
        if len(anns) == 1:
            type_str = anns[0]
        elif len(anns) > 1:
            type_str = "(" + ", ".join(anns) + ")"
        desc_str = " ".join(descs).strip()
    if type_str is None:
        ann = _annotation_str(getattr(callable_obj, "returns", None))
        if ann and ann not in ("None", "NoneType"):
            type_str = ann
    if not type_str and not desc_str:
        return []
    attrs = []
    if type_str:
        attrs.append(f'type="{escape.attr(type_str)}"')
    if desc_str:
        attrs.append(f"description={{'{escape.desc_literal(desc_str)}'}}")
    return [f"<Returns {' '.join(attrs)} />", ""]


def raises_block(doc: ParsedDoc) -> list[str]:
    if not doc.raises:
        return []
    lines = ["<Raises", "  items={["]
    for item in doc.raises:
        ann = _annotation_str(getattr(item, "annotation", None)) or "Exception"
        desc = (getattr(item, "description", "") or "").strip()
        lines.append("    {")
        lines.append(f"      type: '{escape.js_str(ann)}',")
        if desc:
            lines.append(f"      description: '{escape.desc_literal(desc)}',")
        lines.append("    },")
    lines.append("  ]}")
    lines.append("/>")
    lines.append("")
    return lines


def attributes_table(doc: ParsedDoc, attr_members: list[Any]) -> list[str]:
    """Build an "Attributes" table, preferring the docstring section."""
    rows: list[dict] = []
    if doc.attributes:
        for item in doc.attributes:
            name = getattr(item, "name", None)
            if not name:
                continue
            rows.append(
                {
                    "name": name,
                    "type": _annotation_str(getattr(item, "annotation", None)),
                    "default": None,
                    "description": (getattr(item, "description", "") or "").strip(),
                }
            )
    else:
        for member in attr_members:
            name = member.name
            annotation = _annotation_str(getattr(member, "annotation", None))
            value = _default_str(getattr(member, "value", None))
            desc = ""
            if getattr(member, "docstring", None):
                d = analyze_docstring(member)
                desc = (d.summary + (" " + d.body if d.body else "")).strip()
            rows.append(
                {"name": name, "type": annotation, "default": value, "description": desc}
            )
    return params_table(rows, title="Attributes")
