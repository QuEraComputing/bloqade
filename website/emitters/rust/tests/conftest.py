"""Make the script-style ``emit_rust.py`` importable as ``emit_rust``.

``emit_rust.py`` lives one directory up and is normally executed as a script
(``uv run emit_rust.py``). Prepending its directory to ``sys.path`` lets the
tests ``import emit_rust`` and exercise its functions directly. The module has
no import-time side effects (``main()`` only runs under ``__main__``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import emit_rust  # noqa: E402  (path is set up above)


# A minimal, HAND-CRAFTED rustdoc-JSON document: crate root module with one
# struct (`Widget`, one field `size`, one inherent method `area`) and one free
# function (`make_widget`). Pinned to the emitter's EXPECTED_FORMAT_VERSION so
# the tests never need cargo/nightly. Kept deterministic + fast.
def _mini_rustdoc() -> dict:
    return {
        "root": "0",
        "crate_version": "0.1.0",
        "format_version": emit_rust.EXPECTED_FORMAT_VERSION,
        "includes_private": False,
        "index": {
            "0": {
                "id": 0,
                "name": "mini_crate",
                "visibility": "public",
                "docs": "Mini crate root.\n\nSecond paragraph of crate docs.",
                "inner": {
                    "module": {"is_crate": True, "items": ["1", "2"], "is_stripped": False}
                },
            },
            "1": {
                "id": 1,
                "name": "Widget",
                "visibility": "public",
                "docs": "A widget.\n\nMore about the widget.",
                "span": {"filename": "src/lib.rs", "begin": [10, 0], "end": [14, 1]},
                "inner": {
                    "struct": {
                        "kind": {"plain": {"fields": ["3"], "has_stripped_fields": False}},
                        "generics": {"params": [], "where_predicates": []},
                        "impls": ["4"],
                    }
                },
            },
            "2": {
                "id": 2,
                "name": "make_widget",
                "visibility": "public",
                "docs": "Make a widget from a size.",
                "span": {"filename": "src/lib.rs", "begin": [20, 0], "end": [22, 1]},
                "inner": {
                    "function": {
                        "sig": {
                            "inputs": [["size", {"primitive": "usize"}]],
                            "output": {
                                "resolved_path": {"path": "Widget", "id": 1, "args": None}
                            },
                            "is_c_variadic": False,
                        },
                        "generics": {"params": [], "where_predicates": []},
                        "header": {
                            "is_const": False,
                            "is_unsafe": False,
                            "is_async": False,
                            "abi": "Rust",
                        },
                        "has_body": True,
                    }
                },
            },
            "3": {
                "id": 3,
                "name": "size",
                "visibility": "public",
                "docs": "The widget size.",
                "inner": {"struct_field": {"primitive": "usize"}},
            },
            "4": {
                "id": 4,
                "name": None,
                "visibility": "public",
                "docs": None,
                "inner": {
                    "impl": {
                        "is_unsafe": False,
                        "is_negative": False,
                        "is_synthetic": False,
                        "blanket_impl": None,
                        "trait": None,
                        "for": {
                            "resolved_path": {"path": "Widget", "id": 1, "args": None}
                        },
                        "items": ["5"],
                        "generics": {"params": [], "where_predicates": []},
                    }
                },
            },
            "5": {
                "id": 5,
                "name": "area",
                "visibility": "public",
                "docs": "Compute the area.",
                "span": {"filename": "src/lib.rs", "begin": [15, 4], "end": [17, 5]},
                "inner": {
                    "function": {
                        "sig": {
                            "inputs": [
                                [
                                    "self",
                                    {
                                        "borrowed_ref": {
                                            "lifetime": None,
                                            "is_mutable": False,
                                            "type": {"generic": "Self"},
                                        }
                                    },
                                ]
                            ],
                            "output": {"primitive": "usize"},
                            "is_c_variadic": False,
                        },
                        "generics": {"params": [], "where_predicates": []},
                        "header": {
                            "is_const": False,
                            "is_unsafe": False,
                            "is_async": False,
                            "abi": "Rust",
                        },
                        "has_body": True,
                    }
                },
            },
        },
        "paths": {
            "0": {"crate_id": 0, "path": ["mini_crate"], "kind": "module"},
            "1": {"crate_id": 0, "path": ["mini_crate", "Widget"], "kind": "struct"},
            "2": {"crate_id": 0, "path": ["mini_crate", "make_widget"], "kind": "function"},
        },
    }


@pytest.fixture()
def mini_doc() -> dict:
    return _mini_rustdoc()
