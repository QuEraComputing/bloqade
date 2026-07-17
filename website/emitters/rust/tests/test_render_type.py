"""Tests for the rustdoc-JSON ``Type`` renderer in ``emit_rust.py``.

Feeds representative rustdoc type nodes to ``render_type`` (and its helpers)
and asserts the produced Rust type string. Every branch of the renderer is
supposed to degrade gracefully to ``"_"`` so a novel node can never crash the
build; that fallback is pinned too.
"""

from __future__ import annotations


from emit_rust import (
    render_args,
    render_bounds,
    render_const,
    render_generics,
    render_type,
)


def test_none_is_unit():
    assert render_type(None) == "()"


def test_bare_string_passthrough():
    assert render_type("'static") == "'static"


def test_primitive():
    assert render_type({"primitive": "u32"}) == "u32"


def test_generic():
    assert render_type({"generic": "T"}) == "T"


def test_resolved_path_bare():
    # Path is reduced to its last segment.
    assert (
        render_type({"resolved_path": {"path": "std::vec::Vec", "args": None}}) == "Vec"
    )


def test_resolved_path_with_generic_args():
    node = {
        "resolved_path": {
            "path": "std::vec::Vec",
            "args": {
                "angle_bracketed": {
                    "args": [{"type": {"primitive": "u8"}}],
                    "constraints": [],
                }
            },
        }
    }
    assert render_type(node) == "Vec<u8>"


def test_resolved_path_with_lifetime_and_const_args():
    node = {
        "resolved_path": {
            "path": "Array",
            "args": {
                "angle_bracketed": {
                    "args": [
                        {"lifetime": "'a"},
                        {"type": {"generic": "T"}},
                        {"const": {"expr": "N"}},
                    ],
                    "constraints": [],
                }
            },
        }
    }
    assert render_type(node) == "Array<'a, T, N>"


def test_borrowed_ref_plain():
    node = {
        "borrowed_ref": {
            "lifetime": None,
            "is_mutable": False,
            "type": {"primitive": "str"},
        }
    }
    assert render_type(node) == "&str"


def test_borrowed_ref_mut_with_lifetime():
    node = {
        "borrowed_ref": {
            "lifetime": "'a",
            "is_mutable": True,
            "type": {"primitive": "str"},
        }
    }
    assert render_type(node) == "&'a mut str"


def test_slice():
    assert render_type({"slice": {"primitive": "u8"}}) == "[u8]"


def test_array():
    assert (
        render_type({"array": {"type": {"primitive": "u8"}, "len": "4"}}) == "[u8; 4]"
    )


def test_tuple():
    node = {"tuple": [{"primitive": "i32"}, {"primitive": "bool"}]}
    assert render_type(node) == "(i32, bool)"


def test_empty_tuple():
    assert render_type({"tuple": []}) == "()"


def test_impl_trait():
    node = {"impl_trait": [{"trait_bound": {"trait": {"path": "Iterator"}}}]}
    assert render_type(node) == "impl Iterator"


def test_impl_trait_multiple_bounds():
    node = {
        "impl_trait": [
            {"trait_bound": {"trait": {"path": "core::fmt::Debug"}}},
            {"trait_bound": {"trait": {"path": "Clone"}}},
        ]
    }
    assert render_type(node) == "impl Debug + Clone"


def test_dyn_trait_with_lifetime():
    node = {
        "dyn_trait": {
            "traits": [{"trait": {"path": "std::fmt::Debug"}}],
            "lifetime": "'a",
        }
    }
    assert render_type(node) == "dyn Debug + 'a"


def test_raw_pointer_const_and_mut():
    assert (
        render_type({"raw_pointer": {"is_mutable": False, "type": {"primitive": "u8"}}})
        == "*const u8"
    )
    assert (
        render_type({"raw_pointer": {"is_mutable": True, "type": {"primitive": "u8"}}})
        == "*mut u8"
    )


def test_qualified_path_with_trait():
    node = {
        "qualified_path": {
            "self_type": {"generic": "T"},
            "name": "Item",
            "trait": {"path": "Iterator"},
        }
    }
    assert render_type(node) == "<T as Iterator>::Item"


def test_qualified_path_without_trait():
    node = {
        "qualified_path": {"self_type": {"generic": "T"}, "name": "Item", "trait": None}
    }
    assert render_type(node) == "T::Item"


def test_function_pointer():
    node = {
        "function_pointer": {
            "sig": {
                "inputs": [["x", {"primitive": "i32"}]],
                "output": {"primitive": "bool"},
            }
        }
    }
    assert render_type(node) == "fn(i32) -> bool"


def test_unknown_node_falls_back_to_underscore():
    assert render_type({"some_future_variant": {"nope": 1}}) == "_"


def test_empty_dict_falls_back():
    assert render_type({}) == "_"


# -- helpers ---------------------------------------------------------------- #
def test_render_const_variants():
    assert render_const({"expr": "N"}) == "N"
    assert render_const({"value": "3"}) == "3"
    assert render_const("not-a-dict") == "_"


def test_render_bounds_trait_and_outlives():
    bounds = [
        {"trait_bound": {"trait": {"path": "core::clone::Clone"}}},
        {"outlives": "'a"},
    ]
    assert render_bounds(bounds) == "Clone + 'a"


def test_render_args_parenthesized_fn_trait():
    # Fn(A) -> B style parenthesized args.
    args = {
        "parenthesized": {
            "inputs": [{"primitive": "i32"}],
            "output": {"primitive": "bool"},
        }
    }
    assert render_args(args) == "(i32) -> bool"


def test_render_generics_type_and_const_params():
    generics = {
        "params": [
            {"name": "'a", "kind": {"lifetime": {"outlives": []}}},
            {
                "name": "T",
                "kind": {
                    "type": {
                        "bounds": [{"trait_bound": {"trait": {"path": "Clone"}}}],
                        "is_synthetic": False,
                    }
                },
            },
            {"name": "N", "kind": {"const": {"type": {"primitive": "usize"}}}},
        ]
    }
    assert render_generics(generics) == "<'a, T: Clone, const N: usize>"


def test_render_generics_skips_synthetic_impl_trait():
    generics = {
        "params": [
            {
                "name": "impl_arg",
                "kind": {"type": {"bounds": [], "is_synthetic": True}},
            },
        ]
    }
    assert render_generics(generics) == ""
