"""Bind classical task arguments before Gemini's physical compilation."""

from __future__ import annotations

import inspect
from typing import Any, TypeVar, ParamSpec, cast

from kirin import ir, types
from kirin.dialects import py, func, ilist
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.lanes.validation.recursion import check_call_graph

Params = ParamSpec("Params")
RetType = TypeVar("RetType")


def _is_classical_constant(value: Any) -> bool:
    # Exact builtin types exclude qubit handles, mutable containers, and objects
    # whose methods could execute during constant folding.
    if type(value) is ilist.IList:
        return type(value.data) in (list, tuple, range) and all(
            _is_classical_constant(item) for item in value.data
        )
    return type(value) in (bool, int, float, str, type(None)) or (
        type(value) is tuple and all(_is_classical_constant(item) for item in value)
    )


def _constant_type(value: Any) -> types.TypeAttribute:
    if type(value) is ilist.IList:
        # Infer from contents rather than the optional (often Any) elem hint.
        # Bottom makes an empty IList compatible with any element type.
        elem: types.TypeAttribute = types.Bottom
        for item in value.data:
            elem = elem.join(_constant_type(item))
        return ilist.IListType[elem, types.Literal(len(value))]
    if type(value) is tuple:
        return types.Generic(tuple, *(_constant_type(item) for item in value))
    return ir.PyAttr(value).type


def _snapshot_constant(value: Any) -> Any:
    """Detach IList backing storage, including lists nested inside tuples."""
    if type(value) is ilist.IList:
        data = [_snapshot_constant(item) for item in value.data]
        if type(value.data) is tuple:
            data = tuple(data)
        elif type(value.data) is range:
            data = value.data
        value_type = _constant_type(value)
        assert isinstance(value_type, types.Generic)
        return ilist.IList(data, elem=value_type.vars[0])
    if type(value) is tuple:
        return tuple(_snapshot_constant(item) for item in value)
    return value


def bind_task_arguments(
    kernel: ir.Method[Params, RetType],
    *args: Params.args,
    **kernel_args: Params.kwargs,
) -> ir.Method[[], RetType]:
    """Return a zero-argument specialization without changing the caller's IR.

    Python-backed methods use their Python signature (including defaults).
    Deserialized methods use the argument names and types retained in the IR.
    Classical scalars and recursively nested tuples/ILists are accepted;
    no live qubit handles. IList backing storage is copied before binding.
    Pass kernel arguments directly, e.g. ``bind_task_arguments(kernel, 2, flip=True)``.
    """
    if kernel.py_func is not None:
        signature = inspect.signature(kernel.py_func)
    else:
        signature = inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in (kernel.arg_names or [])[1:]
            ]
        )
    bound = signature.bind(*args, **kernel_args)
    bound.apply_defaults()
    names = (kernel.arg_names or [])[1:]
    if len(names) != len(kernel.args) or set(bound.arguments) != set(names):
        raise TypeError("Kernel signature does not match its IR arguments")
    values = [bound.arguments[name] for name in names]
    for name, value, expected in zip(names, values, kernel.arg_types):
        if not _is_classical_constant(value):
            raise TypeError(
                f"Kernel argument {name!r} must be an immutable classical value "
                "(bool, int, float, str, None, or nested tuples/ILists of these)"
            )
        actual = _constant_type(value)
        if not actual.is_subseteq(expected):
            raise TypeError(
                f"Kernel argument {name!r} has type {actual}, expected {expected}"
            )
    if not values:
        # compile_task already copies zero-argument kernels before rewriting.
        # Binding above proved that the method has no remaining inputs.
        return cast(ir.Method[[], RetType], kernel)

    check_call_graph(kernel)
    specialized = kernel.similar()
    if not isinstance(specialized.code, func.Function):
        raise TypeError("Task argument binding requires a func.Function kernel")
    block = specialized.callable_region.blocks[0]
    first = block.first_stmt
    if first is None:
        raise ValueError("Cannot bind arguments to an empty kernel")
    for argument, value in zip(tuple(specialized.args), values):
        value = _snapshot_constant(value)
        constant = py.Constant(ir.PyAttr(value, pytype=_constant_type(value)))
        constant.insert_before(first)
        argument.replace_by(constant.result)
        block.args.delete(argument)
    specialized.code.signature = func.Signature((), specialized.return_type)
    specialized.code.slots = ()
    specialized.nargs = 1  # Kirin counts the implicit method/self argument.
    specialized.arg_names = (kernel.arg_names or ["self"])[:1]
    specialized.py_func = None
    specialized.inferred = False
    block.args[0].type = specialized.self_type
    AggressiveUnroll(specialized.dialects, no_raise=False).fixpoint(specialized)
    specialized.verify()
    # IR rewrites removed the inputs; static typing cannot track that mutation.
    return cast(ir.Method[[], RetType], specialized)
