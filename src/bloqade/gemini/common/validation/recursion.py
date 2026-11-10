"""Deprecated compatibility imports; use ``bloqade.lanes.validation.recursion``."""

from bloqade.lanes.validation.recursion import (
    Cycle as Cycle,
    CallGraph as CallGraph,
    NoRecursionValidation as NoRecursionValidation,
    NoOpaqueCallValidation as NoOpaqueCallValidation,
    _name as _name,
    _sort_key as _sort_key,
    _self_value as _self_value,
    format_cycle as format_cycle,
    _constant_method as _constant_method,
    check_call_graph as check_call_graph,
    _parameter_values as _parameter_values,
    _callable_operands as _callable_operands,
)
