"""Require a Gemini logical *program* to take no arguments.

A kernel that allocates its own qubits is a whole program: the hardware runs it
start to finish with nothing to call it, so a parameter is a value nobody is
left to supply. That is what
:class:`bloqade.lanes.validation.kernel_args.KernelArgumentValidation` rejects.

**Why this wrapper exists rather than the bare pass.** ``gemini.logical.kernel``
decorates sub-kernels too, and a sub-kernel taking arguments is a supported
pattern, not a mistake::

    @gemini.logical.kernel(verify=False)  # allocates nothing, so opts out
    def flip(q):            # a sub-kernel -- inlined into its caller
        squin.x(q)

    @gemini.logical.kernel
    def main():             # a program -- allocates, measures, takes nothing
        q = squin.qalloc(2)
        flip(q[0])
        gemini.logical.terminal_measure(q)

``run_pass`` runs identically on both, so the bare pass applied to the whole
dialect group rejects every helper at its own definition. Two tests pin that
pattern by name (``test_passing_a_parameter_to_a_subkernel_is_allowed``,
``test_non_recursive_subkernel_calls_are_unaffected``).

**The discriminator is qubit allocation**, which is not a new idea here:
``SteaneTerminalMeasurementValidation`` next door already decides "is this a
program?" exactly this way, requiring a terminal measurement only of kernels
whose ``AddressAnalysis`` qubit count is non-zero. Reusing it keeps one meaning
of "program" across the suite -- the same kernels that must terminal-measure
must take no arguments -- rather than introducing a second, differently-drawn
line.

It costs a second ``AddressAnalysis`` run per validated kernel. The suite's
cache cannot avoid it: it stores analysis *frames*, and ``qubit_count`` lives on
the analysis object.

A sub-kernel's arguments go unchecked, by construction. They are not unchecked
in substance -- the call site is inlined before this suite runs, so a mismatched
argument surfaces there.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from kirin import ir
from bloqade.analysis import address
from kirin.validation import ValidationPass
from bloqade.lanes.validation.kernel_args import KernelArgumentValidation


@dataclass
class GeminiLogicalArgumentValidation(ValidationPass):
    """Reject arguments on a logical kernel that allocates its own qubits."""

    def name(self) -> str:
        return "Gemini Logical Argument Validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        analysis = address.AddressAnalysis(dialects=method.dialects)
        analysis.run(method)

        if analysis.qubit_count == 0:
            # A sub-kernel: it is handed its qubits, so it is not a program and
            # its parameters are the caller's business.
            return None, []

        return KernelArgumentValidation().run(method)
