from typing import Annotated

from kirin import ir, rewrite
from kirin.passes import Default
from bloqade.squin import gate, qubit
from kirin.prelude import structural_no_opt
from bloqade.analysis import address
from kirin.validation import ValidationSuite
from typing_extensions import Doc
from kirin.passes.inline import InlinePass
from bloqade.squin.rewrite import WrapAddressAnalysis
from bloqade.lanes.dialects import arch, steane713 as operations
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.decoders.dialects import annotate
from bloqade.lanes.validation.steane import SteaneLogicalValidation
from bloqade.lanes.dialects.steane713 import extensions as extensions
from bloqade.lanes.rewrite.qubit_count import InsertQubitCount
from bloqade.lanes.validation.call_site import InlineOrigins
from bloqade.lanes.validation.recursion import check_call_graph
from bloqade.lanes.validation.const_address import ConstAddressValidation
from bloqade.lanes.validation.duplicate_address import DuplicateAddressValidation
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.lanes.validation.terminal_measurement import (
    SteaneTerminalMeasurementValidation,
)

# Load method tables before constructing the dialect group and its analyses.
from bloqade.gemini.common import impl as _common_impl  # noqa: F401
from bloqade.gemini.logical import impl as _logical_impl  # noqa: F401
from bloqade.gemini.steane_defaults import STEANE7_PHYSICAL_QUBITS


@ir.dialect_group(
    structural_no_opt.union([gate, qubit, operations, extensions, annotate, arch])
)
def kernel(self):
    """Compile a function to a Gemini logical kernel.

    Aggressive inlining and unrolling are enabled by default. Set
    ``aggressive_unroll=False`` to use the standard pass pipeline instead.

    The number of physical qubits per logical qubit is fixed at Steane
    [[7,1,3]]'s seven. It is not configurable: this is the first version of
    gemini-qec, where the logical-to-physical lowering is hand-written against
    that code, so any other width produces a program the rest of the pipeline
    cannot lower.
    """
    address_analysis = address.AddressAnalysis(dialects=self)

    def run_pass(
        mt,
        *,
        verify: Annotated[
            bool, Doc("run `verify` before running passes, default is `True`")
        ] = True,
        typeinfer: Annotated[
            bool,
            Doc("run type inference and apply the inferred type to IR, default `True`"),
        ] = True,
        fold: Annotated[bool, Doc("run folding passes")] = True,
        aggressive: Annotated[
            bool, Doc("run aggressive folding passes if `fold=True`")
        ] = False,
        inline: Annotated[
            bool,
            Doc("inline function calls when `aggressive_unroll=False`, default `True`"),
        ] = True,
        aggressive_unroll: Annotated[
            bool,
            Doc("Run aggressive inlining and unrolling pass on the IR, default `True`"),
        ] = True,
        no_raise: Annotated[bool, Doc("do not raise exception during analysis")] = True,
    ) -> None:

        # NOTE: ahead of every other pass, and deliberately not gated on
        # `verify` -- see `check_call_graph` for why skipping it reintroduces a
        # hang rather than merely losing a diagnostic.
        check_call_graph(mt)

        # NOTE: has to happen before the inliner splices the callees in and the
        # invokes are lost; only the `verify` path consumes the result.
        origins = InlineOrigins.collect(mt) if verify else InlineOrigins()

        if inline and not aggressive_unroll:
            InlinePass(mt.dialects, no_raise=no_raise).fixpoint(mt)

        if aggressive_unroll:
            AggressiveUnroll(mt.dialects, no_raise=no_raise).fixpoint(mt)
        else:
            default_pass = Default(
                self,
                verify=verify,
                fold=fold,
                aggressive=aggressive,
                typeinfer=typeinfer,
                no_raise=no_raise,
            )

            default_pass.fixpoint(mt)

        # NOTE: the fold passes run `Call2Invoke`, which turns a dynamic call
        # into a static `func.Invoke` once const propagation can prove the
        # callee. That can expose a cycle the first check could not see, so
        # re-check before the address analysis, which is what actually diverges.
        check_call_graph(mt)

        if no_raise:
            runner = address_analysis.run_no_raise
        else:
            runner = address_analysis.run

        address_frame, _ = runner(mt)

        # NOTE: the code distance is hard-coded to Steane [[7,1,3]] because this
        # is the first version of gemini-qec, where the logical-to-physical
        # lowering is written out by hand against that code -- the stdlib
        # post-processing kernels index measurements 0..6 directly, and
        # `append_measurements_and_annotations` strides detectors by seven.
        # Nothing downstream honours another width, so it is not a knob the
        # user gets to turn. When the lowering becomes code-generic, this
        # should come from the code/architecture, not from the decorator.
        rewrite.Walk(
            rewrite.Chain(
                WrapAddressAnalysis(address_frame.entries),
                InsertQubitCount(STEANE7_PHYSICAL_QUBITS),
            )
        ).rewrite(mt.code)

        if verify:
            from bloqade.gemini.logical.validation.arguments import (
                GeminiLogicalArgumentValidation,
            )

            validator = ValidationSuite(
                [
                    SteaneLogicalValidation,
                    SteaneTerminalMeasurementValidation,
                    FlatKernelNoCloningValidation,
                    # Before DuplicateAddressValidation: it is that pass's
                    # precondition, and reporting the precondition first reads
                    # better when both fire.
                    ConstAddressValidation,
                    DuplicateAddressValidation,
                    GeminiLogicalArgumentValidation,
                ]
            )
            origins.snapshot(mt)
            validation_result = origins.annotate(
                mt, validator.validate(mt), aggressive_unroll=aggressive_unroll
            )
            validation_result.raise_if_invalid()
            mt.verify()

    return run_pass
