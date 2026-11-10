from typing import Annotated

from kirin import ir
from kirin.passes import Default
from bloqade.squin import gate, qubit
from kirin.prelude import structural_no_opt
from kirin.dialects import debug
from kirin.validation import ValidationSuite
from typing_extensions import Doc
from kirin.passes.inline import InlinePass
from bloqade.lanes.dialects import arch, arrange
from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.decoders.dialects import annotate
from bloqade.lanes.validation.call_site import InlineOrigins
from bloqade.lanes.validation.recursion import check_call_graph
from bloqade.lanes.validation.const_address import ConstAddressValidation
from bloqade.lanes.validation.terminal_measure import (
    PhysicalTerminalMeasurementValidation,
)
from bloqade.lanes.validation.duplicate_address import DuplicateAddressValidation
from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation

# Load method tables before constructing the dialect group and its analyses.
from bloqade.gemini.common import impl as _common_impl  # noqa: F401


@ir.dialect_group(
    structural_no_opt.union(
        [
            qubit,
            gate,
            annotate,
            arch,
            arrange,
            debug,
        ]
    )
)
def kernel(self):
    """Physical SQuIN kernel with explicit allocation and arrangement.

    Physical kernels use ordinary SQuIN gates and end with one
    ``squin.broadcast.measure`` that consumes every allocated qubit. They may
    also use ``new_at``, ``arrange.move_to`` / ``arrange.permute``, and
    architecture-location primitives such as ``loc`` and ``cz_partner``.
    """

    def run_pass(
        mt: ir.Method,
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
        inline: Annotated[bool, Doc("inline function calls, default `True`")] = True,
        aggressive_unroll: Annotated[
            bool,
            Doc(
                "Run aggressive inlining and unrolling pass on the IR, default `False`"
            ),
        ] = False,
        no_raise: Annotated[bool, Doc("do not raise exception during analysis")] = True,
    ) -> None:
        # Reject recursive call graphs up front so a physical kernel cannot
        # defer an unbounded unroll to the compiler.

        check_call_graph(mt)

        # NOTE: has to happen before the inliner splices the callees in and the
        # invokes are lost; only the `verify` path consumes the result.
        origins = InlineOrigins.collect(mt) if verify else InlineOrigins()

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

            # Unlike logical terminal measurement, SQuIN broadcast measurement
            # inlines to qubit.Measure. Type-check it while its concrete list
            # length is still represented by the broadcast helper, then inline
            # so the terminal-measurement validator can inspect it.
            if inline:
                InlinePass(mt.dialects, no_raise=no_raise).fixpoint(mt)

        # The fold passes can expose a cycle by turning a dynamic call into a
        # static invoke.
        check_call_graph(mt)

        if verify:
            validator = ValidationSuite(
                [
                    FlatKernelNoCloningValidation,
                    ConstAddressValidation,
                    DuplicateAddressValidation,
                    PhysicalTerminalMeasurementValidation,
                ]
            )
            origins.snapshot(mt)
            validation_result = origins.annotate(
                mt, validator.validate(mt), aggressive_unroll=aggressive_unroll
            )
            validation_result.raise_if_invalid()
            mt.verify()

    return run_pass
