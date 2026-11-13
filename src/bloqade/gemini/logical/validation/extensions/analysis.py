"""Reject the experimental extension statements on paths that cannot support them.

The statements in :mod:`bloqade.lanes.dialects.steane713.extensions` are part of
the ``@gemini.logical.kernel`` dialect group, so they lower and compile like any
other logical statement. What sets them apart is that they are *not* logical
operations: ``StarRz`` puts physical Z rotations on the three qubits of a
weight-3 logical-Z representative, which takes the state out of the code space.
The state only comes back if the program post-selects on the error-correction
checks around it -- something the compiler neither inserts nor can verify.

That makes them usable, but only deliberately: in simulation, where the user can
post-select on the detectors themselves. This pass is the guard for everywhere
else. It is deliberately **not** part of ``@gemini.logical.kernel``'s own
validation suite -- putting it there would reject the feature at definition
time, everywhere, which is the opposite of what a gated experimental feature
needs. It is added to the suites that submit a program for execution instead,
and adding it to a suite is what turns the guard on.

**The whole call graph is searched**, not just the validated kernel's own body.
A Gemini kernel is aggressively unrolled, so in practice a use inside a helper
has already been inlined into its caller by the time any suite runs -- but a
guard is the wrong place to bank on that. Nothing stops a method reaching a
device undecorated (the CUDA-Q route builds one by conversion), or reaching one
with inlining turned off, and a guard that assumes a compilation shape fails
open exactly when the assumption breaks.

The traversal is kirin's abstract interpreter, the same mechanism
``SteaneTerminalMeasurementValidation`` next door uses: ``impls.Func`` gives this
analysis a ``func.Invoke`` impl, so descending into a statically resolved callee
is the interpreter's own behaviour rather than something this pass arranges. It
inherits that machinery's one limitation -- a callee reached only through a
dynamic ``func.Call`` is not descended into, see :meth:`run_lattice` -- which is
the same line every other Forward analysis in this package draws.

**The check is on the dialect**, not on a list of statement types, so a second
extension is covered the day it is added rather than the day someone remembers
to update this file. That is why it lives in :meth:`eval_fallback` rather than
in a method table: an impl has to name a statement class, and nothing here wants
to know which statements exist -- only which dialect they came from.

One error per offending statement, so a program with three uses reports all
three -- the same report-everything contract the validations next door keep.

Each error is anchored to the statement. The user-facing wrappers are ``@kernel``
stdlib kernels, so the location generally points inside
``bloqade.gemini.logical.extensions`` rather than at the user's call site,
whether or not inlining has run. The message names the statement for that
reason, rather than relying on the caret.
"""

from typing import Any
from dataclasses import dataclass

from kirin import ir
from kirin.lattice import EmptyLattice
from kirin.analysis import Forward, ForwardFrame
from kirin.validation import ValidationPass
from bloqade.lanes.dialects.steane713.extensions import dialect as extensions_dialect

EXPERIMENTAL_HELP = (
    "this is an experimental gadget that acts on the physical qubits inside a "
    "logical qubit, leaving the logical subspace; it is supported only where "
    "the caller can post-select on the surrounding error-correction checks, "
    "e.g. `GeminiLogicalSimulator`"
)


@dataclass
class _NoLogicalExtensionsAnalysis(Forward[EmptyLattice]):
    keys = ("lanes.validate.no_extensions",)

    lattice = EmptyLattice

    def eval_fallback(self, frame: ForwardFrame, node: ir.Statement):
        # Every statement without an impl lands here, which is the point: the
        # rule is about the dialect a statement belongs to, not about any
        # particular statement class. `impls.Func` keeps the call statements out
        # of this path so the interpreter still walks into callees.
        if node.dialect is extensions_dialect:
            self.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"the experimental statement "
                    f"'{extensions_dialect.name}.{node.name}' is not supported "
                    "on this backend",
                    help=EXPERIMENTAL_HELP,
                ),
            )

        return tuple(self.lattice.bottom() for _ in range(len(node.results)))

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return self.lattice.bottom()

    def run_lattice(
        self,
        callee: EmptyLattice,
        inputs: tuple[EmptyLattice, ...],
        keys: tuple[str, ...],
        kwargs: tuple[EmptyLattice, ...],
    ) -> EmptyLattice:
        """Handle a dynamic ``func.Call``.

        Required, not optional: ``impls.Func`` subclasses the address analysis'
        ``func`` method table, so this analysis inherits a ``func.Call`` impl
        that calls ``run_lattice`` on the interpreter. Without an override, any
        kernel containing a dynamic call dies with an ``AttributeError``.

        Dynamic callees are deliberately not descended into, matching
        ``SteaneTerminalMeasurementValidation``: ``EmptyLattice`` carries no
        information, so there is nothing to resolve the callee with. A Gemini
        program cannot reach a device with an unresolved call still in it --
        ``NoStaticCallValidation`` rejects one, and it runs in the same suites
        this pass is added to -- so the gap is closed next door rather than here.
        """
        return self.lattice.bottom()


@dataclass
class NoLogicalExtensionsValidation(ValidationPass):
    """Reject any use of a ``gemini.logical.extensions`` statement."""

    def name(self) -> str:
        return "Gemini Logical Extensions Validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        analysis = _NoLogicalExtensionsAnalysis(method.dialects)
        frame, _ = analysis.run(method)

        return frame, analysis.get_validation_errors()
