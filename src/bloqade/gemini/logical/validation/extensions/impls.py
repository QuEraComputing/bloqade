from kirin.dialects import func
from bloqade.analysis.address.impls import Func as AddressFuncMethodTable


@func.dialect.register(key="lanes.validate.no_extensions")
class Func(AddressFuncMethodTable):
    """Give the analysis the address analysis' ``func`` impls.

    This is what makes the pass a call-graph search rather than a walk of one
    body: without a ``func.Invoke`` impl the interpreter would send the call to
    ``eval_fallback``, which returns bottom without descending, and a use inside
    an un-inlined callee would go unreported.

    Reusing the address analysis' table rather than writing one is the same
    choice ``lanes.validate.terminal_measurement`` makes next door.
    """
