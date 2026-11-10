"""Calls that survive the passes are rejected, and inlined errors say who called.

Two related failures, both about the boundary a call draws through a kernel:

- A static call still standing when the ``verify`` suite runs means the program
  is not flat, and nothing downstream will say so. ``NoStaticCallValidation``
  reports it, and ``SteaneLogicalValidation`` delegates to it. This used to be a
  ``func.Invoke`` impl instead; a call nested inside an ``scf.For`` was invisible
  there, because that analysis' ``scf.For`` impl returns bottom without walking
  the loop body. The nested case below is what a dataflow check cannot see.

- A call that *is* inlined leaves its callee's statements behind carrying the
  callee's source info, so an error in one names a file the user never opened
  (bloqade-internal#449). ``InlineOrigins`` puts the call site back into the
  message and re-points the excerpt at the kernel the code came from.
"""

import pytest
import bloqade.squin as squin
from kirin.validation import ValidationSuite
from kirin.ir.exception import ValidationErrorGroup
from bloqade.lanes.validation.steane import SteaneLogicalValidation
from bloqade.lanes.validation.call_site import CallSite, InlineOrigins
from bloqade.lanes.validation.static_call import NoStaticCallValidation

import bloqade.gemini as gemini
from bloqade.gemini.logical.stdlib import default_post_processing


def _messages(err: ValidationErrorGroup) -> list[str]:
    return [e.args[0] if e.args else str(e) for e in err.errors]


# With aggressive unrolling disabled, `inline=False` keeps static calls; the
# `verify=False` on the fixtures below is because a surviving invoke is exactly
# what is being asserted on, so the fixture must not raise while being built.
@gemini.logical.kernel(verify=False)
def _helper(q):
    squin.x(q)


def test_surviving_call_is_reported():
    @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
    def caller():
        q = squin.qalloc(1)
        _helper(q[0])

    _, errors = NoStaticCallValidation().run(caller)

    assert any("_helper" in str(e) for e in errors)
    assert all("was not inlined" in str(e) for e in errors)


def test_call_nested_in_a_loop_is_reported():
    """The case the dataflow impl this replaced could not reach."""

    @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
    def caller():
        q = squin.qalloc(2)
        for i in range(len(q)):
            _helper(q[i])

    _, errors = NoStaticCallValidation().run(caller)
    assert any("_helper" in str(e) for e in errors)


def test_ilist_map_is_not_a_static_call():
    """`ilist.map` invokes its `fn`, but it is control flow the lowering handles.

    `squin.qalloc` lowers to an `ilist.map` over a kernel, so an inlined kernel
    that allocates is enough to exercise this without hand-building the map.
    """

    @gemini.logical.kernel(verify=False)
    def caller():
        q = squin.qalloc(2)
        squin.broadcast.x(q)

    _, errors = NoStaticCallValidation().run(caller)
    assert errors == []


def test_flat_kernel_has_no_errors():
    @gemini.logical.kernel(verify=False)
    def caller():
        q = squin.qalloc(1)
        _helper(q[0])

    _, errors = NoStaticCallValidation().run(caller)
    assert errors == []


def test_gemini_logical_validation_still_reports_calls_on_its_own():
    """`SteaneLogicalValidation` is public; composing it alone must still work.

    It is re-exported from `bloqade.gemini.logical.validation.clifford`, so a
    caller can assemble their own suite from it. Moving the call check out to
    `NoStaticCallValidation` must not quietly narrow what they get -- this pins
    the delegation so the pass stays a drop-in.
    """

    @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
    def caller():
        q = squin.qalloc(1)
        _helper(q[0])

    _, errors = SteaneLogicalValidation().run(caller)
    assert any("_helper" in str(e) and "was not inlined" in str(e) for e in errors)


def test_delegated_call_check_reaches_into_loop_bodies():
    """Delegating is strictly better than the impl it replaced.

    The old `func.Invoke` impl never saw this call: the `scf.For` impl returns
    bottom without walking the loop body.
    """

    @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
    def caller():
        q = squin.qalloc(2)
        for i in range(len(q)):
            _helper(q[i])

    _, errors = SteaneLogicalValidation().run(caller)
    assert any("_helper" in str(e) for e in errors)


def test_a_single_unresolved_call_is_reported_once():
    """Delegation rather than a second suite entry keeps the count honest."""

    @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
    def caller():
        q = squin.qalloc(1)
        _helper(q[0])

    result = ValidationSuite([SteaneLogicalValidation]).validate(caller)
    reported = [
        e
        for errors in result.errors.values()
        for e in errors
        if "_helper" in str(e.args[0])
    ]
    assert len(reported) == 1, reported


def test_surviving_call_is_rejected_by_the_kernel_group():
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(inline=False, aggressive_unroll=False)
        def caller():
            q = squin.qalloc(1)
            _helper(q[0])

    messages = " ".join(_messages(exc_info.value))
    assert "was not inlined" in messages


def test_inlined_error_names_the_call_site():
    """bloqade-internal#449: the reported file was one the user never opened.

    `default_post_processing` loops over `range(1, len(register))`, which only
    `aggressive_unroll=True` can flatten, so opting out leaves it invalid --
    but the error pointed into the stdlib with no hint of which call put it
    there.
    """
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(aggressive_unroll=False)
        def main():
            qbs = squin.qalloc(2)
            squin.broadcast.sqrt_y(qbs)
            squin.cz(qbs[0], qbs[1])
            return default_post_processing(qbs)

    messages = _messages(exc_info.value)
    assert any("Non-constant iterable in for loop" in m for m in messages)

    inlined = [m for m in messages if "inlined from" in m]
    assert inlined, messages
    for message in inlined:
        assert "'default_post_processing'" in message
        # The call site is in *this* file, which is the whole point.
        assert __file__ in message
        # Unrolling is the fix for the loop, and only for the loop: the
        # measurement-ID error it cascades into is not itself about unrolling.
        if "Non-constant iterable in for loop" in message:
            assert "aggressive_unroll=True" in message
            assert "verify=False" in message
        else:
            assert "aggressive_unroll" not in message


def test_inlined_error_excerpt_matches_the_reported_file():
    """`attach` renders the callee's filename against the caller's source.

    Left alone the excerpt is a chimera: right file, wrong line numbers, wrong
    code. `InlineOrigins` restores the statement's own offset and swaps in the
    defining kernel's source, so file, line and quoted code agree.
    """
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(aggressive_unroll=False)
        def main():
            qbs = squin.qalloc(2)
            squin.broadcast.sqrt_y(qbs)
            squin.cz(qbs[0], qbs[1])
            return default_post_processing(qbs)

    errors = [e for e in exc_info.value.errors if "inlined from" in str(e.args[0])]
    assert errors

    for err in errors:
        assert err.source is not None and err.source.file is not None
        assert err.source.file.endswith("post_processing.py")

        with open(err.source.file) as f:
            file_lines = f.read().splitlines()

        # `hint()` marks `lines[source.lineno - 1]`; that line has to be the one
        # `File "...", line N` names, or the caret points at unrelated code.
        assert err.lines, "excerpt was dropped"
        marked = err.lines[err.source.lineno - 1]
        absolute = err.source.lineno + err.source.lineno_begin
        assert marked.strip() == file_lines[absolute - 1].strip()


@pytest.mark.parametrize("aggressive_unroll", [False, True])
def test_inlined_error_unrolling_cannot_fix_gets_no_unroll_hint(aggressive_unroll):
    """The call site is still named, but the unroll advice would be wrong.

    `squin.ccz` inlines to a gate Gemini does not support; unrolling leaves it
    exactly as unsupported, so telling the user to unroll -- least of all when
    they already did -- sends them the wrong way.
    """
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(aggressive_unroll=aggressive_unroll)
        def main():
            q = squin.qalloc(3)
            squin.ccz(q[0], q[1], q[2])
            gemini.logical.terminal_measure(q)

    messages = _messages(exc_info.value)
    ccz = [m for m in messages if "Gate ccz is not supported" in m]
    assert ccz, messages
    for message in ccz:
        assert "inlined from 'ccz'" in message
        assert __file__ in message

    for message in messages:
        assert "aggressive_unroll" not in message
        assert "verify=False" not in message


def test_unroll_hint_is_dropped_when_the_kernel_already_unrolls():
    """An unroll-fixable error under `aggressive_unroll=True` stays dynamic.

    The loop in the helper runs over a runtime length, so unrolling was already
    tried and could not flatten it; repeating the advice would be circular. The
    kernel group cannot build this case (a Gemini kernel with arguments is
    rejected first), so drive `InlineOrigins` directly.
    """
    from kirin.passes.inline import InlinePass

    @gemini.logical.kernel(verify=False)
    def dynamic_loop(qubits):
        for i in range(1, len(qubits)):
            squin.x(qubits[i])

    def origins_and_result():
        @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
        def caller(qbs):
            dynamic_loop(qbs)

        origins = InlineOrigins.collect(caller)
        InlinePass(caller.dialects).fixpoint(caller)
        origins.snapshot(caller)
        result = ValidationSuite([SteaneLogicalValidation]).validate(caller)
        return caller, origins, result

    def loop_messages(result):
        return [
            e.args[0]
            for errors in result.errors.values()
            for e in errors
            if "Non-constant iterable" in e.args[0]
        ]

    caller, origins, result = origins_and_result()
    unrolled = loop_messages(origins.annotate(caller, result, aggressive_unroll=True))
    assert unrolled
    for message in unrolled:
        assert "inlined from 'dynamic_loop'" in message
        assert "aggressive_unroll" not in message

    # Same error without unrolling: the advice applies, so it is given.
    caller, origins, result = origins_and_result()
    plain = loop_messages(origins.annotate(caller, result))
    assert plain
    assert all("aggressive_unroll=True" in m for m in plain)


def test_aggressive_unroll_compiles_the_same_kernel():
    """The hint the message gives has to actually work."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        qbs = squin.qalloc(2)
        squin.broadcast.sqrt_y(qbs)
        squin.cz(qbs[0], qbs[1])
        return default_post_processing(qbs)

    assert main.sym_name == "main"


def test_origins_ignores_the_entry_method_itself():
    """An error in code the user wrote must not be blamed on a call."""

    # Disable both inlining paths because `collect` reads invokes, and in the
    # real pipeline it runs before the inliner splices them away.
    @gemini.logical.kernel(verify=False, inline=False, aggressive_unroll=False)
    def caller():
        q = squin.qalloc(1)
        _helper(q[0])

    origins = InlineOrigins.collect(caller)
    assert (caller.file, caller.lineno_begin) not in origins.callees
    # ...but the helper next to it in the same file is still a callee, which is
    # what keying on `(file, lineno_begin)` rather than on `file` alone buys.
    assert (_helper.file, _helper.lineno_begin) in origins.callees


def test_helper_in_the_users_own_file_is_attributed():
    """The same-file case: filename alone cannot tell the two kernels apart."""

    @gemini.logical.kernel(verify=False)
    def dynamic_loop(qubits):
        for i in range(1, len(qubits)):
            squin.x(qubits[i])

    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(aggressive_unroll=False)
        def main():
            qbs = squin.qalloc(2)
            dynamic_loop(qbs)
            return default_post_processing(qbs)

    inlined = [m for m in _messages(exc_info.value) if "'dynamic_loop'" in m]
    assert inlined, _messages(exc_info.value)
    assert all("Non-constant iterable in for loop" in m for m in inlined)


def test_call_site_renders_file_and_line():
    site = CallSite(callee="post_processing", file="/tmp/kernel.py", lineno=12)
    assert str(site) == "'post_processing' called at /tmp/kernel.py:12"

    assert str(CallSite(callee="f", file=None, lineno=3)) == "'f' called at line 3"


def test_annotate_is_a_no_op_without_call_sites():
    """A kernel that calls nothing outside its own file gets no note."""

    @gemini.logical.kernel(verify=False)
    def caller(q):
        squin.x(q)

    origins = InlineOrigins()
    result = ValidationSuite([NoStaticCallValidation]).validate(caller)
    assert origins.annotate(caller, result) is result
