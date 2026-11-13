"""Tests for LocationAddress attribute statements + GetAttr rewrite."""

from kirin import ir, types, lowering
from kirin.dialects import py, func
from bloqade.lanes.dialects.arch import (
    SiteId,
    WordId,
    ZoneId,
    LocationAddressType,
    RewriteLocationAttr,
)
from bloqade.lanes.bytecode.encoding import LocationAddress

from bloqade.gemini.logical import loc
from bloqade.gemini.physical import kernel as movement_kernel


def test_statements_are_pure_and_named():
    for stmt_cls, name in [
        (WordId, "word_id"),
        (SiteId, "site_id"),
        (ZoneId, "zone_id"),
    ]:
        assert issubclass(stmt_cls, ir.Statement)
        assert stmt_cls.name == name
        # Pure so the fold pass can evaluate them.
        assert any(isinstance(t, ir.Pure) for t in stmt_cls.traits)
        # Not lowered from a Python call (constructed only by the rewrite).
        assert not any(isinstance(t, lowering.FromPythonCall) for t in stmt_cls.traits)


# -- rule unit tests: the guard cases must NOT rewrite --


def test_rewrite_skips_unsupported_attrname():
    addr = ir.TestValue(type=LocationAddressType)
    node = py.GetAttr(obj=addr, attrname="nonexistent")
    result = RewriteLocationAttr().rewrite_Statement(node)
    assert not result.has_done_something


def test_rewrite_skips_non_location_type():
    obj = ir.TestValue(type=types.Int)
    node = py.GetAttr(obj=obj, attrname="word_id")
    result = RewriteLocationAttr().rewrite_Statement(node)
    assert not result.has_done_something


def test_rewrite_skips_bottom_type():
    # Bottom is a subtype of every type, so without the explicit guard this
    # would wrongly match. It must be skipped.
    obj = ir.TestValue(type=types.Bottom)
    node = py.GetAttr(obj=obj, attrname="word_id")
    result = RewriteLocationAttr().rewrite_Statement(node)
    assert not result.has_done_something


# -- integration: a symbolic (non-constant) address rewrites to the statement --


def test_param_word_id_rewrites_to_statement():
    @movement_kernel(verify=False)
    def k(a: LocationAddress):
        return a.word_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, WordId)


def test_param_site_id_rewrites_to_statement():
    @movement_kernel(verify=False)
    def k(a: LocationAddress):
        return a.site_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, SiteId)


def test_param_zone_id_rewrites_to_statement():
    @movement_kernel(verify=False)
    def k(a: LocationAddress):
        return a.zone_id

    stmts = list(k.callable_region.walk())
    assert not any(isinstance(s, py.GetAttr) for s in stmts)
    ret = next(s for s in stmts if isinstance(s, func.Return))
    assert isinstance(ret.value.owner, ZoneId)


# -- integration: a constant loc resolves against the arch + its attribute
#    reads const-fold to the resolved word/site/zone --
#
# ``loc`` now resolves ``(zone, row, col)`` via the arch spec, so it only folds
# once ``arch_spec`` is bound (by ``BindArchSpec``, as the pipeline does).


def _attr_lattice(k, stmt_cls):
    """Bind the arch spec then run const-prop; return the (lattice, arch) for
    the attribute-read statement's result."""
    from kirin import rewrite
    from kirin.analysis import const
    from bloqade.lanes.dialects.arch import BindArchSpec
    from bloqade.lanes.arch.gemini.physical import get_physical_layout_arch_spec

    arch = get_physical_layout_arch_spec()
    rewrite.Walk(BindArchSpec(arch)).rewrite(k.code)
    frame, _ = const.Propagate(k.dialects).run(k)
    node = next(s for s in k.callable_region.walk() if isinstance(s, stmt_cls))
    return frame.entries.get(node.result), arch


def test_constant_word_id_folds():
    from kirin.analysis import const

    @movement_kernel(verify=False)
    def k():
        return loc(0, 1, 2).word_id

    val, arch = _attr_lattice(k, WordId)
    resolved = arch.location_at(0, 1, 2)
    assert resolved is not None and isinstance(val, const.Value)
    assert val.data == resolved.word_id


def test_constant_site_id_folds():
    from kirin.analysis import const

    @movement_kernel(verify=False)
    def k():
        return loc(0, 0, 4).site_id

    val, arch = _attr_lattice(k, SiteId)
    resolved = arch.location_at(0, 0, 4)
    assert resolved is not None and isinstance(val, const.Value)
    assert val.data == resolved.site_id


def test_constant_zone_id_folds():
    from kirin.analysis import const

    @movement_kernel(verify=False)
    def k():
        return loc(0, 1, 2).zone_id

    val, arch = _attr_lattice(k, ZoneId)
    resolved = arch.location_at(0, 1, 2)
    assert resolved is not None and isinstance(val, const.Value)
    assert val.data == resolved.zone_id
