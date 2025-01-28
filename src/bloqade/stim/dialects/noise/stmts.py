from kirin import ir
from kirin.decl import info, statement
from kirin.ir import types

from ._dialect import dialect


@statement(dialect=dialect)
class Depolarize1(ir.Statement):
    name = "Depolarize1"
    traits = frozenset({ir.FromPythonCall()})
    p: ir.SSAValue = info.argument(ir.types.Float)
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)


@statement(dialect=dialect)
class Depolarize2(ir.Statement):
    name = "Depolarize2"
    traits = frozenset({ir.FromPythonCall()})
    p: ir.SSAValue = info.argument(ir.types.Float)
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)


@statement(dialect=dialect)
class PauliChannel1(ir.Statement):
    name = "PauliChannel1"
    traits = frozenset({ir.FromPythonCall()})
    px: ir.SSAValue = info.argument(ir.types.Float)
    py: ir.SSAValue = info.argument(ir.types.Float)
    pz: ir.SSAValue = info.argument(ir.types.Float)
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)


@statement(dialect=dialect)
class PauliChannel2(ir.Statement):
    name = "PauliChannel2"
    # TODO custom lowering to make sugar for this
    traits = frozenset({ir.FromPythonCall()})
    pix: ir.SSAValue = info.argument(ir.types.Float)
    piy: ir.SSAValue = info.argument(ir.types.Float)
    piz: ir.SSAValue = info.argument(ir.types.Float)
    pxi: ir.SSAValue = info.argument(ir.types.Float)
    pxx: ir.SSAValue = info.argument(ir.types.Float)
    pxy: ir.SSAValue = info.argument(ir.types.Float)
    pxz: ir.SSAValue = info.argument(ir.types.Float)
    pyi: ir.SSAValue = info.argument(ir.types.Float)
    pyx: ir.SSAValue = info.argument(ir.types.Float)
    pyy: ir.SSAValue = info.argument(ir.types.Float)
    pyz: ir.SSAValue = info.argument(ir.types.Float)
    pzi: ir.SSAValue = info.argument(ir.types.Float)
    pzx: ir.SSAValue = info.argument(ir.types.Float)
    pzy: ir.SSAValue = info.argument(ir.types.Float)
    pzz: ir.SSAValue = info.argument(ir.types.Float)
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)


@statement(dialect=dialect)
class XError(ir.Statement):
    name = "X_ERROR"
    traits = frozenset({ir.FromPythonCall()})
    p: ir.SSAValue = info.argument(ir.types.Float)
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)


@statement(dialect=dialect)
class YError(ir.Statement):
    name = "Y_ERROR"
    traits = frozenset({ir.FromPythonCall()})
    p: ir.SSAValue = info.argument(ir.types.Float)
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)


@statement(dialect=dialect)
class ZError(ir.Statement):
    name = "Z_ERROR"
    traits = frozenset({ir.FromPythonCall()})
    p: ir.SSAValue = info.argument(ir.types.Float)
    targets: tuple[ir.SSAValue, ...] = info.argument(types.Int)
