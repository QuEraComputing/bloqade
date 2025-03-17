from typing import cast
from dataclasses import dataclass

from kirin import ir


@dataclass(frozen=True)
class Sized(ir.StmtTrait):
    data: int


@dataclass(frozen=True)
class HasSize(ir.StmtTrait):
    """An operator with a `size` attribute."""

    def get_size(self, stmt: ir.Statement):
        attr = stmt.get_attr_or_prop("size")
        if attr is None:
            raise ValueError(f"Missing size attribute in {stmt}")
        return cast(ir.PyAttr[int], attr).data

    def set_size(self, stmt: ir.Statement, value: int):
        stmt.attributes["size"] = ir.PyAttr(value)
        return


@dataclass(frozen=True)
class Unitary(ir.StmtTrait):
    pass


@dataclass(frozen=True)
class MaybeUnitary(ir.StmtTrait):

    def is_unitary(self, stmt: ir.Statement):
        attr = stmt.get_attr_or_prop("is_unitary")
        if attr is None:
            return False
        return cast(ir.PyAttr[bool], attr).data

    def set_unitary(self, stmt: ir.Statement, value: bool):
        stmt.attributes["is_unitary"] = ir.PyAttr(value)
        return
