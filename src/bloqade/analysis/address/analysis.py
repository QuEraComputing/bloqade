from typing import TypeVar

from kirin import ir, interp
from bloqade.types import QubitType
from kirin.analysis import Forward, const
from kirin.analysis.forward import ForwardFrame

from .lattice import Address


class AddressAnalysis(Forward[Address]):
    keys = ["qubit.address"]
    lattice = Address

    def initialize(self):
        super().initialize()
        self.next_address: int = 0
        return self

    @property
    def qubit_count(self) -> int:
        return self.next_address

    T = TypeVar("T")

    def get_const_value(self, typ: type[T], value: ir.SSAValue) -> T:
        if isinstance(hint := value.hints.get("const"), const.Value):
            data = hint.data
            if isinstance(data, typ):
                return hint.data
            raise interp.InterpreterError(
                f"Expected constant value <type = {typ}>, got {data}"
            )
        raise interp.InterpreterError(
            f"Expected constant value <type = {typ}>, got {value}"
        )

    def eval_stmt_fallback(
        self, frame: ForwardFrame[Address], stmt: ir.Statement
    ) -> tuple[Address, ...] | interp.SpecialValue[Address]:
        return tuple(
            (
                self.lattice.top()
                if result.type.is_subseteq(QubitType)
                else self.lattice.bottom()
            )
            for result in stmt.results
        )

    def run_method(self, method: ir.Method, args: tuple[Address, ...]):
        # NOTE: we do not support dynamic calls here, thus no need to propagate method object
        return self.run_callable(method.code, (self.lattice.bottom(),) + args)
