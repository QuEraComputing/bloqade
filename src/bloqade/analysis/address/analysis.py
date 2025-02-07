from typing import TypeVar

from kirin import ir, types, interp
from bloqade.types import QubitType
from kirin.analysis import Forward, const
from kirin.dialects import cf, py, func, ilist
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
        if isinstance(value.type, types.Hinted) and isinstance(
            value.type.data, const.Value
        ):
            data = value.type.data.data
            if isinstance(data, typ):
                return value.type.data.data
            raise interp.InterpreterError(
                f"Expected constant value <type = {typ}>, got {data}"
            )
        raise interp.InterpreterError(
            f"Expected constant value <type = {typ}>, got {value}"
        )

    def eval_stmt_fallback(
        self, frame: ForwardFrame[Address, None], stmt: ir.Statement
    ) -> tuple[Address, ...] | interp.SpecialValue[Address]:
        return tuple(
            (
                self.lattice.top()
                if result.type.is_subseteq(QubitType)
                else self.lattice.bottom()
            )
            for result in stmt.results
        )

    # def should_exec_stmt(self, stmt: ir.Statement):
    #     return (
    #         stmt.has_trait(ir.ConstantLike)
    #         or stmt.dialect in self.dialects.data
    #         or isinstance(
    #             stmt,
    #             (
    #                 func.Return,
    #                 func.Invoke,
    #                 py.tuple.New,
    #                 ilist.New,
    #                 py.GetItem,
    #                 py.Alias,
    #                 py.Add,
    #                 cf.Branch,
    #                 cf.ConditionalBranch,
    #             ),
    #         )
    #     )

    def run_method(self, method: ir.Method, args: tuple[Address, ...]) -> Address:
        # NOTE: we do not support dynamic calls here, thus no need to propagate method object
        return self.run_callable(method.code, (self.lattice.bottom(),) + args)
