from typing import TypeVar, Iterable

from kirin import ir, types, interp
from bloqade.types import QubitType
from kirin.analysis import Forward, const
from kirin.dialects import cf, py, func, ilist
from kirin.exceptions import InterpreterError
from kirin.analysis.forward import ForwardFrame

from .lattice import Address


class AddressAnalysis(Forward[Address]):
    keys = ["qubit.address"]
    lattice = Address

    def __init__(
        self,
        dialects: ir.DialectGroup | Iterable[ir.Dialect],
        *,
        fuel: int | None = None,
        save_all_ssa: bool = False,
        max_depth: int = 128,
        max_python_recursion_depth: int = 8192,
    ):
        super().__init__(
            dialects,
            fuel=fuel,
            save_all_ssa=save_all_ssa,
            max_depth=max_depth,
            max_python_recursion_depth=max_python_recursion_depth,
        )
        self.next_address: int = 0
        self.constprop_results: dict[ir.SSAValue, const.JointResult] = {}

    def clear(self):
        self.next_address = 0
        self.constprop_results.clear()

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
            raise InterpreterError(
                f"Expected constant value <type = {typ}>, got {data}"
            )
        raise InterpreterError(f"Expected constant value <type = {typ}>, got {value}")

    def run_stmt_fallback(
        self, frame: ForwardFrame[Address, None], stmt: ir.Statement
    ) -> tuple[Address, ...] | interp.SpecialResult[Address]:
        return tuple(
            (
                self.lattice.top()
                if result.type.is_subseteq(QubitType)
                else self.lattice.bottom()
            )
            for result in stmt.results
        )

    def should_exec_stmt(self, stmt: ir.Statement):
        return (
            stmt.has_trait(ir.ConstantLike)
            or stmt.dialect in self.dialects.data
            or isinstance(
                stmt,
                (
                    func.Return,
                    func.Invoke,
                    py.tuple.New,
                    ilist.New,
                    py.GetItem,
                    py.Alias,
                    py.Add,
                    cf.Branch,
                    cf.ConditionalBranch,
                ),
            )
        )

    def run_method(
        self, method: ir.Method, args: tuple[Address, ...]
    ) -> Address | interp.Err[Address]:
        if len(self.state.frames) >= self.max_depth:
            raise InterpreterError("maximum recursion depth exceeded")
        # NOTE: we do not support dynamic calls here, thus no need to propagate method object
        return self.run_callable(method.code, (self.lattice.bottom(),) + args)
