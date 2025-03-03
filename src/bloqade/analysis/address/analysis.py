from typing import TypeVar
from dataclasses import field

from kirin import ir, interp
from bloqade.types import QubitType
from kirin.analysis import Forward, const
from kirin.analysis.forward import ForwardFrame

from .lattice import Address


class AddressAnalysis(Forward[Address]):
    """This analysis pass can be used to track the global addresses of qubits."""

    keys = ["qubit.address"]
    lattice = Address
    address_map: dict[int, ir.SSAValue] = field(default_factory=dict, init=False)
    next_address: int = field(default=0, init=False)

    def initialize(self):
        super().initialize()
        self.next_address: int = 0
        self.address_map = {}
        return self

    @property
    def qubit_count(self) -> int:
        """Total number of qubits found by the analysis."""
        return self.next_address

    @property
    def qubit_ssa_value(self):
        """Map of global qubit addresses to their SSA values."""
        return self.address_map

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
