from kirin import interp
from kirin.analysis import ForwardFrame
from bloqade.lanes.dialects.arch import stmts
from bloqade.analysis.address.lattice import Address, AddressQubit
from bloqade.analysis.address.analysis import AddressAnalysis
from bloqade.lanes.dialects.arch._dialect import dialect


@dialect.register(key="qubit.address")
class _GeminiCommonAddressMethods(interp.MethodTable):
    @interp.impl(stmts.NewAt)
    def new_at(
        self,
        interp_: AddressAnalysis,
        frame: ForwardFrame[Address],
        stmt: stmts.NewAt,
    ):
        addr = AddressQubit(interp_.next_address)
        interp_.next_address += 1
        return (addr,)
