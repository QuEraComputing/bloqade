"""
qubit.address method table for a few builtin dialects.
"""

from kirin import interp
from kirin.dialects import cf, py, func, ilist
from bloqade.qasm2.dialects import core

from .lattice import Address, NotQubit, AddressReg, AddressQubit, AddressTuple
from .analysis import AddressAnalysis


@core.dialect.register(key="qubit.address")
class AddressMethodTable(interp.MethodTable):

    @interp.impl(core.QRegNew)
    def new(
        self,
        interp: AddressAnalysis,
        frame: interp.Frame[Address],
        stmt: core.QRegNew,
    ):
        n_qubits = interp.get_const_value(int, stmt.n_qubits)
        addr = AddressReg(range(interp.next_address, interp.next_address + n_qubits))
        interp.next_address += n_qubits
        return (addr,)

    @interp.impl(core.QRegGet)
    def get(
        self, interp: AddressAnalysis, frame: interp.Frame[Address], stmt: core.QRegGet
    ):
        addr = frame.get(stmt.reg)
        pos = interp.get_const_value(int, stmt.idx)
        if isinstance(addr, AddressReg):
            return (AddressQubit(addr.data[pos]),)
        else:  # this is not reachable
            return (NotQubit(),)


@py.binop.dialect.register(key="qubit.address")
class PyBinOp(interp.MethodTable):

    @interp.impl(py.Add)
    def add(self, interp: AddressAnalysis, frame: interp.Frame, stmt: py.Add):
        lhs = frame.get(stmt.lhs)
        rhs = frame.get(stmt.rhs)

        if isinstance(lhs, AddressTuple) and isinstance(rhs, AddressTuple):
            return (AddressTuple(data=lhs.data + rhs.data),)
        else:
            return (NotQubit(),)


@py.constant.dialect.register(key="qubit.address")
class PyConstant(interp.MethodTable):
    @interp.impl(py.Constant)
    def constant(self, interp: AddressAnalysis, frame: interp.Frame, stmt: py.Constant):
        return (NotQubit(),)


@py.tuple.dialect.register(key="qubit.address")
class PyTuple(interp.MethodTable):
    @interp.impl(py.tuple.New)
    def new_tuple(
        self,
        interp: AddressAnalysis,
        frame: interp.Frame,
        stmt: py.tuple.New,
    ):
        return (AddressTuple(frame.get_values(stmt.args)),)


@ilist.dialect.register(key="qubit.address")
class IList(interp.MethodTable):
    @interp.impl(ilist.New)
    def new_ilist(
        self,
        interp: AddressAnalysis,
        frame: interp.Frame,
        stmt: ilist.New,
    ):
        return (AddressTuple(frame.get_values(stmt.args)),)


@py.list.dialect.register(key="qubit.address")
class PyList(interp.MethodTable):
    @interp.impl(py.list.New)
    def new_ilist(
        self,
        interp: AddressAnalysis,
        frame: interp.Frame,
        stmt: py.list.New,
    ):
        return (AddressTuple(frame.get_values(stmt.args)),)


@py.indexing.dialect.register(key="qubit.address")
class PyIndexing(interp.MethodTable):
    @interp.impl(py.GetItem)
    def getitem(self, interp: AddressAnalysis, frame: interp.Frame, stmt: py.GetItem):
        idx = interp.get_const_value(int, stmt.index)
        obj = frame.get(stmt.obj)
        if isinstance(obj, AddressTuple):
            return (obj.data[idx],)
        elif isinstance(obj, AddressReg):
            return (AddressQubit(obj.data[idx]),)
        else:
            return (NotQubit(),)


@py.assign.dialect.register(key="qubit.address")
class PyAssign(interp.MethodTable):
    @interp.impl(py.Alias)
    def alias(self, interp: AddressAnalysis, frame: interp.Frame, stmt: py.Alias):
        return (frame.get(stmt.value),)


@func.dialect.register(key="qubit.address")
class Func(interp.MethodTable):
    @interp.impl(func.Return)
    def return_(self, _: AddressAnalysis, frame: interp.Frame, stmt: func.Return):
        return interp.ReturnValue(frame.get(stmt.value))


@cf.dialect.register(key="qubit.address")
class Cf(cf.typeinfer.TypeInfer):
    # NOTE: cf just re-use the type infer method table
    # it's the same process as type infer.
    pass
