"""
qubit.address method table for a few builtin dialects.
"""

from kirin import interp
from kirin.analysis import ForwardFrame, const
from kirin.dialects import cf, py, scf, func, ilist

from .lattice import Address, NotQubit, AddressReg, AddressQubit, AddressTuple
from .analysis import AddressAnalysis


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

    # TODO: replace with the generic implementation
    @interp.impl(func.Invoke)
    def invoke(self, interp_: AddressAnalysis, frame: interp.Frame, stmt: func.Invoke):
        _, ret = interp_.run_method(
            stmt.callee,
            interp_.permute_values(
                stmt.callee.arg_names, frame.get_values(stmt.inputs), stmt.kwargs
            ),
        )
        return (ret,)

    # TODO: support lambda?


@cf.dialect.register(key="qubit.address")
class Cf(cf.typeinfer.TypeInfer):
    # NOTE: cf just re-use the type infer method table
    # it's the same process as type infer.
    pass


@scf.dialect.register(key="qubit.address")
class Scf(scf.absint.Methods):

    @interp.impl(scf.For)
    def for_loop(
        self,
        interp_: AddressAnalysis,
        frame: ForwardFrame[Address],
        stmt: scf.For,
    ):
        if not isinstance(hint := stmt.iterable.hints.get("const"), const.Value):
            return interp_.eval_stmt_fallback(frame, stmt)

        iterable = hint.data
        loop_vars = frame.get_values(stmt.initializers)
        body_block = stmt.body.blocks[0]
        block_args = body_block.args

        # NOTE: we need to actually run iteration in case there are
        # new allocations/re-assign in the loop body.
        for _ in iterable:
            with interp_.state.new_frame(interp_.new_frame(stmt)) as body_frame:
                body_frame.entries.update(frame.entries)
                body_frame.set_values(
                    block_args,
                    (NotQubit(),) + loop_vars,
                )
                loop_vars = interp_.run_ssacfg_region(body_frame, stmt.body)

            if loop_vars is None:
                loop_vars = ()
            elif isinstance(loop_vars, interp.ReturnValue):
                return loop_vars

        if isinstance(body_block.last_stmt, func.Return):
            frame.worklist.append(interp.Successor(body_block, NotQubit(), *loop_vars))
            return  # if terminate is Return, there is no result

        return loop_vars
