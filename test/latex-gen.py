from io import StringIO

# from typing import Union
from dataclasses import field, dataclass

from kirin import ir, emit, interp
from bloqade import qasm2
from kirin.dialects import func
from bloqade.qasm2.passes import fold, py2qasm
from bloqade.qasm2.dialects import uop


@dataclass
class LaTeXFrame(emit.EmitStrFrame):
    pass  # customize, add option to select surrounding template (wraps build syntax) via field
    # default arguments should be template + empty list
    # method inside frame generates final string (combine template + list)


@dataclass
class EmitLaTeX(emit.EmitStr):
    keys = ["emit.latex"]
    # have options for "void" (what to do if interpreter touches nothing ),
    # "file" (I guess we can just use a string here for now?, copy from EmitStimMain), prefix, and prefix_if_none
    file: StringIO = field(default_factory=StringIO)


class LaTeXTarget:

    def emit(self, entry: ir.Method) -> str:
        entry = entry.similar()

        fold.QASM2Fold(entry.dialects)(entry)
        py2qasm.Py2QASM(entry.dialects)(entry)

        # skip the second fold for now

        # do we need address analysis? -> Yes, just return an int for methodtable of core (qregget), str format later
        # Used to initialize EmitQourier
        ## address_analysis takes a statement (e.g.: core.QRegNew) and then
        ## gives you an address obj from SSA value... but only seems to be used for qreg?
        ## probably don't need to handle that in LaTeX right now

        latex_str = (
            EmitLaTeX(entry.dialects).run(entry, ()).expect()
        )  # -> could be a list of strs that can later
        # -> be jointed together

        return latex_str


# "main"  is literally a function, have to define this or the other stuff doesn't go through
@func.dialect.register(key="emit.latex")
class Func(interp.MethodTable):

    @interp.impl(func.Function)
    def emit_func(self, emit: EmitLaTeX, frame: LaTeXFrame, stmt: func.Function):
        _ = emit.run_ssacfg_region(frame, stmt.body)  # took this from Stim codegen
        return ()


@uop.dialect.register(key="emit.latex")
class UOp(interp.MethodTable):

    @interp.impl(qasm2.uop.H)
    def emit_h(self, emit: EmitLaTeX, frame: LaTeXFrame, stmt: uop.H):

        emit.writeln(frame, "aye this shit works!")
        return ()  # where does this return value go when you're dealing with building strings?


if __name__ == "__main__":

    @qasm2.main  # the main dialect consists of: uop, expr, core, scf, indexing, func, lowering.func, lowering.call
    # need to
    def main():

        q = qasm2.qreg(2)

        qasm2.h(q[0])

    main.print()

    target = LaTeXTarget()

    str = target.emit(main)
    print(type(str))
    print(str)
