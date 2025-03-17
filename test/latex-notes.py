"""
import io
from dataclasses import dataclass

from kirin import ir, emit
from rich.console import Console
from kirin.analysis import CallGraph
from kirin.dialects import ilist
from bloqade.qasm2.parse import ast, pprint
from bloqade.qasm2.emit.gate import EmitQASM2Gate
from bloqade.qasm2.emit.main import EmitQASM2Main
from bloqade.qasm2.passes.fold import QASM2Fold
from bloqade.qasm2.passes.glob import GlobalToParallel
from bloqade.qasm2.passes.py2qasm import Py2QASM
from bloqade.qasm2.passes.parallel import ParallelToUOp


@dataclass
class LaTeXFrame:  # should be something like emit.EmitFrame[???]
    # EmitFrame inherits from interpreter frame, which has ability to
    # get and set ValueTypes from SSAValues

    # - Seem to use frames for buffering things, like individual gates
    # - in the actual EmitFrame, ValueType only contributes to block_ref var

    # I can get the
    pass


@dataclass
class EmitLaTeX:  # should inherit from emti.EmitABC[LatExFrame, ???]
    pass


class LaTeXTarget:

    def emit(self, entry: ir.Method) -> str:
        entry = entry.similar()

        QASM2Fold(entry.dialects)(entry)
        Py2QASM(entry.dialects)(entry)

        # skip the second fold for now

        # do we need address analysis?
        # Used to initialize EmitQourier
        ## address_analysis takes a statement (e.g.: core.QRegNew) and then
        ## gives you an address obj from SSA value... but only seems to be used for qreg?
        ## probably don't need to handle that in LaTeX right now

        # EmitLaTeX(...).run(entry, ()).expect() -> could be a list of strs that can later
        #                                           be jointed together


if __name__ == "__main__":

    from bloqade import qasm2

    @qasm2.main  # contains uop, expr, core, scf, indexing, func, lowering.func, lowering.call
    # the extended dialect group will have parallel and global in it but we can rewrite that all the way to UOp anyways
    def main():
        q = qasm2.qreg(4)

        qasm2.h(q[0])
        qasm2.cx(q[0], q[1])

    main.print()

    LaTeXTarget().emit(main)

    main.print()
"""
