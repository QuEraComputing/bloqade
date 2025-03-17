from kirin import ir
from kirin.prelude import structural_no_opt

from . import op, wire, qubit


@ir.dialect_group(structural_no_opt.union([op, qubit]))
def kernel(self):
    def run_pass(method):
        pass

    return run_pass


@ir.dialect_group(structural_no_opt.union([op, wire]))
def wired(self):
    def run_pass(method):
        pass

    return run_pass
