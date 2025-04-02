from kirin import ir, passes
from bloqade import squin
from kirin.ir import dialect_group
from kirin.prelude import basic
from bloqade.analysis import address


@dialect_group(basic.add(squin.wire).add(squin.qubit))
def squin_dialect(self):
    # Const prop analysis runs first, then fold pass takes
    # ConstantFold puts in the type hints! Need that for
    fold_pass = passes.Fold(self)
    typeinfer_pass = passes.TypeInfer(self)

    def run_pass(
        method: ir.Method,
        *,
        fold: bool = True,
    ):
        method.verify()
        # TODO make special Function rewrite

        if fold:
            fold_pass(method)

        typeinfer_pass(method)
        method.code.typecheck()

    return run_pass


@squin_dialect
def main():

    # create some new qubits
    qubits = squin.qubit.new(10)

    return qubits


frame, _ = address.AddressAnalysis(main.dialects).run_analysis(main)

print(frame)

for ssa_val, addr_type in frame.entries.items():
    print(f"SSA: {ssa_val}\n Addr: {addr_type}")
