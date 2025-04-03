from kirin import ir, passes
from bloqade import squin
from kirin.ir import dialect_group
from kirin.prelude import basic

# from bloqade.analysis import address


@dialect_group(basic.add(squin.wire).add(squin.qubit).add(squin.op))
def squin_dialect(self):
    # Const prop analysis runs first, then fold pass takes
    # ConstantFold puts in the type hints! Need that for the
    # get_constant_value method in the address analysis pass
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


"""
@squin_dialect
def squin_new_qubits():

    # create some new qubits
    qubits = squin.qubit.new(10)

    return qubits


frame, _ = address.AddressAnalysis(squin_new_qubits.dialects).run_analysis(squin_new_qubits)

print(frame)

for ssa_val, addr_type in frame.entries.items():
    print(f"SSA: {ssa_val}\n Addr: {addr_type}")"
"""


## Avoid using frontend
# squin.qubit.New()


@squin_dialect
def simple_squin_program():

    # Now we wrap to get the qubit back
    ## Can I just reuse the original qubit in the wrapping op?
    ## -> Yes! That's how it's done in Quake
    # squin.wire.wrap(wire=w, qubit=q)

    return


simple_squin_program.print()


def test(*args: int):
    print(args)
    return args
    pass


test(1, 2, 3, 4)


# from kirin.ir import Statement

# Statement(result_types=)
