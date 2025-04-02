from bloqade import squin
from kirin.ir import dialect_group
from kirin.prelude import basic_no_opt
from bloqade.analysis import address


@dialect_group(basic_no_opt.add(squin.wire.dialect).add(squin.qubit.dialect))
def squin_dialect(self):
    # usually plug in Fold pass, hold off for now
    return


@squin_dialect
def main():

    # create some new qubits
    qubits = squin.qubit.new(10)

    return qubits


main.print()
frame, _ = address.AddressAnalysis(main.dialects).run_analysis(main)

print(frame)

for ssa_val, addr_type in frame.entries.items():
    print(f"SSA: {ssa_val}\n Addr: {addr_type}")
