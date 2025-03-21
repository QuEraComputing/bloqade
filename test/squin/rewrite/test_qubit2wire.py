from kirin import rewrite
from bloqade.squin import op, qubit
from bloqade.squin.groups import wired, kernel
from bloqade.squin.rewrite.qubit2wire import Qubit2WireRule, wid


@kernel(fold=True)
def test_1(flag: bool):
    qs = qubit.new(2)
    qubit.apply(op.kron(op.x(), op.x()), [qs[0], qs[1]])
    if flag:
        qubit.apply(op.x(), [qs[0]])


# test_1.print()

rewrite.Fixpoint(rewrite.Walk(rewrite.CommonSubexpressionElimination())).rewrite(
    test_1.code
)

rewrite.Walk(Qubit2WireRule(), region_first=True).rewrite(test_1.code)


test_1.print()

for stmt in test_1.callable_region.blocks[0].stmts:
    print(", ".join(wid(result) for result in stmt.results), ":", stmt)


rewrite.Walk(rewrite.DeadCodeElimination(), region_first=False).rewrite(test_1.code)

test_wired = test_1.similar(wired)


test_wired.print()
