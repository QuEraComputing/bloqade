from kirin import passes, rewrite
from kirin.passes import aggressive
from bloqade.squin import op, qubit
from bloqade.squin.groups import wired, kernel
from bloqade.squin.rewrite import Qubit2WireRule


@kernel(fold=True)
def test_1(flag: bool):
    qs = qubit.new(2)
    qubit.apply(op.kron(op.x(), op.x()), [qs[0], qs[1]])
    if flag:
        qubit.apply(op.x(), [qs[0]])


test_1.print()

rewrite.Fixpoint(rewrite.Walk(rewrite.CommonSubexpressionElimination())).rewrite(
    test_1.code
)
rewrite.Walk(Qubit2WireRule()).rewrite(test_1.code)
rewrite.Walk(rewrite.DeadCodeElimination()).rewrite(test_1.code)

test_wired = test_1.similar(wired)


test_wired.print()
