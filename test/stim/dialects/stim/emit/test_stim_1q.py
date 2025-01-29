from bloqade import stim

from .base import codegen


def test_x():

    @stim.main
    def test_x():
        stim.X(targets=(0, 1, 2, 3), dagger=False)

    test_x.print()
    out = codegen(test_x)

    assert out.strip() == "X 0 1 2 3"

    @stim.main
    def test_x_dag():
        stim.X(targets=(0, 1, 2, 3), dagger=True)

    out = codegen(test_x_dag)

    assert out.strip() == "X 0 1 2 3"
