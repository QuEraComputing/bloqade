from bloqade import stim

from .base import codegen


def test_x():

    @stim.main
    def test_x():
        stim.x(targets=(0, 1, 2, 3), dagger=False)

    test_x.print()
    out = codegen(test_x)

    assert out.strip() == "X 0 1 2 3"

    @stim.main
    def test_x_dag():
        stim.x(targets=(0, 1, 2, 3), dagger=True)

    out = codegen(test_x_dag)

    assert out.strip() == "X 0 1 2 3"


def test_y():

    @stim.main
    def test_y():
        stim.y(targets=(0, 1, 2, 3), dagger=False)

    test_y.print()
    out = codegen(test_y)

    assert out.strip() == "Y 0 1 2 3"

    @stim.main
    def test_y_dag():
        stim.y(targets=(0, 1, 2, 3), dagger=True)

    out = codegen(test_y_dag)

    assert out.strip() == "Y 0 1 2 3"
