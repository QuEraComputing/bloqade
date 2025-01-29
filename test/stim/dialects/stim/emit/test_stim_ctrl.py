from bloqade import stim

from .base import codegen


def test_cx():

    @stim.main
    def test_simple_cx():
        stim.CX(controls=(4, 5, 6, 7), targets=(0, 1, 2, 3), dagger=False)

    out = codegen(test_simple_cx)
    assert out.strip() == "CX 4 0 5 1 6 2 7 3"


def test_cx_cond_on_measure():

    @stim.main
    def test_simple_cx_cond_measure():
        stim.CX(
            controls=(stim.GetRecord(id=-1), 4, stim.GetRecord(id=-2)),
            targets=(0, 1, 2),
            dagger=False,
        )

    out = codegen(test_simple_cx_cond_measure)

    assert out.strip() == "CX rec[-1] 0 4 1 rec[-2] 2"
