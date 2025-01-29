from bloqade import stim

from .base import codegen


def test_detector():

    @stim.main
    def test_simple_cx():
        stim.Detector(coord=(1, 2, 3), targets=(stim.GetRecord(-3), stim.GetRecord(-1)))

    out = codegen(test_simple_cx)

    assert out.strip() == "DETECTOR(1, 2, 3) rec[-3] rec[-1]"


test_detector()
