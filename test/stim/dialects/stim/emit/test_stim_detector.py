from kirin import ir
from bloqade import stim
from bloqade.stim.emit.stim import EmitStimMain

emit = EmitStimMain()


def codegen(mt: ir.Method):
    # method should not have any arguments!
    emit.initialize()
    emit.run(mt=mt, args=())
    return emit.output


def test_detector():

    @stim.main
    def test_simple_cx():
        stim.Detector(coord=(1, 2, 3), targets=(stim.GetRecord(-3), stim.GetRecord(-1)))

    out = codegen(test_simple_cx)

    assert out == "DETECTOR(1, 2, 3) rec[-3] rec[-1]"


test_detector()
