from kirin import ir
from bloqade import stim
from bloqade.stim.emit.stim import EmitStimMain

emit = EmitStimMain()


def codegen(mt: ir.Method):
    # method should not have any arguments!
    emit.initialize()
    emit.run(mt=mt, args=())
    return emit.output


def test_meas():

    @stim.main
    def test_simple_meas():
        stim.MX(p=0.3, targets=(0, 3, 4, 5))

    out = codegen(test_simple_meas)

    assert out == "MX(0.30000000) 0 3 4 5"
