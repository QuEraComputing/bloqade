from kirin import ir

from bloqade import stim
from bloqade.stim.emit.stim import EmitStimMain

emit = EmitStimMain()


def codegen(mt: ir.Method):
    # method should not have any arguments!
    emit.initialize()
    emit.run(mt=mt, args=())
    return emit.output




def test_obs_inc():

    @stim.main
    def test_simple_obs_inc():
        stim.ObservableInclude(idx=3, targets=(stim.GetRecord(-3), stim.GetRecord(-1)))

    out = codegen(test_simple_obs_inc)

    assert out == "OBSERVABLE_INCLUDE(3) rec[-3] rec[-1]"
