from bloqade import stim

from .base import codegen


def test_obs_inc():

    @stim.main
    def test_simple_obs_inc():
        stim.ObservableInclude(idx=3, targets=(stim.GetRecord(-3), stim.GetRecord(-1)))

    out = codegen(test_simple_obs_inc)

    assert out.strip() == "OBSERVABLE_INCLUDE(3) rec[-3] rec[-1]"
