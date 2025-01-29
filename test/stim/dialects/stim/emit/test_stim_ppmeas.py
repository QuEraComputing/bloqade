from bloqade import stim

from .base import codegen


def test_mpp():

    @stim.main
    def test_mpp_main():
        stim.PPMeasurement(
            targets=(
                stim.NewPauliString(
                    string=("X", "X", "Z"),
                    flipped=(True, False, False),
                    targets=(0, 1, 2),
                ),
                stim.NewPauliString(
                    string=("Y", "X", "Y"),
                    flipped=(False, False, True),
                    targets=(3, 4, 5),
                ),
            ),
            p=0.3,
        )

    test_mpp_main.print()
    out = codegen(test_mpp_main)

    assert out.strip() == "MPP(0.30000000) !X0*X1*Z2 Y3*X4*!Y5"


test_mpp()
