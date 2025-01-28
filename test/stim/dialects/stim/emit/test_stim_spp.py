from kirin import ir
from bloqade import stim
from bloqade.stim.emit import EmitStimMain

emit = EmitStimMain()


def codegen(mt: ir.Method):
    # method should not have any arguments!
    emit.initialize()
    emit.run(mt=mt, args=()).expect()
    return emit.output


def test_spp():

    @stim.main
    def test_spp_main():
        stim.SPP(
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
            dagger=False,
        )

    test_spp_main.print()
    out = codegen(test_spp_main)
    print(out)
    assert out == "SPP !X0*X1*Z2 Y3*X4*!Y5"


test_spp()
