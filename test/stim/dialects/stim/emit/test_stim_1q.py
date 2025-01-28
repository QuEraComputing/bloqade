from kirin import ir
from bloqade import stim
from bloqade.stim.emit import EmitStimMain

emit = EmitStimMain()


def codegen(mt: ir.Method):
    # method should not have any arguments!
    emit.initialize()
    emit.run(mt=mt, args=()).expect()
    return emit.output


def test_x():

    @stim.main
    def test_x():
        stim.X(targets=(0, 1, 2, 3), dagger=False)

    test_x.print()
    out = codegen(test_x)

    assert out == "X 0 1 2 3"

    @stim.main
    def test_x_dag():
        stim.X(targets=(0, 1, 2, 3), dagger=True)

    out = codegen(test_x_dag)

    assert out == "X 0 1 2 3"


test_x()
