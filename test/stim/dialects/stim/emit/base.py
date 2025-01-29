from kirin import ir
from bloqade.stim.emit import EmitStimMain

emit = EmitStimMain()


def codegen(mt: ir.Method):
    # method should not have any arguments!
    emit.initialize()
    emit.run(mt=mt, args=()).expect()
    return emit.get_output()
