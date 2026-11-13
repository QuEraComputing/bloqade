import bloqade.squin as squin
import bloqade.types as types
from kirin import types as kirin_types
from kirin.dialects import ilist
from kirin.analysis.typeinfer import TypeInference

import bloqade.gemini as gemini


def test_type_inference():

    @gemini.logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(3)

        for i in range(3):
            squin.x(q[i])

        return gemini.logical.terminal_measure(q)

    _, ret_type = TypeInference(main.dialects).run(main)

    assert ret_type.is_structurally_equal(
        ilist.IListType[
            # Steane [[7,1,3]]: the width `@logical.kernel` always stamps.
            ilist.IListType[types.MeasurementResultType, kirin_types.Literal(7)],
            kirin_types.Literal(3),
        ]
    )
