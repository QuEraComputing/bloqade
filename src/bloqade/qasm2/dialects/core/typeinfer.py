from kirin import types, interp
from kirin.analysis import TypeInference
from kirin.dialects import py
from bloqade.qasm2.types import CRegType, QRegType, QubitType

from ._dialect import dialect


@dialect.register(key="typeinfer")
class TypeInfer(interp.MethodTable):

    @interp.impl(py.indexing.GetItem, QRegType, types.Int)
    def getitem_qreg(
        self, infer: TypeInference, frame: interp.Frame, node: py.indexing.GetItem
    ):
        return (QubitType,)

    @interp.impl(py.indexing.GetItem, CRegType, types.Int)
    def getitem_creg(
        self, infer: TypeInference, frame: interp.Frame, node: py.indexing.GetItem
    ):
        return (QubitType,)
