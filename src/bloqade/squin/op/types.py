from kirin import types


class Op:

    def __matmul__(self, other: "Op") -> "Op":
        raise NotImplementedError("@ can only be used within a squin kernel program")


OpType = types.PyClass(Op)
