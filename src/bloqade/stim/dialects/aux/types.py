from dataclasses import dataclass

from kirin.ir import types


@dataclass
class RecordResult:
    value: int


RecordType = types.PyClass(RecordResult)
