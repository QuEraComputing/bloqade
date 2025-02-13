from dataclasses import dataclass

from kirin import types


@dataclass
class RecordResult:
    value: int


@dataclass
class PauliString:
    string: tuple[str, ...]
    flipped: tuple[bool, ...]
    targets: tuple[int, ...]


RecordType = types.PyClass(RecordResult)
PauliStringType = types.PyClass(PauliString)
