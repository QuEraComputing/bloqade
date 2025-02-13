import os
import pathlib

from bloqade.qasm2.parse import loads, spprint, loadfile
from bloqade.qasm2.parse.parser import qasm2_parser as lark_parser


def roundtrip(file):
    ast1 = loadfile(os.path.join(os.path.dirname(__file__), "programs", file))
    ast2 = loads(spprint(ast1))
    return ast1 == ast2


def test_roundtrip():
    path = pathlib.Path(__file__).parent / "programs"
    for file in path.glob("*.qasm"):
        assert roundtrip(file.name), f"Failed roundtrip for {file}"


if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "programs", "global.qasm")
    print(filepath)
    with open(filepath) as f:
        qasm_str = f.read()

    parse_tree = lark_parser.parse(qasm_str)  # raw parsing seems to be fine
    print(parse_tree.pretty())

    ast = loads(qasm_str)
    print(spprint(ast))
