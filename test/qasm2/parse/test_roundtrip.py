import os
import pathlib

from bloqade.qasm2.parse import loads, spprint, loadfile


def roundtrip(file):
    ast1 = loadfile(os.path.join(os.path.dirname(__file__), "programs", file))
    ast2 = loads(spprint(ast1))
    return ast1 == ast2


def test_roundtrip():
    path = pathlib.Path(__file__).parent / "programs"
    for file in path.glob("*.qasm"):
        assert roundtrip(file.name), f"Failed roundtrip for {file}"
