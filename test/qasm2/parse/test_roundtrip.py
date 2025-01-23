import io
import os
import pathlib

from bloqade.qasm2.parse import loads, pprint, loadfile


def roundtrip(file):
    ast1 = loadfile(os.path.join(os.path.dirname(__file__), "programs", file))
    buf = io.StringIO()
    pprint(ast1, buf)
    ast2 = loads(buf.getvalue())
    return ast1 == ast2


def test_roundtrip():
    path = pathlib.Path(__file__).parent / "programs"
    for file in path.glob("*.qasm"):
        assert roundtrip(file.name), f"Failed roundtrip for {file}"
