import io
import os

from bloqade.qasm2.parse import loads, pprint, loadfile


def roundtrip(file):
    ast1 = loadfile(os.path.join(os.path.dirname(__file__), file))
    buf = io.StringIO()
    pprint(ast1, buf)
    ast2 = loads(buf.getvalue())
    assert ast1 == ast2


def test_roundtrip():
    roundtrip("main.qasm")
    roundtrip("para.qasm")
    roundtrip("qelib1.inc")
