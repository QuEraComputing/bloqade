import os

from bloqade.qasm2.parse import loads, pprint


def test_roundtrip():
    with open(os.path.join(os.path.dirname(__file__), "main.qasm")) as f:
        ast = loads(f.read())

    pprint(ast)

    with open(os.path.join(os.path.dirname(__file__), "para.qasm")) as f:
        ast = loads(f.read())

    pprint(ast)
    assert True
