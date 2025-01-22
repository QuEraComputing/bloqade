from lark import Lark

qasm2_parser = Lark.open(
    "qasm2.lark", rel_to=__file__, parser="lalr", start="mainprogram"
)
