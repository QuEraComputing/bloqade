import io
import difflib

from kirin import ir, print as pprint
from rich.console import Console


def print_diff(node: pprint.Printable, expected_node: pprint.Printable):
    gn_con = Console(record=True, file=io.StringIO())
    node.print(console=gn_con)

    expected_con = Console(record=True, file=io.StringIO())
    expected_node.print(console=expected_con)

    expected = expected_con.export_text()

    gn = gn_con.export_text()
    diff = difflib.Differ().compare(
        expected.splitlines(),
        gn.splitlines(),
    )

    print("\n".join(diff))


def assert_nodes(node: ir.IRNode, expected_node: ir.IRNode):
    try:
        assert node.is_equal(expected_node)
    except AssertionError as e:
        print_diff(node, expected_node)
        raise e


def assert_methods(mt: ir.Method, expected_mt: ir.Method):
    assert_nodes(mt.code, expected_mt.code)
