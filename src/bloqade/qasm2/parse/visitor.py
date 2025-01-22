from typing import Generic, TypeVar

from . import ast

T = TypeVar("T")


class Visitor(Generic[T]):

    def visit(self, node: ast.Node) -> T:
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.Node) -> T:
        raise NotImplementedError(f"No visit_{node.__class__.__name__} method")
