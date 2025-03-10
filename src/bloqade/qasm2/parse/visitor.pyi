from typing import Generic, TypeVar

from . import ast

T = TypeVar("T")

class Visitor(Generic[T]):
    def visit(self, node: ast.Node) -> T: ...
    def generic_visit(self, node: ast.Node) -> T: ...
    def visit_MainProgram(self, node: ast.MainProgram) -> T: ...
    def visit_OPENQASM(self, node: ast.OPENQASM) -> T: ...
    def visit_Kirin(self, node: ast.Kirin) -> T: ...
    def visit_QReg(self, node: ast.QReg) -> T: ...
    def visit_CReg(self, node: ast.CReg) -> T: ...
    def visit_Gate(self, node: ast.Gate) -> T: ...
    def visit_Opaque(self, node: ast.Opaque) -> T: ...
    def visit_IfStmt(self, node: ast.IfStmt) -> T: ...
    def visit_Cmp(self, node: ast.Cmp) -> T: ...
    def visit_Barrier(self, node: ast.Barrier) -> T: ...
    def visit_Include(self, node: ast.Include) -> T: ...
    def visit_Measure(self, node: ast.Measure) -> T: ...
    def visit_Reset(self, node: ast.Reset) -> T: ...
    def visit_Instruction(self, node: ast.Instruction) -> T: ...
    def visit_UGate(self, node: ast.UGate) -> T: ...
    def visit_CXGate(self, node: ast.CXGate) -> T: ...
    def visit_Bit(self, node: ast.Bit) -> T: ...
    def visit_BinOp(self, node: ast.BinOp) -> T: ...
    def visit_UnaryOp(self, node: ast.UnaryOp) -> T: ...
    def visit_Call(self, node: ast.Call) -> T: ...
    def visit_Number(self, node: ast.Number) -> T: ...
    def visit_Pi(self, node: ast.Pi) -> T: ...
    def visit_Name(self, node: ast.Name) -> T: ...
    # extensions
    def visit_ParaU3Gate(self, node: ast.ParaU3Gate) -> T: ...
    def visit_ParaCZGate(self, node: ast.ParaCZGate) -> T: ...
    def visit_ParaRZGate(self, node: ast.ParaRZGate) -> T: ...
    def visit_ParallelQArgs(self, node: ast.ParallelQArgs) -> T: ...
    def visit_GlobUGate(self, node: ast.GlobUGate) -> T: ...
    def visit_NoisePAULI1(self, node: ast.NoisePAULI1) -> T: ...
