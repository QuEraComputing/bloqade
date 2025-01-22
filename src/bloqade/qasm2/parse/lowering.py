from dataclasses import dataclass

from kirin import ir, lowering
from bloqade.types import CRegType, QRegType
from kirin.dialects import cf, func
from kirin.lowering import LoweringState
from kirin.exceptions import DialectLoweringError
from bloqade.qasm2.dialects import uop, core, expr

from . import ast
from .visitor import Visitor


@dataclass
class LoweringQASM(Visitor[lowering.Result]):
    state: LoweringState

    def visit_MainProgram(self, node: ast.MainProgram) -> lowering.Result:
        for stmt in node.statements:
            self.visit(stmt)
        return lowering.Result()

    def visit_QReg(self, node: ast.QReg) -> lowering.Result:
        reg = core.QRegNew(
            self.state.append_stmt(expr.ConstInt(value=node.size)).result
        )
        self.state.append_stmt(reg)
        self.state.current_frame.defs[node.name] = reg.result
        return lowering.Result()

    def visit_CReg(self, node: ast.CReg) -> lowering.Result:
        reg = core.CRegNew(
            self.state.append_stmt(expr.ConstInt(value=node.size)).result
        )
        self.state.append_stmt(reg)
        self.state.current_frame.defs[node.name] = reg.result
        return lowering.Result()

    def visit_Barrier(self, node: ast.Barrier) -> lowering.Result:
        self.state.append_stmt(
            uop.Barrier(
                qargs=tuple(self.visit(qarg).expect_one() for qarg in node.qargs)
            )
        )
        return lowering.Result()

    def visit_CXGate(self, node: ast.CXGate) -> lowering.Result:
        self.state.append_stmt(
            uop.CX(
                ctrl=self.visit(node.ctrl).expect_one(),
                qarg=self.visit(node.qarg).expect_one(),
            )
        )
        return lowering.Result()

    def visit_Measure(self, node: ast.Measure) -> lowering.Result:
        self.state.append_stmt(
            core.Measure(
                qarg=self.visit(node.qarg).expect_one(),
                carg=self.visit(node.carg).expect_one(),
            )
        )
        return lowering.Result()

    def visit_UGate(self, node: ast.UGate) -> lowering.Result:
        self.state.append_stmt(
            uop.UGate(
                theta=self.visit(node.theta).expect_one(),
                phi=self.visit(node.phi).expect_one(),
                lam=self.visit(node.lam).expect_one(),
                qarg=self.visit(node.qarg).expect_one(),
            )
        )
        return lowering.Result()

    def visit_Reset(self, node: ast.Reset) -> lowering.Result:
        self.state.append_stmt(core.Reset(qarg=self.visit(node.qarg).expect_one()))
        return lowering.Result()

    def visit_IfStmt(self, node: ast.IfStmt) -> lowering.Result:
        cond_stmt = core.CRegEq(
            lhs=self.visit(node.cond.lhs).expect_one(),
            rhs=self.visit(node.cond.rhs).expect_one(),
        )
        cond = self.state.append_stmt(cond_stmt).result
        frame = self.state.current_frame
        before_block = frame.current_block
        if frame.next_block is None:
            raise DialectLoweringError("code block is not exiting")
        else:
            before_block_next = frame.next_block
        if_block = self.state.current_frame.append_block()
        for stmt in node.body:
            self.visit(stmt)

        if_block.stmts.append(
            cf.Branch(
                arguments=(),
                successor=before_block_next,
            )
        )
        before_block.stmts.append(
            cf.ConditionalBranch(
                cond=cond,
                then_arguments=(),
                then_successor=if_block,
                else_arguments=(),
                else_successor=before_block_next,
            )
        )
        frame.current_block = before_block_next
        return lowering.Result()

    def visit_BinOp(self, node: ast.BinOp) -> lowering.Result:
        if node.op == "+":
            stmt_type = expr.Add
        elif node.op == "-":
            stmt_type = expr.Sub
        elif node.op == "*":
            stmt_type = expr.Mul
        else:
            stmt_type = expr.Div

        stmt = self.state.append_stmt(
            stmt_type(
                lhs=self.visit(node.lhs).expect_one(),
                rhs=self.visit(node.rhs).expect_one(),
            )
        )
        return lowering.Result(stmt.result)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> lowering.Result:
        if node.op == "-":
            stmt = expr.Neg(value=self.visit(node.operand).expect_one())
            return lowering.Result(stmt.result)
        else:
            return lowering.Result(self.visit(node.operand).expect_one())

    def visit_Bit(self, node: ast.Bit) -> lowering.Result:
        if node.name.id not in self.state.current_frame.defs:
            raise ValueError(f"Bit {node.name} not found")

        addr = self.state.append_stmt(expr.ConstInt(value=node.addr))
        reg = self.state.current_frame.get_local(node.name.id)
        if reg is None:
            raise DialectLoweringError(f"{node.name.id} is not defined")

        if reg.type.is_subseteq(QRegType):
            stmt = core.QRegGet(reg, addr.result)
        elif reg.type.is_subseteq(CRegType):
            stmt = core.CRegGet(reg, addr.result)
        return lowering.Result(self.state.append_stmt(stmt).result)

    def visit_Call(self, node: ast.Call) -> lowering.Result:
        if node.name == "cos":
            stmt = expr.Cos(self.visit(node.args[0]).expect_one())
        elif node.name == "sin":
            stmt = expr.Sin(self.visit(node.args[0]).expect_one())
        elif node.name == "tan":
            stmt = expr.Tan(self.visit(node.args[0]).expect_one())
        elif node.name == "exp":
            stmt = expr.Exp(self.visit(node.args[0]).expect_one())
        elif node.name == "log":
            stmt = expr.Log(self.visit(node.args[0]).expect_one())
        elif node.name == "sqrt":
            stmt = expr.Sqrt(self.visit(node.args[0]).expect_one())
        else:
            raise ValueError(f"Unknown function {node.name}")
        self.state.append_stmt(stmt)
        return lowering.Result(stmt.result)

    def visit_Instruction(self, node: ast.Instruction) -> lowering.Result:
        params = [self.visit(param).expect_one() for param in node.params]
        qargs = [self.visit(qarg).expect_one() for qarg in node.qargs]
        visit_inst = getattr(self, "visit_instruction_" + node.name.id, None)
        if visit_inst is not None:
            self.state.append_stmt(visit_inst(node, params, qargs))
        else:
            value = self.state.get_global(node.name.id).expect(ir.Method)
            # NOTE: QASM expects the return type to be known at call site
            if value.return_type is None:
                raise ValueError(f"Unknown return type for {node.name.id}")
            self.state.append_stmt(
                func.Invoke(
                    callee=value,
                    inputs=tuple(params + qargs),
                    kwargs=tuple(),
                )
            )
        return lowering.Result()

    def visit_instruction_x(self, node: ast.Instruction, params, qargs):
        return uop.X(qarg=qargs[0])

    def visit_instruction_y(self, node: ast.Instruction, params, qargs):
        return uop.Y(qarg=qargs[0])

    def visit_instruction_z(self, node: ast.Instruction, params, qargs):
        return uop.Z(qarg=qargs[0])

    def visit_instruction_h(self, node: ast.Instruction, params, qargs):
        return uop.H(qarg=qargs[0])

    def visit_instruction_s(self, node: ast.Instruction, params, qargs):
        return uop.S(qarg=qargs[0])

    def visit_instruction_t(self, node: ast.Instruction, params, qargs):
        return uop.T(qarg=qargs[0])

    def visit_instruction_rx(self, node: ast.Instruction, params, qargs):
        return uop.RX(theta=params[0], qarg=qargs[0])

    def visit_instruction_ry(self, node: ast.Instruction, params, qargs):
        return uop.RY(theta=params[0], qarg=qargs[0])

    def visit_instruction_rz(self, node: ast.Instruction, params, qargs):
        return uop.RZ(theta=params[0], qarg=qargs[0])

    def visit_instruction_u(self, node: ast.Instruction, params, qargs):
        return uop.UGate(theta=params[0], phi=params[1], lam=params[2], qarg=qargs[0])

    def visit_instruction_cx(self, node: ast.Instruction, params, qargs):
        return uop.CX(ctrl=qargs[0], qarg=qargs[1])

    def visit_instruction_ccx(self, node: ast.Instruction, params, qargs):
        return uop.CCX(ctrl1=qargs[0], ctrl2=qargs[1], qarg=qargs[2])

    def visit_Number(self, node: ast.Number) -> lowering.Result:
        if isinstance(node.value, int):
            stmt = expr.ConstInt(value=node.value)
        else:
            stmt = expr.ConstFloat(value=node.value)
        return lowering.Result(self.state.append_stmt(stmt).result)

    def visit_Pi(self, node: ast.Pi) -> lowering.Result:
        return lowering.Result(self.state.append_stmt(expr.ConstPI()).result)

    def visit_Include(self, node: ast.Include) -> lowering.Result:
        raise NotImplementedError("Include lowering not supported")

    def visit_Gate(self, node: ast.Gate) -> lowering.Result:
        raise NotImplementedError("Gate lowering not supported")

    def visit_Name(self, node: ast.Name) -> lowering.Result:
        if (value := self.state.current_frame.get_local(node.id)) is not None:
            return lowering.Result(value)
        raise ValueError(f"name {node.id} not found")
