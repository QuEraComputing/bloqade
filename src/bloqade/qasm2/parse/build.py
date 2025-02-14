# type: ignore
from lark import Token
from lark.tree import ParseTree

from . import ast


class BuildError(Exception):
    pass


class Build:

    def build(self, tree: ParseTree) -> ast.Node:
        return getattr(self, f"build_{tree.data}", self.build_generic)(tree)

    def build_generic(self, tree: ParseTree) -> ast.Node:
        raise NotImplementedError(f"No build_{tree.data} method")

    def build_mainprogram(self, tree: ParseTree) -> ast.MainProgram:
        header = self.build(tree.children[0])
        return ast.MainProgram(
            header=header,
            statements=[self.build(stmt) for stmt in tree.children[1:]],
        )

    def build_openqasm(self, tree: ParseTree) -> ast.OPENQASM:
        return ast.OPENQASM(version=self.build(tree.children[0]))

    def build_kirin(self, tree: ParseTree) -> ast.Kirin:
        return ast.Kirin(dialects=[self.build_dialect(each) for each in tree.children])

    def build_dialect(self, tree: ParseTree) -> str:
        return ".".join(each.value for each in tree.children)

    def build_version(self, tree: ParseTree) -> ast.Version:
        return ast.Version(
            int(tree.children[0].value),
            int(tree.children[1].value),
        )

    def build_include(self, tree: ParseTree) -> ast.Include:
        return ast.Include(filename=tree.children[0].value[1:-1])

    def build_qreg(self, tree: ParseTree) -> ast.QReg:
        return ast.QReg(name=tree.children[0].value, size=int(tree.children[1].value))

    def build_creg(self, tree: ParseTree) -> ast.CReg:
        return ast.CReg(name=tree.children[0].value, size=int(tree.children[1].value))

    def build_cx_gate(self, tree: ParseTree) -> ast.Gate:
        return ast.CXGate(
            self.build_bit(tree.children[0]), self.build_bit(tree.children[1])
        )

    def build_ugate(self, tree: ParseTree) -> ast.Gate:
        return ast.UGate(
            theta=self.build_expr(tree.children[0]),
            phi=self.build_expr(tree.children[1]),
            lam=self.build_expr(tree.children[2]),
            qarg=self.build_bit(tree.children[3]),
        )

    def build_barrier(self, tree: ParseTree) -> ast.Barrier:
        return ast.Barrier(qargs=self.build(tree.children[0]))

    def build_inst(self, tree: ParseTree) -> ast.Instruction:
        if tree.children[1] is None:
            params = []
        else:
            params = self.build(tree.children[1])
        return ast.Instruction(
            name=ast.Name(tree.children[0].value),
            params=params,
            qargs=self.build(tree.children[2]),
        )

    def build_opaque(self, tree: ParseTree) -> ast.Opaque:
        return ast.Opaque(
            name=tree.children[0].value,
            cparams=self.build(tree.children[1]) if tree.children[1] else [],
            qparams=self.build(tree.children[2]) if tree.children[2] else [],
        )

    def build_gate(self, tree: ParseTree) -> ast.Gate:
        cparams = tree.children[1]
        qparams = tree.children[2]
        return ast.Gate(
            name=tree.children[0].value,
            cparams=self.build(cparams) if cparams else [],
            qparams=self.build(qparams) if qparams else [],
            body=[self.build(stmt) for stmt in tree.children[3:]],
        )

    def build_noise_pauli1(self, tree: ParseTree) -> ast.NoisePAULI1:
        return ast.NoisePAULI1(
            px=self.build_expr(tree.children[0]),
            py=self.build_expr(tree.children[1]),
            pz=self.build_expr(tree.children[2]),
            qarg=self.build_bit(tree.children[3]),
        )

    def build_glob_u_gate(self, tree: ParseTree) -> ast.GlobUGate:
        return ast.GlobUGate(
            [self.build_bit(each) for each in tree.children[3].children],
            self.build_expr(tree.children[0]),
            self.build_expr(tree.children[1]),
            self.build_expr(tree.children[2]),
        )

    def build_para_u_gate(self, tree: ParseTree) -> ast.ParaU3Gate:
        return ast.ParaU3Gate(
            self.build_expr(tree.children[0]),
            self.build_expr(tree.children[1]),
            self.build_expr(tree.children[2]),
            self.build_parallel_body(tree.children[3]),
        )

    def build_para_cz_gate(self, tree: ParseTree) -> ast.ParaCZGate:
        return ast.ParaCZGate(
            self.build_parallel_body(tree.children[0]),
        )

    def build_para_rz_gate(self, tree: ParseTree) -> ast.ParaRZGate:
        return ast.ParaRZGate(
            self.build_expr(tree.children[0]),
            self.build_parallel_body(tree.children[1]),
        )

    def build_parallel_body(self, tree: ParseTree) -> ast.ParallelQArgs:
        return ast.ParallelQArgs([self.build_task_args(each) for each in tree.children])

    def build_task_args(self, tree: ParseTree) -> tuple[ast.Bit | ast.Name, ...]:
        return tuple(self.build(each) for each in tree.children)

    def build_measure(self, tree: ParseTree) -> ast.Measure:
        return ast.Measure(
            qarg=self.build_bit(tree.children[0]),
            carg=self.build_bit(tree.children[1]),
        )

    def build_reset(self, tree: ParseTree) -> ast.Reset:
        return ast.Reset(qarg=self.build_bit(tree.children[0]))

    def build_ifstmt(self, tree: ParseTree) -> ast.IfStmt:
        return ast.IfStmt(
            cond=ast.Cmp(
                lhs=self.build_expr(tree.children[0]),
                rhs=self.build_expr(tree.children[1]),
            ),
            body=self.build(tree.children[2]),
        )

    def build_ifbody(self, tree: ParseTree) -> list[ast.Node]:
        return [self.build(stmt) for stmt in tree.children]

    def build_qparams(self, tree: ParseTree) -> list[ast.Expr]:
        return [param.value for param in tree.children]

    def build_cparams(self, tree: ParseTree) -> list[ast.Expr]:
        return [param.value for param in tree.children]

    def build_qubits(self, tree: ParseTree) -> list[ast.Bit]:
        return [self.build_bit(qubit) for qubit in tree.children]

    def build_bit(self, tree: ParseTree) -> ast.Name | ast.Bit:
        if isinstance(tree, Token) and tree.type == "IDENTIFIER":
            return ast.Name(tree.value)
        else:
            return ast.Bit(
                name=ast.Name(tree.children[0].value), addr=int(tree.children[1].value)
            )

    def build_params(self, tree: ParseTree) -> list[ast.Expr]:
        return [self.build_expr(param) for param in tree.children]

    def build_call(self, tree: ParseTree) -> ast.Call:
        return ast.Call(
            name=tree.children[0].value,
            args=self.build(tree.children[1]),
        )

    def build_arglist(self, tree: ParseTree) -> list[ast.Expr]:
        return [self.build_expr(arg) for arg in tree.children]

    def build_expr(self, tree: ParseTree) -> ast.Expr:
        if isinstance(tree, Token):
            return self._build_token_expr(tree)
        elif tree.data != "expr":
            return self.build(tree)

        if tree.children[1].type == "PLUS":
            lhs = ast.BinOp(
                op="+",
                lhs=self.build_term(tree.children[0]),
                rhs=self.build_term(tree.children[2]),
            )
        elif tree.children[1].type == "MINUS":
            lhs = ast.BinOp(
                op="-",
                lhs=self.build_term(tree.children[0]),
                rhs=self.build_term(tree.children[2]),
            )
        else:
            raise BuildError(f"Unsupported operator {tree.children[1].type}")

        for i in range(3, len(tree.children), 2):
            if tree.children[i].type == "PLUS":
                lhs = ast.BinOp(
                    op="+",
                    lhs=lhs,
                    rhs=self.build_term(tree.children[i + 1]),
                )
            elif tree.children[i].type == "MINUS":
                lhs = ast.BinOp(
                    op="-",
                    lhs=lhs,
                    rhs=self.build_term(tree.children[i + 1]),
                )
            else:
                raise BuildError(f"Unsupported operator {tree.children[i].type}")

        return lhs

    def build_term(self, tree: ParseTree) -> ast.Expr:
        if isinstance(tree, Token):
            return self._build_token_expr(tree)
        elif tree.data != "term":
            return self.build(tree)

        if tree.children[1].type == "TIMES":
            lhs = ast.BinOp(
                op="*",
                lhs=self.build_factor(tree.children[0]),
                rhs=self.build_factor(tree.children[2]),
            )
        elif tree.children[1].type == "DIVIDE":
            lhs = ast.BinOp(
                op="/",
                lhs=self.build_factor(tree.children[0]),
                rhs=self.build_factor(tree.children[2]),
            )
        else:
            raise BuildError(f"Unsupported operator {tree.children[1].type}")

        for i in range(3, len(tree.children), 2):
            if tree.children[i].type == "TIMES":
                lhs = ast.BinOp(
                    op="*",
                    lhs=lhs,
                    rhs=self.build_factor(tree.children[i + 1]),
                )
            elif tree.children[i].type == "DIVIDE":
                lhs = ast.BinOp(
                    op="/",
                    lhs=lhs,
                    rhs=self.build_factor(tree.children[i + 1]),
                )
            else:
                raise BuildError(f"Unsupported operator {tree.children[i].type}")

        return lhs

    def build_factor(self, tree: ParseTree) -> ast.Expr:
        if isinstance(tree, Token):
            return self._build_token_expr(tree)
        elif tree.data != "factor":
            return self.build(tree)

        if tree.children[0].type == "PLUS":
            return ast.UnaryOp(op="+", operand=self.build_expr(tree.children[1]))
        elif tree.children[0].type == "MINUS":
            return ast.UnaryOp(op="-", operand=self.build_expr(tree.children[1]))
        else:
            raise BuildError(f"Unsupported operator {tree.children[0].type}")

    def _build_token_expr(self, token: Token) -> ast.Node:
        if token.type == "NUMBER":
            return ast.Number(float(token.value))
        elif token.type == "IDENTIFIER":
            return ast.Name(token.value)
        elif token.type == "PI":
            return ast.Pi()
        elif token.type == "INT":
            return ast.Number(int(token.value))
        elif token.type == "FLOAT":
            return ast.Number(float(token.value))
        else:
            raise BuildError(f"Unsupported token type {token.type}")
