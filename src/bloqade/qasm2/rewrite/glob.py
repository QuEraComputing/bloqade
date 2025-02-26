from typing import Dict, List
from dataclasses import dataclass

from kirin import ir
from bloqade import qasm2
from kirin.rewrite import abc, result
from kirin.dialects import py, ilist
from bloqade.analysis import address
from bloqade.qasm2.dialects import glob


@dataclass
class GlobalRewriteBase:
    address_analysis: Dict[ir.SSAValue, address.Address]

    def get_qubit_ssa(self, node: glob.UGate):
        new_stmts: List[ir.Statement] = []
        qubit_ssa: List[ir.SSAValue] = []
        # can't rewrite if the registers are coming from a block argument
        if not isinstance(node.registers, ir.ResultValue):
            return new_stmts, None

        if not isinstance(node.registers.owner, ilist.New):
            return new_stmts, None

        register_ssa_values = node.registers.owner.values

        for register_ssa in register_ssa_values:
            addr = self.address_analysis.get(register_ssa, address.Address.top())
            if not isinstance(addr, address.AddressReg):
                new_stmts.clear()
                return new_stmts, None

            for qubit in range(len(addr.data)):
                new_stmts.append(idx_stmt := py.constant.Constant(value=qubit))
                new_stmts.append(
                    qubit_stmt := qasm2.core.QRegGet(
                        reg=register_ssa, idx=idx_stmt.result
                    )
                )
                qubit_ssa.append(qubit_stmt.result)

        return new_stmts, qubit_ssa


@dataclass
class GlobalToParallelRule(abc.RewriteRule, GlobalRewriteBase):

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if type(node) in glob.dialect.stmts:
            return getattr(self, f"rewrite_{node.name}")(node)

        return result.RewriteResult()

    def rewrite_ugate(self, node: glob.UGate):

        new_stmts, qubit_ssa = self.get_qubit_ssa(node)

        if qubit_ssa is None:
            return result.RewriteResult()

        new_stmts.append(qargs := ilist.New(values=qubit_ssa))
        new_stmts.append(
            qasm2.dialects.parallel.UGate(
                qargs=qargs.result, theta=node.theta, phi=node.phi, lam=node.lam
            )
        )

        for stmt in new_stmts:
            stmt.insert_before(node)

        node.delete()

        return result.RewriteResult(has_done_something=True)


@dataclass
class GlobalToUOpRule(abc.RewriteRule, GlobalRewriteBase):

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if type(node) in glob.dialect.stmts:
            return getattr(self, f"rewrite_{node.name}")(node)

        return result.RewriteResult()

    def rewrite_ugate(self, node: glob.UGate):

        new_stmts, qubit_ssa = self.get_qubit_ssa(node)

        if qubit_ssa is None:
            return result.RewriteResult()

        for qarg in qubit_ssa:
            new_stmts.append(
                qasm2.uop.UGate(qarg=qarg, theta=node.theta, phi=node.phi, lam=node.lam)
            )

        for stmt in new_stmts:
            stmt.insert_before(node)

        node.delete()
        return result.RewriteResult(has_done_something=True)
