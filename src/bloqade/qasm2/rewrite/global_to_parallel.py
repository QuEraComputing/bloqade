from typing import List
from dataclasses import dataclass

from kirin import ir
from bloqade import qasm2
from kirin.rewrite import abc, result
from kirin.dialects import py, ilist
from bloqade.analysis import address
from bloqade.qasm2.dialects import glob


@dataclass
class GlobalToParallelRule(abc.RewriteRule):
    address_regs: List[address.AddressReg]
    address_reg_ssas: List[ir.SSAValue]

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if node.dialect == glob.dialect:
            return getattr(self, f"rewrite_{node.name}")(node)

        return result.RewriteResult()

    def rewrite_ugate(self, node: ir.Statement):
        assert isinstance(node, glob.UGate)

        # if there's no register even found, just give up
        if not self.address_regs:
            return result.RewriteResult()

        ilist_stmt = node.registers.owner
        assert isinstance(ilist_stmt, ilist.New)

        active_reg_idx = [self.address_reg_ssas.index(reg) for reg in ilist_stmt.values]

        list_ssa = []
        for idx in active_reg_idx:
            address_reg = self.address_regs[idx]
            address_reg_ssa = self.address_reg_ssas[idx]

            for qubit_idx in address_reg.data:

                qubit_idx = py.constant.Constant(value=qubit_idx)

                qubit_stmt = qasm2.core.QRegGet(
                    reg=address_reg_ssa, idx=qubit_idx.result
                )
                qubit_ssa = qubit_stmt.result
                list_ssa.append(qubit_ssa)

                qubit_idx.insert_before(node)
                qubit_stmt.insert_before(node)

        list_stmt = ilist.New(values=list_ssa)
        list_stmt.insert_before(node)

        new_gate_stmt = qasm2.dialects.parallel.UGate(
            theta=node.theta, phi=node.phi, lam=node.lam, qargs=list_stmt.result
        )

        new_gate_stmt.insert_before(node)

        node.delete()

        return result.RewriteResult(has_done_something=True)
