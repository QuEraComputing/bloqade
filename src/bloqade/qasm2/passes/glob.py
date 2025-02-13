from typing import List
from dataclasses import dataclass

from kirin import ir
from bloqade import qasm2
from kirin.rewrite import abc, cse, dce, walk, result
from bloqade.analysis import address
from kirin.passes.abc import Pass
from bloqade.qasm2.dialects import glob


@dataclass
class GlobalToUOpRule(abc.RewriteRule):
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

        for address_reg, address_reg_ssa in zip(
            self.address_regs, self.address_reg_ssas
        ):

            for qubit_idx in address_reg.data:

                qubit_idx = qasm2.expr.ConstInt(value=qubit_idx)

                qubit_stmt = qasm2.core.QRegGet(
                    reg=address_reg_ssa, idx=qubit_idx.result
                )
                qubit_ssa = qubit_stmt.result

                ugate_node = qasm2.uop.UGate(
                    qarg=qubit_ssa, theta=node.theta, phi=node.phi, lam=node.lam
                )

                qubit_idx.insert_before(node)
                qubit_stmt.insert_before(node)
                ugate_node.insert_after(node)

        node.delete()

        return result.RewriteResult(has_done_something=True)


class GlobalToUOP(Pass):

    def generate_rule(self, mt: ir.Method) -> GlobalToUOpRule:
        results, _ = address.AddressAnalysis(mt.dialects).run_analysis(mt)

        # You can't hash the address register because it contains a Sequence type,
        # need to juggle things around in lists
        encountered_addr_regs = []
        encountered_addr_reg_ssas = []

        in_global_addr_regs = []
        in_global_addr_reg_ssas = []

        for ssa, addr in results.items():

            # Find all the times an AddressReg is encountered
            if isinstance(addr, address.AddressReg):
                encountered_addr_regs.append(addr)
                encountered_addr_reg_ssas.append(ssa)

            # When we encounter an AddressTuple,
            # check which registers are referenced by it and
            # after verifying, use the SSA values and the registers encountered
            # previously to construct to the GlobalToUOpRule
            if isinstance(addr, address.AddressTuple):
                for encountered_addr_reg, encountered_addr_reg_ssa in zip(
                    encountered_addr_regs, encountered_addr_reg_ssas
                ):
                    if encountered_addr_reg in addr.data:
                        in_global_addr_regs.append(encountered_addr_reg)
                        in_global_addr_reg_ssas.append(encountered_addr_reg_ssa)

        return GlobalToUOpRule(
            address_regs=in_global_addr_regs, address_reg_ssas=in_global_addr_reg_ssas
        )

    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:
        rewriter = walk.Walk(self.generate_rule(mt))
        result = rewriter.rewrite(mt.code)

        result = walk.Walk(dce.DeadCodeElimination()).rewrite(mt.code)
        result = walk.Walk(cse.CommonSubexpressionElimination()).rewrite(mt.code)

        return result
