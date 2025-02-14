from kirin import ir
from kirin.rewrite import cse, dce, walk, result
from bloqade.analysis import address
from kirin.passes.abc import Pass
from bloqade.qasm2.rewrite import GlobalToUOpRule


class GlobalToUOP(Pass):

    def generate_rule(self, mt: ir.Method) -> GlobalToUOpRule:
        results, _ = address.AddressAnalysis(mt.dialects).run_analysis(mt)

        # You can't hash the address register because it contains a Sequence type,
        # need to juggle things around in lists
        encountered_addr_regs = []
        encountered_addr_reg_ssas = []

        in_global_addr_regs = []
        in_global_addr_reg_ssas = []

        for ssa, addr in results.entries.items():

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
