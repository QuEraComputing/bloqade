"""
Passes that deal with global gates. As of now, only one rewrite pass exists
which converts global gates to single qubit gates.
"""

from kirin import ir
from kirin.rewrite import cse, dce, walk, result
from bloqade.analysis import address
from kirin.passes.abc import Pass
from bloqade.qasm2.rewrite import GlobalToUOpRule, GlobalToParallelRule


class GlobalToUOP(Pass):
    """Pass to convert Global gates into single gates.

    This pass rewrites the global unitary gate from the `qasm2.glob` dialect into multiple
    single gates in the `qasm2.uop` dialect, bringing the program closer to
    conforming to standard QASM2 syntax.


    ## Usage Examples
    ```
    # Define kernel
    @qasm2.extended
    def main():
        q1 = qasm2.qreg(1)
        q2 = qasm2.qreg(2)

        theta = 1.3
        phi = 1.1
        lam = 1.2

        qasm2.glob.u(theta=theta, phi=phi, lam=lam, registers=[q1, q2])

    GlobalToUOP(dialects=main.dialects)(main)

    # Run rewrite
    GlobalToUOP(main.dialects)(main)
    ```

    The `qasm2.glob.u` statement has been rewritten to individual gates:

    ```
    qasm2.uop.u(q1[0], theta, phi, lam)
    qasm2.uop.u(q2[0], theta, phi, lam)
    qasm2.uop.u(q2[1], theta, phi, lam)
    ```
    """

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


class GlobalToParallel(Pass):
    """Pass to convert Global gates into parallel gates.

    This pass rewrites the global unitary gate from the `qasm2.glob` dialect into multiple
    parallel gates in the `qasm2.parallel` dialect.


    ## Usage Examples
    ```
    # Define kernel
    @qasm2.extended
    def main():
        q1 = qasm2.qreg(1)
        q2 = qasm2.qreg(2)

        theta = 1.3
        phi = 1.1
        lam = 1.2

        qasm2.glob.u(theta=theta, phi=phi, lam=lam, registers=[q1, q2])

    GlobalToParallel(dialects=main.dialects)(main)

    # Run rewrite
    GlobalToParallel(main.dialects)(main)
    ```

    The `qasm2.glob.u` statement has been rewritten to individual gates:

    ```
    qasm2.parallel.u(theta=theta, phi=phi, lam=lam, qargs=[q1[0], q2[0], q2[1]])
    ```
    """

    def generate_rule(self, mt: ir.Method) -> GlobalToParallelRule:
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

        return GlobalToParallelRule(
            address_regs=in_global_addr_regs, address_reg_ssas=in_global_addr_reg_ssas
        )

    def unsafe_run(self, mt: ir.Method) -> result.RewriteResult:
        rewriter = walk.Walk(self.generate_rule(mt))
        result = rewriter.rewrite(mt.code)

        result = walk.Walk(dce.DeadCodeElimination()).rewrite(mt.code)
        result = walk.Walk(cse.CommonSubexpressionElimination()).rewrite(mt.code)

        return result
