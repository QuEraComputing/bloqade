from kirin import ir
from kirin.dialects import py
from kirin.rewrite.abc import RewriteRule, RewriteResult
from bloqade.qasm2.dialects import core


class RaiseRegisterRule(RewriteRule):
    """This rule puts all registers at the top of the block.

    This is required for the UOpToParallel rules to work correctly
    to handle cases where a register is defined in between two statements
    that can be parallelized.

    """

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, core.QRegNew):
            return RewriteResult()

        if node.parent_block is None or node.parent_block.first_stmt is None:
            return RewriteResult()

        first_stmt = node.parent_block.first_stmt

        n_qubits_ref = node.n_qubits

        n_qubits = n_qubits_ref.owner
        if isinstance(n_qubits, py.Constant):
            # case where the n_qubits comes from a constant
            new_n_qubits = n_qubits.from_stmt(n_qubits)
            new_n_qubits.insert_before(first_stmt)
            new_n_qubits_ref = new_n_qubits.result

        elif isinstance(n_qubits, ir.BlockArgument):
            # case where the n_qubits comes from a block argument
            new_n_qubits_ref = n_qubits
        else:
            return RewriteResult()

        new_qreg_stmt = core.QRegNew(n_qubits=new_n_qubits_ref)
        new_qreg_stmt.insert_before(first_stmt)
        node.result.replace_by(new_qreg_stmt.result)
        node.delete()
        return RewriteResult(has_done_something=True)
