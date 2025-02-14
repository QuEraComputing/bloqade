from typing import Dict, List, Optional
from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import abc, result
from bloqade.analysis import address
from bloqade.qasm2.dialects import uop, parallel


@dataclass
class ParallelToUOpRule(abc.RewriteRule):
    id_map: Dict[int, ir.SSAValue]
    address_analysis: Dict[ir.SSAValue, address.Address]

    def rewrite_Statement(self, node: ir.Statement) -> result.RewriteResult:
        if node.dialect == parallel.dialect:
            return getattr(self, f"rewrite_{node.name}")(node)

        return result.RewriteResult()

    def get_qubit_ssa(self, ilist_ref: ir.SSAValue) -> Optional[List[ir.SSAValue]]:
        addr = self.address_analysis.get(ilist_ref)
        if not isinstance(addr, address.AddressTuple):
            return None

        ids = []
        for ele in addr.data:
            if not isinstance(ele, address.AddressQubit):
                return None

            ids.append(ele.data)

        return [self.id_map[ele] for ele in ids]

    def rewrite_cz(self, node: ir.Statement):
        assert isinstance(node, parallel.CZ)

        ctrls = self.get_qubit_ssa(node.ctrls)
        qargs = self.get_qubit_ssa(node.qargs)

        if ctrls is None or qargs is None:
            return result.RewriteResult()

        for ctrl, qarg in zip(ctrls, qargs):
            new_node = uop.CZ(ctrl, qarg)
            new_node.insert_before(node)

        node.delete()

        return result.RewriteResult(has_done_something=True)

    def rewrite_u(self, node: ir.Statement):
        assert isinstance(node, parallel.UGate)

        qargs = self.get_qubit_ssa(node.qargs)

        if qargs is None:
            return result.RewriteResult()

        for qarg in qargs:
            new_node = uop.UGate(qarg, theta=node.theta, phi=node.phi, lam=node.lam)
            new_node.insert_after(node)

        node.delete()

        return result.RewriteResult(has_done_something=True)

    def rewrite_rz(self, node: ir.Statement):
        assert isinstance(node, parallel.RZ)

        qargs = self.get_qubit_ssa(node.qargs)

        if qargs is None:
            return result.RewriteResult()

        for qarg in qargs:
            new_node = uop.RZ(qarg, theta=node.theta)
            new_node.insert_after(node)

        node.delete()

        return result.RewriteResult(has_done_something=True)
