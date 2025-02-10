import networkx as nx
from bloqade import qasm2
from kirin.dialects import ilist
import math

# Define the problem instance as a MaxCut problem on a graph

N = 64
G = nx.random_regular_graph(3, N, seed=42)


def qaoa_sequential(G):
    
    @qasm2.main
    def kernel(gamma:ilist.IList,delta:ilist.IList):
        qreg = qasm2.qreg(N)
        for i in range(N):
            qasm2.h(qreg[i])
        for i in range(len(gamma)):
            for edge in G.edges:
                qasm2.cx(qreg[edge[0]], qreg[edge[1]])
                qasm2.rz(qreg[edge[1]], gamma[i])
                qasm2.cx(qreg[edge[0]], qreg[edge[1]])
            for j in range(N):
                qasm2.rx(qreg[j], delta[i])
        return qreg
    
    return kernel


def qaoa_simd(G):
    
    @qasm2.main
    def cz_phase(q1,q2, gamma):
        qasm2.cx(q1,q2)
        qasm2.rz(q2, gamma)
        qasm2.cx(q1,q2)
    
    @qasm2.main
    def kernel(gamma:ilist.IList[float],beta:ilist.IList[float]):
        qreg = qasm2.qreg(len(G.nodes))
        
        def get_qubit(x: int):
            return qreg[x]
        
        
        
        left = ilist.Map(fn=get_qubit, collection=[edge[0] for edge in G.edges])
        right= ilist.Map(fn=get_qubit, collection=[edge[1] for edge in G.edges])
        all = ilist.Map(fn=get_qubit, collection=range(N))
        
        qasm2.parallel.u(qargs=all, theta=math.pi/2, phi=math.pi/2, lam=0.0)
        for gamma_,beta_ in zip(gamma,beta):
            ilist.Map(fn=cz_phase, collection = ilist.Zip(left, right, [gamma_]*len(G.edges)))
            qasm2.parallel.u(qargs=all, theta=beta_, phi=0.0, lam=0.0)
        return qreg
    return kernel

print("--- Sequential ---")
qaoa_sequential(G).code.print()

print("\n\n--- Simd ---")
qaoa_simd(G).code.print()