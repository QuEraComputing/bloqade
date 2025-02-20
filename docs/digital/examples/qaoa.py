# %% [markdown]
# Lets do a simple example of a prototype circuit that benefits from parallelism: QAOA
# solving the MaxCut problem. For more details, see [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
# and the considerable literature that has developed around this algorithm.

import math
from typing import Any

import kirin
import networkx as nx
from bloqade import qasm2
from kirin.dialects import py, ilist
pi = math.pi

#%% [markdown]
# MaxCut is a combinatorial graph problem that seeks to bi-partition the nodes of some
# graph G such that the number of edges between the two partitions is maximized.
# Here, we choose a random 3 regular graph with 32 nodes [ref](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.042612)
N = 32
G = nx.random_regular_graph(3, N, seed=42)


# %% [markdown]
# To build the quantum program, we use a builder function and use closure to pass variables
# inside of the kernel function (kirin methods).
# In this case, the two variables that are passed inside are the edges and nodes of the graph.
#
# The QAOA first prepares the |+> state as a superposition of all possible bitstrings,
# then repeats between the (diagonal) cost function and the mixer X with angles gamma and beta.
# It is parameterized by gamma and betas, which are each the p length lists of angles.
#
# Lets first implement the sequential version of the QAOA algorithm, which
# does not inform any parallelism to the compiler.

def qaoa_sequential(G:nx.Graph)->kirin.ir.Method:

    edges = list(G.edges)
    nodes = list(G.nodes)
    N = len(nodes)

    @qasm2.extended
    def kernel(gamma: ilist.IList[float, Any], beta: ilist.IList[float, Any]):
        # Initialize the register in the |+> state
        qreg = qasm2.qreg(N)
        for i in range(N): # structural control flow is native to the Kirin compiler
            qasm2.h(qreg[i])
            
        # Repeat the cost and mixer layers
        for i in range(len(gamma)):
            # The cost layer, which corresponds to a ZZ(phase) gate applied
            # to each edge of the graph
            for j in range(len(edges)):
                edge = edges[j]
                qasm2.cx(qreg[edge[0]], qreg[edge[1]])
                qasm2.rz(qreg[edge[1]], gamma[i])
                qasm2.cx(qreg[edge[0]], qreg[edge[1]])
            # The mixer layer, which corresponds to a X(phase) gate applied
            # to each node of the graph
            for j in range(N):
                qasm2.rx(qreg[j], beta[i])

        return qreg

    return kernel


# %% [markdown]
# Next, lets implement a SIMD (Single Instruction, Multiple Data) version of the QAOA algorithm,
# which effectively represents the parallelism in the QAOA algorithm.

def qaoa_simd(G:nx.Graph)->kirin.ir.Method:

    nodes = list(G.nodes)
    
    Gline = nx.line_graph(G)
    colors = nx.algorithms.coloring.equitable_color(Gline,num_colors=5)
    left_ids = ilist.IList([
        ilist.IList([edge[0] for edge in G.edges if colors[edge] == i])
            for i in range(5)
    ])
    right_ids = ilist.IList([
        ilist.IList([edge[1] for edge in G.edges if colors[edge] == i])
            for i in range(5)
    ])

    @qasm2.extended
    def parallel_h(qargs: ilist.IList[qasm2.Qubit, Any]):
        qasm2.parallel.u(qargs=qargs, theta=pi / 2, phi=0.0, lam=pi)

    @qasm2.extended
    def parallel_cx(
        ctrls: ilist.IList[qasm2.Qubit, Any], qargs: ilist.IList[qasm2.Qubit, Any]
    ):
        parallel_h(qargs)
        qasm2.parallel.cz(ctrls, qargs)
        parallel_h(qargs)

    @qasm2.extended
    def parallel_cz_phase(
        ctrls: ilist.IList[qasm2.Qubit, Any],
        qargs: ilist.IList[qasm2.Qubit, Any],
        gamma: float,
    ):
        parallel_cx(ctrls, qargs)
        qasm2.parallel.rz(qargs, gamma)
        parallel_cx(ctrls, qargs)

    

    @qasm2.extended
    def kernel(gamma: ilist.IList[float, Any], beta: ilist.IList[float, Any]):
        qreg = qasm2.qreg(len(nodes))
        qasm2.glob.u(theta=pi / 2, phi=0.0, lam=pi,registers=[qreg])

        def get_qubit(x: int):
            return qreg[x]

        

        
        for i in range(len(gamma)):
            for cind in range(5):
                ctrls = ilist.map(fn=get_qubit, collection=left_ids[cind])
                qargs = ilist.map(fn=get_qubit, collection=right_ids[cind])
                parallel_cz_phase(ctrls, qargs, gamma[i])
            qasm2.glob.u(theta=beta[i],phi=0.0,lam=0.0,registers=[qreg])

        return qreg

    return kernel


# %%
print("--- Sequential ---")
qaoa_sequential(G).code.print()

# %%
kernel = qaoa_simd(G)

print("\n\n--- Simd ---")
kernel.print()


# %%
@qasm2.extended
def main():
    kernel([0.1, 0.2], [0.3, 0.4])


# %%
target = qasm2.emit.QASM2(
    main_target=qasm2.main.union([qasm2.dialects.parallel, qasm2.dialects.glob, ilist, py.constant])
)
ast = target.emit(main)
qasm2.parse.pprint(ast)
