# %% [markdown]
# Lets do a simple example of a prototype circuit that benefits from parallelism: QAOA
# solving the MaxCut problem. For more details, see [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
# and the considerable literature that has developed around this algorithm.

# %%
import math
from typing import Any

import kirin
import networkx as nx
from kirin.dialects import ilist

from bloqade import qasm2

pi = math.pi

# %% [markdown]
# MaxCut is a combinatorial graph problem that seeks to bi-partition the nodes of some
# graph G such that the number of edges between the two partitions is maximized.
# Here, we choose a random 3 regular graph with 32 nodes [ref](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.042612)

# %%
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


# %%
def qaoa_sequential(G: nx.Graph) -> kirin.ir.Method:

    edges = list(G.edges)
    nodes = list(G.nodes)
    N = len(nodes)

    @qasm2.extended
    def kernel(gamma: ilist.IList[float, Any], beta: ilist.IList[float, Any]):
        # Initialize the register in the |+> state
        q = qasm2.qreg(N)
        for i in range(N):  # structural control flow is native to the Kirin compiler
            qasm2.h(q[i])

        # Repeat the cost and mixer layers
        for i in range(len(gamma)):
            # The cost layer, which corresponds to a ZZ(phase) gate applied
            # to each edge of the graph
            for j in range(len(edges)):
                edge = edges[j]
                qasm2.cx(q[edge[0]], q[edge[1]])
                qasm2.rz(q[edge[1]], gamma[i])
                qasm2.cx(q[edge[0]], q[edge[1]])
            # The mixer layer, which corresponds to a X(phase) gate applied
            # to each node of the graph
            for j in range(N):
                qasm2.rx(q[j], beta[i])

        return q

    return kernel


# %% [markdown]
# Next, lets implement a SIMD (Single Instruction, Multiple Data) version of the QAOA algorithm,
# which effectively represents the parallelism in the QAOA algorithm.
# This can be done by coloring the (commuting) ZZ(phase) gates into groups with non-overlapping
# sets of qubits, and then applying each of those groups in parallel.
# By [Vizing's theorem](https://en.wikipedia.org/wiki/Vizing%27s_theorem) the edges of a graph
# can efficiently be colored into $\Delta+1$ colors, where $\Delta$ is the maximum degree of the graph.
# Unfortunately, networkx does not have a native implementation of the algorithm so instead we use
# the lesser [Brooks' theorem]https://en.wikipedia.org/wiki/Brooks%27_theorem) to color the edges
# using an equitable coloring of the line graph.


# %%
def qaoa_simd(G: nx.Graph) -> kirin.ir.Method:

    nodes = list(G.nodes)

    # Note that graph computation is happening /outside/ the kernel function:
    # this is a computation that occurs on your laptop in Python when you generate
    # a program, as opposed to on a piece of quantum hardware, which is what
    # occurs inside of the kernel.
    Gline = nx.line_graph(G)
    colors = nx.algorithms.coloring.equitable_color(Gline, num_colors=5)
    left_ids = ilist.IList(
        [
            ilist.IList([edge[0] for edge in G.edges if colors[edge] == i])
            for i in range(5)
        ]
    )
    right_ids = ilist.IList(
        [
            ilist.IList([edge[1] for edge in G.edges if colors[edge] == i])
            for i in range(5)
        ]
    )
    # We can use composition of kernel functions to simplify repeated code.
    # Small snippets (say, the CX gate) can be written once and then called
    # many times.

    @qasm2.extended
    def parallel_h(qargs: ilist.IList[qasm2.Qubit, Any]):
        qasm2.parallel.u(qargs=qargs, theta=pi / 2, phi=0.0, lam=pi)

    # A parallel CX gate is equivalently a parallel H gate, followed by a parallel CZ gate,
    # followed by another parallel H. the CZ can be done in any order as they permute.
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
        # Declare the register and set it to the |+> state
        q = qasm2.qreg(len(nodes))
        # qasm2.glob.u(theta=pi / 2, phi=0.0, lam=pi,registers=[q])

        def get_qubit(x: int):
            return q[x]

        all_qubits = ilist.map(fn=get_qubit, collection=range(N))

        parallel_h(all_qubits)

        for i in range(len(gamma)):  # For each QAOA layer...
            # Do the ZZ phase gates...
            for cind in range(
                5
            ):  # by applying a parallel CZ phase gate in parallel for each color,
                ctrls = ilist.map(fn=get_qubit, collection=left_ids[cind])
                qargs = ilist.map(fn=get_qubit, collection=right_ids[cind])
                parallel_cz_phase(ctrls, qargs, gamma[i])
            # ...then, do an X phase gate. Observe that because this happens on every
            # qubit, we can do a global rotation, which is higher fidelity than
            # parallel local rotations.
            # qasm2.glob.u(theta=beta[i],phi=0.0,lam=0.0,registers=[q])
            qasm2.parallel.u(qargs=all_qubits, theta=beta[i], phi=0.0, lam=0.0)

        return q

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
target = qasm2.emit.QASM2()
ast = target.emit(main)
qasm2.parse.pprint(ast)
