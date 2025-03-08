# %% [markdown]
# # GHZ State Preparation with Parallelism
# In this example, we will implement a *Greenberger-Horne-Zeilinger* (GHZ) state preparation circuit with $N = 2^n$ qubits.
#
# First, we will present the standard linear-depth construction in Bloqade but later we will present a log-depth
# construction that achieves the same result. We then take this one step further and use the fact that Bloqade
# (and QuEra's neutral atom hardware!) support *parallel* gates, allowing for the application of the same gate
# across multiple qubits simultaneously. Combined with the fact that atom *shuttling* allows for arbitrary
# connectivity, we can also decrease the circuit *execution* depth from $N$ to just $n$.

# %%
import math

from bloqade import qasm2
from kirin.dialects import ilist

# %% [markdown]
# ## Simple Linear Depth Implementation of a GHZ State Preparation Circuit
#
# A simple GHZ state preparation circuit can be built with $N - 1$ CX gates and $1$ H gate.
# This gives the circuit an execution depth of $N$.

# <div align="center">
# <picture>
#    <img src="/digital/examples/ghz_linear_circuit.svg" >
# </picture>
# </div>


# %%
def ghz_linear(n: int):
    n_qubits = int(2**n)

    @qasm2.extended
    def ghz_linear_program():

        qreg = qasm2.qreg(n_qubits)
        # Apply a Hadamard on the first qubit
        qasm2.h(qreg[0])
        # Create a cascading sequence of CX gates
        # necessary for quantum computers that
        # only have nearest-neighbor connectivity between qubits
        for i in range(1, n_qubits):
            qasm2.cx(qreg[i - 1], qreg[i])

    return ghz_linear_program


# %% [markdown]
# ## Log-depth Implementation of a GHZ State Preparation Circuit
#
# Let's take a look how we can rewrite the circuit to take advantage of QuEra's hardware capabilities.
# We can achieve log(N) circuit depth by [rearranging the CX gates (see *Mooney, White, Hill, Hollenberg* - 2021)](https://arxiv.org/abs/2101.08946).
#

# %% [markdown]
# <div class="admonition note">
# <p class="admonition-title">Circuit vs. Execution Depth</p>
# <p>
# Before going any further, it's worth distinguishing between the concept of circuit depth and circuit execution depth.
# For example, in the following implementation, each CX gate instruction inside the for-loop is executed in sequence.
# So even thought the circuit depth is $log(N) = n$, the circuit execution depth is still $N$.
# </p>
# </div>

# <div align="center">
# <picture>
#    <img src="/digital/examples/ghz_log_circuit.svg" >
# </picture>
# </div>


# %%
def ghz_log_depth(n: int):
    n_qubits = int(2**n)

    @qasm2.extended
    def layer_of_cx(i_layer: int, qreg: qasm2.QReg):
        step = n_qubits // (2**i_layer)
        for j in range(0, n_qubits, step):
            qasm2.cx(ctrl=qreg[j], qarg=qreg[j + step // 2])

    @qasm2.extended
    def ghz_log_depth_program():

        qreg = qasm2.qreg(n_qubits)

        qasm2.h(qreg[0])
        for i in range(n):
            layer_of_cx(i_layer=i, qreg=qreg)

    return ghz_log_depth_program


# %% [markdown]
# ## Our Native Gate Set and Parallelism
# By nature, our digital quantum computer can execute native gates in parallel in an single instruction/ execution cycle.
# The concept is very similar to the SIMD (Single Instruction, Multiple Data) in classical computing.
#
# On our hardware, there are two important factors to be considered:
# 1. the native gate set allows for arbitrary (parallel) rotations and (parallel) CZ gates.
# 2. Our atom shuttling architecture allows arbitrary qubit connectivity. This means that our parallel instruction is not limited to fixed connectivity (for example nearest neighbor connectivity).
#
# With this in mind, we can rewrite the `layer` subroutine to now use the `qasm2.parallel` dialect in Bloqade.
# We know that the CX gate can be decomposed into a CZ gate with two single-qubit gates $R_y(-\pi/2)$ and $R_y(\pi/2)$ acting on the target qubits.
# With this decomposition in mind, we can now using our parallel gate instructions `parallel.u` and `parallel.cz`.
# With the following modification, we can further reduce the circuit execution depth to just $n$ (log of the total number of qubits $N$)


# %%
def ghz_log_simd(n: int):
    n_qubits = int(2**n)

    @qasm2.extended
    def layer(i_layer: int, qreg: qasm2.QReg):
        step = n_qubits // (2**i_layer)

        def get_qubit(x: int):
            return qreg[x]

        ctrl_qubits = ilist.Map(fn=get_qubit, collection=range(0, n_qubits, step))
        targ_qubits = ilist.Map(
            fn=get_qubit, collection=range(step // 2, n_qubits, step)
        )

        # Ry(-pi/2)
        qasm2.parallel.u(qargs=targ_qubits, theta=-math.pi / 2, phi=0.0, lam=0.0)

        # CZ gates
        qasm2.parallel.cz(ctrls=ctrl_qubits, qargs=targ_qubits)

        # Ry(pi/2)
        qasm2.parallel.u(qargs=targ_qubits, theta=math.pi / 2, phi=0.0, lam=0.0)

    @qasm2.extended
    def ghz_log_depth_program():

        qreg = qasm2.qreg(n_qubits)

        qasm2.h(qreg[0])
        for i in range(n):
            layer(i_layer=i, qreg=qreg)

    return ghz_log_depth_program


# %% [markdown]
# <div class="admonition warning">
# <p class="admonition-title">Using Closures to Capture Global Variables</p>
# <p>
# While bloqade.qasm2 permits a main program with arguments, standard QASM2 does not.
# To get around this, we need to put the program in a closure.
# Our Kirin compiler toolchain can capture the global variable inside the closure.
# In this case, the n_qubits will be captured upon calling the ghz_log_simd(n_qubits) python function.
# As a result, the returned QASM2 program will not have any arguments.
# </p>
# </div>

# %%
target = qasm2.emit.QASM2(
    allow_parallel=True,
)
ast = target.emit(ghz_log_simd(4))
qasm2.parse.pprint(ast)
