# %% [markdown]
# # Deutsch-Jozsa Algorithm
# In this example, we will implement a version of the [Deutsch-Josza algorithm](https://en.wikipedia.org/wiki/Deutsch–Jozsa_algorithm) using bloqade's squin dialect.
# %% [markdown]

# We start by loading in some stuff and defining some parameters.

# %%
import random
from typing import Any

from bloqade.types import Qubit
from kirin.dialects import ilist
from bloqade.pyqrack import StackMemorySimulator

from bloqade import squin

n_bits = 2

# %% [markdown]
#
# Now, before we can implement the actual algorithm, we need to define the oracles, i.e. the functions we want to check for.
#
# The problem is defined as follows:
# Given a bit string of length $n$, $x \in \{0, 1\}^\otimes n$, we have a function that is either constant or balanced.
#
# A constant function is defined as $f_\text{const}(x) = c \forall x$, where $c \in \{0, 1\}$ is some constant value.
#
# A balanced function, on the other hand, is defined by
#
# $f_\text{balanced}(x) = \begin{cases} 0 \, \forall x \in S(x), \\ 1 \text{ else,} \end{cases}$
#
# where $S(x)$ is an arbitrarily chosen half of all possible bit strings, i.e. $|S(x)| = 2^{n-1}$.


# %% [markdown]
# For our example, we will be using $n + 1$ qubits, where $n$ store the bitstring $x$ and the result is stored in the last qubit.
# We'll be writing our oracle functions as squin kernels, which we can then later use in the actual algorithm implementation.
#
# In order to define our oracle functions, we can simply choose for the constant function to always return $1$, which we achieve by flipping the final qubit using an $X$ gate.
# %%
@squin.kernel
def f_constant(q: ilist.IList[Qubit, Any]):
    x = squin.op.x()

    # flip the final (result) qubit -- every bit string is mapped to 1
    squin.qubit.apply(x, [q[-1]])


# %% [markdown]

# For the balanced oracle we use the following approach: we use the first qubit as control in a $CX$ gate, which is applied to the resulting qubit.
# This means that the result will be $1$ in exactly half the cases.


# %%
@squin.kernel
def f_balanced(q: ilist.IList[Qubit, Any]):
    x = squin.op.x()
    cn_x = squin.op.control(x, n_controls=1)
    squin.qubit.apply(cn_x, [q[0], q[-1]])


# %% [markdown]
#
# Now, we define the actual algorithm as a kernel, which simply takes one of the other kernels as input.
# In the end, we can infer which function was provided by looking at the resulting measurement of the result qubit.
# %%
@squin.kernel
def deutsch_algorithm(f):
    q = squin.qubit.new(n_qubits=n_bits + 1)

    x = squin.op.x()
    squin.qubit.apply(x, [q[-1]])

    h = squin.op.h()
    for i in range(len(q)):
        squin.qubit.apply(h, [q[i]])

    # apply the oracle function
    f(q)

    for i in range(n_bits):
        squin.qubit.apply(h, [q[i]])

    return squin.qubit.measure(q[:-1])


# %% [markdown]
# Finally, we actually run the result.
# To do so, we use the `PyQrack` simulation backend in bloqade.
#
# To make things a bit more interesting, we randomly select which function we are running the algorithm with.

# %%
sim = StackMemorySimulator(min_qubits=n_bits + 1)

f_choice_idx = random.randint(0, 1)
f_choice = (f_constant, f_balanced)[f_choice_idx]

# result = sim.run(deutsch_algorithm, args=(f_balanced, n))
result0 = 0.0
n_shots = 100
for _ in range(n_shots):
    res = sim.run(deutsch_algorithm, args=(f_choice,))
    result0 += res[0] / n_shots

print(
    "Oh magic Deutsch-Jozsa algorithm, tell us if our function is constant or balanced:"
)
print("*drumroll*")
if result0 == 0:
    print("It's constant!")

    # let's make sure we actually did the right thing here
    assert f_choice_idx == 0
else:
    print("It's balanced!")

    # let's make sure we actually did the right thing here
    assert f_choice_idx == 1
