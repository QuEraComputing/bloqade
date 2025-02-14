# %% [markdown]
# # Quantum Fourier Transform
# In this example, we will explore the Quantum Fourier Transform (QFT) circuit using
# recursion and iteration -- a convenient way to implement the QFT circuit using
# our high-level programming features.
#
# To begin, we will import the `qasm2` module from the `bloqade` package and the `PyQrack`
# backend from the `bloqade.pyqrack` module, which can be installed via
#
# ```bash
# pip install bloqade[pyqrack]
# ```
# or
# ```bash
# pip install bloqade[pyqrack-cpu]
# ```
# depending on the hardware and OS you have.
# %%
import math

from bloqade import qasm2
from bloqade.pyqrack import PyQrack

# In the following, we will define the Quantum Fourier Transform (QFT) circuit using recursion
# inside a kernel function `qft`. The `qft` function takes two arguments: a quantum register `qreg`
# and an integer `n` representing the number of qubits we want to apply the QFT circuit to.
# %%
pi = math.pi


@qasm2.extended
def qft(qreg: qasm2.QReg, n: int):
    if n == 0:
        return qreg

    qasm2.h(qreg[0])
    for i in range(1, n):
        qasm2.cu1(qreg[i], qreg[0], 2 * pi / 2**i)
    qft(qreg, n - 1)
    return qreg


# %% [markdown]
# Next, we will call this kernel function `qft` inside a `main` function to check if
# the QFT circuit is correctly implemented. We will use a quantum register of size 3.


# %%
@qasm2.extended
def main():
    return qft(qasm2.qreg(3), 3)


# %% [markdown]
# Finally, we will run the `main` function on the `PyQrack` backend and print the quantum register
# to see the final state of the qubits after applying the QFT circuit.

# %%
device = PyQrack()
qreg = device.run(main)
print(qreg)

# %% [markdown]
# we can also emit the QASM2 code for the `main` function and print it to see the QASM2 code
# that corresponds to the QFT circuit.

# %%
from bloqade.qasm2.emit import QASM2  # noqa: E402
from bloqade.qasm2.parse import pprint  # noqa: E402

target = QASM2()
ast = target.emit(main)
pprint(ast)
