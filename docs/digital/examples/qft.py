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
# pip install bloqade-pyqrack[backend]
# ```
# with the `backend` being one of ` pyqrack`, `pyqrack-cpu`, `pyqrack-cuda` depending on
# the hardware and OS you have. see [README](https://github.com/QuEraComputing/bloqade-pyqrack?tab=readme-ov-file#which-extra-do-i-install)
# for mote details.
# %%
import math

from bloqade import qasm2
from bloqade.pyqrack import PyQrack

# %% [markdown]
# In the following, we will define the Quantum Fourier Transform (QFT) circuit using recursion
# inside a kernel function `qft`. The `qft` function takes two arguments: a quantum register `qreg`
# and an integer `n` representing the number of qubits we want to apply the QFT circuit to.
# %%
pi = math.pi


@qasm2.extended
def qft(qreg: qasm2.QReg, n: int, k: int):
    if k == n:
        return qreg

    qasm2.h(qreg[k])
    for i in range(k + 1, n):
        qasm2.cu1(qreg[i], qreg[k], 2 * math.pi / 2**i)
    qft(qreg, n, k + 1)  # recursion
    return qreg


# %% [markdown]
# Next, we will call this kernel function `qft` inside a `main` function to check if
# the QFT circuit is correctly implemented. We will use a quantum register of size 3.


# %%
@qasm2.extended
def main():
    return qft(qasm2.qreg(3), 3, 0)


# %% [markdown]
# Finally, we will run the `main` function on the `PyQrack` backend and print the quantum register
# to see the final state of the qubits after applying the QFT circuit.
# <div align="center">
# <picture>
#    <img src="/digital/examples/qft.svg" >
# </picture>
# </div>


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
