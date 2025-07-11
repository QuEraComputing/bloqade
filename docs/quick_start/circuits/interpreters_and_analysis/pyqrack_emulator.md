!!! warning
    This page is under construction. The content may be incomplete or incorrect. Submit an issue
    on [GitHub](https://github.com/QuEraComputing/bloqade/issues/new) if you need help or want to
    contribute.

## Running simulations

The program can be executed via a simulator backend, e.g. PyQrack, you can install it for M-series Macs and other machines via:


```bash
pip install pyqrack
```

!!! warning

    If you are using a Mac with an Intel CPU you will need to instead install the following:

    ```bash
    pip install pyqrack-cpu
    ```

    Alternatively, if you have access to a GPU with CUDA support you can leverage that via:

    ```bash
    pip install pyqrack-cuda
    ```
    


```python

from bloqade import qasm2
from bloqade.pyqrack import PyQrack

@qasm2.extended
def main():
    return qft(qasm2.qreg(3), 3)

device = PyQrack()
qreg = device.run(main)
print(qreg)
```
