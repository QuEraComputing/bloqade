!!! warning
    This page is under construction. The content may be incomplete or incorrect. Submit an issue
    on [GitHub](https://github.com/QuEraComputing/bloqade/issues/new) if you need help or want to
    contribute.

## Running simulations

The program can be executed via a simulator backend, e.g. PyQrack, you can install it via


```bash
pip install bloqade-pyqrack[backend]
```

with the `backend` being one of ` pyqrack`, `pyqrack-cpu`, `pyqrack-cuda` depending on
the hardware and OS you have. See [README](https://github.com/QuEraComputing/bloqade-pyqrack?tab=readme-ov-file#which-extra-do-i-install) for mote details.

```python
@qasm2.extended
def main():
    return qft(qasm2.qreg(3), 3)

device = PyQrack()
qreg = device.run(main)
print(qreg)
```
