!!! warning
    This page is under construction. The content may be incomplete or incorrect. Submit an issue
    on [GitHub](https://github.com/QuEraComputing/bloqade/issues/new) if you need help or want to
    contribute.

## Running simulations

The program can be executed via a simulator backend, e.g. PyQrack, you can install it via

```bash
pip install bloqade[pyqrack]
# or if you want to use the CPU only version
pip install bloqade[pyqrack-cpu]
```

```python
@qasm2.extended
def main():
    return qft(qasm2.qreg(3), 3)

device = PyQrack()
qreg = device.run(main)
print(qreg)
```
