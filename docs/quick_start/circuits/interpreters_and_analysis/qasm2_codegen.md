!!! warning
    This page is under construction. The content may be incomplete or incorrect. Submit an issue
    on [GitHub](https://github.com/QuEraComputing/bloqade/issues/new) if you need help or want to
    contribute.

## Emitting QASM2 code

You can also emit QASM2 code from the IR code:

```python
from bloqade.qasm2.emit import QASM2 # the QASM2 target
from bloqade.qasm2.parse import pprint # the QASM2 pretty printer

target = QASM2()
ast = target.emit(main)
pprint(ast)
```

![QFT QASM2](../qft-qasm2.png)
