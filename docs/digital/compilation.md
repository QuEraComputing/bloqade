# Understanding the compilation process

The compilation process is divided into several stages:

1. **Lowering**: a decorator such as `qasm2.extended` takes the Python Abstract Syntax Tree (AST) and lowers it into Kirin's Intermediate Representation (IR) which follows Static Single Assignment (SSA) form.
2. **Interpretation**: when invoking a backend, such as the PyQrack simulator, the IR code is interpreted by an interpreter featuring the corresponding method tables for the runtime evaluation of the dialect statements.
3. **Target code generation**: when emitting code, several steps can be involved before the actual code emission, depending on the target. Similar to interpretation, each statement will be translated to one that fits the chosen target. For example, when emitting QASM2, the following steps occur:
   1. The IR code gets aggressively inlined and all constant expressions are evaluated.
   2. All loops and control flow are unrolled.
   3. All compatible Python expressions (e.g `sin`, arithmetics) are translated into QASM2 expressions.
   4. The QASM2 code is emitted as QASM2 AST for pretty printing.

### Progressive compilation

As well as writing circuit executions, you can also progressively transform and compile that circuit.
While it is possible to write your own compiler passes and optimizations - for that, please refer to the [`kirin`](https://queracomputing.github.io/kirin/latest/) documentation - `bloqade-circuit` also offers a number of different, pre-defined optimizations.

!!! warning
    Compiler and optimization passes are currently under development.
    While quite a lot of them are used internally, they are not in a user-friendly state.
    Please proceed with caution!


## Dialect groups

Bloqade provides a set of [dialects](../dialects_and_kernels/) for QASM2 and our custom extensions to model parallel gates in neutral atom architectures. The basic QASM2 functionality can be enabled via

```bash
pip install bloqade[qasm2]
```

### Extended QASM

The decorator `qasm2.extended` is a group of smaller dialects:

```python
extended = structural_no_opt.union(
     [
         inline,
         uop,
         glob,
         noise,
         parallel,
         core,
     ]
 )
```

where `structural_no_opt` is the base dialect group (defined in [`kirn`](https://queracomputing.github.io/kirin/latest/)) that provides the basic control flow, common Python expressions (but not all), then:

- `core` provides the core QASM2 operations such as register allocation, measurement and reset.
- `uop` provides the unary operations, such as standard Pauli gates, rotation gates, etc.

The following dialects are specific to neutral atom quantum computing as an extension:

- `glob` provides the global gates (Rydberg specific)
- `noise` provides the noise channels
- `parallel` provides the parallel gate support (Rydberg specific).
- `inline` provides the inline QASM string

### Strict QASM2 mode

While the `qasm2.extended` decorator provides a lot of high-level features as an extension of QASM2, you may want to program in strict QASM2 mode for compatibility reasons. You can do this by using the `qasm2.main` and `qasm2.gate` decorators.
Note that `qasm2.main` features all standard QASM2 instructions, whereas `qasm2.gate` adds the functionality for defining custom gate subroutines.

```python
@qasm2.main
def main():
    qasm2.h(0)
    qasm2.cx(0, 1)
    qasm2.measure(0)
    qasm2.measure(1)
    return qasm2.qreg(2)
```

which corresponds to the following QASM2 code:

```qasm
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
```

Note that `return` is not supported in QASM2 and are therefore omitted in the code above.
