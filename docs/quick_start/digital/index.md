!!! warning
    This page is under construction. The content may be incomplete or incorrect. Submit an issue
    on [GitHub](https://github.com/QuEraComputing/bloqade/issues/new) if you need help or want to
    contribute.

# Digital Quantum Computing

This section provides the quick start guide for programming digital quantum circuits using Bloqade.


## Open Quantum Assembly Language (QASM2)

Bloqade provides a set of dialects for QASM2 and our custom extensions to model parallel gates in neutral atom architectures. The QASM2 dialect is a simple quantum assembly language that allows you to write quantum circuits in a human-readable format. However, one should note that QASM2 is a very restricted language and does not support all the features of a high-level language.

For example, there is a separation of **gate routines** declared with `gate` and main program written as a sequence of gate applications. While the gate routine is similar to a function in many ways, it does not support high-level features such as recursions (due to lack of `if` statement support inside) or control flows.

While in our initial release, we support QASM2 as the first eDSL, we plan to use it as a compile target instead of a programming language for long-term development. We are working on a more expressive language that will be more suitable for quantum programming at error-corrected era.
