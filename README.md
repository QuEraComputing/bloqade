# Welcome to Bloqade -- QuEra's Neutral Atom SDK

[![CI](https://github.com/QuEraComputing/bloqade/actions/workflows/ci.yml/badge.svg)](https://github.com/QuEraComputing/bloqade/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/QuEraComputing/bloqade/graph/badge.svg?token=BpHsAYuzdo)](https://codecov.io/gh/QuEraComputing/bloqade)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/bloqade.svg?color=%2334D058)](https://pypi.org/project/bloqade)
[![Documentation](https://img.shields.io/badge/Documentation-6437FF)](https://bloqade.quera.com/)
[![DOI](https://zenodo.org/badge/629628885.svg)](https://zenodo.org/doi/10.5281/zenodo.11114109)


Bloqade is a Python SDK for neutral atom quantum computing. It provides a set of embedded domain-specific languages (eDSLs) for programming neutral atom quantum computers. Bloqade is designed to be a high-level, user-friendly SDK that abstracts away the complexities of neutral atom quantum computing, allowing users to focus on developing quantum algorithms and compilation strategies for neutral atom quantum computers.

> [!IMPORTANT]
>
> This project is in the early stage of development. API and features are subject to change.

## Installation

### Install via `uv` (Recommended)

```py
uv add bloqade
```

## Documentation

The documentation is available at [https://bloqade.quera.com/latest/](https://bloqade.quera.com/latest/). We are at an early stage of completing the documentation with more details and examples, so comments and contributions are most welcome!

## Roadmap

We use github issues to track the roadmap. There are more feature requests and proposals in the issues. Here are some of the most wanted features we wish to implement by 2025 summer (July):

- [x] QASM2 dialect (dialect, parser, pyqrack backend, ast, codegen)
- [x] QASM2 extensions (e.g. parallel gates, noise, etc.)
- [x] STIM dialect (dialect, codegen)
- [ ] structural gate dialect (language proposal, dialect, passes)
- [ ] atom-move dialect (language proposal, dialect, passes)
- [ ] atom move animation backend

Proposal for the roadmap and feature requests are welcome!

## License

Apache License 2.0 with LLVM Exceptions
