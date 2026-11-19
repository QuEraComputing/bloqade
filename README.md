# Welcome to Bloqade -- QuEra's Neutral Atom SDK

[![CI](https://github.com/QuEraComputing/bloqade/actions/workflows/ci.yml/badge.svg)](https://github.com/QuEraComputing/bloqade/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/QuEraComputing/bloqade/graph/badge.svg?token=BpHsAYuzdo)](https://codecov.io/gh/QuEraComputing/bloqade)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/bloqade.svg?color=%2334D058)](https://pypi.org/project/bloqade)
[![Documentation](https://img.shields.io/badge/Documentation-6437FF)](https://queracomputing.github.io/bloqade/)
[![DOI](https://zenodo.org/badge/629628885.svg)](https://zenodo.org/doi/10.5281/zenodo.11114109)


Bloqade is QuEra's Python SDK for neutral atom quantum computing. This
repository contains the [Gemini SDK](https://github.com/QuEraComputing/bloqade/tree/release-0-37/src/bloqade/gemini),
which provides logical and physical kernels, device and simulator interfaces,
and decoding workflows. Installing `bloqade` also installs the companion
packages for circuit and analog programming, compilation, simulation, and decoding.

> [!IMPORTANT]
>
> This project is in the early stage of development. API and features are subject to change.

## Installation

### Install via `uv` (Recommended)

```bash
uv add bloqade
```

Import the Gemini SDK with `from bloqade import gemini`. For the default
dependencies and optional extras, see the [package overview](https://github.com/QuEraComputing/bloqade/blob/release-0-37/src/bloqade/README.md).
The [Gemini migration notes](GEMINI_MIGRATION.md) describe which APIs moved
from `bloqade-lanes` and which versions work together.

## Documentation

The documentation is available at [https://queracomputing.github.io/bloqade/latest/](https://queracomputing.github.io/bloqade/latest/). We are at an early stage of completing the documentation with more details and examples, so comments and contributions are most welcome!

## Contributing

We welcome contributions! Please check the [contribution guidelines](https://queracomputing.github.io/bloqade/latest/contrib/) for a detailed guide.

> [!NOTE]
>
> We welcome contributions that use AI assistance. However, **we ask that first-time contributors do not submit AI-generated PRs**.


## License

Apache License 2.0 with LLVM Exceptions
