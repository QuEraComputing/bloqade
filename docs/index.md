<div align="center">
<picture>
  <img id="logo_light_mode" src="assets/logo.svg" style="width: 70%" alt="Bloqade Logo">
  <img id="logo_dark_mode" src="assets/logo-dark.svg" style="width: 70%" alt="Bloqade Logo">
</picture>
<!--pad the following div a bit top-->
<div style="padding-top: -100px">
<h2>the Software Development Kit for neutral atom quantum computers</h2>
</div>
</div>

Bloqade is [QuEra Computing](https://quera.com)'s software development kit (SDK) for neutral atom quantum computers. It is designed to be a hub of embedded domain-specific languages (eDSLs) for neutral atom quantum computing. Bloqade is built on top of [Kirin](https://github.com/QuEraComputing/kirin), the Kernel Intermediate Representation Infrastructure.

!!! warning
    Bloqade is currently in the early stages of development. The APIs and features are subject to change. While we do not promise stability and backward compatibility at the moment, we will try to minimize breaking changes as much as possible. If you are concerned about the stability of the APIs, consider pin the version of Bloqade in your project.

!!! info
    The old version (<= 0.15) of Bloqade is still available as a sub-package `bloqade-analog`. You can keep using it via `bloqade.analog` module. For example `from bloqade import start` becomes `from bloqade.analog import start`. See [Installation](install.md) for more information.

## Installation

To install Bloqade, you can use the following command:

```bash
pip install bloqade
```

To install the extensions or extras for Bloqade and to setup the development environment, please refer to the [installation guide](install.md).

## Getting Started

To get started with Bloqade, you can refer to the following tutorials:

- [Background](background.md): Background information on neutral atom quantum computing.
- [Digital quick start](quick_start/circuits/index.md): A quick start guide for QASM2 and its extensions.
- [Analog quick start](quick_start/analog/index.md): A quick start guide for the analog quantum computing eDSL (same as older `bloqade` versions).

## Contributing

We welcome contributions to Bloqade. Please refer to the [contribution guide](contrib.md) for more information.

## License

Bloqade is licensed under the Apache License 2.0.
