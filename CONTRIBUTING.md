# Contributing

Please see [Installation](install.md) for instructions on how to set up your development environment.

## AI policy

We welcome contributions that use AI assistance. However, **we ask that first-time contributors do not submit AI-generated PRs**. We want to get to know you and your understanding of the project before AI-assisted contributions enter the mix. Once you have made a few contributions, feel free to use AI tools as you see fit.

## Pre-commit hooks

We use `pre-commit` to run the linter checks before you commit your changes. The pre-commit hooks are installed as part of the development dependencies. You can setup `pre-commit` using the following command:

```bash
pre-commit install
```

This will run the linter checks before you commit your changes. If the checks fail, the commit will be
rejected. Most of the following sections can be checked by the pre-commit hooks.

## Running the tests

We use `pytest` for testing. To run the tests, simply run:

```bash
pytest
```

or for a specific test file with the `-s` flag to show the output of the program:

```bash
pytest -s tests/test_program.py
```

lots of tests contain pretty printing of the IR themselves, so it's useful to see the output.

## Code style

We use `black` for code formatting. Besides the linter requirements, we also require the following
good-to-have practices:

### Naming

- try not to use abbreviation as names, unless it's a common abbreviation like `idx` for `index`
- try not to create a lot of duplicated name prefix unless the extra information is necessary when accessing the class object.
- try to use `snake_case` for naming variables and functions, and `CamelCase` for classes.

### Comments

- try not to write comments, unless it's really necessary. The code should be self-explanatory.
- if you have to write comments, try to use `NOTE:`, `TODO:` `FIXME:` tags to make it easier to search for them.

## Documentation

We use [`mise`](https://mise.jdx.dev) to manage the toolchain and command-line tasks (`brew install mise`, then `mise install`). Run `mise run` to list every task. To preview the existing MkDocs documentation, run:

```bash
mise run mkdocs:serve
```

This will launch a local server to preview the documentation. You can also run `mise run mkdocs:build` to build the documentation without launching the server. The new Astro + Starlight docs site under `website/` has its own `docs:*` tasks (`mise run docs:dev`, `mise run docs:build`, …) — see [`website/README.md`](website/README.md). The bare top-level tasks act on the Python package: `mise run build` (`uv build`) and `mise run test`.

## License

By contributing to this project, you agree to license your contributions under the Apache License 2.0 with LLVM Exceptions.
