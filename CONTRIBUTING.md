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

We use `just` for managing command line tools and scripts. It should be installed when you run `uv sync`. To build the documentation, simply run:

```bash
just doc
```

This will launch a local server to preview the documentation. You can also run `just doc-build` to build the documentation without launching the server.

## Backporting to release branches

Use the same labels as Bloqade Lanes: `backport v<major>.<minor>` targets
`release-<major>-<minor>`. For example, `backport v0.36` targets `release-0-36`.

1. Merge the change into `main`. Add one or more backport labels either before
   merging or afterward. A maintainer can create a missing label in GitHub's
   Labels page; the corresponding release branch must already exist.
2. The **Backport merged PR** workflow cherry-picks the change and opens one PR
   per target branch. Both squash and rebase merges are supported. Backport PRs
   retain the original title and non-backport labels, and use branches named
   `codex/backport-<PR>-to-release-<major>-<minor>`.
3. Review each backport and run the release branch's checks before merging it.
   The workflow does not auto-merge, bump versions, create release branches, or
   publish a release.

The source PR receives a summary with links or conflict-recovery instructions.
Conflicts are not resolved automatically. Already-applied changes and existing
backport PRs are skipped. To retry, run **Backport merged PR** from the Actions
tab on `main`, supplying the original PR number. It reads the PR's current labels.
Removing a label does not close or delete an existing backport PR.

### Repository setup and CI credentials

In **Settings > Actions > General > Workflow permissions**, allow GitHub Actions
to create and approve pull requests (subject to organization policy). The
workflow uses `GITHUB_TOKEN` by default, matching the no-extra-secret approach
in Lanes. GitHub creates `pull_request` workflow runs for these automated PRs
in an **approval-required** state: a maintainer with write access must select
**Approve workflows to run** in the PR's merge box. See
[GitHub's workflow-trigger documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow#triggering-a-workflow-from-a-workflow).

For automatic PR checks, configure a repository Actions secret named
`BACKPORT_TOKEN` with a bot's fine-grained personal access token scoped to this
repository, with **Contents: read/write** and **Pull requests: read/write**.
Include **Workflows: read/write** if backports may modify `.github/workflows/`.
The workflow uses this token for both checkout/push and PR creation. Never put
the token in a file or PR. Without this secret, approve the generated PR's
workflows before merging; do not merge an untested backport. If no approval
banner or checks appear, a maintainer can close and reopen the PR to trigger
its `pull_request` checks.

The privileged backport job checks out only `main`, validates the source is
already merged into this repository's `main`, and accepts only numeric release
labels. It never executes source-PR or release-branch code after cherry-picking.
Keep that boundary intact when modifying the workflow. Local validation:

```bash
node --test .github/scripts/backport-targets.test.cjs
```

## License

By contributing to this project, you agree to license your contributions under the Apache License 2.0 with LLVM Exceptions.
