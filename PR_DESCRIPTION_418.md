# refactor(gemini): own the move-program debuggers, the architecture visualizer and the architecture builders

## Summary

Move the atom-movement visualization and architecture tooling from
`bloqade-lanes` into bloqade, so changes to them ship with a bloqade release:

| What | New home | Was |
| --- | --- | --- |
| Move-program debuggers (`debugger`, `animated_debugger`, `plotly_debugger`) | `bloqade.gemini.visualize` | `bloqade.lanes.visualize` |
| Architecture visualizer (`ArchVisualizer`, `bus_preview_label`) and its plotly JavaScript controller | `bloqade.gemini.visualize.arch` | `bloqade.lanes.visualize.arch` |
| Architecture builders (`ArchBuilder`, `ZoneBuilder`, `build_arch`, blueprints, topologies, `build.v2`) | `bloqade.gemini.arch` | `bloqade.lanes.arch` |
| Example architectures `gemini_full`, `generic_full` | `bloqade.gemini.arch` | `bloqade.lanes.arch` |

On top of the move, the debuggers can now **step through state preparation**
(`move.PhysicalInitialize`) instead of showing it as a single step in which
nothing moves.

Everything here works against the released `bloqade-lanes` 0.12.2 pinned today.
The companion PR, QuEraComputing/bloqade-lanes#1161, removes the code from
lanes.

## Move-program debuggers: `bloqade.gemini.visualize`

Ported from lanes: `artist.py` (step collection and matplotlib renderers),
`debug.py` (static and animated matplotlib debuggers), `plotly_circuit.py`
(executed-circuit panel) and `plotly_debug.py` (plotly debugger).

The matplotlib debuggers still subclass `DebuggerController` from
`bloqade.lanes.visualize.app`, which stays in lanes because lanes' entropy-tree
viewer is built on it.

### Stepping through state preparation

Each debugger takes an optional `initialize_kernel`: the kernel that
`move.PhysicalInitialize` stands for. When given, each such statement becomes:

- the kernel's single-qubit gates, as `LocalR` / `LocalRz` steps;
- for every CZ, one move bringing the paired atoms together, the CZ pulse, and
  one move back.

How it works:

- `initialize.py` applies the kernel to the statement's blocks with its actual
  angles, lowers it with the compiler's squin-to-native lowering, and reads off
  the `R`, `Rz` and `CZ` operations. There is no hand-written schedule.
- The controls of a CZ are the atoms that move; each goes to the CZ partner
  site of its target.
- `DebugStep` gains an optional `moves` field, so a step can carry routes the
  architecture has no lane for. The plotly debugger labels them
  "Initialization move".
- Rotations by zero are dropped, and equal rotations on different blocks share
  a step.
- If the kernel can't be laid out (for example a CZ partner site is occupied),
  the statement stays one step and a warning says why.

Without `initialize_kernel` the debuggers behave as before.

### `task.visualize()`

Uses `bloqade.gemini.visualize`. Logical simulator tasks pass the noisy
initialization kernel from their noise model, which names the movers first in
every CZ (the clean kernel only pairs the qubits). Physical simulator tasks
pass no kernel, since their programs have no `move.PhysicalInitialize`.

## Architecture visualizer: `bloqade.gemini.visualize.arch`

`arch.py` and `_arch_interactive.js` are ported unchanged, except that the
JavaScript is now loaded from the `bloqade.gemini.visualize` package. The
debuggers draw on top of it: the plotly debugger starts from
`plot_interactive()`, and the matplotlib debuggers use `plot()` and the bounds.

## Architecture builders: `bloqade.gemini.arch`

Ported with imports repointed and no other code changes: the declarative
builder (`ArchBlueprint`, `ZoneSpec`, `DeviceLayout`, `build_arch`), the
imperative builder (`ArchBuilder`, `ZoneBuilder`), the topologies, word factory
and geometry helpers, the spec-shaped `build/v2` `ArchBuilder`, and the
three-zone examples `gemini_full` and `generic_full`.

`bloqade.gemini.arch` re-exports the builder API, as `bloqade.lanes.arch` did.
The builders still produce `bloqade.lanes.arch.spec.ArchSpec`, which the lanes
compiler consumes, so compilation is unchanged.

## Packaging, tooling and docs

- **`.gitignore`**: the `build/` rule matched the new `arch/build/` package, so
  git and the wheel would have left the builder out. A `!src/**/build/`
  exception keeps source subpackages named `build`, as lanes already does.
- **Dependencies**:
  - `rustworkx>=0.17.1` is a direct dependency, because the builder imports it
    (it was already installed through lanes);
  - the `atom-vis` extra now declares `plotly>=5.24.0,<8` itself, since lanes'
    `visualization` extra no longer includes it.

  `uv.lock` only gains those two edges; no versions change.
- **JavaScript checks**: `just check-visualization-js` (`node --check` on the
  controller plus its `node --test` regression tests), a new `javascript` job
  in `lint.yml`, and a pre-commit hook, all carried over from lanes.
- **Docs**: three new pages under "Architecture examples": the two builder
  tutorials from lanes' `demo/`, and the interactive-visualization and HTML
  export guide from lanes' docs.
- **Migration notes**: `GEMINI_MIGRATION.md` has the old and new import paths
  for all three moves. `src/bloqade/gemini/AGENT.md` lists the new packages as
  owned by Gemini.

Importing `bloqade.gemini` loads none of the new packages, so matplotlib,
plotly and the builder stay out of the default import.

## Testing

- `test/gemini/visualize/` and `test/gemini/arch/`: every lanes test for the
  moved code, ported, plus new tests for the state-preparation steps and
  `task.visualize()`. 488 passed, 1 xfailed (the xfail is carried over from
  lanes).
- The same tests, plus `test/gemini/device`, pass against the lanes branch of
  #1161 with the code removed from lanes: 651 passed, 1 xfailed.
- `test/test_dependencies.py` and the package-boundary tests pass;
  `uv lock --check` passes.
- black, isort, ruff and typos are clean.
- A locally built wheel contains `_arch_interactive.js` and all nine builder
  modules.
- The builder tutorials run to completion as scripts.
- Checked in a browser: on a compiled 3-qubit logical kernel, the plotly
  debugger shows state preparation in 29 steps (12 before), with the moves
  labelled "Initialization move".

Not run locally: the JavaScript tests (no Node.js here; the new CI job runs
them) and the docs build, which will execute the new tutorial pages for the
first time.

## Notes for reviewers

- **Release order.** This PR doesn't need a lanes release. Until bloqade pins a
  lanes version without this code, both import paths work.
- **Lanes internals.** The builder constructs grids, buses and zones through
  `bloqade.lanes.bytecode._native`, lanes' Rust bindings, and the plotly
  debugger calls `ArchVisualizer._iter_locations()`. A change to the bindings
  in lanes can break the builder; the second call is now within bloqade.
- **Downstream repos** that import any of this from lanes need to switch to the
  new paths and depend on `bloqade` (with `atom-vis` for visualization). The
  list is in #1161.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
