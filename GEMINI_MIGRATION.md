# Gemini package ownership

`bloqade.gemini` is now shipped by the `bloqade` distribution, from
`src/bloqade/gemini`. Install `bloqade` to use `@logical.kernel`, physical
kernels, devices, simulators, and decoding. Kernel import paths are unchanged;
devices, simulators and the decoding workflow moved under `bloqade.gemini.logical`
and `bloqade.gemini.physical` (see "Devices, simulators and experiments" below).
Lanes itself can be imported and used without installing Gemini.

Compiler definitions belong to `bloqade.lanes`:

| Previous definition | Canonical definition |
| --- | --- |
| `gemini.common.dialects.qubit` | `lanes.dialects.arch` (`NewAt`, `new_at`) |
| `gemini.common.dialects.arrange` | `lanes.dialects.arrange` |
| `gemini.logical.dialects.operations` | `lanes.dialects.steane713` |
| `gemini.logical.dialects.extensions` | `lanes.dialects.steane713.extensions` |
| `gemini.star` | `lanes.dialects.steane713.star` |
| `GeminiLogicalValidation` | `lanes.validation.steane.SteaneLogicalValidation` |
| `GeminiTerminalMeasurementValidation` | `lanes.validation.terminal_measurement.SteaneTerminalMeasurementValidation` |

The `bloqade.gemini.logical.dialects` compatibility package is removed;
import compiler definitions directly from `bloqade.lanes.dialects.steane713`.
`common` contains Gemini's shared Cirq conversion support. Its
`common.validation` compatibility package is removed; import validation passes
directly from `bloqade.lanes.validation`.
Device-only extension rejection and kernel argument policies remain in Gemini.
The experimental STAR dialect remains separate from the ordinary Steane dialect.

## Allocation and movement API

Use `bloqade.gemini.new_at(zone_id, word_id, site_id)` for pinned allocation,
`bloqade.gemini.move_to(qubits, locations)` for explicit destinations and
`bloqade.gemini.arrange.permute(qubits, perm)` for permutations. The frontend
`arrange` module exports only `permute`; it does not expose compiler internals.
The old `bloqade.gemini.common.dialects` compatibility package is removed,
including its `qubit` and `arrange` subpackages.
Compiler consumers should import statements, the dialect object, and other
implementation details directly from `bloqade.lanes.dialects.arch` or
`bloqade.lanes.dialects.arrange`.

## Devices, simulators and experiments

The `bloqade.gemini.device` and `bloqade.gemini.decoding` packages are removed.
Their contents are grouped by the kind of kernel they run:

| Previous import | New import |
| --- | --- |
| `bloqade.gemini.device.GeminiLogicalDevice` (also `GeminiLogicalFuture`, `GeminiLogicalResult`) | `bloqade.gemini.logical.device` |
| `bloqade.gemini.device.GeminiLogicalSimulator` (also `GeminiLogicalSimulatorTask`, `Result`, `DetectorResult`, `SimulatorResult`) | `bloqade.gemini.logical.simulation` |
| `bloqade.gemini.device.GeminiPhysicalSimulator` (also `PhysicalSimulator`, `PhysicalSimulatorTask`, `PhysicalResult`) | `bloqade.gemini.physical.simulation` |
| `bloqade.gemini.device.TsimSimulatorBackend` (also `PPVMSimulatorBackend`, `CliffTSimulatorBackend`, `AbstractSimulatorBackend`, `BackendSample`) | `bloqade.simulation` |
| `bloqade.gemini.decoding.PostSelectionExperiment` (also `PostselectionCurveData`) | `bloqade.gemini.logical.experiments` |
| `bloqade.gemini.decoding.magic_state_dist_steane` | `bloqade.gemini.logical.kernels.build_msd_steane` |
| `bloqade.gemini.decoding.single_qubit_state_tomography` | `bloqade.gemini.logical.kernels.build_singleq_tomo` |
| `bloqade.gemini.decoding.empty_logical_circuit` | `bloqade.gemini.logical.kernels.build_empty_logical_circuit` |
| `bloqade.gemini.decoding.TableDecoder` | `bloqade.decoders.TableDecoder` |
| `bloqade.gemini.decoding.TomographyResult` | `bloqade.analysis.tomography.TomographyResult` |
| `bloqade.gemini.decoding.GurobiDecoder` | removed; use `bloqade.decoders.MILPDecoder(dem, solver="GUROBI")` |

The top-level re-exports (`bloqade.gemini.GeminiLogicalSimulator`,
`bloqade.gemini.TsimSimulatorBackend` and the rest) still work.

The lower-level MSD helpers are now public:

- `bloqade.gemini.logical.kernels`: `build_msd_kernels` (was
  `_build_msd_primitives`), `build_tomography_kernels` (was
  `_build_tomography_primitives`), `compose_tomography_kernels` (was
  `_build_decoder_kernel_bundle`) and `PostselectionExperimentKernels` (was
  `_DecoderPrimitiveSet`), plus new kernel builders for a 10-qubit GHZ state
  (`build_bell_state_kernels`), a logical Toffoli (`build_logical_toffoli`)
  and two-level MSD (`build_concatenated_msd_kernels`, which needs 25 logical
  qubits and so exceeds the current 10-qubit Gemini limit);
- `bloqade.gemini.logical.utils`: `MSD_TARGET_BLOCH`, `bloch_to_u3`,
  `msd_postselection_obs` (the ancilla pattern to pass to
  `decode_and_postselect`) and `plot_decoder_curves`.

`PostSelectionExperiment` takes the device to sample from as `device=`
(default: `GeminiLogicalSimulator()`), so `make_tasks()` needs no argument.
Passing a device to `make_tasks(device)` still overrides it.

## Move-program debuggers and architecture drawing

The step-by-step debuggers for compiled move programs and the architecture
visualizer are owned by Gemini:

- import `debugger`, `animated_debugger` and `plotly_debugger` from
  `bloqade.gemini.visualize` instead of `bloqade.lanes.visualize`;
- import `ArchVisualizer` and `bus_preview_label` from
  `bloqade.gemini.visualize.arch` instead of `bloqade.lanes.visualize.arch`.

`task.visualize()` uses them. Install the `atom-vis` extra for plotly and
matplotlib. The entropy-tree search viewer and the matplotlib stepper it builds
on (`bloqade.lanes.visualize.app.DebuggerController`) remain in lanes.

Each debugger accepts an `initialize_kernel`. When given, a
`move.PhysicalInitialize` is stepped through instead of shown as one step: the
kernel's gates, and for every CZ one move bringing the paired atoms together,
the pulse, and one move back. The controls of a CZ are taken to be the atoms
that move. Logical simulator tasks pass the noisy initialization kernel of
their noise model, which names the movers first.

## Architecture builder

Import the spec-shaped builder with `from bloqade.gemini.arch import ArchBuilder`.
Update imports from `bloqade.gemini.arch.build.v2` to this public path.

**Breaking change.** Callers using `ZoneBuilder`, the zero-argument
`ArchBuilder()`, or `build_arch(ArchBlueprint(...))` must migrate to the
shared word template. Define words once in **(row, column)** order, then add
each zone's physical coordinates and configure its buses by zone name.

```python
from bloqade.gemini.arch import ArchBuilder

builder = ArchBuilder(grid_shape=(1, 4), word_shape=(1, 2))
builder.add_word(rows=0, columns=[0, 1])
builder.add_word(rows=0, columns=[2, 3])
builder.add_zone(
    "storage", rows=[0.0], columns=[0.0, 10.0, 20.0, 30.0],
    x_clearance=1.0, y_clearance=1.0,
)
builder.add_word_bus("storage", src=[0], dst=[1])
spec = builder.build()
```

By default, a word bus preserves the site index. To move between sites,
pass a Lanes `SiteBus` as `site_map`. This second bus moves site 1 of word 0
to site 0 of word 1. Both sites must
belong to the zone's `sites_with_word_buses`. The same map applies to every
word pair on the bus.

```python
from bloqade.lanes.bytecode import SiteBus

builder.add_word_bus("storage", src=[0], dst=[1], site_map=SiteBus(src=[1], dst=[0]))
spec = builder.build()
```

Use `ArchBuilder.from_spec(spec)` to edit an existing Lanes `ArchSpec`.
The `gemini_full` and `generic_full` example factories still return an
`ArchResult` containing the spec and zone metadata. For a rectangular
grid of interleaved words, pass `num_rows`, `num_cols`, and `sites_per_word`
as keywords to `create_zone_words` in place of `ZoneSpec` and `DeviceLayout`.
See the [architecture example](docs/digital/examples/arch/arch_cust_vis_demo.py)
for a complete walkthrough.

## Method-table registration

Lanes 0.12.2 owns and registers the shared compiler method tables, including
qubit-address analysis, Steane type inference, measurement analysis, and
validation. Its dialect and validation modules load these implementations;
compiler callers do not need to import Gemini. Gemini's kernel groups reuse
these registrations without replacing them. Registration attaches tables to
shared lanes/Kirin dialect objects and is process-wide, not per kernel group.
Generic move/place and architecture evaluation/constant-propagation tables also
remain in lanes, which never imports Gemini.

Gemini retains the Cirq emission tables (`emit.cirq`) loaded by its Cirq
conversion modules and the device-only extension-rejection table
(`lanes.validate.no_extensions`) loaded by its extension-validation package.
Kernel argument validation and device policy also remain in Gemini.

## Coordinated release

This change requires `bloqade-lanes>=0.12.2,<0.13` and
`bloqade-circuit~=0.15.6`. The committed `uv.lock` resolves lanes 0.12.2,
circuit 0.15.6, and decoders 0.7.0 from published packages; no prerequisite
release or deferred lockfile regeneration remains for this migration.
Lanes 0.12.2 supplies the compiler-owned method-table registrations. Lanes 0.11
still owns the old Gemini package files and must not be installed alongside
the new bloqade wheel.

For testing unreleased source, install the two checkouts into the same environment
with compatible circuit/Kirin dependencies. A local uv source override for
`bloqade-lanes` can point at the lanes checkout; it should not be committed as a
machine-specific absolute path.

## Saved IR and backend compatibility

Python aliases preserve imports, not the old serialized IR protocol. Statement
module paths and dialect names changed (`lanes.arch`, `lanes.arrange`,
`lanes.steane713`, and `lanes.steane713.extensions`). Recompile saved kernels from
source. Producers and consumers of serialized pre-lowering IR must upgrade
together; old serialized kernels are not promised to load. Compiled move/bytecode
formats are unchanged. New IR is covered by serialization round-trip tests.

The recursion, static-call, and call-site utilities currently live in lanes.
Upstreaming them to Kirin remains a separate proposal requiring its maintainer's
review; this change does not implement that proposal.

## Decoder API

Decoding requires `bloqade-decoders>=0.7.0`. Use its `BaseDecoder` interface
instead of the removed Gemini `ConfidenceDecoder` interface, and call
`decode_confidence(bits)` instead of `decode_with_confidence(bits)`.

`GurobiDecoderWithConfidence` and Gemini's own `GurobiDecoder` are removed.
For MILP decoding with Gurobi, use `bloqade.decoders.MILPDecoder(dem,
solver="GUROBI")`; install `bloqade[decoding]` for the optional solver
dependencies. (`bloqade.decoders.GurobiDecoder` is upstream's deprecated
direct-gurobipy decoder, not a drop-in replacement.) MILP confidence follows the
upstream normalized logical-gap score, `tanh(max(gap, 0) / 2)`, rather than the
raw gap; update any absolute postselection thresholds accordingly.

Use `bloqade.decoders.TableDecoder` directly. The local
`TableDecoderWithConfidence` subclass is removed. Upstream 0.7 defaults to
10,000 training shots and reports zero confidence for unseen syndromes. It does
not implement the old `seed` or `step_size` training options. For reproducible
training, construct with `num_shots=0` and pass seeded DEM samples to
`update_det_obs_counts()`.

## Tomography API

Use `bloqade.analysis.tomography.TomographyResult`; the
`bloqade.gemini.decoding` re-export is removed. Pass one-dimensional
shot arrays per basis. Gemini's experiment helpers convert their internal shot
arrays and reject confidence cuts that leave a tomography basis empty.

The declared circuit minimum is 0.15.6, which permits decoders 0.7.0. These
versions are resolved together in `uv.lock`; the earlier circuit/decoder
dependency conflict is resolved.

The `bloqade.gemini.logical.dialects.extensions` compatibility package is removed.
Compiler consumers should use `bloqade.lanes.dialects.steane713.extensions`.
User kernels can call `logical.star_rz`, which aliases the existing frontend
`logical.extensions.star_rz` wrapper; both accept an angle in radians.
`logical.kernel` continues to include the lanes extension dialect directly.

`logical.terminal_measure` remains a direct public export of the lanes operation
and works in `@logical.kernel`; it does not require the removed compatibility modules.

Use `logical.star_rz(theta, qubit)` for one qubit or
`logical.broadcast.star_rz(theta, qubits)` for a register, alongside the standard
broadcast gates. Both take angles in radians.

The `bloqade.gemini.logical.rewrite` compatibility package is removed. Import
initialization, qubit-count, postprocessing-removal, and Steane transversal
rewrites directly from `bloqade.lanes.rewrite`.

The deprecated `GeminiLogicalValidation` and `GeminiTerminalMeasurementValidation`
re-exports and `bloqade.gemini.star` shim are removed. Import the validation
passes from `bloqade.lanes.validation` and STAR support definitions from
`bloqade.lanes.dialects.steane713.star`. Their compiler method tables are
registered by lanes. Gemini retains its device-only extension check and kernel
argument validation.
