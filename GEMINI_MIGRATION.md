# Gemini package ownership

`bloqade.gemini` is now shipped by the `bloqade` distribution, from
`src/bloqade/gemini`. Install `bloqade` to use `@logical.kernel`, physical
kernels, devices, simulators, and decoding. The kernel and device Python import paths remain
unchanged. Lanes itself can be imported and used without installing Gemini.

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

`bloqade.gemini.decoding.GurobiDecoder` replaces `GurobiDecoderWithConfidence`.
It is a thin `MILPDecoder` subclass selecting `solver="GUROBI"`, including when
constructed through `instantiate(dem)`. Install `bloqade[decoding]` for the
optional solver dependencies. Confidence now follows the upstream normalized
logical-gap score, `tanh(max(gap, 0) / 2)`, rather than returning the raw gap;
update any absolute postselection thresholds accordingly.

`bloqade.gemini.decoding.TableDecoder` (also available from `table_decoders`)
now directly re-exports `bloqade.decoders.TableDecoder`. The local
`TableDecoderWithConfidence` subclass is removed. Upstream 0.7 defaults to
10,000 training shots and reports zero confidence for unseen syndromes. It does
not implement the old `seed` or `step_size` training options. For reproducible
training, construct with `num_shots=0` and pass seeded DEM samples to
`update_det_obs_counts()`.

## Tomography API

`bloqade.gemini.decoding.TomographyResult`, including the `tomography` module
export, is now the class from `bloqade.analysis.tomography`. Pass one-dimensional
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
