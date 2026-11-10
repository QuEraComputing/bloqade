# Gemini package ownership

`bloqade.gemini` is now shipped by the `bloqade` distribution, from
`src/bloqade/gemini`. Install `bloqade` to use `@logical.kernel`, physical
kernels, devices, simulators, and decoding. Their Python import paths remain
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

The Gemini package retains deprecated Python re-exports for the relocated
modules. These refer to the same class and dialect objects; they do not define
or register duplicate statements. The old `common` compatibility namespace is
therefore retained for now. New code should use the canonical lanes paths.
Device-only extension rejection and kernel argument policies remain in Gemini.
The experimental STAR dialect remains separate from the ordinary Steane dialect.

## Method-table registration

Gemini owns its qubit-address, Steane type-inference/measurement, and validation
method tables. Importing its logical or physical kernel groups loads these
implementations before constructing the groups. Registration attaches them to
the shared lanes/Kirin dialect objects; it is process-wide, not per kernel group.
Validation passes and analysis classes remain in lanes, which never imports
Gemini. Direct compiler callers must first load their frontend's tables;
missing required registrations raise a clear error. Generic move/place and
architecture evaluation/constant-propagation tables remain in lanes.

## Coordinated release

This change requires lanes 0.12 and circuit 0.15.5 or newer within their declared
compatible ranges. Publish the updated lanes package before releasing this
bloqade change. Regenerate `uv.lock` against that release; an older lanes wheel
both lacks the new compiler modules and owns the old Gemini package files.
Do not install the new bloqade wheel alongside lanes 0.11.

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
