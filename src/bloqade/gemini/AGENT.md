# Ownership

`bloqade.lanes` owns the compiler — dialects, rewrites, analysis.
`bloqade.gemini` owns the product — interfaces, simulation, user flow.
Gemini uses lanes to implement those; lanes owns the infrastructure that makes it possible.

Import compiler definitions from lanes. Keep product kernel decorators, device
policies, simulation, and decoding here. A deferred lanes import must have a
comment explaining a real optional-dependency or type-only requirement;
working around a lanes-to-Gemini cycle is not acceptable.

Lanes owns and registers the shared compiler method tables, including validation,
qubit-address analysis, Steane type inference, and measurement analysis. Compiler
callers do not need to import Gemini to register them, and Gemini must not
register duplicate implementations.

Gemini retains its Cirq emission tables (`emit.cirq`) and the device-only
extension-rejection table (`lanes.validate.no_extensions`). These belong to
Gemini's conversion and device policies, not the shared compiler registration.
