# Ownership

`bloqade.lanes` owns the compiler — dialects, rewrites, analysis.
`bloqade.gemini` owns the product — interfaces, simulation, user flow.
Gemini uses lanes to implement those; lanes owns the infrastructure that makes it possible.

Import compiler definitions from lanes. Keep product kernel decorators, device
policies, simulation, and decoding here. A deferred lanes import must have a
comment explaining a real optional-dependency or type-only requirement;
working around a lanes-to-Gemini cycle is not acceptable.

Gemini owns its frontend method tables, including validation implementations.
Import them before constructing kernel groups. Validation passes/analyses remain
in lanes; generic lanes move/place and arch evaluation tables stay there.
