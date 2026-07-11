/**
 * Shared API component library — the emitter <-> Astro contract.
 *
 * These components are auto-imported into every `.mdx` file (see the
 * `astro-auto-import` config in astro.config.mjs), so generated API pages can
 * use `<ApiFn>`, `<Signature>`, etc. with NO per-file import lines.
 *
 * Downstream emitters (Python griffe -> MDX, Rust rustdoc-json -> MDX) MUST
 * only rely on the prop shapes documented in each component file.
 */
export { default as ApiModule } from './ApiModule.astro';
export { default as ApiClass } from './ApiClass.astro';
export { default as ApiFn } from './ApiFn.astro';
export { default as Signature } from './Signature.astro';
export { default as Params } from './Params.astro';
export { default as Returns } from './Returns.astro';
export { default as Raises } from './Raises.astro';
export { default as Source } from './Source.astro';
export { default as ApiXref } from './ApiXref.astro';
