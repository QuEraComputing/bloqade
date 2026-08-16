/// <reference types="vitest" />
/**
 * vitest.config.ts — fast, isolated UNIT tests for the docs site's components
 * and lib helpers.
 *
 * This is the THIRD docs gate, deliberately separate from (and much faster
 * than) the two build-based harnesses:
 *   * `mise run docs:check` — strict-xref build + internal link/anchor crawl.
 *   * `mise run docs:test`  — build-warning gate + Playwright runtime crawl.
 *   * `mise run docs:unit`  — THIS: `vitest run`, no browser, no full build.
 *
 * We build the vite config via Astro's own `getViteConfig` so that:
 *   * `.astro` components compile (Astro's core compiler vite plugin is added),
 *   * `astro:components` (the `<Code>` used by Signature) resolves,
 *   * `.svg` asset imports (`import logo from '../assets/logo.svg'`) resolve to
 *     an ImageMetadata object, and
 *   * CSS side-effect imports (`import './api.css'`) are handled (not executed
 *     as JS), so component modules load cleanly under SSR.
 *
 * We pass `{ configFile: false }` so it does NOT load the real
 * `astro.config.mjs`: that would boot the full Starlight integration and run
 * the xref integration's `astro:config:setup` hook (which regenerates
 * `src/generated/xref-inventory.json`). Unit tests must be side-effect-free and
 * fast, and none of the components under test need Starlight's config to
 * COMPILE — the few thin Starlight-wrapper overrides (Header/Footer/Sidebar/
 * MermaidHead) that need Starlight's route context at RENDER time are covered
 * by the build/runtime harness instead (see tests/unit/README notes).
 *
 * The `@` / `@components` path aliases mirror astro.config.mjs `vite.resolve`
 * and tsconfig.json `compilerOptions.paths`, supplied here directly since we
 * skip the config file.
 */
import { fileURLToPath } from 'node:url';
import { getViteConfig } from 'astro/config';

const src = fileURLToPath(new URL('./src', import.meta.url));
const components = fileURLToPath(new URL('./src/components', import.meta.url));

export default getViteConfig(
  {
    resolve: {
      alias: {
        '@components': components,
        '@': src,
      },
    },
    test: {
      // Node env is enough: the Container API renders components to an HTML
      // STRING, which we assert on directly — no jsdom/browser DOM required.
      environment: 'node',
      globalSetup: ['./tests/unit/global-setup.ts'],
      include: ['tests/unit/**/*.test.ts', 'src/**/*.test.ts'],
    },
  },
  { configFile: false },
);
