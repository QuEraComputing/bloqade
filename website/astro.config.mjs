// @ts-check
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mdx from '@astrojs/mdx';
import AutoImport from 'astro-auto-import';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
// Phase C1 post-build integration: regenerates the merged xref inventory
// (astro:config:setup), reports unresolved <ApiXref>s + emits a Sphinx v2
// objects.inv (astro:build:done). Ordered LAST so it does not disturb the
// delicate [AutoImport(), starlight(), mdx()] sequence below.
import xrefIntegration from './src/integrations/xref.mjs';

// Path aliases, kept in sync with tsconfig.json `compilerOptions.paths`.
const src = fileURLToPath(new URL('./src', import.meta.url));
const components = fileURLToPath(new URL('./src/components', import.meta.url));

// Phase C2 — the `latest` API alias. `src/generated/versions.json` is written
// by scripts/build_versions.py and is GITIGNORED (regenerated from whatever
// `api/<V>` trees exist), so it may be absent on a fresh clone: read it
// defensively and fall back to `dev`. `latest` maps `/api/latest/**` onto the
// newest release's concrete version via the `redirects` below.
function readLatestVersion() {
  const file = fileURLToPath(new URL('./src/generated/versions.json', import.meta.url));
  try {
    if (existsSync(file)) {
      const manifest = JSON.parse(readFileSync(file, 'utf8'));
      if (manifest && typeof manifest.latest === 'string') return manifest.latest;
    }
  } catch {
    /* fall through to default */
  }
  return 'dev';
}
const LATEST_VERSION = readLatestVersion();

// Phase D2 — env-driven deploy target so the SAME build artifact can be
// published at the site root (production) or under a gh-pages preview subpath
// (e.g. `/next/` for the WIP branch or `/astro-preview/pr-N/` for a PR). Both
// vars are injected by the `.github/workflows/website-*.yml` pipelines and both
// default to the production values, so local builds and `astro dev` are
// unchanged. `base` MUST match the publish subpath or every asset/link 404s.
//
// `base` is normalized to always carry a leading AND trailing slash (Astro's
// canonical form) so `withBase()` can splice it onto redirect destinations
// without doubling slashes.
function normalizeBase(raw) {
  if (!raw || raw === '/') return '/';
  const withLead = raw.startsWith('/') ? raw : `/${raw}`;
  return withLead.endsWith('/') ? withLead : `${withLead}/`;
}
const SITE_BASE = normalizeBase(process.env.SITE_BASE);
const SITE_URL = process.env.SITE_URL ?? 'https://bloqade.quera.com';

// Astro prefixes the redirect SOURCE keys with `base` automatically (the
// generated meta-refresh page is emitted at `<base>/api/latest/…`), but it does
// NOT prefix the string DESTINATION — so we splice `base` on here. No-op when
// base is `/`. Verified against `SITE_BASE=/next/ astro build` output.
const withBase = (path) =>
  SITE_BASE === '/' ? path : `${SITE_BASE.slice(0, -1)}${path}`;

// The shared API component contract (see src/components/api/). These names are
// auto-imported into every .mdx file so generated API pages need NO import lines.
const API_COMPONENTS = [
  'ApiModule',
  'ApiClass',
  'ApiFn',
  'Signature',
  'Params',
  'Returns',
  'Raises',
  'Source',
  'ApiXref',
];

// https://astro.build/config
export default defineConfig({
  site: SITE_URL,
  base: SITE_BASE,

  // Phase C2 — `/api/latest/**` aliases onto the newest release's concrete
  // version (LATEST_VERSION, read from versions.json) so the sidebar links
  // never need editing on a new release. Static keys only: a spread-param
  // redirect (`/api/latest/[...slug]`) is not enumerable in a static build, and
  // the <VersionSwitcher> already navigates between CONCRETE versions, so only
  // these landing aliases are needed. Astro emits them as meta-refresh pages.
  redirects: {
    '/api/latest': withBase(`/api/${LATEST_VERSION}/python/`),
    '/api/latest/python': withBase(`/api/${LATEST_VERSION}/python/`),
    '/api/latest/rust': withBase(`/api/${LATEST_VERSION}/rust/`),
  },

  // Global markdown pipeline. `@astrojs/mdx` reads these at `astro:config:done`
  // (after every integration's setup hook), so KaTeX + auto-import + Starlight
  // plugins are all collected regardless of integration ordering.
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },

  integrations: [
    // IMPORTANT ordering: [AutoImport(), starlight(), mdx()].
    //   * AutoImport must be first: it bails out unless `@astrojs/mdx` is
    //     already present in the integrations array when its setup hook runs,
    //     so an explicit `mdx()` must exist somewhere in this array.
    //   * The explicit `mdx()` is listed AFTER `starlight()`: Starlight inserts
    //     its own built-ins (astro-expressive-code, sitemap, ...) immediately
    //     after itself, and astro-expressive-code REQUIRES being ordered before
    //     `@astrojs/mdx`. Putting our mdx() after starlight() keeps EC < mdx.
    //   * Starlight only adds its own `mdx()` if one isn't already present, so
    //     our explicit mdx() prevents a double-registration.
    //   * `@astrojs/mdx` reads `markdown.remarkPlugins` at config:done (after
    //     every setup hook), so the remark plugin AutoImport injects AND
    //     Starlight's plugins are all picked up regardless of ordering.
    AutoImport({
      imports: [
        {
          // Bare alias (kept verbatim in the generated import) -> resolved by
          // the `@components` Vite alias below. Emits into each .mdx:
          //   import { ApiModule, ApiClass, ... } from "@components/api";
          '@components/api': API_COMPONENTS,
        },
      ],
    }),
    starlight({
      title: 'Bloqade',
      // Starlight has no dedicated "tagline" field; the tagline lives in the
      // site description (used for meta + the default hero on splash pages).
      description: 'The Neutral Atom SDK',
      logo: {
        light: './src/assets/logo.svg',
        dark: './src/assets/logo-dark.svg',
        replacesTitle: true,
      },
      favicon: '/favicon.ico',
      customCss: [
        // Fonts first, then KaTeX styles, then brand overrides (loaded last so
        // brand.css wins the cascade).
        '@fontsource/lato/400.css',
        '@fontsource/lato/700.css',
        'katex/dist/katex.min.css',
        './src/styles/brand.css',
      ],
      social: [
        { icon: 'x.com', label: 'X', href: 'https://x.com/QueraComputing' },
        {
          icon: 'linkedin',
          label: 'LinkedIn',
          href: 'https://www.linkedin.com/company/quera-computing-inc/',
        },
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/QuEraComputing/bloqade',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/QuEraComputing/bloqade/edit/main/website/',
      },
      // Mermaid support: override Starlight's <Head> to also inject a bundled,
      // code-split client-side renderer (see src/components/MermaidHead.astro).
      // This is the LOW-RISK path — no build-time browser deps (rehype-mermaid /
      // Playwright) and it does NOT touch the delicate integration ordering.
      components: {
        Head: './src/components/MermaidHead.astro',
        // Phase C2 — wrap the sidebar to mount the API <VersionSwitcher> at its
        // top (the switcher self-hides off `/api/<version>/**`).
        Sidebar: './src/components/Sidebar.astro',
        // Design unification — replace the top/bottom chrome with the bespoke
        // brand header/footer (shared with SiteLayout via SiteHeader/SiteFooter).
        // Header composes Starlight's <Search /> so the search modal + ⌘K keep
        // working; the mobile menu toggle is rendered by PageFrame (not the
        // Header) so mobile navigation is unaffected. Footer renders Starlight's
        // default footer (prev/next pagination) above the brand footer.
        Header: './src/components/Header.astro',
        Footer: './src/components/Footer.astro',
      },
      sidebar: [
        // Top-level links back to the bespoke (non-Starlight) routes so docs
        // readers can reach the landing page and the blog.
        { label: 'Home', link: '/' },
        { label: 'Blog', link: '/blog/' },
        {
          label: 'Guides',
          autogenerate: { directory: 'guides' },
        },
        {
          // Phase C2 — the API reference is versioned (single `bloqade`
          // meta-version axis). These links point at the `/api/latest/**`
          // alias (redirected to the newest release) + the evergreen landing
          // and compatibility pages; the <VersionSwitcher> handles per-version
          // navigation. `api/<V>` trees are generated + gitignored, so this
          // group is explicit links (NOT autogenerate, which would break on a
          // fresh clone with no generated content).
          label: 'API Reference',
          items: [
            { label: 'Overview', link: '/api/' },
            { label: 'Python (latest)', link: '/api/latest/python' },
            { label: 'Rust (latest)', link: '/api/latest/rust' },
            { label: 'Compatibility', link: '/api/compatibility/' },
          ],
        },
        {
          label: 'Reference',
          autogenerate: { directory: 'reference' },
        },
      ],
    }),
    // Explicit mdx() AFTER starlight() (see ordering note above). Present so
    // AutoImport doesn't bail and so Starlight doesn't double-register mdx.
    mdx(),
    // Post-build only (config:setup + build:done). Safe to append here — it
    // registers no markdown/vite plugins that depend on the ordering above.
    xrefIntegration(),
  ],

  vite: {
    resolve: {
      alias: {
        '@components': components,
        '@': src,
      },
    },
  },
});
