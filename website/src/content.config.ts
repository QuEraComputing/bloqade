import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

/**
 * Phase C2 — preserve dotted path segments in docs slugs.
 *
 * Astro's default `generateId` runs `github-slugger` on every path segment,
 * which STRIPS the dot from a version directory (`api/0.35/...` -> `api/035/`).
 * That would break the versioned API layout: the emitters build inventory URLs
 * as `/api/0.35/python/...` (mount = the literal version label), so the built
 * route MUST keep the dot or every release-version xref + switcher link 404s.
 *
 * Our docs slugs are already URL-safe (module names, the version label, the
 * rust crate — all word chars / dots / underscores), so we generate the id
 * verbatim from the file path (extension + trailing `/index` stripped) instead
 * of slugifying. Explicit `slug` frontmatter still wins. Hand-written pages are
 * unaffected (their filenames are already lowercase, space-free).
 */
function generateId({ entry, data }: { entry: string; data: Record<string, unknown> }): string {
  if (typeof data?.slug === 'string' && data.slug) return data.slug;
  const path = entry.replace(/\\/g, '/').replace(/\.[^./]+$/, ''); // strip extension
  if (path === 'index') return '';
  return path.replace(/\/index$/, '');
}

/**
 * Custom frontmatter fields for generated API pages.
 *
 * THIS IS PART OF THE CONTRACT — downstream emitters (Python/Rust -> MDX) set
 * these fields and downstream components/plugins read them. All fields are
 * OPTIONAL so ordinary hand-written docs pages are unaffected.
 */
const apiFrontmatter = z.object({
  /** Which language this API page documents. */
  language: z.enum(['python', 'rust']).optional(),
  /** Fully-qualified symbol/module id, e.g. "bloqade.squin.kernel". */
  fqName: z.string().optional(),
  /** API doc version, e.g. "dev", "0.35". */
  apiVersion: z.string().optional(),
  /** Source repo slug, e.g. "QuEraComputing/bloqade-circuit". */
  sourceRepo: z.string().optional(),
  /** Source git ref (tag/branch/sha) the page was generated from. */
  sourceRef: z.string().optional(),
  /** Canonical URL to the source (file/line) on the forge. */
  sourceUrl: z.string().optional(),
});

export const collections = {
  // Starlight docs collection. Loads from src/content/docs and extends the
  // default Starlight docs schema with our optional API frontmatter fields.
  docs: defineCollection({
    loader: docsLoader({ generateId }),
    schema: docsSchema({ extend: apiFrontmatter }),
  }),

  // Blog collection. Content comes in a later phase; an empty dir + schema is
  // enough for Phase A.
  blog: defineCollection({
    loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
    schema: z.object({
      title: z.string(),
      date: z.coerce.date(),
      authors: z.array(z.string()).optional(),
      excerpt: z.string().optional(),
    }),
  }),
};
