/**
 * versions.ts — read the generated API version manifest (Phase C2).
 *
 * `src/generated/versions.json` is written by `scripts/build_versions.py` and
 * is GITIGNORED (it is regenerated from whatever `api/<V>` trees exist). It may
 * therefore be absent on a fresh clone that has not run build_versions yet.
 *
 * We load it with Vite's `import.meta.glob` (eager) rather than a static
 * `import` or an `import.meta.url`-relative `fs` read: the glob resolves
 * relative to THIS source file at build time (reliable even after the component
 * is bundled), and gracefully yields NOTHING when the file is absent — so we
 * transparently fall back to a committed-safe default.
 */

export interface VersionEntry {
  /** Version label, e.g. "dev" or "0.35". */
  label: string;
  /** Mount path, e.g. "/api/0.35". */
  path: string;
  /** True for the newest release (what `/api/latest` aliases). */
  latest: boolean;
  /** True for the dev line (built from main). */
  dev: boolean;
  /** True when the label looks like a release (vs. dev). */
  release: boolean;
}

export interface VersionsManifest {
  /** Label the `latest` alias points at. */
  latest: string;
  /** The dev label if a dev build exists, else null. */
  dev: string | null;
  /** All built versions, releases newest-first then dev. */
  versions: VersionEntry[];
}

/** Committed-safe default used when versions.json has not been generated. */
const DEFAULT_MANIFEST: VersionsManifest = {
  latest: 'dev',
  dev: 'dev',
  versions: [{ label: 'dev', path: '/api/dev', latest: true, dev: true, release: false }],
};

// Eagerly glob-import the (optional) generated manifest. Matches 0 or 1 file;
// the value is the parsed JSON (default export). Empty when it hasn't been
// generated yet, in which case loadVersions() returns DEFAULT_MANIFEST.
const generated = import.meta.glob<VersionsManifest>('../generated/versions.json', {
  eager: true,
  import: 'default',
});

/** Load the version manifest, falling back to the default when absent/invalid. */
export function loadVersions(): VersionsManifest {
  const found = Object.values(generated)[0];
  if (found && Array.isArray(found.versions) && found.versions.length > 0) {
    return found;
  }
  return DEFAULT_MANIFEST;
}

/**
 * Given a pathname, return the current API version segment if the page is a
 * versioned API page (`/api/<version>/...`), else null. `latest` is resolved to
 * the concrete latest label. Non-version segments (the evergreen `/api/`
 * landing, `/api/compatibility/`) return null so the switcher stays hidden.
 */
export function currentApiVersion(pathname: string, manifest: VersionsManifest): string | null {
  const m = /^\/api\/([^/]+)(?:\/|$)/.exec(pathname);
  if (!m) return null;
  const seg = m[1];
  if (seg === 'latest') return manifest.latest;
  const known = manifest.versions.some((v) => v.label === seg);
  return known ? seg : null;
}
