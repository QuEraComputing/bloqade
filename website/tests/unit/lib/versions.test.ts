/**
 * versions.ts — version manifest loading + current-version-from-path.
 *
 * `currentApiVersion` is a PURE function of (pathname, manifest), so it is
 * tested exhaustively against hand-built manifests — fully hermetic. `loadVersions`
 * reads the (gitignored, possibly-absent) generated manifest via `import.meta.glob`
 * with a committed-safe default fallback; we assert the STRUCTURAL invariants it
 * must always satisfy, which hold whether it reads a real manifest or the default.
 */
import { describe, expect, it } from 'vitest';
import { currentApiVersion, loadVersions, type VersionsManifest } from '@/lib/versions';

const MANIFEST: VersionsManifest = {
  latest: '0.35',
  dev: 'dev',
  versions: [
    { label: '0.35', path: '/api/0.35', latest: true, dev: false, release: true },
    { label: 'dev', path: '/api/dev', latest: false, dev: true, release: false },
  ],
};

describe('currentApiVersion', () => {
  it('returns the concrete version segment on a versioned API page', () => {
    expect(currentApiVersion('/api/dev/python/bloqade/', MANIFEST)).toBe('dev');
    expect(currentApiVersion('/api/0.35/python/', MANIFEST)).toBe('0.35');
    expect(currentApiVersion('/api/0.35', MANIFEST)).toBe('0.35'); // no trailing slash
  });

  it('resolves the `latest` alias segment to the concrete latest label', () => {
    expect(currentApiVersion('/api/latest/python/bloqade/', MANIFEST)).toBe('0.35');
    expect(currentApiVersion('/api/latest', MANIFEST)).toBe('0.35');
  });

  it('returns null for the evergreen /api/ landing (no version segment)', () => {
    expect(currentApiVersion('/api/', MANIFEST)).toBeNull();
    expect(currentApiVersion('/api', MANIFEST)).toBeNull();
  });

  it('returns null for non-version API segments (e.g. compatibility)', () => {
    expect(currentApiVersion('/api/compatibility/', MANIFEST)).toBeNull();
  });

  it('returns null for an unknown version segment', () => {
    expect(currentApiVersion('/api/9.99/python/', MANIFEST)).toBeNull();
  });

  it('returns null off the API tree entirely (keeps the switcher hidden)', () => {
    expect(currentApiVersion('/guides/example/', MANIFEST)).toBeNull();
    expect(currentApiVersion('/', MANIFEST)).toBeNull();
    expect(currentApiVersion('/blog/', MANIFEST)).toBeNull();
  });
});

describe('loadVersions', () => {
  it('returns a structurally valid manifest', () => {
    const m = loadVersions();
    expect(typeof m.latest).toBe('string');
    expect(m.latest.length).toBeGreaterThan(0);
    expect(Array.isArray(m.versions)).toBe(true);
    expect(m.versions.length).toBeGreaterThan(0);
    for (const v of m.versions) {
      expect(typeof v.label).toBe('string');
      expect(v.path.startsWith('/api/')).toBe(true);
      expect(typeof v.latest).toBe('boolean');
      expect(typeof v.dev).toBe('boolean');
      expect(typeof v.release).toBe('boolean');
    }
  });

  it('keeps `latest` consistent — the latest label is one of the built versions', () => {
    const m = loadVersions();
    expect(m.versions.some((v) => v.label === m.latest)).toBe(true);
  });

  it('composes with currentApiVersion: /api/latest resolves to a real version', () => {
    const m = loadVersions();
    const resolved = currentApiVersion('/api/latest/python/', m);
    expect(resolved).toBe(m.latest);
    expect(m.versions.some((v) => v.label === resolved)).toBe(true);
  });
});
