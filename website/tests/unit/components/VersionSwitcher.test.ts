/**
 * VersionSwitcher — the API version dropdown (Phase C2).
 *
 * We test the SERVER-RENDERED behaviour reachable via `Astro.url` (set through
 * the Container's `request`) + the manifest:
 *   * self-hides off `/api/<version>/**` (renders no switcher markup);
 *   * on a versioned API page, renders one <option> per version and marks the
 *     one derived from the URL segment as `selected`.
 *
 * HERMETIC: `loadVersions` is mocked with a controlled manifest (its real impl
 * reads the gitignored generated manifest); `currentApiVersion` — the actual
 * version-derivation logic — is kept REAL via importOriginal.
 *
 * CLIENT-ONLY (NOT covered here): the `<script>` that navigates to the same
 * sub-path under the chosen version on `change` runs only in the browser, so it
 * is exercised by the Playwright runtime crawl (`mise run docs:test`), not this
 * unit gate.
 */
import { describe, expect, it, vi } from 'vitest';

const { MANIFEST } = vi.hoisted(() => ({
  MANIFEST: {
    latest: '0.35',
    dev: 'dev',
    versions: [
      { label: '0.35', path: '/api/0.35', latest: true, dev: false, release: true },
      { label: 'dev', path: '/api/dev', latest: false, dev: true, release: false },
    ],
  },
}));

vi.mock('@/lib/versions', async (orig) => {
  const actual = await orig<typeof import('@/lib/versions')>();
  return { ...actual, loadVersions: () => MANIFEST };
});

import VersionSwitcher from '@components/VersionSwitcher.astro';
import { render } from '../render';

const at = (pathname: string) =>
  render(VersionSwitcher, { request: new Request(`https://bloqade.test${pathname}`) });

describe('VersionSwitcher visibility', () => {
  it('renders the switcher on a versioned API page', async () => {
    const html = await at('/api/dev/python/bloqade/');
    expect(html).toContain('class="version-switcher"');
    expect(html).toContain('id="api-version-select"');
  });

  it('hides itself on the evergreen /api/ landing', async () => {
    const html = await at('/api/');
    expect(html).not.toContain('class="version-switcher"');
    expect(html).not.toContain('<select');
  });

  it('hides itself on /api/compatibility/ (non-version segment)', async () => {
    const html = await at('/api/compatibility/');
    expect(html).not.toContain('class="version-switcher"');
  });

  it('hides itself off the API tree (guides/blog/home)', async () => {
    for (const p of ['/guides/example/', '/blog/', '/']) {
      const html = await at(p);
      expect(html).not.toContain('class="version-switcher"');
    }
  });
});

describe('VersionSwitcher options', () => {
  it('renders one option per version with (dev)/(latest) suffixes', async () => {
    const html = await at('/api/dev/python/');
    expect(html).toMatch(/<option value="0\.35"[^>]*>0\.35 \(latest\)<\/option>/);
    expect(html).toMatch(/<option value="dev"[^>]*>dev \(dev\)<\/option>/);
  });

  it('marks the version derived from the URL segment as selected (dev)', async () => {
    const html = await at('/api/dev/python/bloqade/');
    expect(html).toMatch(/<option value="dev"[^>]*\sselected[^>]*>/);
    expect(html).not.toMatch(/<option value="0\.35"[^>]*\sselected/);
  });

  it('marks the concrete version as selected (0.35)', async () => {
    const html = await at('/api/0.35/rust/');
    expect(html).toMatch(/<option value="0\.35"[^>]*\sselected[^>]*>/);
    expect(html).not.toMatch(/<option value="dev"[^>]*\sselected/);
  });

  it('resolves /api/latest to the concrete latest (0.35) and marks it selected', async () => {
    const html = await at('/api/latest/python/');
    expect(html).toMatch(/<option value="0\.35"[^>]*\sselected[^>]*>/);
  });
});
