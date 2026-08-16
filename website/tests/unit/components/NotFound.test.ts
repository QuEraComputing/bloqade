/**
 * NotFound — the on-brand 404 body AND the `/api/latest/<symbol>` deep-link
 * fallback embedded by 404.mdx.
 *
 * SERVER-RENDERED assertions (this gate):
 *   * the friendly 404 markup + the three CTA links render;
 *   * the deep-link fallback <script> is present with the BAKED `latest` (from
 *     the version manifest, via define:vars) and `base` (from BASE_URL), plus
 *     the `/api/latest/<rest>` rewrite regex.
 *
 * HERMETIC: `loadVersions` is mocked to return a DISTINCTIVE latest ("9.9")
 * that differs from any on-disk manifest, so the assertion proves the bake
 * flows from the manifest (not from a stale generated file). `base` is "/" —
 * vitest's default `import.meta.env.BASE_URL`.
 *
 * CLIENT-ONLY (NOT covered here): the script's actual URL rewrite runs in the
 * browser on a real 404 hit; that end-to-end redirect is exercised by the
 * Playwright runtime crawl (`mise run docs:test`, which loads 404.html) and by
 * `scripts/check_links.mjs` (which reimplements the same rewrite for /api/latest
 * deep links). We assert the baked INPUTS + the rewrite pattern here.
 */
import { describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/versions', async (orig) => {
  const actual = await orig<typeof import('@/lib/versions')>();
  return {
    ...actual,
    loadVersions: () => ({
      latest: '9.9',
      dev: 'dev',
      versions: [{ label: '9.9', path: '/api/9.9', latest: true, dev: false, release: true }],
    }),
  };
});

import NotFound from '@components/NotFound.astro';
import { render } from '../render';

describe('NotFound friendly 404', () => {
  it('renders the on-brand 404 body', async () => {
    const html = await render(NotFound);
    expect(html).toContain('class="notfound"');
    expect(html).toMatch(/class="notfound__code"[^>]*>404</);
    expect(html).toContain('This page could not be found');
  });

  it('renders the three CTA links (Home, Guides, API reference)', async () => {
    const html = await render(NotFound);
    expect(html).toMatch(/href="\/"[^>]*>Home<\/a>/);
    expect(html).toMatch(/href="\/guides\/"[^>]*>Guides<\/a>/);
    expect(html).toMatch(/href="\/api\/"[^>]*>API reference<\/a>/);
  });
});

describe('NotFound /api/latest deep-link fallback script', () => {
  it('bakes the concrete latest label from the version manifest', async () => {
    const html = await render(NotFound);
    expect(html).toContain('const latest = "9.9"');
  });

  it('bakes the deploy base (root "/" under vitest)', async () => {
    const html = await render(NotFound);
    expect(html).toContain('const base = "/"');
  });

  it('contains the /api/latest/<rest> rewrite matcher', async () => {
    const html = await render(NotFound);
    // The regex that matches deep links to be forwarded to the concrete version.
    expect(html).toContain('/^\\/api\\/latest\\/(.+)$/');
    // ...and it forwards to /api/<latest>/... preserving search + hash.
    expect(html).toContain("'api/' + latest + '/'");
    expect(html).toContain('window.location.search');
    expect(html).toContain('window.location.hash');
  });
});
