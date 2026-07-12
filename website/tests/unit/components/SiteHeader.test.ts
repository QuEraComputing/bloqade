/**
 * SiteHeader — the SINGLE source of the top-chrome markup, shared by the landing
 * (SiteLayout) and the docs (Starlight Header override). We render it directly
 * (the Starlight `Header.astro` wrapper that feeds it `<Search/>` needs
 * Starlight's route context, so that thin wrapper is covered by the build/runtime
 * harness instead).
 *
 * Assertions cross-check the rendered nav against nav.ts (the shared data), so a
 * drift between the two would fail here.
 *
 * CLIENT-ONLY (NOT covered here): the theme-toggle `<script>` (writes
 * localStorage + flips <html data-theme>) runs only in the browser -> covered by
 * the Playwright runtime crawl.
 */
import { describe, expect, it } from 'vitest';
import SiteHeader from '@components/SiteHeader.astro';
import { githubUrl, primaryNav } from '@/lib/nav';
import { render } from '../render';

describe('SiteHeader', () => {
  it('renders the brand home link and both (light/dark) logos', async () => {
    const html = await render(SiteHeader);
    expect(html).toMatch(/class="site-brand"[^>]*href="\/"/);
    expect(html).toContain('site-logo-light');
    expect(html).toContain('site-logo-dark');
  });

  it('renders every primaryNav link with its exact href + label', async () => {
    const html = await render(SiteHeader);
    for (const link of primaryNav) {
      // Tie the label to its href so a mismatch (wrong label on a link) fails.
      const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      expect(html).toMatch(new RegExp(`href="${esc(link.href)}"[^>]*>\\s*${esc(link.label)}\\s*<`));
    }
  });

  it('renders the GitHub icon link (noopener) and the theme toggle button', async () => {
    const html = await render(SiteHeader);
    expect(html).toMatch(new RegExp(`href="${githubUrl.replace(/[.]/g, '\\.')}"[^>]*rel="noopener"`));
    expect(html).toContain('data-theme-toggle');
  });

  it('omits the search region by default, and renders it (with slot) when withSearch', async () => {
    const without = await render(SiteHeader);
    expect(without).not.toContain('site-header__search');

    const withSearch = await render(SiteHeader, {
      props: { withSearch: true },
      slots: { search: '<div id="SEARCH-SLOT"></div>' },
    });
    expect(withSearch).toContain('site-header__search');
    expect(withSearch).toContain('id="SEARCH-SLOT"');
  });
});
