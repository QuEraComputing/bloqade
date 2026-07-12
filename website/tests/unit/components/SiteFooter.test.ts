/**
 * SiteFooter — the SINGLE source of the brand-footer markup, shared by the
 * landing (SiteLayout) and the docs (Starlight Footer override). Rendered
 * directly here; the Starlight `Footer.astro` wrapper (which also renders
 * Starlight's prev/next pager) needs route context -> covered by the build
 * harness.
 *
 * Cross-checks the rendered links against footerLinks (the shared data): every
 * link renders with its exact href, external links carry rel="noopener", and
 * internal links do NOT.
 */
import { describe, expect, it } from 'vitest';
import SiteFooter from '@components/SiteFooter.astro';
import { footerLinks } from '@/lib/nav';
import { render } from '../render';

describe('SiteFooter', () => {
  it('renders the brand tagline and the current-year copyright', async () => {
    const html = await render(SiteFooter);
    expect(html).toContain('The Neutral Atom SDK, by QuEra Computing.');
    // The © is emitted as the &copy; HTML entity.
    expect(html).toContain(`&copy; ${new Date().getFullYear()} QuEra Computing Inc.`);
    expect(html).toContain('Apache 2.0');
  });

  it('renders every footer link with its href, and the right rel per internal/external', async () => {
    const html = await render(SiteFooter);
    for (const link of footerLinks) {
      // Each link's anchor, matched non-greedily up to its close tag.
      const anchor = new RegExp(`<a href="${link.href.replace(/[.]/g, '\\.')}"[^>]*>`, 'i');
      expect(html).toMatch(anchor);
      const tag = html.match(anchor)![0];
      if (link.external) {
        expect(tag).toContain('rel="noopener"');
      } else {
        expect(tag).not.toContain('rel=');
      }
    }
  });
});
