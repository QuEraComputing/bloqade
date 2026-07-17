/**
 * nav.ts — the shared navigation source of truth consumed by BOTH the bespoke
 * SiteHeader/SiteFooter and the Starlight Header/Footer overrides. Pure data, so
 * these tests just pin the contract (labels, hrefs, internal-vs-external flag).
 */
import { describe, expect, it } from 'vitest';
import { footerLinks, githubUrl, primaryNav, type NavLink } from '@/lib/nav';

describe('primaryNav', () => {
  it('lists Guides, API, Blog with root-relative hrefs and no external flag', () => {
    expect(primaryNav.map((l) => l.label)).toEqual(['Guides', 'API', 'Blog']);
    const byLabel = Object.fromEntries(primaryNav.map((l) => [l.label, l]));
    expect(byLabel.Guides.href).toBe('/guides/example/');
    expect(byLabel.API.href).toBe('/api/');
    expect(byLabel.Blog.href).toBe('/blog/');
    for (const l of primaryNav) expect(l.external).toBeUndefined();
  });
});

describe('githubUrl', () => {
  it('points at the QuEraComputing/bloqade repo', () => {
    expect(githubUrl).toBe('https://github.com/QuEraComputing/bloqade');
  });
});

describe('footerLinks', () => {
  it('extends the primary nav with QuEra + GitHub external links', () => {
    expect(footerLinks.map((l) => l.label)).toEqual(['Guides', 'API', 'Blog', 'QuEra', 'GitHub']);
    const byLabel = Object.fromEntries(footerLinks.map((l) => [l.label, l]));
    expect(byLabel.QuEra.external).toBe(true);
    expect(byLabel.GitHub.external).toBe(true);
    expect(byLabel.GitHub.href).toBe(githubUrl);
  });

  it('invariant: external links are absolute (https), internal are root-relative', () => {
    const check = (l: NavLink) =>
      l.external ? l.href.startsWith('https://') : l.href.startsWith('/');
    expect([...primaryNav, ...footerLinks].every(check)).toBe(true);
  });
});
