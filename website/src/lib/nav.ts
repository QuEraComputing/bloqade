/**
 * Shared navigation source of truth.
 *
 * Consumed by BOTH the bespoke SiteLayout chrome (landing + blog, via
 * SiteHeader/SiteFooter) AND the Starlight docs Header/Footer overrides, so the
 * two chromes render literally the same links. Do NOT duplicate these arrays in
 * the components — import from here.
 *
 * Note: hrefs are root-relative (no `base` prefix), matching the historical
 * SiteLayout behaviour. Production deploys at `base: '/'`.
 */
export interface NavLink {
  label: string;
  href: string;
  /** External links get rel="noopener"; internal links do not. */
  external?: boolean;
}

/** Primary top-nav links, shown in the header on every page. */
export const primaryNav: NavLink[] = [
  { label: 'Guides', href: '/guides/example/' },
  { label: 'API', href: '/api/' },
  { label: 'Blog', href: '/blog/' },
];

/** Canonical GitHub repo link (header icon + footer). */
export const githubUrl = 'https://github.com/QuEraComputing/bloqade';

/** Footer link row (primary nav + external destinations). */
export const footerLinks: NavLink[] = [
  { label: 'Guides', href: '/guides/example/' },
  { label: 'API', href: '/api/' },
  { label: 'Blog', href: '/blog/' },
  { label: 'QuEra', href: 'https://quera.com', external: true },
  { label: 'GitHub', href: githubUrl, external: true },
];
