/**
 * Source — a compact "view source" link. FROZEN prop contract.
 */
import { describe, expect, it } from 'vitest';
import Source from '@components/api/Source.astro';
import { render } from '../../render';

describe('Source', () => {
  it('renders a new-tab, noopener link to href with the default "source" label', async () => {
    const html = await render(Source, { props: { href: 'https://github.com/x/y/blob/main/z.py#L1' } });
    expect(html).toContain('class="api-source"');
    expect(html).toContain('href="https://github.com/x/y/blob/main/z.py#L1"');
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener noreferrer"');
    expect(html).toMatch(/<span[^>]*>source<\/span>/);
    // The inline SVG icon is present (no external icon dependency).
    expect(html).toContain('class="api-source__icon"');
  });

  it('honors a custom label', async () => {
    const html = await render(Source, { props: { href: 'https://example.com', label: 'view on GitHub' } });
    expect(html).toMatch(/<span[^>]*>view on GitHub<\/span>/);
  });
});
