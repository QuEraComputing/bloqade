/**
 * Raises — exception table. FROZEN prop contract.
 */
import { describe, expect, it } from 'vitest';
import Raises from '@components/api/Raises.astro';
import { render } from '../../render';

describe('Raises', () => {
  it('renders a Type/Description row per exception, with HTML descriptions', async () => {
    const html = await render(Raises, {
      props: {
        items: [
          { type: 'ValueError', description: 'when <code>shots</code> is negative' },
          { type: 'RuntimeError', description: 'on backend failure' },
        ],
      },
    });
    expect(html).toContain('class="api-raises"');
    expect(html).toContain('>Raises<');
    // Header cells.
    expect(html).toContain('>Type<');
    expect(html).toContain('>Description<');
    // Rows.
    expect(html).toMatch(/class="api-table__type"[^>]*><code[^>]*>ValueError</);
    expect(html).toMatch(/class="api-table__type"[^>]*><code[^>]*>RuntimeError</);
    // Description injected as raw HTML (the inline <code> tag is NOT escaped).
    expect(html).toContain('when <code>shots</code> is negative');
    expect(html).toContain('on backend failure');
  });

  it('renders an empty table body for no items (header still present)', async () => {
    const html = await render(Raises, { props: { items: [] } });
    expect(html).toContain('>Raises<');
    expect(html).toContain('>Type<');
    expect(html).not.toContain('api-table__type');
  });
});
