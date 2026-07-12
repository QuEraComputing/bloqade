/**
 * Params — parameter table. FROZEN prop contract. Key behaviour: the Default
 * column is omitted entirely unless at least one item declares a default;
 * descriptions are injected as raw HTML (set:html).
 */
import { describe, expect, it } from 'vitest';
import Params from '@components/api/Params.astro';
import { render } from '../../render';

// Count `<th ...>` header cells (the `[\s>]` guard avoids matching `<thead>`).
const countTh = (h: string) => (h.match(/<th[\s>]/g) ?? []).length;

describe('Params', () => {
  it('renders a row per item with name, type, and (HTML) description', async () => {
    const html = await render(Params, {
      props: {
        items: [
          { name: 'shots', type: 'int', description: 'number of <em>shots</em>' },
          { name: 'seed', type: 'int | None', description: 'the seed' },
        ],
      },
    });
    expect(html).toMatch(/class="api-table__name"[^>]*><code[^>]*>shots</);
    expect(html).toMatch(/class="api-table__type"[\s\S]*?<code[^>]*>int</);
    expect(html).toMatch(/class="api-table__name"[^>]*><code[^>]*>seed</);
    expect(html).toContain('int | None');
    // Description injected as raw HTML, not escaped.
    expect(html).toContain('number of <em>shots</em>');
  });

  it('omits the Default column entirely when no item declares a default', async () => {
    const html = await render(Params, {
      props: { items: [{ name: 'x', type: 'int' }, { name: 'y', type: 'str' }] },
    });
    expect(html).not.toContain('>Default<');
    expect(html).not.toContain('api-table__default');
    // Three header cells: Name, Type, Description.
    expect(countTh(html)).toBe(3);
  });

  it('shows the Default column when any item has a default, marking others "required"', async () => {
    const html = await render(Params, {
      props: {
        items: [
          { name: 'x', type: 'int', default: '5', description: 'has default' },
          { name: 'y', type: 'str', description: 'no default' },
        ],
      },
    });
    expect(html).toContain('>Default<');
    // Four header cells: Name, Type, Default, Description.
    expect(countTh(html)).toBe(4);
    // The item WITH a default shows its value in the default cell.
    expect(html).toMatch(/class="api-table__default"[\s\S]*?<code[^>]*>5</);
    // The item WITHOUT a default is marked required.
    expect(html).toMatch(/class="api-table__muted"[^>]*>required</);
  });

  it('honors a custom title (also used as the scroll-container aria-label)', async () => {
    const html = await render(Params, {
      props: { title: 'Attributes', items: [{ name: 'x', type: 'int' }] },
    });
    expect(html).toMatch(/class="api-params__title"[^>]*>Attributes</);
    expect(html).toContain('aria-label="Attributes"');
  });
});
