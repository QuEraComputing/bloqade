/**
 * Returns — a callable's return value (compact labeled block). FROZEN contract.
 */
import { describe, expect, it } from 'vitest';
import Returns from '@components/api/Returns.astro';
import { render } from '../../render';

describe('Returns', () => {
  it('renders the type as inline code and the description as raw HTML', async () => {
    const html = await render(Returns, {
      props: { type: 'BatchResult', description: 'the aggregated <em>result</em>' },
    });
    expect(html).toContain('>Returns<');
    expect(html).toMatch(/class="api-returns__type"[^>]*>BatchResult</);
    expect(html).toMatch(/class="api-returns__description"[^>]*>the aggregated <em>result<\/em></);
  });

  it('renders the type alone when there is no description', async () => {
    const html = await render(Returns, { props: { type: 'int' } });
    expect(html).toMatch(/class="api-returns__type"[^>]*>int</);
    expect(html).not.toContain('api-returns__description');
  });

  it('renders the description alone when there is no type', async () => {
    const html = await render(Returns, { props: { description: 'nothing useful' } });
    expect(html).not.toContain('api-returns__type');
    expect(html).toContain('nothing useful');
  });
});
