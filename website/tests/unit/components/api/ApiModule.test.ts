/**
 * ApiModule — wraps a generated module page. FROZEN prop contract.
 */
import { describe, expect, it } from 'vitest';
import ApiModule from '@components/api/ApiModule.astro';
import { render } from '../../render';

describe('ApiModule', () => {
  it('renders the module heading with id={fqName}, the fqName line, and slot members', async () => {
    const html = await render(ApiModule, {
      props: { name: 'kernel', fqName: 'bloqade.squin.kernel', summary: 'The squin kernel.' },
      slots: { default: '<p>member list</p>' },
    });
    expect(html).toMatch(/class="api-module__heading"[^>]*id="bloqade\.squin\.kernel"/);
    expect(html).toMatch(/<code[^>]*>kernel<\/code>/);
    expect(html).toMatch(/class="api-module__fqname"[\s\S]*?>bloqade\.squin\.kernel</);
    expect(html).toContain('data-fq-name="bloqade.squin.kernel"');
    expect(html).toMatch(/class="api-module__summary"[^>]*>The squin kernel\./);
    expect(html).toContain('<p>member list</p>');
    expect(html).toContain('href="#bloqade.squin.kernel"'); // permalink
  });

  it('omits the summary paragraph when not provided', async () => {
    const html = await render(ApiModule, {
      props: { name: 'kernel', fqName: 'bloqade.squin.kernel' },
    });
    expect(html).not.toContain('api-module__summary');
  });
});
