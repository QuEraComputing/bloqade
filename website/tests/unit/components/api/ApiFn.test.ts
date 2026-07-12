/**
 * ApiFn — generated function/method/property entry. The prop contract is FROZEN
 * (emitters depend on it), so these assertions pin the rendered structure that
 * makes each entry linkable + searchable.
 */
import { describe, expect, it } from 'vitest';
import ApiFn from '@components/api/ApiFn.astro';
import { render } from '../../render';

describe('ApiFn', () => {
  it('renders the short name + kind badge, and the fqName as the searchable line', async () => {
    const html = await render(ApiFn, {
      props: { name: 'result', fqName: 'bloqade.task.BatchFuture.result', kind: 'method' },
      slots: { default: '<p>docstring body</p>' },
    });
    // Kind badge shows the kind with its modifier class.
    expect(html).toContain('class="api-badge api-badge--method"');
    expect(html).toMatch(/api-badge api-badge--method[^>]*>method</);
    // Short name in the monospace heading name.
    expect(html).toMatch(/class="api-fn__name"[^>]*>result</);
    // The fully-qualified name is rendered as its own visible line (searchable).
    expect(html).toMatch(/class="api-fn__fqname"[\s\S]*?>bloqade\.task\.BatchFuture\.result</);
    // Slot body renders.
    expect(html).toContain('<p>docstring body</p>');
  });

  it('sets id={fqName} on the heading and mirrors it in data-fq-name/data-kind', async () => {
    const html = await render(ApiFn, {
      props: { name: 'run', fqName: 'bloqade.task.run', kind: 'function' },
    });
    expect(html).toContain('id="bloqade.task.run"');
    expect(html).toContain('data-fq-name="bloqade.task.run"');
    expect(html).toContain('data-kind="function"');
    // Permalink anchor points at the same id.
    expect(html).toContain('href="#bloqade.task.run"');
  });

  it('defaults kind to "function" when omitted', async () => {
    const html = await render(ApiFn, {
      props: { name: 'run', fqName: 'bloqade.task.run' },
    });
    expect(html).toContain('api-badge--function');
    expect(html).toContain('data-kind="function"');
  });

  it('renders a Source link only when sourceUrl is given', async () => {
    const withSource = await render(ApiFn, {
      props: { name: 'run', fqName: 'bloqade.task.run', sourceUrl: 'https://github.com/x/y#L10' },
    });
    expect(withSource).toContain('class="api-fn__source"');
    expect(withSource).toContain('class="api-source"');
    expect(withSource).toContain('href="https://github.com/x/y#L10"');

    const withoutSource = await render(ApiFn, {
      props: { name: 'run', fqName: 'bloqade.task.run' },
    });
    expect(withoutSource).not.toContain('api-fn__source');
    expect(withoutSource).not.toContain('class="api-source"');
  });
});
