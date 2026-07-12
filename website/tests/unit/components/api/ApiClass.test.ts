/**
 * ApiClass — generated class entry. FROZEN prop contract.
 */
import { describe, expect, it } from 'vitest';
import ApiClass from '@components/api/ApiClass.astro';
import { render } from '../../render';

describe('ApiClass', () => {
  it('renders the class badge, short name, and fqName searchable line + id', async () => {
    const html = await render(ApiClass, {
      props: { name: 'BatchFuture', fqName: 'bloqade.task.BatchFuture' },
      slots: { default: '<p>members</p>' },
    });
    expect(html).toMatch(/class="api-badge api-badge--class"[^>]*>class</);
    expect(html).toMatch(/class="api-class__name"[^>]*>BatchFuture</);
    expect(html).toMatch(/class="api-class__fqname"[\s\S]*?>bloqade\.task\.BatchFuture</);
    expect(html).toContain('id="bloqade.task.BatchFuture"');
    expect(html).toContain('data-fq-name="bloqade.task.BatchFuture"');
    expect(html).toContain('<p>members</p>');
  });

  it('renders the Bases line with each base as inline code, joined by commas', async () => {
    const html = await render(ApiClass, {
      props: { name: 'C', fqName: 'm.C', bases: ['Base', 'Mixin'] },
    });
    expect(html).toContain('class="api-class__bases"');
    expect(html).toContain('Bases:');
    expect(html).toMatch(/<code[^>]*>Base<\/code>/);
    expect(html).toMatch(/<code[^>]*>Mixin<\/code>/);
    // Comma separator between the two bases.
    expect(html.replace(/\s+/g, ' ')).toMatch(/Base<\/code> , <code/);
  });

  it('omits the Bases line when there are no bases', async () => {
    const html = await render(ApiClass, { props: { name: 'C', fqName: 'm.C', bases: [] } });
    expect(html).not.toContain('api-class__bases');
    const noneGiven = await render(ApiClass, { props: { name: 'C', fqName: 'm.C' } });
    expect(noneGiven).not.toContain('api-class__bases');
  });

  it('renders the summary when given, and a Source link when sourceUrl is given', async () => {
    const html = await render(ApiClass, {
      props: {
        name: 'C',
        fqName: 'm.C',
        summary: 'A one-line summary.',
        sourceUrl: 'https://github.com/x/y#L1',
      },
    });
    expect(html).toMatch(/class="api-class__summary"[^>]*>A one-line summary\./);
    expect(html).toContain('class="api-class__source"');
    expect(html).toContain('href="https://github.com/x/y#L1"');
  });

  it('omits summary + source when not provided', async () => {
    const html = await render(ApiClass, { props: { name: 'C', fqName: 'm.C' } });
    expect(html).not.toContain('api-class__summary');
    expect(html).not.toContain('api-class__source');
  });
});
