/**
 * ApiXref — cross-reference to another API symbol by fully-qualified id.
 *
 * The component resolves the reference at render time via `resolveXref` from
 * `@/lib/xref`, which imports the generated, gitignored inventory. To test the
 * component's RESOLUTION-DRIVEN markup hermetically (independent of any built
 * inventory), we `vi.mock('@/lib/xref')` with a controlled resolver:
 *   * a known id -> a fake url  => href = url,          class api-xref--resolved
 *   * an unknown id -> null      => href = '#' + to,     class api-xref--unresolved
 *
 * `@/lib/xref` resolves to the SAME module the component imports as
 * `@/lib/xref`, so the mock intercepts it. The real resolveXref ALGORITHM is
 * covered separately (hermetically) in tests/unit/lib/xref.test.ts.
 */
import { describe, expect, it, vi } from 'vitest';

const RESOLVED_URL = '/api/dev/python/bloqade/task/#bloqade.task.BatchFuture.result';

vi.mock('@/lib/xref', () => ({
  resolveXref: (to: string) =>
    to === 'bloqade.task.BatchFuture.result' ? RESOLVED_URL : null,
  hasXref: (to: string) => to === 'bloqade.task.BatchFuture.result',
}));

import ApiXref from '@components/api/ApiXref.astro';
import { render } from '../../render';

describe('ApiXref', () => {
  it('resolved: uses the target url + api-xref--resolved, keeping data-xref-to', async () => {
    const html = await render(ApiXref, {
      props: { to: 'bloqade.task.BatchFuture.result', label: 'result' },
    });
    expect(html).toContain('class="api-xref api-xref--resolved"');
    expect(html).toContain(`href="${RESOLVED_URL}"`);
    expect(html).toContain('data-xref-to="bloqade.task.BatchFuture.result"');
    expect(html).toMatch(/>result<\/a>/); // custom label
    expect(html).not.toContain('api-xref--unresolved');
  });

  it('unresolved: falls back to href="#<to>" + api-xref--unresolved', async () => {
    const html = await render(ApiXref, { props: { to: 'bloqade.gone.Missing' } });
    expect(html).toContain('class="api-xref api-xref--unresolved"');
    expect(html).toContain('href="#bloqade.gone.Missing"');
    expect(html).toContain('data-xref-to="bloqade.gone.Missing"');
    // Label defaults to the fully-qualified id when omitted.
    expect(html).toMatch(/>bloqade\.gone\.Missing<\/a>/);
    expect(html).not.toContain('api-xref--resolved');
  });

  it('defaults the visible label to `to` even when resolved', async () => {
    const html = await render(ApiXref, { props: { to: 'bloqade.task.BatchFuture.result' } });
    expect(html).toMatch(/>bloqade\.task\.BatchFuture\.result<\/a>/);
  });

  it('renders data-xref-origin only when the origin prop is provided', async () => {
    const html = await render(ApiXref, {
      props: { to: 'bloqade.task.BatchFuture.result', label: 'result', origin: 'docstring' },
    });
    expect(html).toContain('data-xref-origin="docstring"');
    // origin is purely informational: resolution + classes are unchanged.
    expect(html).toContain('class="api-xref api-xref--resolved"');
    expect(html).toContain('data-xref-to="bloqade.task.BatchFuture.result"');
  });

  it('omits data-xref-origin when the origin prop is absent (backward compatible)', async () => {
    const html = await render(ApiXref, { props: { to: 'bloqade.gone.Missing' } });
    expect(html).not.toContain('data-xref-origin');
    // Unchanged unresolved behavior.
    expect(html).toContain('class="api-xref api-xref--unresolved"');
    expect(html).toContain('href="#bloqade.gone.Missing"');
  });

  it('origin is orthogonal to resolution: also renders on an unresolved ref', async () => {
    const html = await render(ApiXref, {
      props: { to: 'bloqade.gone.Missing', origin: 'docstring' },
    });
    expect(html).toContain('data-xref-origin="docstring"');
    expect(html).toContain('class="api-xref api-xref--unresolved"');
  });
});
