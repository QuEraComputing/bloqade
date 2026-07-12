/**
 * Signature — syntax-highlighted signature block via Astro's built-in Shiki
 * `<Code>` (astro:components). FROZEN prop contract.
 *
 * "Highlighted" is asserted via the Shiki output markers: a `<pre class="astro-code
 * css-variables">` whose tokens carry `--astro-code-token-*` custom properties
 * (the css-variables theme). The collapsible variant wraps the block in a native
 * `<details>` disclosure.
 */
import { describe, expect, it } from 'vitest';
import Signature from '@components/api/Signature.astro';
import { render } from '../../render';

describe('Signature', () => {
  it('highlights python and keeps a short signature as a plain (non-details) block', async () => {
    const html = await render(Signature, { props: { code: 'def f(x): ...', lang: 'python' } });
    expect(html).toContain('data-lang="python"');
    expect(html).toContain('class="astro-code css-variables"');
    expect(html).toContain('data-language="python"');
    // Highlighting: the `def` keyword token gets the keyword custom property.
    expect(html).toMatch(/--astro-code-token-keyword[^>]*>def</);
    // Short + single-line => rendered as a div, not a <details>.
    expect(html).toContain('<div class="api-signature"');
    expect(html).not.toContain('<details');
  });

  it('highlights rust and tags the block with data-lang="rust"', async () => {
    const html = await render(Signature, { props: { code: 'fn foo() -> u32 { 0 }', lang: 'rust' } });
    expect(html).toContain('data-lang="rust"');
    expect(html).toContain('data-language="rust"');
    expect(html).toMatch(/--astro-code-token-keyword[^>]*>fn</);
  });

  it('defaults lang to python', async () => {
    const html = await render(Signature, { props: { code: 'x = 1' } });
    expect(html).toContain('data-lang="python"');
  });

  it('wraps a long (multiline) signature in a collapsible <details> disclosure', async () => {
    const code = 'def f(\n    a: int,\n    b: int,\n    c: int,\n) -> int: ...';
    const html = await render(Signature, { props: { code, lang: 'python' } });
    expect(html).toContain('<details class="api-signature"');
    expect(html).toContain('class="api-signature__summary"');
    expect(html).toContain('Signature'); // summary label
  });

  it('wraps in <details> when collapsible=true even for a short signature', async () => {
    const html = await render(Signature, {
      props: { code: 'def f(x): ...', lang: 'python', collapsible: true },
    });
    expect(html).toContain('<details class="api-signature"');
    expect(html).toContain('class="api-signature__summary"');
  });
});
