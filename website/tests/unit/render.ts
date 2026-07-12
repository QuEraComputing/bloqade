/**
 * Tiny wrapper around Astro's experimental Container API for rendering a single
 * `.astro` component to an HTML string in isolation (no full build, no browser).
 *
 * Usage:
 *   const html = await render(ApiFn, { props: { ... }, slots: { default: '...' } });
 *
 * NOTE: the Container renders in a dev-like mode, so the HTML carries extra
 * `data-astro-source-file` / `data-astro-cid-*` debug attributes that a
 * production build strips. Assertions therefore match on stable structure
 * (class names, ids, text, attribute values) rather than exact whole-tag
 * equality.
 */
import type { AstroComponentFactory } from 'astro/runtime/server/index.js';
import { experimental_AstroContainer as AstroContainer } from 'astro/container';

export interface RenderOptions {
  props?: Record<string, unknown>;
  slots?: Record<string, unknown>;
  /** Sets `Astro.url` — pass a full URL so components can read the pathname. */
  request?: Request;
}

export async function render(
  Component: AstroComponentFactory,
  { props = {}, slots = {}, request }: RenderOptions = {},
): Promise<string> {
  const container = await AstroContainer.create();
  return container.renderToString(Component, { props, slots, request });
}
