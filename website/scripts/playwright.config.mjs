// @ts-check
/**
 * playwright.config.mjs — configuration for the docs runtime crawl.
 *
 * This is the tunable config for the Playwright-based per-page runtime crawler
 * (`scripts/crawl_runtime.mjs`). It is a plain ESM config module consumed by
 * that script — NOT the `@playwright/test` runner (the crawl is a standalone
 * script so it can enumerate ~1000 built pages, pool a single browser across
 * workers, and print one aggregate pass/fail report). Every value can be
 * overridden by the matching environment variable so CI can tune throughput
 * without editing code.
 */

import os from 'node:os';

const num = (envVal, dflt) => {
  const n = Number(envVal);
  return Number.isFinite(n) && n > 0 ? n : dflt;
};

export default {
  // Static preview server (astro preview) — host + fixed port. astro preview
  // auto-increments to the next free port if PORT is taken; the crawler parses
  // the actual URL from the server banner, so a clash is handled gracefully.
  host: process.env.CRAWL_HOST ?? '127.0.0.1',
  port: num(process.env.CRAWL_PORT, 4322),

  // Concurrency: number of parallel browser contexts crawling the page list.
  // The crawl is localhost-I/O-bound, so a handful of workers saturates it.
  workers: num(process.env.CRAWL_CONCURRENCY, Math.min(8, os.cpus().length || 4)),

  // Per-navigation timeout and the post-load settle window that lets deferred
  // module scripts (mermaid, katex, the version switcher, …) run and throw.
  navTimeoutMs: num(process.env.CRAWL_NAV_TIMEOUT, 30000),
  settleMs: num(process.env.CRAWL_SETTLE, 400),

  // Optional cap for local debugging. When >0 the crawler processes only the
  // first N pages and LOGS exactly which pages were skipped (no silent
  // truncation). 0 / unset = crawl every page.
  limit: num(process.env.CRAWL_LIMIT, 0),

  // How long to wait for the preview server banner before giving up (ms).
  serverStartTimeoutMs: num(process.env.CRAWL_SERVER_TIMEOUT, 60000),

  // Chromium launch options.
  launchOptions: {
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  },
};
