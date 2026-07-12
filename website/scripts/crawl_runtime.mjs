// @ts-check
/**
 * crawl_runtime.mjs — per-page RUNTIME error crawl for the Bloqade docs site.
 *
 * Catches "errors that only happen when you open a certain page": things a
 * static HTML/link check can't see, because they only surface once a real
 * browser parses, styles and RUNS the page. This is the "(b) per-page runtime
 * errors" half of `mise run docs:test`. It is SEPARATE from `docs:check`
 * (link/xref reachability) and complements it.
 *
 * WHAT IT DOES
 *   1. Enumerates EVERY built page by walking dist/**\/*.html (index.html for
 *      every route + the top-level 404.html + the /api/latest redirect stubs),
 *      reusing check_links.mjs's page-enumeration approach (base handling,
 *      pagefind/_astro skip, redirect-stub detection).
 *   2. Serves dist/ with `astro preview` on a fixed port and parses the real
 *      server URL (incl. any base) from its banner.
 *   3. Loads each page in headless Chromium (a pool of parallel workers) and
 *      collects, per page:
 *        · console messages of type `error`      -> ERROR (fails)
 *        · console messages of type `warning`    -> WARNING (reported, no fail)
 *        · uncaught `pageerror` exceptions        -> ERROR (fails)
 *        · unhandled promise rejections           -> ERROR (fails)
 *        · same-origin `requestfailed`            -> ERROR (fails)
 *        · same-origin HTTP responses status>=400 -> ERROR (fails)
 *      External/offline resources are EXCLUDED (see sameOrigin + route block).
 *   4. FAILS (exit 1) if any page produced an error-class problem; prints the
 *      offending pages + details. Warnings are printed separately and never fail.
 *
 * USAGE   node scripts/crawl_runtime.mjs [distDir=dist]
 * Tunables (port, workers, timeouts, limit) live in scripts/playwright.config.mjs
 * and are all env-overridable. Requires the `playwright` dev dep + its chromium
 * browser (`npx playwright install chromium`).
 */

import { spawn } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import CONFIG from './playwright.config.mjs';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const WEBSITE_ROOT = resolve(SCRIPT_DIR, '..');

const distArg = process.argv[2] || 'dist';
const DIST = resolve(process.cwd(), distArg);
if (!existsSync(DIST) || !statSync(DIST).isDirectory()) {
  console.error(`crawl_runtime: dist directory not found: ${DIST}`);
  process.exit(2);
}

// Site base handling — identical to check_links.mjs so a based deploy build
// (SITE_BASE=/next/ etc.) is crawled correctly. astro preview serves the site
// UNDER `base`, but the on-disk file tree + our page paths are base-stripped,
// so we splice `base` back on when building request URLs. `base` comes from
// $SITE_BASE if set, else is auto-detected from the built index.html's /_astro/
// asset URL (so this does NOT depend on the preview banner echoing the base).
function normalizeBase(raw) {
  if (!raw || raw === '/') return '/';
  const withLead = raw.startsWith('/') ? raw : `/${raw}`;
  return withLead.endsWith('/') ? withLead : `${withLead}/`;
}
function detectBase() {
  const idx = join(DIST, 'index.html');
  if (existsSync(idx)) {
    const m = /(?:href|src)="([^"]*\/)_astro\//.exec(readFileSync(idx, 'utf8'));
    if (m) return normalizeBase(m[1]);
  }
  return '/';
}
const BASE = process.env.SITE_BASE ? normalizeBase(process.env.SITE_BASE) : detectBase();
const BASE_PREFIX = BASE === '/' ? '' : BASE.slice(0, -1); // "/next" (no trailing slash)

// ---------------------------------------------------------------------------
// RUNTIME ALLOWLIST — benign console noise that must NOT fail the crawl.
//
// START EMPTY. Only add an entry with a written justification (like the build
// allowlist in check_build.mjs). Each entry: { class?, match, why }.
//   · match  — RegExp tested against the problem's text.
//   · class  — optional; restrict to a problem class ('console.error', ...).
// A captured ERROR whose text (and class, if given) matches an entry is
// downgraded to "allowlisted" (counted, printed, not failed).
// ---------------------------------------------------------------------------
/** @type {{class?:string, match:RegExp, why:string}[]} */
const RUNTIME_ALLOWLIST = [
  // (empty — the clean tree is runtime-error-free)
];

// Structural noise filter (NOT a user allowlist): Chromium logs a console.error
// mirroring every failed subresource ("Failed to load resource: …"). The
// AUTHORITATIVE signal for a failed resource is the requestfailed/response
// listeners, which already do same-origin filtering — so these mirror lines are
// dropped to avoid double-counting and to swallow external/offline resource
// noise. A genuine same-origin asset failure is still caught by those listeners.
const CONSOLE_RESOURCE_MIRROR = /Failed to load resource/i;

// ---------------------------------------------------------------------------
// page enumeration (mirrors check_links.mjs: skip pagefind/_astro, redirect
// stubs via <meta http-equiv=refresh>, dir/index.html served at "/dir/").
// ---------------------------------------------------------------------------

/** Recursively collect *.html under `dir` (skipping non-navigable trees). */
function walkHtml(dir, acc = /** @type {string[]} */ ([])) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (name === 'pagefind' || name === '_astro') continue;
      walkHtml(p, acc);
    } else if (name.endsWith('.html')) {
      acc.push(p);
    }
  }
  return acc;
}

/** Base-stripped served URL path for a source file (dir/index.html -> /dir/). */
function pageUrlPath(absFile) {
  let rel = relative(DIST, absFile).split('\\').join('/');
  if (rel.endsWith('index.html')) rel = rel.slice(0, -'index.html'.length);
  return '/' + rel.replace(/^\/+/, '');
}

/** True when a file is a meta-refresh redirect stub (e.g. /api/latest). */
function isRedirectStub(absFile) {
  const html = readFileSync(absFile, 'utf8');
  return /<meta[^>]*http-equiv=["']?refresh["']?/i.test(html);
}

function enumeratePages() {
  const files = walkHtml(DIST).sort();
  return files.map((file) => ({
    file,
    rel: relative(DIST, file).split('\\').join('/'),
    path: pageUrlPath(file),
    redirect: isRedirectStub(file),
  }));
}

// ---------------------------------------------------------------------------
// preview server (astro preview) — spawn + parse the real base URL
// ---------------------------------------------------------------------------

function startPreview() {
  return new Promise((resolvePromise, rejectPromise) => {
    const args = ['exec', 'astro', 'preview', '--port', String(CONFIG.port), '--host', CONFIG.host];
    const child = spawn('pnpm', args, { cwd: WEBSITE_ROOT, env: process.env });
    let out = '';
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill('SIGTERM');
      rejectPromise(new Error(`astro preview did not report a URL within ${CONFIG.serverStartTimeoutMs}ms`));
    }, CONFIG.serverStartTimeoutMs);

    const onData = (d) => {
      out += d.toString();
      // Astro prints a banner line like:  "Local    http://127.0.0.1:4322/"
      const m = /(https?:\/\/[^\s]+)/.exec(out);
      if (m && !settled) {
        settled = true;
        clearTimeout(timer);
        resolvePromise({ child, url: m[1] });
      }
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('exit', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      rejectPromise(new Error(`astro preview exited early (code ${code}). Output:\n${out}`));
    });
  });
}

// ---------------------------------------------------------------------------
// crawl
// ---------------------------------------------------------------------------

/** @typedef {{class:string, text:string}} Problem */

async function crawl() {
  const t0 = Date.now();
  const all = enumeratePages();
  const redirectCount = all.filter((p) => p.redirect).length;

  // optional cap (documented, no silent truncation)
  let pages = all;
  let skipped = [];
  if (CONFIG.limit && CONFIG.limit < all.length) {
    pages = all.slice(0, CONFIG.limit);
    skipped = all.slice(CONFIG.limit);
    console.warn(
      `crawl_runtime: CRAWL_LIMIT=${CONFIG.limit} — crawling ${pages.length}/${all.length} page(s); ` +
        `SKIPPING ${skipped.length}:`,
    );
    for (const s of skipped) console.warn(`  skipped: ${s.path}  (${s.rel})`);
  }

  console.log(
    `crawl_runtime: ${all.length} built page(s) enumerated ` +
      `(${redirectCount} redirect stub(s)); crawling ${pages.length} with ${CONFIG.workers} worker(s).`,
  );

  const { child: preview, url: serverBaseUrl } = await startPreview();
  const serverOrigin = new URL(serverBaseUrl).origin;
  // Build request URLs from the parsed ORIGIN + the base detected from dist,
  // not from the banner's path — robust whether or not astro echoes the base.
  const urlBase = serverOrigin + BASE_PREFIX; // e.g. http://127.0.0.1:4322  (+ "/next")
  console.log(
    `crawl_runtime: preview server at ${serverBaseUrl} (origin ${serverOrigin}, base ${BASE}).`,
  );

  const sameOrigin = (u) => {
    try {
      return new URL(u).origin === serverOrigin;
    } catch {
      return false;
    }
  };

  /** @type {{page:{path:string,rel:string,redirect:boolean}, errors:Problem[], warnings:Problem[], allowlisted:Problem[]}[]} */
  const results = [];
  let processed = 0;

  const browser = await chromium.launch(CONFIG.launchOptions);

  // shared work queue (single-threaded JS => plain counter is safe)
  let idx = 0;
  const nextPage = () => (idx < pages.length ? pages[idx++] : null);

  async function worker() {
    const context = await browser.newContext();

    // Block external/offline requests: keep the crawl deterministic (works
    // offline in CI) and avoid hangs on third-party badges/links. External
    // resource failures are out of scope for the gate anyway.
    await context.route('**/*', (route) => {
      const u = route.request().url();
      if (sameOrigin(u) || u.startsWith('data:') || u.startsWith('blob:') || u.startsWith('about:')) {
        route.continue();
      } else {
        route.abort();
      }
    });

    // Unhandled promise rejections are not reliably surfaced by Playwright's
    // 'pageerror' event across versions, so capture them via an init script that
    // forwards to this exposed function. `state.sink` points at the current
    // page's problem list (workers are serial, so this is unambiguous).
    const state = { sink: /** @type {Problem[]|null} */ (null) };
    await context.exposeFunction('__harnessReport', (payload) => {
      if (!state.sink) return;
      const stack = payload && payload.stack ? `\n${payload.stack}` : '';
      state.sink.push({ class: 'unhandledrejection', text: `${payload && payload.message}${stack}` });
    });
    await context.addInitScript(() => {
      window.addEventListener('unhandledrejection', (e) => {
        try {
          const r = e && e.reason;
          const message = r && r.message ? r.message : String(r);
          const stack = r && r.stack ? r.stack : '';
          // @ts-ignore — injected by exposeFunction
          window.__harnessReport({ message, stack });
        } catch {
          /* ignore */
        }
      });
    });

    for (let pg = nextPage(); pg; pg = nextPage()) {
      /** @type {Problem[]} */ const errors = [];
      /** @type {Problem[]} */ const warnings = [];
      state.sink = errors;

      const page = await context.newPage();
      page.on('console', (msg) => {
        const type = msg.type();
        if (type !== 'error' && type !== 'warning') return;
        const text = msg.text();
        if (type === 'error') {
          if (CONSOLE_RESOURCE_MIRROR.test(text)) return; // handled by response/requestfailed
          errors.push({ class: 'console.error', text });
        } else {
          warnings.push({ class: 'console.warning', text });
        }
      });
      page.on('pageerror', (err) => {
        errors.push({ class: 'pageerror', text: (err && err.stack) || String(err) });
      });
      page.on('requestfailed', (req) => {
        const url = req.url();
        if (!sameOrigin(url)) return; // external/offline excluded (+ our own route aborts)
        const f = req.failure();
        const errText = f ? f.errorText : 'unknown';
        if (errText === 'net::ERR_ABORTED') return; // navigation/redirect abort, not a failure
        errors.push({ class: 'requestfailed', text: `${errText}  ${url}` });
      });
      page.on('response', (resp) => {
        const status = resp.status();
        if (status < 400) return;
        const url = resp.url();
        if (!sameOrigin(url)) return; // external >=400 excluded
        errors.push({ class: `http-${status}`, text: `${status}  ${url}` });
      });

      const target = urlBase + pg.path;
      try {
        // Redirect stubs meta-refresh immediately; 'commit' lets us verify the
        // stub itself responds without racing the client-side redirect follow.
        await page.goto(target, {
          waitUntil: pg.redirect ? 'commit' : 'load',
          timeout: CONFIG.navTimeoutMs,
        });
        if (!pg.redirect) await page.waitForTimeout(CONFIG.settleMs);
      } catch (e) {
        const m = String((e && e.message) || e);
        // A stub navigating away can abort its own goto — expected, not a failure.
        if (!(pg.redirect && /ERR_ABORTED|interrupt|navigation/i.test(m))) {
          errors.push({ class: 'goto', text: m });
        }
      }
      await page.close();
      state.sink = null;

      // apply the runtime allowlist
      const kept = [];
      const allowed = [];
      for (const p of errors) {
        const hit = RUNTIME_ALLOWLIST.find(
          (a) => (!a.class || a.class === p.class) && a.match.test(p.text),
        );
        (hit ? allowed : kept).push(p);
      }

      results.push({ page: pg, errors: kept, warnings, allowlisted: allowed });
      processed++;
      if (processed % 100 === 0 || processed === pages.length) {
        console.log(`crawl_runtime: ${processed}/${pages.length} page(s) crawled …`);
      }
    }

    await context.close();
  }

  try {
    await Promise.all(Array.from({ length: CONFIG.workers }, () => worker()));
  } finally {
    await browser.close();
    preview.kill('SIGTERM');
  }

  return { results, all, skipped, redirectCount, elapsedMs: Date.now() - t0 };
}

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

const { results, all, skipped, redirectCount, elapsedMs } = await crawl();

const failing = results.filter((r) => r.errors.length > 0);
const withWarnings = results.filter((r) => r.warnings.length > 0);
const withAllowlisted = results.filter((r) => r.allowlisted.length > 0);
const secs = (elapsedMs / 1000).toFixed(1);

console.log(
  `\ncrawl_runtime: crawled ${results.length} of ${all.length} built page(s) ` +
    `(${redirectCount} redirect stub(s)) in ${secs}s ` +
    `[${CONFIG.workers} worker(s)]` +
    (skipped.length ? `; SKIPPED ${skipped.length} (CRAWL_LIMIT)` : ''),
);

if (withWarnings.length) {
  const total = withWarnings.reduce((n, r) => n + r.warnings.length, 0);
  console.log(`\nWARNINGS (non-failing) on ${withWarnings.length} page(s), ${total} message(s):`);
  for (const r of withWarnings) {
    console.log(`  ⚠ ${r.page.path}`);
    for (const w of r.warnings) console.log(`      [${w.class}] ${w.text.split('\n')[0]}`);
  }
}

if (withAllowlisted.length) {
  const total = withAllowlisted.reduce((n, r) => n + r.allowlisted.length, 0);
  console.log(`\nALLOWLISTED (suppressed) on ${withAllowlisted.length} page(s), ${total} message(s).`);
}

if (failing.length === 0) {
  console.log('\ncrawl_runtime OK: no per-page runtime errors across all crawled pages.');
  process.exit(0);
}

const totalErrors = failing.reduce((n, r) => n + r.errors.length, 0);
console.error(`\ncrawl_runtime FAIL: ${totalErrors} runtime error(s) on ${failing.length} page(s):\n`);
for (const r of failing) {
  console.error(`  ✗ ${r.page.path}   (${r.page.rel})`);
  for (const e of r.errors) {
    const first = e.text.split('\n')[0];
    console.error(`      [${e.class}] ${first}`);
    for (const extra of e.text.split('\n').slice(1, 4)) console.error(`          ${extra}`);
  }
}
console.error(`\nFAIL: ${totalErrors} runtime error(s) on ${failing.length}/${results.length} crawled page(s).`);
process.exit(1);
