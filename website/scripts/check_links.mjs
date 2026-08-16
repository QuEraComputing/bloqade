// @ts-check
/**
 * check_links.mjs — post-build internal link + fragment-anchor reachability
 * check for the Bloqade docs site (Astro 5 + Starlight).
 *
 * Zero runtime deps (node: builtins only). Scans the generated HTML and FAILS
 * (exit 1) if any internal link, fragment anchor, or resolved API cross-
 * reference points at something that does not exist. Content-agnostic: it
 * reports breakage, it does not edit anything.
 *
 * USAGE
 *   node scripts/check_links.mjs [distDir=dist]
 *
 *   SITE_BASE=/next/ node scripts/check_links.mjs dist   # match a deploy subpath
 *
 * WHAT IT CHECKS (only `<a href>` navigational links — not asset <link>/<script>)
 *   • root-relative      /guides/x/            -> <dist>/guides/x/index.html
 *   • relative           ../y/                 (resolved against the source page)
 *   • extensionless      /guides/x             -> tries file, /index.html, .html
 *   • fragments          /page/#anchor, #anchor-> the target has id= / name= anchor
 *   • redirect stubs     /api/latest[/python]  -> follows the <meta refresh> to its
 *                                                 destination and validates THAT
 *   • deep latest links  /api/latest/<deep>    -> VALID when the runtime 404
 *                                                 fallback target /api/<latest>/<deep>
 *                                                 exists (see notes below)
 *
 * WHAT IT SKIPS
 *   external (http/https), protocol-relative (//host), mailto:, tel:, data:,
 *   javascript:, and bare `#` / `?query` (same-page) links.
 *
 * ------------------------------------------------------------------------
 * SITE `base`
 *   Astro prefixes every internal link with the configured `base` (`/` in
 *   production, e.g. `/next/` or `/astro-preview/pr-N/` for gh-pages previews).
 *   The on-disk file tree does NOT include that prefix, so we strip it before
 *   resolving. `base` is taken from $SITE_BASE if set, else auto-detected from
 *   the built index.html's `/_astro/` asset URL.
 *
 * `/api/latest/**` RUNTIME FALLBACK  (why deep latest links are not flagged)
 *   Only three `/api/latest*` landing pages are real redirect stubs
 *   (astro.config.mjs `redirects`): `/api/latest`, `/api/latest/python`,
 *   `/api/latest/rust`. DEEP links such as `/api/latest/python/bloqade/foo/`
 *   cannot be enumerated in a static build, so they are served by
 *   `src/content/docs/404.mdx` (dist/404.html), whose inline script rewrites
 *   `/api/latest/<rest>` -> `/api/<latest>/<rest>` at runtime. This checker
 *   mirrors that: a deep `/api/latest/<rest>` link is VALID iff
 *   `/api/<latest>/<rest>` exists on disk (+ its fragment). `<latest>` is read
 *   from src/generated/versions.json (fallback: "dev").
 *
 * IGNORE-LIST  (for genuinely-acceptable breakage)
 *   Optional file `scripts/check_links.ignore.json`:
 *       {
 *         "targets": ["^/api/rust/", "..."],   // regex, tested against the
 *                                              //   base-stripped target path
 *         "sources": ["^some/page/"]           // regex, tested against the
 *                                              //   source page's dist-relative path
 *       }
 *   A broken link is suppressed (counted as "ignored", not failing) when its
 *   target matches any `targets` regex OR its source matches any `sources`
 *   regex. Missing file / empty arrays = ignore nothing. Document every entry.
 * ------------------------------------------------------------------------
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const WEBSITE_ROOT = resolve(SCRIPT_DIR, '..');

// ---------------------------------------------------------------------------
// config / inputs
// ---------------------------------------------------------------------------

const distArg = process.argv[2] || 'dist';
const DIST = resolve(process.cwd(), distArg);

if (!existsSync(DIST) || !statSync(DIST).isDirectory()) {
  console.error(`check_links: dist directory not found: ${DIST}`);
  process.exit(2);
}

/** Normalize a base to Astro's canonical leading+trailing-slash form. */
function normalizeBase(raw) {
  if (!raw || raw === '/') return '/';
  const withLead = raw.startsWith('/') ? raw : `/${raw}`;
  return withLead.endsWith('/') ? withLead : `${withLead}/`;
}

/** Auto-detect the site base from a built page's /_astro/ asset URL. */
function detectBase() {
  const idx = join(DIST, 'index.html');
  if (existsSync(idx)) {
    const html = readFileSync(idx, 'utf8');
    const m = /(?:href|src)="([^"]*\/)_astro\//.exec(html);
    if (m) return normalizeBase(m[1]);
  }
  return '/';
}

const BASE = process.env.SITE_BASE ? normalizeBase(process.env.SITE_BASE) : detectBase();

/** Concrete label the `/api/latest` alias resolves to (e.g. "0.35" / "dev"). */
function readLatest() {
  const file = join(WEBSITE_ROOT, 'src', 'generated', 'versions.json');
  try {
    if (existsSync(file)) {
      const m = JSON.parse(readFileSync(file, 'utf8'));
      if (m && typeof m.latest === 'string') return m.latest;
    }
  } catch {
    /* fall through */
  }
  return 'dev'; // matches DEFAULT_MANIFEST in src/lib/versions.ts
}
const LATEST = readLatest();

/** Load the optional ignore-list config. */
function loadIgnore() {
  const file = join(SCRIPT_DIR, 'check_links.ignore.json');
  const out = { targets: /** @type {RegExp[]} */ ([]), sources: /** @type {RegExp[]} */ ([]) };
  if (!existsSync(file)) return out;
  try {
    const cfg = JSON.parse(readFileSync(file, 'utf8'));
    for (const s of cfg.targets ?? []) out.targets.push(new RegExp(s));
    for (const s of cfg.sources ?? []) out.sources.push(new RegExp(s));
  } catch (err) {
    console.warn(`check_links: could not read ignore-list: ${err}`);
  }
  return out;
}
const IGNORE = loadIgnore();

// Anchors that Starlight/Astro synthesize on every page — always considered
// present so we don't false-positive on `#_top` "back to top" links.
const SYNTHETIC_IDS = new Set(['_top', 'top']);

// ---------------------------------------------------------------------------
// filesystem helpers
// ---------------------------------------------------------------------------

/** Recursively collect *.html files under `dir`. */
function walkHtml(dir, acc = /** @type {string[]} */ ([])) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (name === 'pagefind' || name === '_astro') continue; // no navigable HTML
      walkHtml(p, acc);
    } else if (name.endsWith('.html')) {
      acc.push(p);
    }
  }
  return acc;
}

/** Per-file parse cache: { ids: Set<string>, refresh: string|null }. */
const fileCache = new Map();

/** @param {string} absFile */
function parseFile(absFile) {
  let cached = fileCache.get(absFile);
  if (cached) return cached;
  const html = readFileSync(absFile, 'utf8');

  const ids = new Set();
  const idRe = /\s(?:id|name)\s*=\s*(?:"([^"]*)"|'([^']*)')/gi;
  let m;
  while ((m = idRe.exec(html)) !== null) ids.add(m[1] ?? m[2]);

  // meta-refresh redirect stub: <meta http-equiv="refresh" content="0;url=...">
  let refresh = null;
  const meta = /<meta[^>]*http-equiv=["']?refresh["']?[^>]*content=["']([^"']*)["'][^>]*>/i.exec(
    html,
  );
  if (meta) {
    const u = /url=\s*(.+?)\s*$/i.exec(meta[1]);
    if (u) refresh = u[1];
  }

  cached = { ids, refresh };
  fileCache.set(absFile, cached);
  return cached;
}

// ---------------------------------------------------------------------------
// path resolution (all in base-STRIPPED space; paths start with "/")
// ---------------------------------------------------------------------------

/** Strip the site base prefix from an absolute-URL pathname. */
function stripBase(pathname) {
  if (BASE === '/') return pathname;
  const noSlash = BASE.slice(0, -1); // "/next"
  if (pathname === noSlash) return '/';
  if (pathname.startsWith(BASE)) return '/' + pathname.slice(BASE.length);
  if (pathname.startsWith(noSlash + '/')) return pathname.slice(noSlash.length);
  return pathname; // relative-resolved path: already base-free
}

/**
 * Map a base-stripped URL path to the file that serves it, or null.
 * Trailing-slash "ignore" semantics: try the exact file, then <path>/index.html,
 * then <path>.html.
 * @param {string} p url path beginning with "/"
 * @returns {string|null} absolute file path
 */
function resolveToFile(p) {
  let rel;
  try {
    rel = decodeURIComponent(p.replace(/^\/+/, ''));
  } catch {
    rel = p.replace(/^\/+/, '');
  }
  const candidates = p.endsWith('/')
    ? [join(rel, 'index.html')]
    : [rel, join(rel, 'index.html'), rel + '.html'];
  for (const c of candidates) {
    if (!c) continue;
    const abs = join(DIST, c);
    if (existsSync(abs) && statSync(abs).isFile()) return abs;
  }
  return null;
}

/**
 * Follow meta-refresh redirect stubs to the final target file.
 * @param {string} absFile
 * @returns {{ file: string|null, path: string }} final target file (null when a
 *   redirect points at a missing destination) + its url path
 */
function followRefresh(absFile, urlPath) {
  let file = absFile;
  let path = urlPath;
  for (let hops = 0; hops < 8; hops++) {
    const { refresh } = parseFile(file);
    if (!refresh) break;
    const dest = stripBase(refresh.split('#')[0].split('?')[0]);
    const next = resolveToFile(dest);
    if (!next) return { file: null, path: dest }; // dangling redirect; report dest
    file = next;
    path = dest;
  }
  return { file, path };
}

/** Verify a fragment id exists on the target file. */
function hasFragment(absFile, frag) {
  if (!frag) return true;
  let f = frag;
  try {
    f = decodeURIComponent(frag);
  } catch {
    /* keep raw */
  }
  if (SYNTHETIC_IDS.has(frag) || SYNTHETIC_IDS.has(f)) return true;
  const { ids } = parseFile(absFile);
  return ids.has(frag) || ids.has(f);
}

// ---------------------------------------------------------------------------
// link extraction
// ---------------------------------------------------------------------------

const SKIP_SCHEME = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i; // scheme: or protocol-relative

/**
 * Extract every `<a>` link from an HTML string as { href, unresolvedXref }.
 * `unresolvedXref` is true for `<a class="... api-xref--unresolved ...">`, i.e.
 * an <ApiXref> whose target id was not found in the merged inventory.
 * @returns {{ href: string, unresolvedXref: boolean }[]}
 */
function extractLinks(html) {
  const out = [];
  const aRe = /<a\s([^>]*?)>/gi;
  let m;
  while ((m = aRe.exec(html)) !== null) {
    const attrs = m[1];
    const hm = /href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i.exec(attrs);
    if (!hm) continue;
    out.push({
      href: hm[1] ?? hm[2] ?? hm[3],
      unresolvedXref: /\bapi-xref--unresolved\b/.test(attrs),
    });
  }
  return out;
}

/**
 * dist-relative served url path (base-stripped) for a source file, used as the
 * base for resolving relative + same-page (`#frag`) links. `dir/index.html` is
 * served at `/dir/`; a flat file like `404.html` is served at `/404.html` (so a
 * same-page `#frag` resolves back to that very file, not a nonexistent `/404/`).
 */
function pageUrlPath(absFile) {
  let rel = relative(DIST, absFile).split('\\').join('/');
  if (rel.endsWith('index.html')) rel = rel.slice(0, -'index.html'.length);
  return '/' + rel.replace(/^\/+/, '');
}

/** Section bucket for categorization. */
function section(p) {
  if (p.startsWith('/guides/') || p.startsWith('guides/')) return 'guide';
  if (p.startsWith('/api/') || p.startsWith('api/')) return 'api';
  if (p.startsWith('/blog/') || p.startsWith('blog/')) return 'blog';
  if (p.startsWith('/reference/') || p.startsWith('reference/')) return 'reference';
  if (p.startsWith('/tutorials/') || p.startsWith('tutorials/')) return 'tutorial';
  return 'other';
}

// ---------------------------------------------------------------------------
// main scan
// ---------------------------------------------------------------------------

const files = walkHtml(DIST);
let totalLinks = 0;
let ignoredCount = 0;

/** @type {Map<string, { category: string, target: string, reason: string, sources: Set<string> }>} */
const broken = new Map();

function recordBroken(category, target, reason, sourceRel) {
  const key = category + "|" + target + "|" + reason;
  let e = broken.get(key);
  if (!e) {
    e = { category, target, reason, sources: new Set() };
    broken.set(key, e);
  }
  e.sources.add(sourceRel);
}

for (const file of files) {
  const sourceRel = relative(DIST, file).split('\\').join('/');
  const srcUrl = pageUrlPath(file);
  const html = readFileSync(file, 'utf8');

  for (const link of extractLinks(html)) {
    // An <ApiXref> whose id wasn't in the inventory: same failure the xref
    // integration reports under XREF_STRICT=1, surfaced here so check_links is a
    // complete gate on its own (its href is a dead `#fqName` self-anchor).
    if (link.unresolvedXref) {
      const to = link.href.replace(/^#/, '') || '(unknown)';
      recordBroken('xref-unresolved', to, 'ApiXref target not in inventory', sourceRel);
      continue;
    }

    let raw = link.href;
    if (raw == null) continue;
    raw = raw.trim().replace(/&amp;/g, '&');
    if (raw === '' || raw === '#' || raw.startsWith('?')) continue; // same-page
    if (SKIP_SCHEME.test(raw)) continue; // external / protocol-relative / mailto: / etc.

    totalLinks++;

    // Resolve (relative or root-relative) against the source page URL.
    let url;
    try {
      url = new URL(raw, 'http://local' + srcUrl);
    } catch {
      recordBroken('other', raw, 'unparseable href', sourceRel);
      continue;
    }
    const targetPath = stripBase(url.pathname);
    const frag = url.hash ? url.hash.slice(1) : '';

    // ignore-list (target- or source-scoped)
    if (
      IGNORE.targets.some((re) => re.test(targetPath)) ||
      IGNORE.sources.some((re) => re.test(sourceRel))
    ) {
      ignoredCount++;
      continue;
    }

    const category = `${section(sourceRel)}→${section(targetPath)}`;

    // 1. direct file resolution (+ follow redirect stubs)
    let targetFile = resolveToFile(targetPath);
    let finalPath = targetPath;
    if (targetFile) {
      // followRefresh returns a non-null file only when it resolved on disk, so
      // a null result means a redirect stub pointed at a missing destination.
      const followed = followRefresh(targetFile, targetPath);
      if (!followed.file) {
        recordBroken(category, `${targetPath} -> ${followed.path}`, 'redirect stub target missing', sourceRel);
        continue;
      }
      targetFile = followed.file;
      finalPath = followed.path;
    } else {
      // 2. /api/latest/<deep> runtime 404 fallback -> /api/<latest>/<deep>
      const lm = /^\/api\/latest\/(.+)$/.exec(targetPath);
      if (lm) {
        const fallback = `/api/${LATEST}/${lm[1]}`;
        const fb = resolveToFile(fallback);
        if (fb) {
          targetFile = fb;
          finalPath = fallback;
        } else {
          recordBroken(
            category,
            targetPath,
            `no target file (api-latest fallback ${fallback} also missing)`,
            sourceRel,
          );
          continue;
        }
      } else {
        recordBroken(category, targetPath, 'target file not found', sourceRel);
        continue;
      }
    }

    // 3. fragment anchor
    if (frag && !hasFragment(targetFile, frag)) {
      recordBroken('anchor', `${finalPath}#${frag}`, 'fragment id/name not found', sourceRel);
    }
  }
}

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

console.log(`check_links: scanned ${files.length} HTML file(s), ${totalLinks} internal <a> link(s)`);
console.log(`  distDir=${relative(process.cwd(), DIST) || '.'}  base=${BASE}  api-latest=${LATEST}`);
if (ignoredCount) console.log(`  ignore-list suppressed ${ignoredCount} link(s)`);

if (broken.size === 0) {
  console.log('\nOK: every internal link, fragment anchor and resolved xref is reachable.');
  process.exit(0);
}

// group entries by category
/** @type {Map<string, { target: string, reason: string, sources: string[] }[]>} */
const byCategory = new Map();
let totalBroken = 0;
for (const e of broken.values()) {
  const list = byCategory.get(e.category) ?? [];
  list.push({ target: e.target, reason: e.reason, sources: [...e.sources].sort() });
  byCategory.set(e.category, list);
  totalBroken += e.sources.size;
}

console.log(`\nBROKEN (${totalBroken} link occurrence(s), ${broken.size} unique target(s)):\n`);
for (const category of [...byCategory.keys()].sort()) {
  const list = byCategory.get(category).sort((a, b) => a.target.localeCompare(b.target));
  const occ = list.reduce((n, x) => n + x.sources.length, 0);
  console.log(`[${category}]  ${list.length} unique target(s), ${occ} occurrence(s)`);
  for (const x of list) {
    console.log(`  x ${x.target}   (${x.reason})`);
    const shown = x.sources.slice(0, 5);
    for (const s of shown) console.log(`      <- ${s}`);
    if (x.sources.length > shown.length) {
      console.log(`      <- ... and ${x.sources.length - shown.length} more page(s)`);
    }
  }
  console.log('');
}

console.log(
  `FAIL: ${totalBroken} broken link occurrence(s) / ${broken.size} unique target(s) across ${byCategory.size} categor${byCategory.size === 1 ? 'y' : 'ies'}.`,
);
process.exit(1);
