#!/usr/bin/env node
// check_search_scope.mjs — verify site search is scoped to the LATEST API version.
//
// Every generated page under dist/api/<latest>/** must be search-indexable
// (carry `data-pagefind-body`); every page under a NON-latest dist/api/<V>/**
// must be EXCLUDED from search (no `data-pagefind-body`). Guides / blog /
// reference live outside api/<V>/ and are unaffected.
//
// This guards `build_versions.py`'s search-scoping (it injects `pagefind: false`
// into non-latest API pages): a regression that re-indexed old versions, or that
// stopped indexing the latest API, fails here.
//
// Usage: node scripts/check_search_scope.mjs [distDir=dist]
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const dist = process.argv[2] || 'dist';
const MARK = 'data-pagefind-body';

// Read the real version labels from the manifest — NOT "every subdir of
// dist/api/", because evergreen pages (api/index, api/compatibility) and the
// api/latest redirect stub are also directories in dist but are not versions
// (and should stay searchable / are irrelevant to scoping).
function readManifest() {
  try {
    const m = JSON.parse(readFileSync('src/generated/versions.json', 'utf8'));
    if (m && typeof m.latest === 'string' && Array.isArray(m.versions)) {
      return { latest: m.latest, versions: m.versions.map((v) => v.label) };
    }
  } catch {
    /* fall through */
  }
  return { latest: 'dev', versions: ['dev'] };
}

function walkHtml(dir) {
  const out = [];
  if (!existsSync(dir)) return out;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...walkHtml(p));
    else if (e.name.endsWith('.html')) out.push(p);
  }
  return out;
}

const apiRoot = join(dist, 'api');
if (!existsSync(apiRoot)) {
  console.error(`check_search_scope: no ${apiRoot} — generate the API docs first (mise run docs:api).`);
  process.exit(2);
}

const { latest, versions } = readManifest();

let indexedLatest = 0;
const leaked = []; // non-latest pages still search-indexable (a scope violation)

for (const v of versions) {
  for (const p of walkHtml(join(apiRoot, v))) {
    const has = readFileSync(p, 'utf8').includes(MARK);
    if (v === latest) {
      if (has) indexedLatest++;
    } else if (has) {
      leaked.push(p);
    }
  }
}

let ok = true;
if (leaked.length) {
  ok = false;
  console.error(
    `check_search_scope: FAIL — ${leaked.length} non-latest API page(s) are still in the search index (should be excluded):`,
  );
  for (const p of leaked.slice(0, 20)) console.error('  ' + p);
  if (leaked.length > 20) console.error(`  … and ${leaked.length - 20} more`);
}
if (indexedLatest === 0) {
  ok = false;
  console.error(
    `check_search_scope: FAIL — no searchable pages under api/${latest}/ (the latest API must be indexed).`,
  );
}
if (!ok) process.exit(1);

const excluded = versions.filter((v) => v !== latest);
console.log(
  `check_search_scope: OK — latest=${latest}: ${indexedLatest} searchable latest-API page(s); ` +
    `non-latest API excluded from search (${excluded.join(', ') || 'none'}).`,
);
