// @ts-check
/**
 * build_inventory.mjs — regenerate the merged API cross-reference inventory.
 *
 * Phase C1. Node only, NO heavy deps (uses node:fs / node:path / node:url).
 *
 * WHAT IT DOES
 *   Collects every per-language / per-version inventory JSON file, merges them
 *   into a single lookup module and writes it to:
 *       src/generated/xref-inventory.json
 *   That module is imported by src/lib/xref.ts, which the <ApiXref> component
 *   uses to turn a fully-qualified id into a real page URL + anchor.
 *
 * SOURCES (loaded in this order; later files win for the version-agnostic map)
 *   1. Dev fallback: emitters/<lang>/_out/inventory.*.json
 *        The raw output of the local emitters. Handy so `astro dev` / a bare
 *        `astro build` resolve xrefs with zero extra setup.
 *   2. Authoritative: .api-inventory/*.json
 *        Where Phase C2 drops one inventory file PER VERSION before building,
 *        e.g. inventory.python.0.35.json, inventory.python.dev.json,
 *        inventory.rust.dev.json. These OVERRIDE the dev fallback.
 *
 * INVENTORY FILE SHAPE (both accepted)
 *   a) a flat array:               [ { fqName, kind, url, apiVersion? }, ... ]
 *   b) an object with entries:     { apiVersion?, language?, entries: [ ... ] }
 *   The `url` already encodes "<page path>#<fqName>" (the emitters build it).
 *
 * VERSION RESOLUTION (per file)
 *   fileVersion  = object.apiVersion ?? object.version
 *                  ?? <version> parsed from name "inventory.<lang>.<version>.json"
 *                  ?? null (version-agnostic only)
 *   entry.apiVersion overrides the file version for that single entry.
 *   Filenames without a version segment (e.g. "inventory.python.json") are
 *   treated as version-agnostic — they only populate the `any` fallback map.
 *
 * OUTPUT SHAPE (src/generated/xref-inventory.json)
 *   {
 *     "byVersion": { "<apiVersion>": { "<fqName>": "<url>" } },
 *     "any":       { "<fqName>": "<url>" }   // version-agnostic fallback
 *   }
 *
 * C2 FLOW
 *   For each version V: drop inventory.<lang>.<V>.json into .api-inventory/,
 *   then run `node website/scripts/build_inventory.mjs` before `astro build`.
 *   (The xref integration also regenerates this at astro:config:setup, so a
 *   plain `astro build` stays self-contained.)
 */

import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

/** Absolute path to the `website/` root (this file lives in website/scripts). */
export const WEBSITE_ROOT = fileURLToPath(new URL('..', import.meta.url));

const INVENTORY_DIR = join(WEBSITE_ROOT, '.api-inventory');
const EMITTERS_DIR = join(WEBSITE_ROOT, 'emitters');
const OUTPUT_FILE = join(WEBSITE_ROOT, 'src', 'generated', 'xref-inventory.json');

/**
 * @typedef {Object} InventoryEntry
 * @property {string} fqName
 * @property {string} url
 * @property {string|null} [kind]
 * @property {string|null} [apiVersion]
 */

/**
 * Parse "inventory.<lang>.<version>.json" -> { lang, version }.
 * A name without a version segment yields version === null.
 * @param {string} basename
 * @returns {{ lang: string|null, version: string|null }}
 */
function parseName(basename) {
  const stem = basename.replace(/\.json$/i, '');
  if (!stem.startsWith('inventory.')) return { lang: null, version: null };
  const tokens = stem.slice('inventory.'.length).split('.');
  const lang = tokens.shift() ?? null;
  const version = tokens.length > 0 ? tokens.join('.') : null;
  return { lang, version };
}

/**
 * Ordered list of inventory source files: dev fallback first, then the
 * authoritative .api-inventory/ drops (so the latter win the `any` map).
 * @returns {string[]}
 */
function collectSourceFiles() {
  /** @type {string[]} */
  const files = [];

  // 1. dev fallback: emitters/<lang>/_out/inventory.*.json
  if (existsSync(EMITTERS_DIR)) {
    for (const lang of readdirSync(EMITTERS_DIR).sort()) {
      const outDir = join(EMITTERS_DIR, lang, '_out');
      if (!existsSync(outDir) || !statSync(outDir).isDirectory()) continue;
      for (const name of readdirSync(outDir).sort()) {
        if (/^inventory\..*\.json$/i.test(name)) files.push(join(outDir, name));
      }
    }
  }

  // 2. authoritative: .api-inventory/*.json (sorted; last wins for `any`)
  if (existsSync(INVENTORY_DIR) && statSync(INVENTORY_DIR).isDirectory()) {
    for (const name of readdirSync(INVENTORY_DIR).sort()) {
      if (name.toLowerCase().endsWith('.json')) files.push(join(INVENTORY_DIR, name));
    }
  }

  return files;
}

/**
 * Read + normalise one inventory file into entries tagged with a version.
 * @param {string} filePath
 * @returns {InventoryEntry[]}
 */
function loadFile(filePath) {
  let data;
  try {
    data = JSON.parse(readFileSync(filePath, 'utf8'));
  } catch (err) {
    console.warn(`[xref] skipping unreadable inventory ${filePath}: ${err}`);
    return [];
  }

  let rawEntries;
  let fileVersion = null;
  if (Array.isArray(data)) {
    rawEntries = data;
  } else if (data && Array.isArray(data.entries)) {
    rawEntries = data.entries;
    fileVersion = data.apiVersion ?? data.version ?? null;
  } else {
    console.warn(`[xref] skipping malformed inventory ${filePath}`);
    return [];
  }

  const parsed = parseName(filePath.split(/[\\/]/).pop() ?? '');
  const version = fileVersion ?? parsed.version;

  /** @type {InventoryEntry[]} */
  const out = [];
  for (const e of rawEntries) {
    if (!e || typeof e.fqName !== 'string' || typeof e.url !== 'string') continue;
    out.push({
      fqName: e.fqName,
      url: e.url,
      kind: e.kind ?? null,
      apiVersion: e.apiVersion ?? version ?? null,
    });
  }
  return out;
}

/**
 * Collect every inventory entry (with kind + version) from all sources, in
 * load order. Consumers that need `kind` (e.g. objects.inv) use this.
 * @returns {InventoryEntry[]}
 */
export function collectEntries() {
  /** @type {InventoryEntry[]} */
  const all = [];
  for (const file of collectSourceFiles()) all.push(...loadFile(file));
  return all;
}

/**
 * Build the merged lookup: { byVersion, any }.
 * @returns {{ byVersion: Record<string, Record<string,string>>, any: Record<string,string> }}
 */
export function buildInventory() {
  /** @type {Record<string, Record<string,string>>} */
  const byVersion = {};
  /** @type {Record<string,string>} */
  const any = {};

  for (const e of collectEntries()) {
    any[e.fqName] = e.url; // last write wins (authoritative overrides fallback)
    if (e.apiVersion) {
      (byVersion[e.apiVersion] ??= {})[e.fqName] = e.url;
    }
  }
  return { byVersion, any };
}

/**
 * Write src/generated/xref-inventory.json and return the merged inventory.
 * @returns {{ byVersion: Record<string, Record<string,string>>, any: Record<string,string> }}
 */
export function writeGeneratedInventory() {
  const inv = buildInventory();
  mkdirSync(dirname(OUTPUT_FILE), { recursive: true });
  writeFileSync(OUTPUT_FILE, JSON.stringify(inv, null, 2) + '\n');
  return inv;
}

// Run as a script: regenerate + report.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const inv = writeGeneratedInventory();
  const versions = Object.keys(inv.byVersion);
  console.log(
    `[xref] wrote ${OUTPUT_FILE}\n` +
      `       ${Object.keys(inv.any).length} symbol(s) in the version-agnostic map\n` +
      `       ${versions.length} versioned map(s): ${versions.join(', ') || '(none)'}`,
  );
}
