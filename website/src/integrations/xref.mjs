// @ts-check
/**
 * xref.mjs — Astro integration for API cross-references (Phase C1).
 *
 * Registered from astro.config.mjs. It does three things:
 *
 *   1. astro:config:setup  — regenerate src/generated/xref-inventory.json from
 *      all available inventory sources, so both `astro dev` and `astro build`
 *      resolve <ApiXref> without a separate pre-step. (CI / Phase C2 may still
 *      run scripts/build_inventory.mjs explicitly; same result.)
 *
 *   2. astro:build:done — scan the emitted HTML for `a.api-xref--unresolved`
 *      (references <ApiXref> could not resolve). WARN by default; FAIL the
 *      build when XREF_STRICT=1.
 *
 *   3. astro:build:done — emit a Sphinx v2 `objects.inv` (zlib-compressed) at
 *      the site root so external Sphinx docs can `intersphinx` into Bloqade.
 */

import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';

import { collectEntries, writeGeneratedInventory } from '../../scripts/build_inventory.mjs';

/**
 * Map an inventory entry to a Sphinx `domain:role` token.
 * @param {string|null|undefined} kind
 * @param {string} url
 */
function domainRole(kind, url) {
  if (url.startsWith('/api/python')) {
    switch (kind) {
      case 'module':
        return 'py:module';
      case 'class':
        return 'py:class';
      case 'exception':
        return 'py:exception';
      case 'method':
        return 'py:method';
      case 'function':
        return 'py:function';
      case 'attribute':
        return 'py:attribute';
      case 'property':
        return 'py:property';
      case 'data':
        return 'py:data';
      default:
        return 'py:obj';
    }
  }
  if (url.startsWith('/api/rust')) {
    /** @type {Record<string,string>} */
    const map = {
      module: 'module',
      struct: 'struct',
      enum: 'enum',
      trait: 'trait',
      fn: 'fn',
      function: 'fn',
      method: 'method',
      field: 'field',
      variant: 'variant',
      const: 'constant',
      constant: 'constant',
      static: 'static',
      macro: 'macro',
      type: 'type',
      typedef: 'type',
    };
    return 'rust:' + (map[kind ?? ''] ?? 'object');
  }
  return 'std:doc';
}

/**
 * Turn an inventory URL into a Sphinx inventory location (relative to the site
 * root). Uses the `#$` shorthand when the anchor equals the object name.
 * @param {string} url
 * @param {string} name
 */
function toLocation(url, name) {
  let uri = url.replace(/^\//, '');
  const hash = uri.indexOf('#');
  if (hash >= 0 && uri.slice(hash + 1) === name) {
    uri = uri.slice(0, hash + 1) + '$';
  }
  return uri;
}

/**
 * Build the Sphinx v2 objects.inv payload (header + zlib-compressed body).
 * @param {import('../../scripts/build_inventory.mjs').InventoryEntry[]} entries
 * @param {string} project
 */
function buildObjectsInv(entries, project) {
  // Dedupe by domain:name (Sphinx keys on this); last entry wins.
  /** @type {Map<string,string>} */
  const lines = new Map();
  /** @type {Set<string>} */
  const versions = new Set();
  for (const e of entries) {
    if (e.apiVersion) versions.add(e.apiVersion);
    const type = domainRole(e.kind, e.url);
    const location = toLocation(e.url, e.fqName);
    // "<name> <domain:role> <priority> <uri> <dispname>"
    lines.set(`${type}\t${e.fqName}`, `${e.fqName} ${type} 1 ${location} -`);
  }

  const version = [...versions].sort().join(', ') || 'dev';
  const header =
    `# Sphinx inventory version 2\n` +
    `# Project: ${project}\n` +
    `# Version: ${version}\n` +
    `# The remainder of this file is compressed using zlib.\n`;

  const body = [...lines.values()].sort().join('\n') + '\n';
  const compressed = deflateSync(Buffer.from(body, 'utf8'));
  const buf = Buffer.concat([Buffer.from(header, 'utf8'), compressed]);
  return { buf, count: lines.size };
}

/** Recursively collect *.html files under a directory. */
function walkHtml(dir, /** @type {string[]} */ acc = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walkHtml(p, acc);
    else if (name.endsWith('.html')) acc.push(p);
  }
  return acc;
}

/**
 * Scan built HTML for unresolved API cross-references.
 * @param {string} outDir
 * @returns {{ page: string, to: string }[]}
 */
function findUnresolved(outDir) {
  /** @type {{ page: string, to: string }[]} */
  const hits = [];
  const anchorRe = /<a\b[^>]*class="[^"]*\bapi-xref\b[^"]*"[^>]*>/g;
  for (const file of walkHtml(outDir)) {
    const html = readFileSync(file, 'utf8');
    let m;
    while ((m = anchorRe.exec(html)) !== null) {
      const tag = m[0];
      if (!/\bapi-xref--unresolved\b/.test(tag)) continue;
      const to = /data-xref-to="([^"]*)"/.exec(tag);
      hits.push({ page: relative(outDir, file), to: to ? to[1] : '(unknown)' });
    }
  }
  return hits;
}

/** @returns {import('astro').AstroIntegration} */
export default function xrefIntegration() {
  return {
    name: 'bloqade-xref',
    hooks: {
      'astro:config:setup': ({ logger }) => {
        try {
          const inv = writeGeneratedInventory();
          const versions = Object.keys(inv.byVersion);
          logger.info(
            `xref inventory: ${Object.keys(inv.any).length} symbol(s), ` +
              `versions [${versions.join(', ') || 'none'}]`,
          );
        } catch (err) {
          logger.warn(`could not regenerate xref inventory: ${err}`);
        }
      },

      'astro:build:done': ({ dir, logger }) => {
        const outDir = fileURLToPath(dir);

        // 1. broken cross-reference check
        const unresolved = findUnresolved(outDir);
        if (unresolved.length > 0) {
          const sample = unresolved
            .slice(0, 25)
            .map((u) => `    - ${u.to}  (in ${u.page})`)
            .join('\n');
          const more = unresolved.length > 25 ? `\n    ... and ${unresolved.length - 25} more` : '';
          const msg = `${unresolved.length} unresolved API cross-reference(s):\n${sample}${more}`;
          if (process.env.XREF_STRICT === '1') {
            logger.error(msg);
            throw new Error(
              `XREF_STRICT=1: build failed with ${unresolved.length} unresolved API cross-reference(s).`,
            );
          }
          logger.warn(msg);
        } else {
          logger.info('all API cross-references resolved');
        }

        // 2. Sphinx v2 objects.inv
        try {
          const entries = collectEntries();
          const { buf, count } = buildObjectsInv(entries, 'Bloqade');
          writeFileSync(join(outDir, 'objects.inv'), buf);
          logger.info(`wrote objects.inv (${count} objects, ${buf.length} bytes)`);
        } catch (err) {
          logger.warn(`could not write objects.inv: ${err}`);
        }
      },
    },
  };
}
