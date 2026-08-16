// @ts-check
/**
 * check_build.mjs — build-warning gate for the Bloqade docs site.
 *
 * Runs the production build (or scans an already-captured build log), then
 * FAILS (exit 1) on any warning/error-class line in the combined stdout+stderr
 * — EXCEPT a small, documented allowlist (see ALLOWLIST below). This is the
 * "(a) any build warning/error" half of `mise run docs:test`. It is SEPARATE
 * from `docs:check` (link/xref reachability) and does not replace it.
 *
 * The clean tree is warning-free except one third-party Node deprecation that
 * Starlight emits when it launches Pagefind; that single line is allowlisted.
 * Everything else is treated as a failure.
 *
 * USAGE
 *   node scripts/check_build.mjs                 # run `pnpm build`, capture its
 *                                                # combined output, scan, and
 *                                                # LEAVE dist/ in place (so the
 *                                                # runtime crawl can reuse it)
 *   node scripts/check_build.mjs --log <file>    # scan an already-captured
 *                                                # combined build log instead of
 *                                                # building (no build is run)
 *
 * ENV
 *   BUILD_CMD   override the build command (run via a shell). Default: the
 *               website's `pnpm build` (= `astro build`), matching docs:build.
 *
 * DESIGN
 *   • SIGNALS  — regexes that mark a line as a warning/error. A line that
 *                matches ANY signal is a candidate failure.
 *   • ALLOWLIST — regexes for genuinely-benign lines. A candidate line that
 *                also matches ANY allowlist entry is SUPPRESSED (counted, not
 *                failed). Keep this list tiny and document every entry.
 *   • A line fails iff it matches a signal AND no allowlist entry.
 *   • A non-zero build exit code also fails the gate.
 */

import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const WEBSITE_ROOT = resolve(SCRIPT_DIR, '..');

// ---------------------------------------------------------------------------
// SIGNALS — a line matching any of these is a candidate warning/error.
// ---------------------------------------------------------------------------
const SIGNALS = [
  // Astro / Vite / pnpm structured "[WARN]" (and "[ERROR]") tags.
  { name: 'log [WARN]/[ERROR] tag', re: /\[(?:WARN|ERROR)\]/i },
  // The word "warning" — includes Node's "DeprecationWarning" (the DEP0190 line
  // is caught here and then suppressed by the allowlist).
  { name: 'the word "warning"', re: /warning/i },
  // The word "error" (word-boundaried, case-insensitive): Astro/Rollup/Vite
  // build errors, "Error:", "✘ [ERROR]", etc.
  { name: 'the word "error"', re: /\berror\b/i },
  { name: '"Could not …"', re: /could not/i },
  { name: '"Unrecognized …"', re: /unrecognized/i },
  // Expressive Code failing to highlight a fenced block / falling back for an
  // unknown language (the thing astro.config.mjs's committed grammars prevent).
  {
    name: 'Expressive Code highlighting / unknown-language fallback',
    re: /highlighting|unknown language|does not (?:support|highlight)|no grammar/i,
  },
  // KaTeX render diagnostics. Matched CASE-SENSITIVELY on the brandcased
  // "KaTeX" so the lowercase emitted asset filename `_astro/katex.*.js` in
  // Vite's bundle-size table is NOT mistaken for a warning.
  { name: 'KaTeX diagnostic (brandcased)', re: /KaTeX/ },
  { name: 'KaTeX character-metrics / LaTeX-incompat', re: /character metrics|latex-incompatible/i },
  // Pagefind walking a page with no <html> (Astro's meta-refresh redirect
  // stubs) — silenced via PAGEFIND_ROOT_SELECTOR in astro.config.mjs; kept as a
  // signal so a regression reappears here.
  { name: 'Pagefind "found without an <html>"', re: /without an <html>/i },
];

// ---------------------------------------------------------------------------
// ALLOWLIST — benign lines that match a signal but must NOT fail the gate.
// Keep tiny; document WHY for every entry. Matched against the raw line.
// ---------------------------------------------------------------------------
const ALLOWLIST = [
  {
    // Node DEP0190. Emitted by STARLIGHT'S OWN Pagefind launcher, which does
    // `spawn('npx', […], { shell: true })` during astro:build:done. This is
    // third-party (Starlight + Node core); nothing in this repo passes
    // shell:true. Matched narrowly on the DEP code + the exact wording.
    re: /\[DEP0190\] DeprecationWarning: Passing args to a child process with shell option true/,
    why: "Starlight spawns Pagefind with {shell:true}; upstream code, not ours.",
  },
  {
    // Node prints this stock hint EXACTLY ONCE, right after the first
    // deprecation warning of the process (i.e. the DEP0190 above). It is not a
    // diagnostic of its own; any NON-allowlisted deprecation is still caught by
    // its own "[DEPxxxx] DeprecationWarning: …" line.
    re: /\(Use `node --trace-deprecation .* to show where the warning was created\)/,
    why: "Node's stock trace-deprecation footer that trails the allowlisted DEP0190.",
  },
  {
    // Vite's post-build bundle-size table lists every EMITTED asset's filename +
    // size (e.g. `_astro/katex.*.js   261 kB │ gzip: …`). Those are filenames,
    // not diagnostics — a chunk whose name happens to contain "error"/"warn"
    // must not trip the word signals above.
    re: /\[vite\].*\b(?:kB|gzip:)/,
    why: 'Vite bundle-size report rows are asset filenames, not warnings.',
  },
];

// ---------------------------------------------------------------------------
// scan
// ---------------------------------------------------------------------------

/**
 * @param {string} text combined stdout+stderr
 * @returns {{ offending: {lineNo:number,line:string,signal:string}[], allowlisted: {line:string,why:string}[] }}
 */
function scan(text) {
  const lines = text.split(/\r?\n/);
  const offending = [];
  const allowlisted = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const sig = SIGNALS.find((s) => s.re.test(line));
    if (!sig) continue;
    const allow = ALLOWLIST.find((a) => a.re.test(line));
    if (allow) {
      allowlisted.push({ line, why: allow.why });
      continue;
    }
    offending.push({ lineNo: i + 1, line, signal: sig.name });
  }
  return { offending, allowlisted };
}

function report(text) {
  const { offending, allowlisted } = scan(text);

  if (allowlisted.length) {
    console.log(`\ncheck_build: suppressed ${allowlisted.length} allowlisted line(s):`);
    // group by justification for a compact summary
    const byWhy = new Map();
    for (const a of allowlisted) byWhy.set(a.why, (byWhy.get(a.why) ?? 0) + 1);
    for (const [why, n] of byWhy) console.log(`  · ${n}× ${why}`);
  }

  if (offending.length === 0) {
    console.log('\ncheck_build OK: no non-allowlisted build warning/error lines.');
    return true;
  }

  console.error(`\ncheck_build FAIL: ${offending.length} non-allowlisted warning/error line(s):`);
  for (const o of offending) {
    console.error(`  line ${o.lineNo} [${o.signal}]`);
    console.error(`    ${o.line.trim()}`);
  }
  return false;
}

// ---------------------------------------------------------------------------
// modes
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const out = { logFile: /** @type {string|null} */ (null) };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--log') out.logFile = argv[++i];
  }
  return out;
}

/** Run the build, streaming + capturing combined output. Resolves the text + exit code. */
function runBuild() {
  return new Promise((resolvePromise) => {
    const buildCmd = process.env.BUILD_CMD ?? 'pnpm build';
    console.log(`check_build: running build (\`${buildCmd}\`) in ${WEBSITE_ROOT} …\n`);
    // Run via a shell so BUILD_CMD (and the default `pnpm build`) resolve on PATH
    // consistently across macOS/Linux CI.
    const child = spawn(buildCmd, {
      cwd: WEBSITE_ROOT,
      shell: true,
      env: process.env,
    });
    let captured = '';
    child.stdout.on('data', (d) => {
      const s = d.toString();
      captured += s;
      process.stdout.write(s); // stream through so CI logs show the build
    });
    child.stderr.on('data', (d) => {
      const s = d.toString();
      captured += s;
      process.stderr.write(s);
    });
    child.on('error', (err) => {
      captured += `\n[check_build] failed to spawn build: ${err}\n`;
      resolvePromise({ text: captured, code: 127 });
    });
    child.on('close', (code) => resolvePromise({ text: captured, code: code ?? 0 }));
  });
}

async function main() {
  const { logFile } = parseArgs(process.argv.slice(2));

  if (logFile) {
    const file = resolve(process.cwd(), logFile);
    if (!existsSync(file)) {
      console.error(`check_build: log file not found: ${file}`);
      process.exit(2);
    }
    console.log(`check_build: scanning captured build log ${file}`);
    const ok = report(readFileSync(file, 'utf8'));
    process.exit(ok ? 0 : 1);
  }

  const { text, code } = await runBuild();
  const clean = report(text);
  if (code !== 0) {
    console.error(`\ncheck_build FAIL: build command exited with code ${code}.`);
    process.exit(1);
  }
  process.exit(clean ? 0 : 1);
}

main();
