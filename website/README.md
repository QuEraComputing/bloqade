# Bloqade docs website (Astro + Starlight)

The new Bloqade documentation site, built with [Astro 5](https://astro.build/) +
[Starlight](https://starlight.astro.build/). It lives alongside the existing
MkDocs site (repo-root `mkdocs.yml` / `mise run mkdocs:serve`) during the
migration and will eventually replace it. Tracking PR:
**[#367](https://github.com/QuEraComputing/bloqade/pull/367)**.

Everything is driven by [`mise`](https://mise.jdx.dev) via the repo-root
[`mise.toml`](../mise.toml), which also pins the toolchain (Node, pnpm, uv). Run
`mise run` (or `mise tasks`) to see every task.

> **Task naming:** the docs-site tasks are namespaced under **`docs:`** (e.g.
> `mise run docs:dev`). The bare top-level names are reserved for the **`bloqade`
> package** — `mise run build` builds the *package* (`uv build`), `mise run test`
> runs the *test suite*. Docs tasks run from the repo root (they're scoped to
> `website/` for you).

## Prerequisites

- **[`mise`](https://mise.jdx.dev)** — `brew install mise` (or see their install
  docs). Then `mise install` provisions the pinned **Node**, **pnpm**, and **uv**.
- **Rust nightly** — only for the *Rust* API (`cargo +nightly rustdoc`); managed
  by `rustup`, not mise (`rustup toolchain install nightly`). Python-only
  generation needs no Rust toolchain.

## Quick start (shell only — no Python/Rust)

```sh
mise install         # provision Node + pnpm + uv (first time only)
mise run docs:install  # pnpm install
mise run docs:dev      # serve with hot-reload; open the printed URL (default :4321)
```

The landing page, guides, blog, and all design/styling work **without generating
any API content**. A fresh clone simply has an empty `/api` section until you
generate it (next section).

## Full site, including the API reference

```sh
mise run docs:bootstrap    # = docs:install -> docs:api -> docs:dev, in one go
# ...or run the steps individually:
mise run docs:install
mise run docs:api          # generate the `dev` API for all languages (Python + Rust)
mise run docs:dev
```

`mise run docs:api` generates the versioned API reference for the `dev` version.
Sources are resolved two ways:

- **`--prefer-local`** (what `docs:api` / `docs:api-quick` use): documents your
  **sibling checkouts** — `../bloqade-circuit`, `../bloqade-analog`,
  `../bloqade-lanes`. Fast, no network, always reflects your local working copies.
- **`--clone-dir` (CI path):** `git clone`s each source at its **pinned tag**
  (resolved from the *installed* `bloqade-*` package versions via `uv pip show`).
  Because it reads installed versions, it requires `uv sync` at the **repo root**
  first. Invoke it directly when reproducing a release build:

  ```sh
  uv sync                                                    # repo root, once
  cd website && uv run scripts/build_versions.py --versions dev --clone-dir sources
  ```

  To build a specific release version instead of `dev`, pass `--versions 0.35`.

### Fast iteration

Regenerating the full API (especially the Rust half) is the slow part. For a
quick loop, generate a Python-only slice of one small package:

```sh
mise run docs:api-quick    # python only, bloqade-lanes only — finishes in seconds
```

`mise run docs:api-dry` prints the emitter commands without writing anything.

## What's generated vs. committed

The `/api` reference is generated, so a fresh clone or a Python-free setup shows
an **empty API section** — that's expected. Regenerate any time with
`mise run docs:api` (or reset with `mise run docs:clean` first; see below).

**Generated & git-ignored** (never show up in `git status`):

- `src/content/docs/api/<version>/**` — the per-version API trees (large).
- `.api-inventory/inventory.<lang>.<version>.json` — the per-version symbol
  inventories (the merged cross-reference source; `.api-inventory/.gitkeep`
  keeps the dir).
- `src/generated/versions.json` — the version-switcher manifest.
- `src/generated/xref-inventory.json` — the merged cross-reference lookup
  (rebuilt automatically on every `astro dev` / `astro build`).
- `dist/` — the production build output.

**Committed** (tracked, edited by hand):

- `src/content/docs/api/index.mdx` and `api/compatibility.mdx` — evergreen API
  landing + compatibility matrix.
- everything under `guides/`, `reference/`, `src/pages/`, `src/components/`,
  `src/styles/`, plus `astro.config.mjs`, `docs.sources.toml`, the emitters, and
  the scripts.

## Reset a stale state

```sh
mise run docs:clean        # remove regenerable generated content, then regenerate:
mise run docs:api
```

`mise run docs:clean` removes the generated `api/<version>/` trees, the per-version
inventories, `versions.json`, `xref-inventory.json`, and `dist/`. It **keeps**
the committed `api/index.mdx`, `api/compatibility.mdx`, and `.api-inventory/.gitkeep`.

## Production build & preview

```sh
mise run docs:build        # astro build: inventory merge + Pagefind index + dist/objects.inv
mise run docs:preview      # serve the built site from dist/
```

## Where things live

| Path | What |
| --- | --- |
| `src/pages/` | Landing page (`index.astro`) and blog (`blog/`). |
| `src/content/docs/404.mdx` | Site 404 page (Starlight-owned `/404` route); embeds `NotFound.astro`, which also runs the `/api/latest` deep-link fallback. |
| `src/content/docs/guides/` | Evergreen guides & tutorials (MDX). |
| `src/content/docs/api/` | Versioned API reference (committed `index.mdx` + `compatibility.mdx`; generated `<version>/` trees). |
| `src/content/docs/reference/` | Reference index. |
| `src/components/api/` | API rendering components (`ApiClass`, `ApiFn`, `ApiModule`, `Params`, …). |
| `src/styles/brand.css` | **Single source of truth for design tokens** (color, spacing, typography) shared by the landing page and Starlight. |
| `src/lib/` | Helpers: `versions.ts`, `xref.ts`, `nav.ts`. |
| `src/generated/` | Generated manifests (`versions.json`, `xref-inventory.json`). |
| `emitters/` | Doc emitters: `python/` (griffe → MDX), `rust/` (rustdoc-json → MDX), `notebooks/`. |
| `scripts/` | Orchestration: `build_versions.py`, `resolve_sources.py`, `build_inventory.mjs`, `gen_compat_matrix.py`, `publish.sh`. |
| `astro.config.mjs` | Astro + Starlight config. |
| `docs.sources.toml` | API source manifest (which repos/tags each version documents). |

## Troubleshooting

- **Empty API section** → you haven't generated it: `mise run docs:api` (or `mise run docs:api-quick`).
- **Stale or odd API state** (missing symbols, wrong version list) → `mise run docs:clean` then `mise run docs:api`.
- **Port 4321 already in use** → Astro automatically picks the next free port; use the URL it prints.
- **`mise run docs:api` can't find sources** → `--prefer-local` expects the sibling
  `../bloqade-circuit`, `../bloqade-analog`, `../bloqade-lanes` checkouts next to
  this repo. Clone them, or use the `--clone-dir` path above.
- **Rust API missing / `cargo rustdoc` errors** → install a Rust **nightly**
  toolchain (`rustup toolchain install nightly`), or iterate Python-only with
  `mise run docs:api-quick`.
- **`mise install` fails on pnpm** → this repo installs pnpm via mise's npm
  backend (`"npm:pnpm"`) to sidestep a stale-registry bug in older mise releases;
  if you still hit issues, `mise self-update` and retry.
