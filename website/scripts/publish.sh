#!/usr/bin/env bash
#
# publish.sh — the ONE host-specific seam of the Bloqade docs deploy.
# =====================================================================
# Given an already-built Astro site (`website/dist/`) and a target SUBPATH,
# publish it under that subpath on the hosting target. Everything upstream of
# this script (install / generate API content / `astro build`) is host-agnostic;
# only this file knows *where* the bytes go.
#
# USAGE
#   website/scripts/publish.sh <dist-dir> <subpath>
#
#   <dist-dir>  Built Astro output (e.g. website/dist). Must already exist.
#   <subpath>   Path under the site root to publish INTO, WITHOUT surrounding
#               slashes. It MUST equal the `SITE_BASE` the site was built with,
#               or every asset/link 404s. Examples:
#                   next                  -> https://<host>/next/
#                   astro-preview/pr-42   -> https://<host>/astro-preview/pr-42/
#
# ENV KNOBS (all optional)
#   PUBLISH_BACKEND   github-pages (default) | vercel   — see "SWAP TARGET".
#   GH_PAGES_BRANCH   Branch to publish to (default: gh-pages).
#   GH_PAGES_REMOTE   Remote to push to    (default: origin).
#   PUBLISH_MESSAGE   Commit message       (default: derived from subpath).
#   PUBLISH_DRY_RUN   If set to 1, do everything EXCEPT the final commit+push
#                     (prints what would happen; used by local verification).
#
# ------------------------------------------------------------------------------
# SWAP TARGET (GitHub Pages -> Vercel) — a documented ONE-LINER.
# ------------------------------------------------------------------------------
# The publish backend is selected by `PUBLISH_BACKEND` (default `github-pages`).
# To move production onto Vercel at cutover, set that env var to `vercel` — the
# body below already delegates to `publish_vercel`, whose entire implementation
# is the single line:
#
#       exec vercel deploy --prebuilt ${VERCEL_PROD:+--prod} --token "$VERCEL_TOKEN"
#
# (Vercel serves the prebuilt `dist/` at its own domain/root, so at that point
# the site is built with SITE_BASE=/ and <subpath> becomes advisory only.)
# ------------------------------------------------------------------------------
set -euo pipefail

DIST_DIR="${1:?usage: publish.sh <dist-dir> <subpath>}"
SUBPATH_RAW="${2:?usage: publish.sh <dist-dir> <subpath>}"

# Normalize: strip leading/trailing slashes so joins never double-slash.
SUBPATH="${SUBPATH_RAW#/}"
SUBPATH="${SUBPATH%/}"
if [[ -z "$SUBPATH" ]]; then
  echo "publish.sh: refusing to publish to the site ROOT (subpath was empty)." >&2
  echo "  Root deploys are gated behind the Phase F cutover; pass e.g. 'next'." >&2
  exit 2
fi

if [[ ! -d "$DIST_DIR" ]]; then
  echo "publish.sh: dist dir not found: $DIST_DIR (did 'astro build' run?)" >&2
  exit 1
fi
if [[ -z "$(ls -A "$DIST_DIR" 2>/dev/null)" ]]; then
  echo "publish.sh: dist dir is empty: $DIST_DIR" >&2
  exit 1
fi

DIST_DIR="$(cd "$DIST_DIR" && pwd)"  # absolutize before we chdir anywhere

PUBLISH_BACKEND="${PUBLISH_BACKEND:-github-pages}"

# ------------------------------------------------------------------------------
# Backend: GitHub Pages branch model (dev/preview during the transition).
#
# Publishes into `<branch>:<subpath>/` while PRESERVING every sibling path, so
# it never disturbs the live MkDocs site that `mike` deploys to the branch root
# (`index.html`, `dev/`, `latest/`, `v*/`, `versions.json`) or the MkDocs PR
# previews under `pr-preview/`. It only ever touches `<subpath>/`.
# ------------------------------------------------------------------------------
publish_github_pages() {
  local branch="${GH_PAGES_BRANCH:-gh-pages}"
  local remote="${GH_PAGES_REMOTE:-origin}"
  local message="${PUBLISH_MESSAGE:-Publish Astro site to /${SUBPATH}/}"

  # Identity for the commit (no-op if the runner already configured one).
  git config user.name  >/dev/null 2>&1 || git config user.name  "github-actions[bot]"
  git config user.email >/dev/null 2>&1 || git config user.email "github-actions[bot]@users.noreply.github.com"

  local worktree
  worktree="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "git worktree remove --force '$worktree' >/dev/null 2>&1 || true; rm -rf '$worktree'" EXIT

  # Materialize the branch into a throwaway worktree (does not touch the
  # currently checked-out branch). Create it as an orphan if it doesn't exist.
  git fetch --depth=1 "$remote" "$branch" >/dev/null 2>&1 || true
  if git show-ref --verify --quiet "refs/remotes/$remote/$branch"; then
    git worktree add --force -B "$branch" "$worktree" "refs/remotes/$remote/$branch"
  else
    echo "publish.sh: '$remote/$branch' not found; creating an orphan branch." >&2
    git worktree add --force --detach "$worktree"
    ( cd "$worktree" && git checkout --orphan "$branch" && git reset --hard && git clean -fdx )
  fi

  # Replace ONLY the target subpath; leave every sibling path untouched.
  rm -rf "${worktree:?}/${SUBPATH}"
  mkdir -p "$worktree/$SUBPATH"
  cp -R "$DIST_DIR"/. "$worktree/$SUBPATH"/
  # GitHub Pages skips `_`-prefixed dirs (e.g. `_astro/`) without this marker.
  touch "$worktree/.nojekyll"

  ( cd "$worktree"
    git add -A
    if git diff --cached --quiet; then
      echo "publish.sh: no changes for /$SUBPATH/ — nothing to publish."
      exit 0
    fi
    if [[ "${PUBLISH_DRY_RUN:-}" == "1" ]]; then
      echo "publish.sh: [dry-run] would commit + push the following to $remote/$branch:"
      git status --short
      exit 0
    fi
    git commit -m "$message"
    # Serialized upstream via the `gh-pages-write` concurrency group, but retry
    # a rebase-on-remote anyway so a stray concurrent write can't lose the push.
    local attempt
    for attempt in 1 2 3; do
      if git push "$remote" "HEAD:$branch"; then
        echo "publish.sh: published $DIST_DIR -> $remote/$branch:/$SUBPATH/"
        exit 0
      fi
      echo "publish.sh: push failed (attempt $attempt); rebasing on $remote/$branch and retrying." >&2
      git fetch --depth=1 "$remote" "$branch"
      git rebase "$remote/$branch" || { git rebase --abort || true; }
    done
    echo "publish.sh: push failed after retries." >&2
    exit 1
  )
}

# Backend: Vercel (production cutover). See "SWAP TARGET" above.
publish_vercel() {
  exec vercel deploy --prebuilt ${VERCEL_PROD:+--prod} --token "$VERCEL_TOKEN"
}

case "$PUBLISH_BACKEND" in
  github-pages) publish_github_pages ;;
  vercel)       publish_vercel ;;
  *) echo "publish.sh: unknown PUBLISH_BACKEND '$PUBLISH_BACKEND'." >&2; exit 2 ;;
esac
