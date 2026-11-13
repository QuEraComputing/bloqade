#!/usr/bin/env python3
"""Select stale mike doc versions to prune from the live site.

Reads `mike list -j` JSON on stdin and prints, one per line, the release
versions that should be deleted so that only the newest ``KEEP_RELEASES``
releases remain live (``dev`` and any non-release entries are always kept).

Older versions stay rebuildable from their git release tags and are captured
in the ``docs-archive-*`` GitHub release asset, so pruning them from the
deployed site loses nothing permanent.
"""

import json
import os
import re
import sys

RELEASE_RE = re.compile(r"^v?\d+\.\d+")


def is_release(version: str) -> bool:
    return RELEASE_RE.match(version) is not None


def sem_key(version: str):
    return [int(n) for n in re.findall(r"\d+", version)]


def main() -> int:
    keep = int(os.environ.get("KEEP_RELEASES", "3"))
    entries = json.load(sys.stdin)
    releases = sorted(
        (e["version"] for e in entries if is_release(e["version"])),
        key=sem_key,
        reverse=True,
    )
    for version in releases[keep:]:
        print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
