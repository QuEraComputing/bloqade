"""Strip author attribution from GitHub's auto-generated release notes.

``gh api .../releases/generate-notes`` returns a body whose entries read
``* <title> by @<author> in <pr-url>`` and which ends with a purely
author-based "New Contributors" section. Release notes cut from a release
branch are built from backport PRs authored by the automation bot, so the
per-line attribution and the contributors list are noise.

Reads the raw body on stdin and writes a cleaned body (no per-line authors,
no "New Contributors" section) to stdout. The category structure from
``.github/release.yml`` and the "Full Changelog" footer are left intact.
"""

from __future__ import annotations

import re
import sys


def clean(body: str) -> str:
    body = body.replace("\r\n", "\n")
    # Remove the trailing, purely author-based "New Contributors" section,
    # keeping any following heading or the "Full Changelog" footer.
    body = re.sub(
        r"^#+ New Contributors[ \t]*\n.*?(?=^#+ |^\*\*Full Changelog\*\*|\Z)",
        "",
        body,
        flags=re.MULTILINE | re.DOTALL,
    )
    # Match the attribution before the PR link so mentions in titles survive.
    body = re.sub(
        r" by @[A-Za-z0-9-]+(?:\[bot\])?(?= in https://github\.com/[^/\s]+/[^/\s]+/pull/\d+)",
        "",
        body,
    )
    # Collapse any blank-line run the section removal introduced.
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def main() -> None:
    sys.stdout.write(clean(sys.stdin.read()))


if __name__ == "__main__":
    main()
