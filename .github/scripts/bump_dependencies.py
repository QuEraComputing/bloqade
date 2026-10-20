"""Bump the minimum compatible versions of direct PyPI dependencies.

Only three-component stable releases are considered. Major releases and direct
Git dependencies are intentionally left for manual review.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tomllib
from pathlib import Path
from urllib.request import Request, urlopen

PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?~=(?P<version>\d+\.\d+\.\d+)$"
)
VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
PROJECT_VERSION = re.compile(
    r'(?m)^(version\s*=\s*")(?P<version>\d+\.\d+\.\d+[^"\n]*)(")$'
)


def version_tuple(value: str) -> tuple[int, int, int] | None:
    match = VERSION.fullmatch(value)
    return tuple(map(int, match.groups())) if match else None


def latest_compatible_release(
    releases: dict[str, list[dict]], current: tuple[int, int, int]
) -> tuple[int, int, int] | None:
    """Return the newest non-yanked stable release in the current major series."""
    candidates = (
        parsed
        for name, files in releases.items()
        if (parsed := version_tuple(name)) is not None
        and parsed[0] == current[0]
        and parsed > current
        and any(not file.get("yanked", False) for file in files)
    )
    return max(candidates, default=None)


def fetch_releases(name: str) -> dict[str, list[dict]]:
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    request = Request(
        f"https://pypi.org/pypi/{normalized}/json",
        headers={"User-Agent": "bloqade-dependency-updater/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)["releases"]


def update_text(
    text: str, release_fetcher=fetch_releases
) -> tuple[str, list[tuple[str, str, str]], bool]:
    project = tomllib.loads(text)["project"]
    changes: list[tuple[str, str, str]] = []
    minor_bump = False

    for requirement in project["dependencies"]:
        if " @ git+" in requirement:
            print(f"Skipping Git-source dependency: {requirement}")
            continue
        match = PIN.fullmatch(requirement)
        if not match:
            raise ValueError(f"Unsupported direct dependency constraint: {requirement}")
        name, current_str = match.group("name", "version")
        current = version_tuple(current_str)
        assert current is not None
        latest = latest_compatible_release(release_fetcher(name), current)
        if latest is None:
            continue

        new_str = ".".join(map(str, latest))
        replacement = requirement[: match.start("version")] + new_str
        old_toml, new_toml = json.dumps(requirement), json.dumps(replacement)
        if text.count(old_toml) != 1:
            raise ValueError(
                f"Could not uniquely find {requirement!r} in pyproject.toml"
            )
        text = text.replace(old_toml, new_toml, 1)
        changes.append((name, current_str, new_str))
        minor_bump |= latest[1] > current[1]

    if minor_bump:
        match = PROJECT_VERSION.search(text)
        if match is None:
            raise ValueError("Could not find the Bloqade project version")
        current = match.group("version")
        release, suffix = re.match(r"(\d+\.\d+\.\d+)(.*)", current).groups()
        major, minor, _ = version_tuple(release)
        bumped = f"{major}.{minor + 1}.0{suffix}"
        text = text[: match.start("version")] + bumped + text[match.end("version") :]

    tomllib.loads(text)  # Validate before writing anything.
    return text, changes, minor_bump


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--pr-body", type=Path)
    args = parser.parse_args()

    original = args.pyproject.read_text()
    updated, changes, minor_bump = update_text(original)
    if updated != original:
        args.pyproject.write_text(updated)

    for name, old, new in changes:
        print(f"{name}: {old} -> {new}")
    if minor_bump:
        print("At least one minor dependency release; bumped Bloqade minor version")
    if not changes:
        print("No new compatible PyPI patch or minor releases")

    if args.pr_body:
        lines = [
            "## Automated direct-dependency updates",
            "",
            *[f"- `{name}`: `{old}` → `{new}`" for name, old, new in changes],
            "",
            (
                "Bloqade's minor version was bumped because at least one direct "
                "dependency crossed a minor-version boundary."
                if minor_bump
                else "Bloqade's version is unchanged: these are patch-only updates."
            ),
            "",
            "The `uv.lock` file and CI checks are included for compatibility review.",
            "Git-source dependencies are not release-monitored by this workflow.",
        ]
        args.pr_body.write_text("\n".join(lines) + "\n")

    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            stream.write(f"changed={'true' if changes else 'false'}\n")


if __name__ == "__main__":
    main()
