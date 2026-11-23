"""Event reader for the old-build store (stores.yaml): one JSON line per superseded build in dist/.

A wheel or sdist directly under dist/ is superseded once a release tag (vX.Y.Z) newer than its version exists;
its time is that tag's creation time, the earliest such tag. Read-only: it lists dist/ and reads the tags.
Runs from the repo checkout. Exits 1 when dist/ or the tags cannot be read.

``superseded(dist)`` is the same reading as a mapping (file name to time); scripts/archive_removal.py's keep rule
asks it again, so the removal keeps exactly what this event would no longer name.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

BUILD = re.compile(r"^llm_utils-(\d+)\.(\d+)\.(\d+)(?:-[^/]+\.whl|\.tar\.gz)$")
TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def superseded(dist: Path) -> dict[str, str]:
    """Each superseded build file directly under ``dist`` and its time. Raises OSError or CalledProcessError when
    ``dist`` or the tags of the repository holding it cannot be read."""
    entries = sorted(dist.iterdir())
    out = subprocess.run(
        ["git", "-C", str(dist.parent), "for-each-ref", "refs/tags/v*",
         "--format=%(refname:strip=2) %(creatordate:iso-strict)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tags = []
    for line in out.splitlines():
        name, _, at = line.partition(" ")
        m = TAG.match(name)
        if m and at:
            tags.append((tuple(int(x) for x in m.groups()), at))
    tags.sort()
    found = {}
    for entry in entries:
        m = BUILD.match(entry.name)
        if not m or entry.is_symlink() or not entry.is_file():
            continue
        version = tuple(int(x) for x in m.groups())
        newer = [at for tag, at in tags if tag > version]
        if newer:
            found[entry.name] = newer[0]
    return found


def main() -> int:
    try:
        found = superseded(Path("dist").absolute())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"superseded_builds: {exc}", file=sys.stderr)
        return 1
    for name, at in found.items():
        print(json.dumps({"item": name, "at": at}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
