"""Archive removal command for this repo's stores (stores.yaml `removal`): ``archive_removal.py <part id>``.

devices' archive asks it through the seam disk-archive-removal v1: one request on stdin (preview, remove or status),
one receipt on stdout. The checks every project shares (the receipt, the bytes, open files, other claims, the removal
itself and its record) are datastore's ``datastore.archive_removal.serve``; this file holds only this repo's keep rule
per part. It runs on datastore's own environment (the argv in stores.yaml), so this public package declares no
dependency on it.

Parts:
- ``old-build`` (dist/): a build file may go while its event still names it: an llm_utils wheel or sdist, a regular
  file directly under dist/, that a newer vX.Y.Z release tag supersedes (scripts/superseded_builds.py, read again
  now). Nothing in this repo or its consumers reads dist/; consumers install from the release tags. An entry that is
  not such a file is kept as ``other``; a build no newer tag supersedes is kept as ``still_read`` (the current build a
  consumer may install); unreadable tags keep it as ``other``.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = "llm_utils"
_HERE = Path(__file__).resolve().parent


def _superseded_builds():
    """scripts/superseded_builds.py, loaded by path (``python -I`` puts no script folder on sys.path)."""
    spec = importlib.util.spec_from_file_location("superseded_builds", _HERE / "superseded_builds.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keep_old_build(entry: dict, paths: list[Path]):
    """None when the build may go; (reason_code, plain words) when it stays."""
    builds = _superseded_builds()
    name = entry["name"]
    if entry["part"] != f"{REPO}:old-build":
        return ("other", f"this keep rule is old-build's; the request names part {entry['part']}")
    if Path(entry["folder"]).resolve() != (_HERE.parent / "dist").resolve():
        return ("other", f"{entry['folder']} is not this checkout's dist/ folder")
    if entry["kind"] != "item" or list(entry["paths"]) != [name] or len(paths) != 1:
        return ("other", "an old-build entry is one build file directly under dist/, never a unit")
    if not builds.BUILD.match(name):
        return ("other", f"{name} is not an llm_utils wheel or sdist file name")
    path = paths[0]
    if path.is_symlink() or not path.is_file():
        return ("other", f"{name} is not a regular file, so it is no build item of old-build")
    try:
        found = builds.superseded(path.parent)
    except (OSError, subprocess.CalledProcessError) as exc:
        return ("other", f"the release tags could not be read, so no newer release is proved: {exc}")
    if name not in found:
        return ("still_read", f"no release tag newer than {name}'s version exists, so it is the current build a "
                              "consumer may still install")
    return None


RULES = {"old-build": keep_old_build}


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in RULES:
        print(f"usage: archive_removal.py {{{','.join(RULES)}}}", file=sys.stderr)
        return 2
    from datastore.archive_removal import serve

    return serve(RULES[argv[0]])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
