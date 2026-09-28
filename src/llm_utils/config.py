"""Package configuration: packaged YAML defaults plus a per-project override.

Precedence, lowest to highest:

1. ``defaults.yaml`` shipped inside the package;
2. a project file: the path in ``LLM_UTILS_CONFIG``, else the nearest
   ``llm_utils.yaml`` found by walking up from the working directory;
3. ``configure(...)`` calls made by the running process.

Individual knobs may add an environment override on top (the spend cap does);
the knob's reader documents it. An override may only set keys the packaged
defaults declare: an unknown key raises ``ValueError`` at load, never a silent
fallback to the default.
"""
from __future__ import annotations

import copy
import os
import stat
import threading
from importlib.resources import files
from typing import Any, Optional

import yaml

CONFIG_ENV = "LLM_UTILS_CONFIG"
PROJECT_FILE = "llm_utils.yaml"

# Keys whose value may be null (off). Every other override must keep the
# packaged default's type: a number for a number, a string for a string.
NULLABLE: frozenset = frozenset()

_lock = threading.Lock()
_cache: Optional[dict] = None
_runtime: dict = {}


def _packaged_defaults() -> dict:
    return yaml.safe_load(files("llm_utils").joinpath("defaults.yaml").read_text()) or {}


def _readable(path: str) -> bool:
    # A FIFO counts: some secret/environment managers serve files that way
    # (the same rule as the .env lookup in constants.py).
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) or stat.S_ISFIFO(mode)


def _find_project_file() -> Optional[str]:
    explicit = os.getenv(CONFIG_ENV)
    if explicit:
        if not _readable(explicit):
            raise ValueError(f"{CONFIG_ENV}={explicit!r} is not a readable file")
        return explicit
    d = os.getcwd()
    while True:
        p = os.path.join(d, PROJECT_FILE)
        if _readable(p):
            return p
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _merge(base: dict, override: dict, where: str, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        dotted = f"{path}{key}"
        if key not in base:
            raise ValueError(f"{where}: unknown llm_utils config key {dotted!r}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{where}: {dotted!r} must be a mapping")
            out[key] = _merge(base[key], value, where, dotted + ".")
        else:
            _check_type(base[key], value, where, dotted)
            out[key] = value
    return out


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_type(default: Any, value: Any, where: str, dotted: str) -> None:
    if value is None:
        if dotted in NULLABLE or default is None:
            return
        raise ValueError(f"{where}: {dotted!r} may not be null")
    if default is None or (_is_number(default) and _is_number(value)):
        return
    if type(value) is not type(default):
        raise ValueError(f"{where}: {dotted!r} must be a {type(default).__name__}, "
                         f"got {value!r}")


def _load() -> dict:
    cfg = _packaged_defaults()
    project = _find_project_file()
    if project:
        with open(project) as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {}, project)
    return _merge(cfg, _runtime, "configure()")


def get_config() -> dict:
    """The effective configuration (a fresh copy; mutating it changes nothing)."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = _load()
        return copy.deepcopy(_cache)


def get(path: str) -> Any:
    """One value by dotted path, e.g. ``get("routes.probe_timeout_s")``."""
    node: Any = get_config()
    for part in path.split("."):
        node = node[part]
    return node


def configure(**sections: dict) -> None:
    """Process-level override, merged over the project file.

    Example: ``configure(spend_cap={"max_usd_per_run": 20.0})``. Validated
    immediately against the packaged keys.
    """
    global _cache, _runtime
    with _lock:
        merged = _merge(_packaged_defaults(), _runtime, "configure()")
        _merge(merged, sections, "configure()")          # validation only
        for name, value in sections.items():
            if isinstance(value, dict):
                _runtime[name] = {**_runtime.get(name, {}), **value}
            else:
                _runtime[name] = value
        _cache = None


def reload_config() -> None:
    """Drop the cached configuration and every ``configure`` override; the
    next read re-reads the packaged defaults and the project file."""
    global _cache, _runtime
    with _lock:
        _cache = None
        _runtime = {}
