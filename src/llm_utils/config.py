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
import threading
from importlib.resources import files
from typing import Any, Optional

import yaml

CONFIG_ENV = "LLM_UTILS_CONFIG"
PROJECT_FILE = "llm_utils.yaml"

_lock = threading.Lock()
_cache: Optional[dict] = None
_runtime: dict = {}


def _packaged_defaults() -> dict:
    return yaml.safe_load(files("llm_utils").joinpath("defaults.yaml").read_text()) or {}


def _find_project_file() -> Optional[str]:
    explicit = os.getenv(CONFIG_ENV)
    if explicit:
        if not os.path.isfile(explicit):
            raise ValueError(f"{CONFIG_ENV}={explicit!r} is not a file")
        return explicit
    d = os.getcwd()
    while True:
        p = os.path.join(d, PROJECT_FILE)
        if os.path.isfile(p):
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
            out[key] = value
    return out


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
