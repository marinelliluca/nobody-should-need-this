"""Central configuration.

Loads `.env` (secrets, per-environment overrides) and `config.toml`
(defaults, tunable parameters) exactly once at import time. Modules
elsewhere in the package read from `CONFIG` rather than reaching for
os.environ or hardcoding values.

Precedence: env var > config.toml. This means CI, container deploys,
and ad-hoc shell overrides all work without editing the TOML file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


# Load .env from the project root or current working directory. python-dotenv
# silently no-ops if the file is missing, which is what we want — .env is
# optional in environments where vars are injected by the orchestrator.
load_dotenv()


def config_path() -> Path:
    """Resolve the active config.toml path.

    Public so callers (e.g. the match dump) can snapshot the exact file
    used for a run. Precedence: RAG_CONFIG_PATH env var, then a default
    of `<package_parent>/config.toml`.
    """
    override = os.getenv("RAG_CONFIG_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "config.toml"


def _load() -> dict[str, Any]:
    """Read and parse config.toml. Called once at import time.

    Not cached: the module-level `CONFIG` is the singleton. If you need
    to reload (e.g. in a long-running notebook after editing config.toml),
    call `reload()` rather than caching here, which would otherwise hide
    the file from being re-read on demand.
    """
    path = config_path()
    if not path.exists():
        raise FileNotFoundError(
            f"config.toml not found at {path}. "
            f"Set RAG_CONFIG_PATH or place config.toml at the project root."
        )
    with path.open("rb") as f:
        return tomllib.load(f)


CONFIG: dict[str, Any] = _load()


def reload() -> dict[str, Any]:
    """Re-read config.toml and update CONFIG in place.

    Useful for editing config.toml in a Jupyter session without restarting
    the kernel. Mutates the existing dict so any module that did
    `from .config import CONFIG` keeps a live view — reassigning the name
    would leave stale references behind.
    """
    fresh = _load()
    CONFIG.clear()
    CONFIG.update(fresh)
    return CONFIG


# Convenience accessors. Keeping these as functions (not module-level
# constants) means tests can monkeypatch os.environ and re-call without
# reloading the module.

def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST") or CONFIG["llm"]["default_host"]


def ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL") or CONFIG["llm"]["model"]


def chroma_dir() -> str | None:
    """Persistent Chroma directory, or None for an ephemeral in-memory index."""
    return os.getenv("CHROMA_DIR") or None
