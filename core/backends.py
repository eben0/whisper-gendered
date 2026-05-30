"""Translation-backend selection and de-magicked constants.

The concrete backends live in ``core/backend_claude.py`` and ``core/backend_local.py``.
``core/backend_factory.py`` owns the match-case factory and the module-level singleton.
This module re-exports what the rest of the codebase needs for backward compat.
"""

from __future__ import annotations

from config import settings
from core.backend_factory import backend, create_backend, CLAUDE, LOCAL

# Re-export the singleton for callers that import from core.backends
__all__ = ["backend", "create_backend", "is_local", "LOCAL", "CLAUDE"]


def is_local() -> bool:
    """True when the configured translation backend is the on-device model."""
    return settings.TRANSLATION_BACKEND.strip().lower() == LOCAL
