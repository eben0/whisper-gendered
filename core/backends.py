"""Translation-backend selection and de-magicked constants.

The concrete backends live in ``core/backend_claude.py`` and ``core/backend_local.py``.
``core/backend_factory.py`` owns the match-case factory and the module-level singleton.
This module re-exports what the rest of the codebase needs for backward compat.
"""

from __future__ import annotations

from core.backend_base import TranslationBackend
from core.backend_factory import backend, create_backend, CLAUDE, LOCAL

# Re-export the singleton and type for callers that import from core.backends
__all__ = ["backend", "create_backend", "LOCAL", "CLAUDE", "TranslationBackend"]
