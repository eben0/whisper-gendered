"""Backend factory — resolves TRANSLATION_BACKEND config to a concrete backend instance."""

from __future__ import annotations

from config import settings
from core.backend_base import TranslationBackend
from core.backend_claude import ClaudeBackend
from core.backend_local import LocalBackend

CLAUDE = "claude"
LOCAL = "local"


def create_backend() -> TranslationBackend:
    """Instantiate the backend named by ``settings.TRANSLATION_BACKEND``.

    Raises ``ValueError`` for unrecognised backend names so misconfiguration
    is caught at startup rather than at the first translation request.
    """
    match settings.TRANSLATION_BACKEND.strip().lower():
        case "claude":
            return ClaudeBackend()
        case "local":
            return LocalBackend()
        case other:
            raise ValueError(
                f"Unknown TRANSLATION_BACKEND {other!r}. "
                f"Valid values: 'claude', 'local'."
            )


# Module-level singleton — resolved once at import time from current settings.
backend: TranslationBackend = create_backend()
