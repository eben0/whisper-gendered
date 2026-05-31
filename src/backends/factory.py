"""Translation backend factory, abstract base class, and type constants.

Merged from: core/backend_base.py + core/backends.py + core/backend_factory.py
(per PR #2 requirement: merge backends.py + backend_base.py -> factory.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings

# Named constants — replace all inline "local"/"claude" string literals.
LOCAL = "local"
CLAUDE = "claude"


class TranslationBackend(ABC):
    """Common interface for all translation backends.

    Use ``backend.is_(LOCAL)`` or ``backend.is_(CLAUDE)`` for identity checks.
    """

    def is_(self, backend_type: str) -> bool:
        """Return True if this instance matches ``backend_type``."""
        return getattr(self, "_backend_type", None) == backend_type

    @abstractmethod
    async def translate_batch_async(
        self, texts: list[str], gender: str | None, target: str, **kwargs: Any
    ) -> list[str]: ...

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    def model_name(self) -> str: ...


def create_backend(settings: "Settings") -> TranslationBackend:
    """Instantiate the backend named by ``settings.TRANSLATION_BACKEND``.

    No module-level singleton — caller creates and holds the instance.
    Raises ``ValueError`` for unrecognised backend names.
    """
    from src.backends.claude import ClaudeBackend
    from src.backends.local import LocalBackend

    match settings.TRANSLATION_BACKEND.strip().lower():
        case "claude":
            return ClaudeBackend(settings)
        case "local":
            return LocalBackend(settings)
        case other:
            raise ValueError(
                f"Unknown TRANSLATION_BACKEND {other!r}. Valid: 'claude', 'local'."
            )
