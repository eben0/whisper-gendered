"""Abstract base class for translation backends.

All concrete backends (``ClaudeBackend``, ``LocalBackend``) inherit from this class
and implement ``is_()`` for type-safe identity checks without isinstance calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TranslationBackend(ABC):
    """Common interface shared by all translation backends.

    Use ``backend.is_(LOCAL)`` or ``backend.is_(CLAUDE)`` (from
    ``core.backends``) to branch on the active backend type.
    """

    def is_(self, backend_type: str) -> bool:
        """Return True if this backend matches ``backend_type``.

        ``backend_type`` should be one of the constants ``LOCAL`` or ``CLAUDE``
        defined in ``core.backends``. Example::

            if backends.backend.is_(backends.LOCAL):
                ...

        The default implementation compares against :attr:`_backend_type`,
        which concrete subclasses must set as a class attribute.
        """
        return getattr(self, "_backend_type", None) == backend_type

    @abstractmethod
    async def translate_batch_async(
        self, texts: list[str], gender: str | None, target: str, **kwargs: Any
    ) -> list[str]: ...

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    def model_name(self) -> str: ...
