"""Translation backend package.

Public API re-exports for clean external imports.
"""

from src.backends.factory import (
    TranslationBackend,
    LOCAL,
    CLAUDE,
    create_backend,
)

__all__ = ["TranslationBackend", "LOCAL", "CLAUDE", "create_backend"]
