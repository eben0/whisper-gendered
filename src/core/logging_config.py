"""Process-wide logging configuration."""

from __future__ import annotations

import logging

from src.config import settings


def configure() -> None:
    """Apply logging config. Call once at entrypoint start."""
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
