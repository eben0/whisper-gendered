"""Logging configuration (level + format) for the whole process."""

from __future__ import annotations

import logging

from config import settings


def configure() -> None:
    """Apply the process-wide logging config. Call once at entrypoint start."""
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
