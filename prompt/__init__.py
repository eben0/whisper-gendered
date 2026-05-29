"""Prompt templates loaded from .md files with ``{placeholder}`` substitution.

Templates live under ``prompt/<category>/<name>.md`` so they're easy to
diff and review without scanning Python. Call ``load("translate/base",
target_language=...)`` to read a template and substitute variables via
``str.format``.

Why separate files: subtitle prompt engineering happens by re-reading the
exact wording the LLM sees. Inline f-strings make the whole prompt hard
to read at a glance and harder to diff across changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).parent


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    """Read ``prompt/<name>.md`` once and cache the raw text."""
    return (_HERE / f"{name}.md").read_text(encoding="utf-8").strip()


def load(name: str, **vars: object) -> str:
    """Read ``prompt/<name>.md`` and substitute ``{placeholder}`` with vars.

    No vars → returns the raw template unchanged (so templates without
    placeholders don't need to escape literal braces).
    """
    text = _read(name)
    return text.format(**vars) if vars else text
