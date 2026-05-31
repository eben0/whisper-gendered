"""Pytest root conftest — makes repo root the rootdir and adds src/ to sys.path."""
import sys
from pathlib import Path

# Add src/ so tests can import from both the old root-level modules (during
# migration) and the new src.* paths simultaneously.
sys.path.insert(0, str(Path(__file__).parent / "src"))
