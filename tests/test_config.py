import importlib

import src.config as config_module


def test_new_settings_have_defaults():
    s = config_module.Settings()
    assert s.CHUNK_DURATION_SEC == 300
    assert s.TRANSLATE_CONCURRENCY == 3
    assert s.CLAUDE_MAX_RETRIES == 4


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("CHUNK_DURATION_SEC", "120")
    monkeypatch.setenv("TRANSLATE_CONCURRENCY", "2")
    monkeypatch.setenv("CLAUDE_MAX_RETRIES", "1")
    importlib.reload(config_module)
    s = config_module.Settings()
    assert s.CHUNK_DURATION_SEC == 120
    assert s.TRANSLATE_CONCURRENCY == 2
    assert s.CLAUDE_MAX_RETRIES == 1
    importlib.reload(config_module)  # restore module-level singleton
