"""Pin the defaults of the new env keys added for Plan 'improve-gender-
detection'. Defaults are deliberately conservative so existing
deployments behave identically until the operator opts in.
"""
import importlib

import config


def test_default_gender_classifier_is_pitch(monkeypatch):
    monkeypatch.delenv("GENDER_CLASSIFIER", raising=False)
    importlib.reload(config)
    assert config.settings.GENDER_CLASSIFIER == "pitch"


def test_default_gender_ml_model_is_alefiury_wav2vec2(monkeypatch):
    monkeypatch.delenv("GENDER_ML_MODEL", raising=False)
    importlib.reload(config)
    assert config.settings.GENDER_ML_MODEL == (
        "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    )


def test_default_translate_context_lines_is_4(monkeypatch):
    monkeypatch.delenv("TRANSLATE_CONTEXT_LINES", raising=False)
    importlib.reload(config)
    assert config.settings.TRANSLATE_CONTEXT_LINES == 4


def test_default_gender_ab_output_is_false(monkeypatch):
    monkeypatch.delenv("GENDER_AB_OUTPUT", raising=False)
    importlib.reload(config)
    assert config.settings.GENDER_AB_OUTPUT is False
