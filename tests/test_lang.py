"""Tests for the ISO 639-1 -> display-name helper."""

from pipeline.lang import language_name


def test_iso_to_name():
    assert language_name("en") == "English"
    assert language_name("he") == "Hebrew"
    assert language_name("fr") == "French"


def test_uppercase_iso_still_resolves():
    assert language_name("EN") == "English"
    assert language_name("He") == "Hebrew"


def test_strips_region_subtag():
    assert language_name("en-US") == "English"
    assert language_name("en_us") == "English"
    assert language_name("zh-Hans") == "Chinese"


def test_already_a_display_name_passes_through():
    assert language_name("English") == "English"
    assert language_name("Hebrew") == "Hebrew"
    # Case-insensitive canonicalisation.
    assert language_name("english") == "English"


def test_empty_and_none_use_default():
    assert language_name("") == "English"
    assert language_name(None) == "English"
    assert language_name("   ") == "English"
    assert language_name("", default="French") == "French"


def test_unknown_passes_through_verbatim():
    # Real Bazarr would never send this, but we want Claude/NLLB to still
    # get a non-empty string rather than crash.
    assert language_name("xx") == "xx"
    assert language_name("Klingon") == "Klingon"
