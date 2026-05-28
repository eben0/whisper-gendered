"""Tests for the ISO 639-1 -> display-name helper and the
target-script-ratio sanity check."""

from pipeline.lang import language_name, target_script_ratio


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


# -- target_script_ratio --------------------------------------------------- #

def test_target_script_ratio_hebrew_high_on_pure_hebrew():
    # Real Hebrew sentence translated by NLLB (in conversation).
    r = target_script_ratio("תחרות מעלה את הטוב ביותר באנשים.", "Hebrew")
    assert r is not None and r == 1.0


def test_target_script_ratio_hebrew_low_on_latin_text():
    # The exact failure mode we hit in production: NLLB wrote Spanish when
    # asked for Hebrew. The check must flag this with a near-zero ratio.
    r = target_script_ratio("La competencia saca lo mejor de la gente.", "Hebrew")
    assert r is not None and r == 0.0


def test_target_script_ratio_hebrew_mixed_text():
    # Hebrew with embedded English brand name — still mostly Hebrew letters.
    r = target_script_ratio("השם שלי הוא Claude.", "Hebrew")
    assert r is not None and 0.5 < r < 0.95


def test_target_script_ratio_ignores_digits_and_punctuation():
    # Only letters count, not "1.", ":", spaces.
    r = target_script_ratio("1. שלום, עולם!", "Hebrew")
    assert r is not None and r == 1.0


def test_target_script_ratio_returns_none_for_latin_targets():
    # French/Spanish/etc. have no usable Unicode signal.
    assert target_script_ratio("Bonjour le monde", "French") is None
    assert target_script_ratio("Hola mundo", "Spanish") is None
    assert target_script_ratio("Guten Tag", "German") is None


def test_target_script_ratio_handles_empty_or_no_letters():
    # Empty input -> 0.0 (no letters to be in the target script).
    assert target_script_ratio("", "Hebrew") == 0.0
    # Only digits/punctuation -> 0.0 (same denominator-protection).
    assert target_script_ratio("1 2 3 !!!", "Hebrew") == 0.0


def test_target_script_ratio_unknown_target_returns_none():
    # Unmapped language passes through as None — the caller must handle it.
    assert target_script_ratio("anything", "Klingon") is None
    assert target_script_ratio("anything", "") is None


def test_target_script_ratio_arabic_russian_greek():
    # Spot-check a few other non-Latin scripts so the regex bounds work.
    assert target_script_ratio("مرحبا بالعالم", "Arabic") == 1.0
    assert target_script_ratio("Привет мир", "Russian") == 1.0
    assert target_script_ratio("Γειά σου κόσμε", "Greek") == 1.0
    # And cross-script: Hebrew text scored against Arabic must be ~0.
    assert target_script_ratio("שלום עולם", "Arabic") == 0.0
