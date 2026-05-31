"""Unit tests for the side-file path translation helper.

These tests pin down the URL-decoded, prefix-translated, suffix-replaced path
the server uses to write a Hebrew SRT next to the source video on the share.
Pure function, no filesystem I/O — the file-write side effect is exercised by
manual E2E. Path assertions use Windows separators (this is a Windows-only
project; the server runs on the host that owns the share mount).
"""

from pathlib import Path

from src.side_file import _compute_side_file_path, _compute_summary_path


def test_translates_linux_prefix_to_windows_prefix():
    target = _compute_side_file_path(
        "/media/tv/Show/episode.mp4",
        src_prefix="/media",
        dst_prefix="Z:\\media",
        suffix=".he.srt",
    )
    assert str(target) == "Z:\\media\\tv\\Show\\episode.he.srt"


def test_handles_url_encoded_characters():
    target = _compute_side_file_path(
        "/media/tv/Oz%20%281997%29/S04E13%20-%20Blizzard.mp4",
        src_prefix="/media",
        dst_prefix="Z:\\media",
        suffix=".he.srt",
    )
    assert str(target) == "Z:\\media\\tv\\Oz (1997)\\S04E13 - Blizzard.he.srt"


def test_replaces_only_the_last_extension():
    # Two-dot filename stays correct: only ".mp4" is removed, ".he.srt" appended.
    target = _compute_side_file_path(
        "/media/Show.S01E01.mp4",
        src_prefix="/media",
        dst_prefix="Z:\\media",
        suffix=".he.srt",
    )
    assert str(target) == "Z:\\media\\Show.S01E01.he.srt"


def test_returns_none_when_video_file_url_does_not_match_prefix():
    assert _compute_side_file_path(
        "/var/data/x.mp4",
        src_prefix="/media",
        dst_prefix="Z:\\media",
        suffix=".he.srt",
    ) is None


def test_returns_none_when_disabled_via_empty_prefix():
    # Either prefix empty -> feature off.
    assert _compute_side_file_path(
        "/media/x.mp4", src_prefix="", dst_prefix="Z:\\media", suffix=".he.srt",
    ) is None
    assert _compute_side_file_path(
        "/media/x.mp4", src_prefix="/media", dst_prefix="", suffix=".he.srt",
    ) is None


def test_returns_none_when_video_file_url_is_empty():
    assert _compute_side_file_path(
        "", src_prefix="/media", dst_prefix="Z:\\media", suffix=".he.srt",
    ) is None


# -- _compute_summary_path ------------------------------------------------- #

def test_summary_path_pairs_with_he_srt():
    # The common case: SRT was written as foo.he.srt, summary lives at
    # foo.he.summary.txt — same directory, paired by filename.
    srt = Path(r"Z:\media\tv\Show\episode.he.srt")
    assert _compute_summary_path(srt) == Path(
        r"Z:\media\tv\Show\episode.he.summary.txt"
    )


def test_summary_path_preserves_language_suffix():
    # Multi-language scenario: episode has .ar.srt, .fr.srt, etc.; the
    # summary must NOT clobber the language tag (the obvious .with_suffix
    # trap that did the wrong thing in the first draft of this code).
    for lang_suffix in (".he", ".ar", ".fr", ".es", ".ja"):
        srt = Path(rf"Z:\media\tv\Show\episode{lang_suffix}.srt")
        expected = Path(rf"Z:\media\tv\Show\episode{lang_suffix}.summary.txt")
        assert _compute_summary_path(srt) == expected


def test_summary_path_handles_uppercase_extension():
    # Some clients send .SRT on Windows shares; we treat that as still an SRT.
    srt = Path(r"Z:\media\tv\Show\episode.he.SRT")
    assert _compute_summary_path(srt) == Path(
        r"Z:\media\tv\Show\episode.he.summary.txt"
    )


def test_summary_path_with_no_srt_extension_appends():
    # Unusual but defended: configured suffix without .srt (e.g. just .he
    # if someone overrides SAVE_SRT_SUFFIX). Append .summary.txt rather than
    # clobber an unknown extension.
    srt = Path(r"Z:\media\tv\Show\episode.he")
    assert _compute_summary_path(srt) == Path(
        r"Z:\media\tv\Show\episode.he.summary.txt"
    )


def test_summary_path_with_dots_in_filename():
    # Multi-dot stems (very common in release names: "Show.S01E01.WEBRip-1080p")
    # must not lose any of those dots — only the trailing ``.srt`` goes.
    srt = Path(r"Z:\media\Show.S01E01.WEBRip-1080p.he.srt")
    assert _compute_summary_path(srt) == Path(
        r"Z:\media\Show.S01E01.WEBRip-1080p.he.summary.txt"
    )
