"""Unit tests for the side-file path translation helper.

These tests pin down the URL-decoded, prefix-translated, suffix-replaced path
the server uses to write a Hebrew SRT next to the source video on the share.
Pure function, no filesystem I/O — the file-write side effect is exercised by
manual E2E. Path assertions use Windows separators (this is a Windows-only
project; the server runs on the host that owns the share mount).
"""

from server import _compute_side_file_path


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
