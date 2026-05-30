"""Side-file I/O: path resolution, atomic writes, and translation summaries.

All path construction is pure (no filesystem I/O) and unit-testable. Atomic writes
use a ``.tmp``->``os.replace`` pattern so partial writes are never visible.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path

from config import settings
from pipeline.lang import target_script_ratio
from pipeline.segment import Segment

log = logging.getLogger("core.side_file")

TMP_SUFFIX = ".tmp"
ENCODING = "utf-8"
SUMMARY_SUFFIX = ".summary.txt"
ALT_CLASSIFIER_SRT_SUFFIX = ".he.alt-classifier.srt"


def _compute_side_file_path(
    video_file_url: str,
    src_prefix: str,
    dst_prefix: str,
    suffix: str,
) -> Path | None:
    """Translate the Bazarr ``video_file`` URL into a local SRT path on the share.

    Returns ``None`` when the feature is disabled (either prefix empty), the URL
    is empty, or the URL does not start with the configured source prefix.
    Pure function — no filesystem I/O — so it is unit-testable without mounting
    a share. The actual write happens in ``_try_save_side_file``.
    """
    if not src_prefix or not dst_prefix or not video_file_url:
        return None
    decoded = urllib.parse.unquote(video_file_url)
    if not decoded.startswith(src_prefix):
        return None
    rel = decoded[len(src_prefix):].replace("/", "\\")
    local = Path(dst_prefix + rel)
    stem = local.with_suffix("")
    return stem.with_name(stem.name + suffix)


def _compute_summary_path(srt_path: Path) -> Path:
    """Derive the summary file path from an SRT path, paired by name.

    ``episode.he.srt`` -> ``episode.he.summary.txt`` (preserving the language
    suffix so the pair stays visually grouped in a directory listing).
    Non-``.srt`` inputs just get ``.summary.txt`` appended.

    Pure function so the naming contract is testable without filesystem I/O.
    """
    name = srt_path.name
    if name.lower().endswith(".srt"):
        return srt_path.with_name(name[: -len(".srt")] + SUMMARY_SUFFIX)
    return srt_path.with_name(name + SUMMARY_SUFFIX)


def _try_save_side_file(
    body: str,
    summary: str | None,
    video_file_url: str,
    suffix: str | None = None,
) -> None:
    """Write the SRT body — and optionally a sibling summary file — to the
    translated path next to the source video.

    ``suffix`` overrides ``settings.SAVE_SRT_SUFFIX`` when set. Used by Plan
    Task 5 to emit a parallel alt-classifier SRT (``*.he.alt-classifier.srt``)
    for A/B comparison.

    Failures (missing share, permission denied, malformed URL) log a warning
    and do not affect the HTTP response. Atomic-ish: write to ``<target>.tmp``
    first, then ``os.replace`` onto the final name. The summary file uses the
    same atomic pattern with a ``.summary.txt`` sibling derived via
    ``_compute_summary_path`` so e.g. ``Show.S01E01.he.srt`` pairs with
    ``Show.S01E01.he.summary.txt``.
    """
    effective_suffix = suffix or settings.SAVE_SRT_SUFFIX
    try:
        target = _compute_side_file_path(
            video_file_url,
            settings.SAVE_SRT_VIDEO_PREFIX,
            settings.SAVE_SRT_LOCAL_PREFIX,
            effective_suffix,
        )
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + TMP_SUFFIX)
        tmp.write_text(body, encoding=ENCODING)
        os.replace(str(tmp), str(target))
        log.info("side-file saved: %s", target)
        if summary:
            summary_target = _compute_summary_path(target)
            tmp_sum = summary_target.with_name(summary_target.name + TMP_SUFFIX)
            tmp_sum.write_text(summary, encoding=ENCODING)
            os.replace(str(tmp_sum), str(summary_target))
            log.info("side-file summary saved: %s", summary_target)
    except Exception:
        log.exception("side-file save failed; continuing without it")


def _fmt_timestamp(t: float) -> str:
    """Render a float-seconds time as ``HH:MM:SS`` for the summary file."""
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_translation_summary(
    *,
    request_id: str,
    video_file_url: str,
    source_language_iso: str,
    source_language_name: str,
    target_language: str,
    backend: str,
    backend_model: str | None,
    segments: list[Segment],
    wall_seconds: float,
) -> str:
    """Produce a multi-line human-readable summary of a completed translation.

    Designed to live next to the produced SRT and to be eyeball-checkable
    without opening the full subtitle file. The most important line is the
    target-script ratio — a ratio < ~0.5 on a known non-Latin target almost
    always indicates a backend mis-translation (we hit exactly this when
    NLLB's ``forced_bos_token_id`` plumbing broke and the model produced
    Spanish/Romanian/Tswana instead of Hebrew).
    """
    # Joined text for ratio/length stats. ``\n`` joins keep per-line counts
    # roughly comparable to the SRT body.
    text = "\n".join(s.text for s in segments)
    total_chars = len(text)
    n_segs = len(segments)
    ratio = target_script_ratio(text, target_language)

    if ratio is None:
        script_line = (
            f"Script check:    n/a (target language has no non-Latin script signal)"
        )
    else:
        pct = ratio * 100
        flag = " ✓" if ratio >= 0.50 else " ⚠ WRONG LANGUAGE?"
        script_line = (
            f"Script check:    {pct:5.1f}% of letters are in the {target_language} "
            f"script{flag}"
        )

    # Sample lines — first, two from middle quarters, last. Useful for a quick
    # eyeball without opening the file.
    samples: list[tuple[str, Segment]] = []
    if n_segs:
        idxs = sorted({0, n_segs // 4, (3 * n_segs) // 4, n_segs - 1})
        labels = ["first", "early", "late", "last"]
        for label, idx in zip(labels, idxs):
            samples.append((label, segments[idx]))

    sample_block = ""
    if samples:
        rows: list[str] = []
        for label, seg in samples:
            ts = _fmt_timestamp(seg.start)
            # Truncate very long lines to keep the file scannable.
            shown = seg.text if len(seg.text) <= 120 else seg.text[:117] + "..."
            rows.append(f"  {label:<6} ({ts}) {shown}")
        sample_block = "Sample lines:\n" + "\n".join(rows)

    decoded_video = (
        urllib.parse.unquote(video_file_url) if video_file_url else "(none)"
    )
    backend_str = f"{backend}" + (f" ({backend_model})" if backend_model else "")

    lines = [
        "=== Translation summary ===",
        f"Request:         {request_id}",
        f"Video:           {decoded_video}",
        f"Source language: {source_language_name} (lang={source_language_iso!r})",
        f"Target language: {target_language}",
        f"Backend:         {backend_str}",
        f"Segments:        {n_segs}",
        f"Output chars:    {total_chars}",
        script_line,
        f"Wall time:       {wall_seconds:.1f}s ({int(wall_seconds)//60:02d}:{int(wall_seconds)%60:02d})",
    ]
    if sample_block:
        lines.append("")
        lines.append(sample_block)
    return "\n".join(lines) + "\n"
