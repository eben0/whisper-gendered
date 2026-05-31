"""Side-file I/O: path resolution, atomic writes, and translation summaries."""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings
    from pipeline.segment import Segment

from pipeline.lang import target_script_ratio

log = logging.getLogger("side_file")

TMP_SUFFIX = ".tmp"
ENCODING = "utf-8"
SUMMARY_SUFFIX = ".summary.txt"
ALT_CLASSIFIER_SRT_SUFFIX = ".he.alt-classifier.srt"


class SideFile:
    """Handles side-file path resolution, atomic writes, and summary generation."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    def compute_path(
        self,
        video_file_url: str,
        suffix: str | None = None,
    ) -> Path | None:
        """Translate a Bazarr video URL to a local SRT path. Pure — no filesystem I/O."""
        src_prefix = self._settings.SAVE_SRT_VIDEO_PREFIX
        dst_prefix = self._settings.SAVE_SRT_LOCAL_PREFIX
        effective_suffix = suffix or self._settings.SAVE_SRT_SUFFIX

        if not src_prefix or not dst_prefix or not video_file_url:
            return None
        decoded = urllib.parse.unquote(video_file_url)
        if not decoded.startswith(src_prefix):
            return None
        rel = decoded[len(src_prefix):].replace("/", "\\")
        local = Path(dst_prefix + rel)
        stem = local.with_suffix("")
        return stem.with_name(stem.name + effective_suffix)

    def compute_summary_path(self, srt_path: Path) -> Path:
        """Derive the .summary.txt path from an SRT path. Pure."""
        name = srt_path.name
        if name.lower().endswith(".srt"):
            return srt_path.with_name(name[: -len(".srt")] + SUMMARY_SUFFIX)
        return srt_path.with_name(name + SUMMARY_SUFFIX)

    def try_save(
        self,
        body: str,
        summary: str | None,
        video_file_url: str,
        suffix: str | None = None,
    ) -> None:
        """Atomically write SRT body and optional summary. Failures are logged, not raised."""
        try:
            target = self.compute_path(video_file_url, suffix)
            if target is None:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + TMP_SUFFIX)
            tmp.write_text(body, encoding=ENCODING)
            os.replace(str(tmp), str(target))
            log.info("side-file saved: %s", target)
            if summary:
                summary_target = self.compute_summary_path(target)
                tmp_sum = summary_target.with_name(summary_target.name + TMP_SUFFIX)
                tmp_sum.write_text(summary, encoding=ENCODING)
                os.replace(str(tmp_sum), str(summary_target))
                log.info("side-file summary saved: %s", summary_target)
        except Exception:
            log.exception("side-file save failed; continuing without it")

    def build_summary(
        self,
        *,
        request_id: str,
        video_file_url: str,
        source_language_iso: str,
        source_language_name: str,
        target_language: str,
        backend: str,
        backend_model: str | None,
        segments: list["Segment"],
        wall_seconds: float,
    ) -> str:
        """Build a human-readable translation summary string."""
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

        samples: list[tuple[str, "Segment"]] = []
        if n_segs:
            idxs = sorted({0, n_segs // 4, (3 * n_segs) // 4, n_segs - 1})
            labels = ["first", "early", "late", "last"]
            for label, idx in zip(labels, idxs):
                samples.append((label, segments[idx]))

        sample_block = ""
        if samples:
            rows: list[str] = []
            for label, seg in samples:
                ts = self.fmt_timestamp(seg.start)
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

    @staticmethod
    def fmt_timestamp(t: float) -> str:
        """Render float seconds as HH:MM:SS."""
        if t < 0:
            t = 0.0
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
