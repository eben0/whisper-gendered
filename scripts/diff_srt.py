"""Diff two SRT files of identical structure (same timestamps + indices), showing
only the entries whose text differs. Prints old vs new for each diverging line.

Usage:
    python scripts/diff_srt.py <old.srt> <new.srt>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_srt(path: Path) -> list[tuple[int, str, str]]:
    """Return [(index, time, text)]. Text is joined with spaces (multi-line cues)."""
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    entries: list[tuple[int, str, str]] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        time = lines[1].strip()
        cue = " ".join(line.rstrip() for line in lines[2:])
        entries.append((idx, time, cue))
    return entries


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])
    old = parse_srt(old_path)
    new = parse_srt(new_path)

    if len(old) != len(new):
        print(f"WARNING: entry count differs: old={len(old)} new={len(new)}")
    # Align by index where possible.
    old_by_idx = {idx: (t, txt) for idx, t, txt in old}
    new_by_idx = {idx: (t, txt) for idx, t, txt in new}
    common = sorted(set(old_by_idx) & set(new_by_idx))
    only_old = sorted(set(old_by_idx) - set(new_by_idx))
    only_new = sorted(set(new_by_idx) - set(old_by_idx))

    diffs = []
    for i in common:
        ot, ox = old_by_idx[i]
        nt, nx = new_by_idx[i]
        if ox != nx:
            diffs.append((i, ot, ox, nx))

    print(f"Total entries: old={len(old)}  new={len(new)}  common={len(common)}")
    print(f"Differing text entries: {len(diffs)}  ({len(diffs)/max(1,len(common))*100:.1f}% of common)")
    if only_old:
        print(f"Only in old: {only_old[:10]}{'...' if len(only_old)>10 else ''}")
    if only_new:
        print(f"Only in new: {only_new[:10]}{'...' if len(only_new)>10 else ''}")
    print()
    print(f"--- {old_path.name}")
    print(f"+++ {new_path.name}")
    print("=" * 72)
    for idx, time, ox, nx in diffs:
        print(f"#{idx}  {time}")
        print(f"  -  {ox}")
        print(f"  +  {nx}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
