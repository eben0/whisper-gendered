"""Filter SRT diff output to entries that contain Hebrew 2nd-person markers in
either the old or new version. Highlights where addressee-aware translation
likely had effect.

Usage:
    python scripts/diff_srt_filter_you.py <old.srt> <new.srt>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Hebrew 2nd-person markers (independent pronouns + common possessive/object
# enclitics). Conservative set — won't catch every 2nd-person form but covers
# the discriminating cases.
SECOND_PERSON_RE = re.compile(
    r"(?:"
    r"אתה|אתם|אתן|"          # masc-sg, masc-pl, fem-pl (vowelled "at" overlaps with too many other words to include bare)
    r"אליך|אלייך|אליכם|אליכן|"  # to-you (m-sg / f-sg / m-pl / f-pl)
    r"איתך|אתך|איתכם|איתכן|"    # with-you variants
    r"שלך|שלכם|שלכן|"        # of-yours (m/f-sg, m-pl, f-pl)
    r"לך|לכם|לכן"             # to-you (preposition forms; lekha/lakh/lakhem/lakhen)
    r")\b"
)


def parse_srt(path: Path):
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    out = []
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
        out.append((idx, time, cue))
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    old = {i: (t, x) for i, t, x in parse_srt(Path(sys.argv[1]))}
    new = {i: (t, x) for i, t, x in parse_srt(Path(sys.argv[2]))}
    common = sorted(set(old) & set(new))

    hits = []
    for i in common:
        ot, ox = old[i]
        nt, nx = new[i]
        if ox == nx:
            continue
        if SECOND_PERSON_RE.search(ox) or SECOND_PERSON_RE.search(nx):
            hits.append((i, ot, ox, nx))

    total_diffs = sum(1 for i in common if old[i][1] != new[i][1])
    print(f"Diffs total: {total_diffs}")
    print(f"Diffs touching 2nd-person markers: {len(hits)}")
    print("=" * 72)
    for idx, time, ox, nx in hits:
        print(f"#{idx}  {time}")
        print(f"  -  {ox}")
        print(f"  +  {nx}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
