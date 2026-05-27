from pipeline.transcribe import Segment
from pipeline.chunk import Chunk, make_chunks


def _seg(start, end, text="x"):
    return Segment(start=start, end=end, text=text)


def test_empty_returns_no_chunks():
    assert make_chunks([], target_sec=300) == []


def test_single_chunk_when_under_target():
    segs = [_seg(0, 5), _seg(5, 10), _seg(10, 15)]
    chunks = make_chunks(segs, target_sec=300)
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == 15
    assert chunks[0].segments == segs


def test_splits_when_current_span_reaches_target():
    # target 10s: seg c starts a new chunk because [0,10) already spans >= 10.
    a, b, c, d = _seg(0, 4), _seg(4, 10), _seg(10, 14), _seg(14, 20)
    chunks = make_chunks([a, b, c, d], target_sec=10)
    assert [ [s for s in ch.segments] for ch in chunks ] == [[a, b], [c, d]]
    assert (chunks[0].start, chunks[0].end) == (0, 10)
    assert (chunks[1].start, chunks[1].end) == (10, 20)


def test_long_single_segment_is_its_own_chunk():
    a, b = _seg(0, 50), _seg(50, 55)
    chunks = make_chunks([a, b], target_sec=10)
    assert chunks[0].segments == [a]
    assert chunks[1].segments == [b]


def test_tiny_trailing_chunk_merges_into_previous():
    # target 10, merge_ratio 0.5 -> trailing chunk shorter than 5s merges back.
    a, b, c = _seg(0, 6), _seg(6, 12), _seg(12, 13)  # c spans 1s, < 5s
    chunks = make_chunks([a, b, c], target_sec=10, merge_ratio=0.5)
    assert len(chunks) == 1
    assert chunks[0].segments == [a, b, c]
    assert chunks[0].end == 13
