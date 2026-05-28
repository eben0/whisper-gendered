import numpy as np
import pytest
from pyannote.core import Annotation, Segment as PSegment

import server
from pipeline.transcribe import Segment


@pytest.fixture
def two_chunk_segments():
    # Two 6s segments -> with target 5s these land in separate chunks.
    return [
        Segment(start=0.0, end=6.0, text="hello"),
        Segment(start=6.0, end=12.0, text="world"),
    ]


def _install_fakes(monkeypatch, segments, gender_aware):
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew" if gender_aware else "Japanese")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 5)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)

    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segments))
    monkeypatch.setattr(server, "_load_wav_mono", lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "SPEAKER_00"
        return ann

    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)
    monkeypatch.setattr(server.diarize, "assign_speaker", lambda seg, ann: "SPEAKER_00")
    monkeypatch.setattr(server.gender, "detect_genders", lambda audio, sr, ann: {"SPEAKER_00": "female"})

    async def fake_translate(texts, gender, target, client, addressee_gender=None):
        return [f"{t}|{gender}" for t in texts]

    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)


@pytest.mark.asyncio
async def test_gender_aware_preserves_order_and_applies_gender(monkeypatch, two_chunk_segments):
    _install_fakes(monkeypatch, two_chunk_segments, gender_aware=True)
    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    assert [s.text for s in out] == ["hello|female", "world|female"]
    assert [(s.start, s.end) for s in out] == [(0.0, 6.0), (6.0, 12.0)]


@pytest.mark.asyncio
async def test_plain_translate_no_diarization(monkeypatch, two_chunk_segments):
    _install_fakes(monkeypatch, two_chunk_segments, gender_aware=False)
    # If diarization were called in the plain path this would raise.
    monkeypatch.setattr(server.diarize, "diarize_waveform", lambda *a: (_ for _ in ()).throw(AssertionError("diarize called in plain path")))
    monkeypatch.setattr(server, "_load_wav_mono", lambda *a: (_ for _ in ()).throw(AssertionError("_load_wav_mono called in plain path")))
    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    assert [s.text for s in out] == ["hello|None", "world|None"]


@pytest.mark.asyncio
async def test_transcription_only_when_target_none(monkeypatch, two_chunk_segments):
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "none")
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(two_chunk_segments))
    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    assert [s.text for s in out] == ["hello", "world"]


@pytest.mark.asyncio
async def test_chunk_local_offset_maps_speakers(monkeypatch, two_chunk_segments):
    # Verifies _translate_chunk subtracts chunk.start before assign_speaker, using
    # the REAL assign_speaker (not mocked). Each chunk is one 6s segment; the
    # per-chunk annotation is slice-local with a turn over [0, 6).
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 5)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(two_chunk_segments))
    monkeypatch.setattr(server, "_load_wav_mono", lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "S"  # slice-local turn
        return ann

    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)
    monkeypatch.setattr(server.gender, "detect_genders", lambda audio, sr, ann: {"S": "female"})

    async def fake_translate(texts, gender, target, client, addressee_gender=None):
        return [f"{t}|{gender}" for t in texts]

    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)

    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Chunk 2's segment is absolute 6-12. Only if chunk_start (6.0) is subtracted
    # does its local midpoint (3.0) fall inside the [0,6) turn -> speaker "S" ->
    # "female". Without the subtraction it would miss the turn -> default "male".
    assert [s.text for s in out] == ["hello|female", "world|female"]


@pytest.mark.asyncio
async def test_addressee_rotates_within_chunk(monkeypatch):
    # Three consecutive 1-second segments, three different speakers M, F, M.
    # With CHUNK_DURATION_SEC=30, they stay in a single chunk.
    segs = [
        Segment(start=0.0, end=1.0, text="a"),
        Segment(start=1.0, end=2.0, text="b"),
        Segment(start=2.0, end=3.0, text="c"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segs))
    monkeypatch.setattr(server, "_load_wav_mono", lambda path: (np.zeros(16000 * 5, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    # Slice-local annotation: speakers cover [0,1), [1,2), [2,3) within the chunk.
    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "S_M1"
    ann[PSegment(1.0, 2.0)] = "S_F"
    ann[PSegment(2.0, 3.0)] = "S_M2"
    monkeypatch.setattr(server.diarize, "diarize_waveform", lambda *a, **k: ann)
    monkeypatch.setattr(
        server.gender, "detect_genders",
        lambda audio, sr, a: {"S_M1": "male", "S_F": "female", "S_M2": "male"},
    )

    addressees: list[str | None] = []

    async def fake_translate(texts, gender, target, client, addressee_gender=None):
        addressees.append(addressee_gender)
        return [f"{t}|{gender}|{addressee_gender}" for t in texts]

    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)

    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Three groups -> three translate calls, addressee = previous group's speaker gender.
    # First group: no prior speaker at all -> None. Then "male", then "female".
    assert addressees == [None, "male", "female"]
    assert [s.text for s in out] == ["a|male|None", "b|female|male", "c|male|female"]


@pytest.mark.asyncio
async def test_addressee_carries_across_chunks(monkeypatch):
    # Two chunks. Each chunk has one segment / one speaker. Chunk 1 = female,
    # chunk 2 = male. The cross-chunk carry should give chunk 2's first group
    # addressee_gender="female" (chunk 1's last group's speaker gender).
    segs = [
        Segment(start=0.0, end=6.0, text="a"),
        Segment(start=6.0, end=12.0, text="b"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 5)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segs))
    monkeypatch.setattr(server, "_load_wav_mono", lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "S"
        return ann
    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)

    chunk_index = {"i": 0}
    def fake_detect(audio, sr, ann):
        result = {"S": "female" if chunk_index["i"] == 0 else "male"}
        chunk_index["i"] += 1
        return result
    monkeypatch.setattr(server.gender, "detect_genders", fake_detect)

    async def fake_translate(texts, gender, target, client, addressee_gender=None):
        return [f"{t}|{gender}|{addressee_gender}" for t in texts]
    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)

    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Output order is preserved by the orchestrator's final list comprehension.
    # Chunk 1's segment "a": speaker female, addressee None (first chunk's first group).
    # Chunk 2's segment "b": speaker male, addressee female (carried).
    text_by_seg = {s.text.split("|")[0]: s.text for s in out}
    assert text_by_seg["a"] == "a|female|None"
    assert text_by_seg["b"] == "b|male|female"
