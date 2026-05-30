import numpy as np
import pytest
from pyannote.core import Annotation, Segment as PSegment

import server
from core import backends
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
    monkeypatch.setattr(server.audio, "_load_wav_mono",lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))
    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "SPEAKER_00"
        return ann

    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)
    monkeypatch.setattr(server.diarize, "assign_speaker", lambda seg, ann: "SPEAKER_00")
    monkeypatch.setattr(server.gender, "detect_genders", lambda audio, sr, ann: {"SPEAKER_00": "female"})

    async def fake_translate(texts, gender, target, addressee_gender=None, source_language="English", previous_context=None):
        return [f"{t}|{gender}" for t in texts]

    monkeypatch.setattr(backends.backend, "translate_batch_async", fake_translate)


@pytest.mark.asyncio
async def test_gender_aware_preserves_order_and_applies_gender(monkeypatch, two_chunk_segments):
    _install_fakes(monkeypatch, two_chunk_segments, gender_aware=True)
    source, target = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Source list keeps the raw transcript text.
    assert [s.text for s in source] == ["hello", "world"]
    # Target list carries the translated text with gender applied.
    assert [s.text for s in target] == ["hello|female", "world|female"]
    # Both share the same instants.
    assert [(s.start, s.end) for s in target] == [(0.0, 6.0), (6.0, 12.0)]
    assert [(s.start, s.end) for s in source] == [(0.0, 6.0), (6.0, 12.0)]


@pytest.mark.asyncio
async def test_plain_translate_no_diarization(monkeypatch, two_chunk_segments):
    _install_fakes(monkeypatch, two_chunk_segments, gender_aware=False)
    # If diarization were called in the plain path this would raise.
    monkeypatch.setattr(server.diarize, "diarize_waveform", lambda *a: (_ for _ in ()).throw(AssertionError("diarize called in plain path")))
    monkeypatch.setattr(server.audio, "_load_wav_mono",lambda *a: (_ for _ in ()).throw(AssertionError("_load_wav_mono called in plain path")))
    source, target = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    assert [s.text for s in source] == ["hello", "world"]
    assert [s.text for s in target] == ["hello|None", "world|None"]


@pytest.mark.asyncio
async def test_transcription_only_when_target_none(monkeypatch, two_chunk_segments):
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "none")
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(two_chunk_segments))
    source, target = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Translation disabled -> target is None, source is the raw transcript.
    assert target is None
    assert [s.text for s in source] == ["hello", "world"]


@pytest.mark.asyncio
async def test_chunk_local_offset_maps_speakers(monkeypatch, two_chunk_segments):
    # Verifies _translate_chunk subtracts chunk.start before assign_speaker, using
    # the REAL assign_speaker (not mocked). Each chunk is one 6s segment; the
    # per-chunk annotation is slice-local with a turn over [0, 6).
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 5)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(two_chunk_segments))
    monkeypatch.setattr(server.audio, "_load_wav_mono",lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))

    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "S"  # slice-local turn
        return ann

    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)
    monkeypatch.setattr(server.gender, "detect_genders", lambda audio, sr, ann: {"S": "female"})

    async def fake_translate(texts, gender, target, addressee_gender=None, source_language="English", previous_context=None):
        return [f"{t}|{gender}" for t in texts]

    monkeypatch.setattr(backends.backend, "translate_batch_async", fake_translate)

    _, target = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Chunk 2's segment is absolute 6-12. Only if chunk_start (6.0) is subtracted
    # does its local midpoint (3.0) fall inside the [0,6) turn -> speaker "S" ->
    # "female". Without the subtraction it would miss the turn -> default "male".
    assert [s.text for s in target] == ["hello|female", "world|female"]


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
    monkeypatch.setattr(server.audio, "_load_wav_mono",lambda path: (np.zeros(16000 * 5, dtype=np.float32), 16000))

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

    async def fake_translate(texts, gender, target, addressee_gender=None, source_language="English", previous_context=None):
        addressees.append(addressee_gender)
        return [f"{t}|{gender}|{addressee_gender}" for t in texts]

    monkeypatch.setattr(backends.backend, "translate_batch_async", fake_translate)

    _, target = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Three groups -> three translate calls, addressee = previous group's speaker gender.
    # First group: no prior speaker at all -> None. Then "male", then "female".
    assert addressees == [None, "male", "female"]
    assert [s.text for s in target] == ["a|male|None", "b|female|male", "c|male|female"]


@pytest.mark.skip(reason=(
    "Cross-chunk addressee carry removed in Plan Task 7 — pyannote re-numbers "
    "speakers per chunk so the previous chunk's last speaker gender is not "
    "necessarily the same person opening the next chunk. See the new test "
    "test_addressee_does_not_carry_across_chunks for the current contract."
))
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
    monkeypatch.setattr(server.audio, "_load_wav_mono",lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))
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

    async def fake_translate(texts, gender, target, addressee_gender=None, source_language="English", previous_context=None):
        return [f"{t}|{gender}|{addressee_gender}" for t in texts]
    monkeypatch.setattr(backends.backend, "translate_batch_async", fake_translate)

    _, target = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Output order is preserved by the orchestrator's final list comprehension.
    # Chunk 1's segment "a": speaker female, addressee None (first chunk's first group).
    # Chunk 2's segment "b": speaker male, addressee female (carried).
    text_by_seg = {s.text.split("|")[0]: s.text for s in target}
    assert text_by_seg["a"] == "a|female|None"
    assert text_by_seg["b"] == "b|male|female"


@pytest.mark.asyncio
async def test_addressee_does_not_carry_across_chunks(monkeypatch):
    """Plan Task 7. pyannote re-numbers speakers per chunk, so the previous
    chunk's last speaker gender can't be trusted as the addressee for the
    first group of the next chunk. With cross-chunk carry removed, both
    chunks' first group must see ``addressee_gender=None``.
    """
    segs = [
        Segment(start=0.0, end=6.0, text="a"),
        Segment(start=6.0, end=12.0, text="b"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 5)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.settings, "ADDRESSEE_GENDER_HINT_ENABLED", True)
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda path, language="en": list(segs))
    monkeypatch.setattr(server.audio, "_load_wav_mono",
                        lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))

    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "S"
        return ann
    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)

    chunk_idx = {"i": 0}
    def fake_detect(audio, sr, ann):
        # Chunk 0 = female, chunk 1 = male; the OLD behavior would have
        # carried "female" into chunk 1's first group.
        result = {"S": "female" if chunk_idx["i"] == 0 else "male"}
        chunk_idx["i"] += 1
        return result
    monkeypatch.setattr(server.gender, "detect_genders", fake_detect)

    addressees: list[str | None] = []
    async def fake_translate(texts, gender, target,
                             addressee_gender=None, source_language="English",
                             previous_context=None):
        addressees.append(addressee_gender)
        return [f"{t}|{gender}|{addressee_gender}" for t in texts]
    monkeypatch.setattr(backends.backend, "translate_batch_async",
                        fake_translate)

    await server.run_pipeline_async(server.Path("ignored.wav"), "en")

    # Two chunks, one group each — both first-of-chunk, so addressee None.
    assert addressees == [None, None], (
        f"cross-chunk addressee carry should be removed; got {addressees}"
    )


@pytest.mark.asyncio
async def test_addressee_hint_disabled_by_flag(monkeypatch):
    # Same setup as test_addressee_rotates_within_chunk, but with the feature
    # flag OFF: addressee_gender should be None for every translate call, even
    # though prev_group_gender would otherwise carry forward.
    segs = [
        Segment(start=0.0, end=1.0, text="a"),
        Segment(start=1.0, end=2.0, text="b"),
        Segment(start=2.0, end=3.0, text="c"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.settings, "ADDRESSEE_GENDER_HINT_ENABLED", False)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segs))
    monkeypatch.setattr(server.audio, "_load_wav_mono",lambda path: (np.zeros(16000 * 5, dtype=np.float32), 16000))

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

    async def fake_translate(texts, gender, target, addressee_gender=None, source_language="English", previous_context=None):
        addressees.append(addressee_gender)
        return [f"{t}|{gender}|{addressee_gender}" for t in texts]

    monkeypatch.setattr(backends.backend, "translate_batch_async", fake_translate)

    _, target = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Flag off -> every call sees addressee_gender=None regardless of rotation.
    assert addressees == [None, None, None]
    assert [s.text for s in target] == ["a|male|None", "b|female|None", "c|male|None"]


@pytest.mark.asyncio
async def test_source_language_from_request_reaches_translator(monkeypatch):
    # Bazarr-style ISO code "fr" should map to "French" and arrive at the
    # translator's source_language kwarg.
    segs = [Segment(start=0.0, end=1.0, text="x")]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segs))
    monkeypatch.setattr(server.audio, "_load_wav_mono",lambda path: (np.zeros(16000 * 2, dtype=np.float32), 16000))

    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "S"
    monkeypatch.setattr(server.diarize, "diarize_waveform", lambda *a, **k: ann)
    monkeypatch.setattr(server.gender, "detect_genders", lambda audio, sr, a: {"S": "male"})

    received: list[str] = []

    async def fake_translate(texts, gender, target, addressee_gender=None, source_language="English", previous_context=None):
        received.append(source_language)
        return [f"{t}|{source_language}" for t in texts]

    monkeypatch.setattr(backends.backend, "translate_batch_async", fake_translate)

    _, target = await server.run_pipeline_async(server.Path("ignored.wav"), "fr")
    assert received == ["French"]
    assert [s.text for s in target] == ["x|French"]


@pytest.mark.asyncio
async def test_pipeline_preserves_source_alongside_translation(monkeypatch):
    # The pipeline must hand back the source-language transcript intact even
    # though the orchestrator mutates the working segment list in place during
    # translation. The two returned lists share start/end stamps but carry
    # different .text and are independent Segment instances (so a downstream
    # mutation to one cannot bleed into the other).
    segs = [
        Segment(start=0.0, end=1.0, text="hello world"),
        Segment(start=1.0, end=2.0, text="goodbye"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segs))
    monkeypatch.setattr(server.audio, "_load_wav_mono",lambda path: (np.zeros(16000 * 3, dtype=np.float32), 16000))

    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "S"
    ann[PSegment(1.0, 2.0)] = "S"
    monkeypatch.setattr(server.diarize, "diarize_waveform", lambda *a, **k: ann)
    monkeypatch.setattr(server.gender, "detect_genders", lambda audio, sr, a: {"S": "male"})

    async def fake_translate(texts, gender, target, addressee_gender=None, source_language="English", previous_context=None):
        return [f"HE:{t}" for t in texts]
    monkeypatch.setattr(backends.backend, "translate_batch_async", fake_translate)

    source, target = await server.run_pipeline_async(server.Path("ignored.wav"), "en")

    # English transcript preserved verbatim.
    assert [s.text for s in source] == ["hello world", "goodbye"]
    # Target carries the translation.
    assert [s.text for s in target] == ["HE:hello world", "HE:goodbye"]
    # Same timestamps in both.
    assert [(s.start, s.end) for s in source] == [(0.0, 1.0), (1.0, 2.0)]
    assert [(s.start, s.end) for s in target] == [(0.0, 1.0), (1.0, 2.0)]
    # Independent instances — mutating target must not leak into source.
    assert source[0] is not target[0]
    target[0].text = "MUTATED"
    assert source[0].text == "hello world"


def test_asr_response_is_source_side_file_is_target(monkeypatch):
    # End-to-end at the /asr route: the HTTP body is the source-language
    # transcript (so Bazarr writes English content to its .en.srt), and the
    # side-file save receives the translated body (which lands at .he.srt).
    # This was the bug: previously both used the translated segments and
    # Bazarr's storage clobbered existing English subtitles.
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)

    source_segs = [Segment(start=0.0, end=1.0, text="hello world")]
    target_segs = [Segment(start=0.0, end=1.0, text="שלום עולם")]

    async def fake_pipeline(audio_path, language, _artifacts_out=None):
        if _artifacts_out is not None:
            _artifacts_out.append(server._PipelineArtifacts(
                raw_segments=list(source_segs), annotations=[],
            ))
        return source_segs, target_segs
    monkeypatch.setattr(server, "run_pipeline_async", fake_pipeline)

    # Stub the audio prep so the upload doesn't hit ffmpeg.
    monkeypatch.setattr(server.audio, "encode_to_wav", lambda src, dst: None)
    monkeypatch.setattr(server.audio, "prepare_unencoded", lambda src, dst: None)

    captured: dict[str, object] = {}
    def fake_save(body, summary, video_file_url):
        captured["body"] = body
        captured["summary"] = summary
        captured["video_file_url"] = video_file_url
    monkeypatch.setattr(server, "_try_save_side_file", fake_save)

    client = TestClient(server.app)
    response = client.post(
        "/asr",
        params={
            "task": "transcribe",
            "language": "en",
            "output": "srt",
            "encode": "false",
            "video_file": "/media/tv/x.mp4",
        },
        files={"audio_file": ("x.wav", b"RIFF0000WAVE", "audio/wav")},
    )

    assert response.status_code == 200
    # Response body MUST be the source-language transcript.
    assert "hello world" in response.text
    assert "שלום עולם" not in response.text
    # Side-file body MUST be the translated content.
    assert "שלום עולם" in captured["body"]
    assert "hello world" not in captured["body"]
    # Summary stats are about the target language (script ratio uses the
    # target segments).
    assert "Hebrew" in captured["summary"]
    assert "Script check" in captured["summary"]


@pytest.mark.asyncio
async def test_orchestrator_logs_segment_counts(monkeypatch, caplog):
    """run_pipeline_async must log input-segment and output-segment counts
    at INFO so a missing-segment investigation has hard numbers (Plan
    Task 4 — diagnostic for the 'missing sentences' user-reported issue).
    """
    import logging
    caplog.set_level(logging.INFO, logger="server")

    segs = [
        Segment(start=0.0, end=1.0, text="line 0"),
        Segment(start=1.0, end=2.0, text="line 1"),
        Segment(start=2.0, end=3.0, text="line 2"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 1)
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda path, language="en": list(segs))
    monkeypatch.setattr(server.audio, "_load_wav_mono",
                        lambda path: (np.zeros(16000 * 4, dtype=np.float32), 16000))

    ann = Annotation()
    ann[PSegment(0.0, 3.0)] = "S"
    monkeypatch.setattr(server.diarize, "diarize_waveform",
                        lambda *a, **k: ann)
    monkeypatch.setattr(server.gender, "detect_genders",
                        lambda audio, sr, a: {"S": "male"})

    async def fake_translate(texts, gender, target,
                             addressee_gender=None, source_language="English",
                             previous_context=None):
        return [f"HE: {t}" for t in texts]
    monkeypatch.setattr(backends.backend, "translate_batch_async",
                        fake_translate)

    await server.run_pipeline_async(server.Path("ignored.wav"), "en")

    msgs = [r.getMessage() for r in caplog.records]
    # New required log: transcribe count.
    assert any("transcribed" in m.lower() and "3" in m and "segments" in m.lower()
               for m in msgs), (
        f"missing transcribe-count log; messages: {msgs}"
    )
    # New required log: translate count.
    assert any("translated" in m.lower() and "3" in m for m in msgs), (
        f"missing translate-count log; messages: {msgs}"
    )


@pytest.mark.asyncio
async def test_orchestrator_logs_error_on_segment_count_mismatch(monkeypatch, caplog):
    """If translation returns fewer segments than transcribe produced, the
    orchestrator must emit an ERROR-level log line so the missing-
    sentence investigation has a clear trigger.
    """
    import logging
    caplog.set_level(logging.ERROR, logger="server")

    segs = [
        Segment(start=0.0, end=1.0, text="a"),
        Segment(start=1.0, end=2.0, text="b"),
        Segment(start=2.0, end=3.0, text="c"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 1)
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda path, language="en": list(segs))
    monkeypatch.setattr(server.audio, "_load_wav_mono",
                        lambda path: (np.zeros(16000 * 4, dtype=np.float32), 16000))

    ann = Annotation()
    ann[PSegment(0.0, 3.0)] = "S"
    monkeypatch.setattr(server.diarize, "diarize_waveform",
                        lambda *a, **k: ann)
    monkeypatch.setattr(server.gender, "detect_genders",
                        lambda audio, sr, a: {"S": "male"})

    # Simulate a chunked-translate flow that loses one segment. The
    # existing alignment fallback in pipeline/translate.py pads with the
    # source text on length mismatch, so in practice the loss would
    # surface as duplicate-text rather than a true count mismatch — but
    # we test the explicit-mismatch error path here by short-circuiting
    # the pipeline at the count level.
    async def fake_pipeline_inner_drop(audio_path, segments, target, client,
                                       source_language="English",
                                       precomputed_annotations=None):
        # Return one fewer segment than was passed in. ``_run_gender_aware``
        # now returns ``(segments, annotations_used)``; the empty list
        # satisfies the contract without exercising the alt-pass path.
        dropped = [
            Segment(s.start, s.end, f"HE: {s.text}") for s in segments[:-1]
        ]
        return dropped, []
    monkeypatch.setattr(server, "_run_gender_aware", fake_pipeline_inner_drop)

    await server.run_pipeline_async(server.Path("ignored.wav"), "en")

    err_msgs = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.ERROR]
    assert any("mismatch" in m.lower() or "lost" in m.lower()
               for m in err_msgs), (
        f"expected ERROR-level mismatch log; got: {err_msgs}"
    )


def test_asr_skips_side_file_when_target_is_none(monkeypatch):
    # When TARGET_LANGUAGE=none the pipeline returns (source, None). The /asr
    # route should return the source transcript and NOT call the side-file
    # save (no translation to save).
    from fastapi.testclient import TestClient

    source_segs = [Segment(start=0.0, end=1.0, text="just english")]

    async def fake_pipeline(audio_path, language, _artifacts_out=None):
        # ``target_segments=None`` → no A/B alt-pass invoked, so artifact
        # capture is irrelevant; just match the signature.
        return source_segs, None
    monkeypatch.setattr(server, "run_pipeline_async", fake_pipeline)
    monkeypatch.setattr(server.audio, "encode_to_wav", lambda src, dst: None)
    monkeypatch.setattr(server.audio, "prepare_unencoded", lambda src, dst: None)

    save_called = {"hit": False}
    def fake_save(body, summary, video_file_url):
        save_called["hit"] = True
    monkeypatch.setattr(server, "_try_save_side_file", fake_save)

    client = TestClient(server.app)
    response = client.post(
        "/asr",
        params={
            "task": "transcribe",
            "language": "en",
            "output": "srt",
            "encode": "false",
            "video_file": "/media/tv/x.mp4",
        },
        files={"audio_file": ("x.wav", b"RIFF0000WAVE", "audio/wav")},
    )

    assert response.status_code == 200
    assert "just english" in response.text
    assert save_called["hit"] is False


@pytest.mark.asyncio
async def test_translate_context_window_is_passed_to_each_batch(monkeypatch):
    """Each translate_batch_async call (one per group) receives a
    previous_context list containing the most recent TRANSLATE_CONTEXT_LINES
    source-language segments handled so far.
    """
    segs = [
        Segment(start=0.0, end=1.0, text="line A"),
        Segment(start=1.0, end=2.0, text="line B"),
        Segment(start=2.0, end=3.0, text="line C"),
        Segment(start=3.0, end=4.0, text="line D"),
        Segment(start=4.0, end=5.0, text="line E"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 1)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONTEXT_LINES", 2)
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda path, language="en": list(segs))
    monkeypatch.setattr(server.audio, "_load_wav_mono",
                        lambda path: (np.zeros(16000 * 6, dtype=np.float32), 16000))

    # Force 5 separate groups (one per segment) by having each segment
    # belong to a different "speaker".
    ann = Annotation()
    for i in range(5):
        ann[PSegment(float(i), float(i+1))] = f"S{i}"
    monkeypatch.setattr(server.diarize, "diarize_waveform",
                        lambda *a, **k: ann)
    monkeypatch.setattr(server.gender, "detect_genders",
                        lambda audio, sr, a: {f"S{i}": "male" for i in range(5)})

    received: list[list[str]] = []
    async def fake_translate(texts, gender, target,
                             addressee_gender=None, source_language="English",
                             previous_context=None):
        received.append(list(previous_context) if previous_context else [])
        return [f"HE: {t}" for t in texts]
    monkeypatch.setattr(backends.backend, "translate_batch_async",
                        fake_translate)

    await server.run_pipeline_async(server.Path("ignored.wav"), "en")

    # Context window now carries (speaker_gender, target_text) tuples.
    # Fake_translate returns ``f"HE: {t}"`` so target_text == "HE: line X".
    # All speakers in this fixture are "male" per the gender stub.
    # Group 1 (A): no prior context.
    # Group 2 (B): [("male", "HE: line A")].
    # Group 3 (C): [..., ("male", "HE: line B")] — window=2 reached.
    # Group 4 (D): rolled.
    # Group 5 (E): rolled.
    assert received == [
        [],
        [("male", "HE: line A")],
        [("male", "HE: line A"), ("male", "HE: line B")],
        [("male", "HE: line B"), ("male", "HE: line C")],
        [("male", "HE: line C"), ("male", "HE: line D")],
    ], received


@pytest.mark.asyncio
async def test_translate_context_disabled_when_setting_zero(monkeypatch):
    """TRANSLATE_CONTEXT_LINES=0 must produce no previous_context at all."""
    segs = [
        Segment(start=0.0, end=1.0, text="A"),
        Segment(start=1.0, end=2.0, text="B"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONTEXT_LINES", 0)
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda path, language="en": list(segs))
    monkeypatch.setattr(server.audio, "_load_wav_mono",
                        lambda path: (np.zeros(16000 * 3, dtype=np.float32), 16000))

    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "S0"
    ann[PSegment(1.0, 2.0)] = "S1"
    monkeypatch.setattr(server.diarize, "diarize_waveform",
                        lambda *a, **k: ann)
    monkeypatch.setattr(server.gender, "detect_genders",
                        lambda audio, sr, a: {"S0": "male", "S1": "male"})

    received = []
    async def fake_translate(texts, gender, target,
                             addressee_gender=None, source_language="English",
                             previous_context=None):
        received.append(list(previous_context) if previous_context else [])
        return [f"HE: {t}" for t in texts]
    monkeypatch.setattr(backends.backend, "translate_batch_async",
                        fake_translate)

    await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Every call must see empty context.
    assert all(c == [] for c in received), received


@pytest.mark.asyncio
async def test_translate_context_window_threads_through_plain_translate(monkeypatch):
    """``_run_plain_translate`` (non-gender-aware path, e.g. Japanese) must
    plumb the same rolling window into each chunk's translate call.
    """
    # Two short segments — with CHUNK_DURATION_SEC=1 they land in separate chunks,
    # so we see chunk-N → chunk-N+1 context flow without invoking diarization.
    segs = [
        Segment(start=0.0, end=1.0, text="line A"),
        Segment(start=1.0, end=2.0, text="line B"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Japanese")  # not gender-aware
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 1)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 1)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONTEXT_LINES", 4)
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda path, language="en": list(segs))

    received: list[list[str]] = []
    async def fake_translate(texts, gender, target,
                             addressee_gender=None, source_language="English",
                             previous_context=None):
        received.append(list(previous_context) if previous_context else [])
        return [f"JA: {t}" for t in texts]
    monkeypatch.setattr(backends.backend, "translate_batch_async",
                        fake_translate)

    await server.run_pipeline_async(server.Path("ignored.wav"), "en")

    # Chunk 1 (A): empty context.
    # Chunk 2 (B): one prior tuple — None gender (plain-translate path),
    # target text from the fake "JA: line A".
    assert received == [[], [(None, "JA: line A")]], received


def test_asr_emits_alt_classifier_srt_when_ab_output_enabled(monkeypatch):
    """When GENDER_AB_OUTPUT=true, the /asr handler must call
    _try_save_side_file a SECOND time with a different filename
    (``*.he.alt-classifier.srt``) using the alternate classifier's
    output.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "GENDER_AB_OUTPUT", True)
    monkeypatch.setattr(server.settings, "GENDER_CLASSIFIER", "pitch")

    source_segs = [Segment(start=0.0, end=1.0, text="hello")]
    target_segs = [Segment(start=0.0, end=1.0, text="שלום")]
    alt_target  = [Segment(start=0.0, end=1.0, text="שלום-ALT")]

    async def fake_pipeline(audio_path, language, _artifacts_out=None):
        if _artifacts_out is not None:
            _artifacts_out.append(server._PipelineArtifacts(
                raw_segments=list(source_segs), annotations=[],
            ))
        return source_segs, target_segs
    monkeypatch.setattr(server, "run_pipeline_async", fake_pipeline)

    async def fake_alt(audio_path, language, artifacts):
        # New signature post-OOM fix: artifacts param replaces the
        # redundant re-transcribe / re-diarize. The fake doesn't need to
        # use them but must accept the kwarg so /asr's call compiles.
        return source_segs, alt_target
    monkeypatch.setattr(server, "run_pipeline_alt_classifier", fake_alt)

    monkeypatch.setattr(server.audio, "encode_to_wav", lambda src, dst: None)
    monkeypatch.setattr(server.audio, "prepare_unencoded", lambda src, dst: None)

    saved: list[tuple[str, str]] = []  # (suffix, body)
    def fake_save(body, summary, video_file_url, suffix=None):
        saved.append((suffix or ".he.srt", body))
    monkeypatch.setattr(server, "_try_save_side_file", fake_save)

    client = TestClient(server.app)
    response = client.post(
        "/asr",
        params={
            "task": "transcribe", "language": "en",
            "output": "srt", "encode": "false",
            "video_file": "/media/tv/x.mp4",
        },
        files={"audio_file": ("x.wav", b"RIFF0000WAVE", "audio/wav")},
    )

    assert response.status_code == 200
    suffixes = [s for s, _ in saved]
    assert ".he.srt" in suffixes, (
        f"primary side-file was not saved with the default .he.srt suffix; "
        f"got: {suffixes}"
    )
    assert ".he.alt-classifier.srt" in suffixes, (
        f"alt-classifier side-file not saved with the expected suffix; "
        f"got: {suffixes}"
    )
    # Bodies differ between primary and alt.
    alt_body = next(b for s, b in saved if ".alt-classifier" in s)
    assert "שלום-ALT" in alt_body


@pytest.mark.asyncio
async def test_alt_classifier_reuses_artifacts_without_retranscribe(monkeypatch):
    """A/B alt-classifier pass MUST NOT call transcribe.transcribe or
    diarize.diarize_waveform — it should reuse the primary pass's
    artifacts. This is the fix for the CUDA OOM that fires on 8GB cards
    when the alt pass triggers a back-to-back Whisper inference.
    """
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "GENDER_CLASSIFIER", "pitch")
    monkeypatch.setattr(server.settings, "TRANSLATION_BACKEND", "claude")

    # Counters: if these go above 0 the regression is back.
    transcribe_calls = []
    diarize_calls = []
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda *a, **k: transcribe_calls.append(a) or [])
    monkeypatch.setattr(server.diarize, "diarize_waveform",
                        lambda *a, **k: diarize_calls.append(a) or None)

    # Stub _run_gender_aware to (a) capture its precomputed_annotations
    # arg and (b) return without touching real diarize.
    captured: dict = {}
    async def fake_gender_aware(audio_path, segments, target, client,
                                source_language="English",
                                precomputed_annotations=None):
        captured["precomputed_annotations"] = precomputed_annotations
        captured["segments_texts"] = [s.text for s in segments]
        # Mutate segments to simulate translation, return (segs, anns).
        for s in segments:
            s.text = f"ALT: {s.text}"
        return segments, []
    monkeypatch.setattr(server, "_run_gender_aware", fake_gender_aware)

    # Pre-fabricated artifacts as if a primary pass had produced them.
    raw = [
        Segment(start=0.0, end=1.0, text="hello"),
        Segment(start=1.0, end=2.0, text="world"),
    ]
    sentinel_ann_chunk0 = object()  # opaque per-chunk annotation marker
    artifacts = server._PipelineArtifacts(
        raw_segments=raw,
        annotations=[sentinel_ann_chunk0],
    )

    source, target = await server.run_pipeline_alt_classifier(
        server.Path("ignored.wav"), "en", artifacts,
    )

    # The fix's core guarantee: zero re-transcribe, zero re-diarize.
    assert transcribe_calls == [], (
        f"alt pass called transcribe.transcribe — defeats the fix "
        f"({len(transcribe_calls)} calls)"
    )
    assert diarize_calls == [], (
        f"alt pass called diarize.diarize_waveform — defeats the fix "
        f"({len(diarize_calls)} calls)"
    )
    # The precomputed annotations from the primary pass reached the
    # gender-aware orchestrator.
    assert captured["precomputed_annotations"] == [sentinel_ann_chunk0], (
        f"primary's annotations did not reach _run_gender_aware; got "
        f"{captured['precomputed_annotations']!r}"
    )
    # Fresh segments were built from artifacts (not the primary's mutated
    # outputs).
    assert captured["segments_texts"] == ["hello", "world"]
    # Target segments hold the alt-pass translation; source segments hold
    # the artifact's raw text (independent of any in-place mutation).
    assert [s.text for s in target] == ["ALT: hello", "ALT: world"]
    assert [s.text for s in source] == ["hello", "world"]
