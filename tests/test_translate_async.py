import json

import pytest

from pipeline import translate


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = 0

    async def create(self, **kwargs):
        text = self._payloads[self.calls]
        self.calls += 1
        return _FakeResponse(text)


class _FakeAsyncClient:
    def __init__(self, payloads):
        self.messages = _FakeMessages(payloads)


@pytest.mark.asyncio
async def test_translate_batch_async_returns_translations():
    client = _FakeAsyncClient([json.dumps({"translations": ["a-he", "b-he"]})])
    out = await translate.translate_batch_async(["a", "b"], "male", "Hebrew", client)
    assert out == ["a-he", "b-he"]


@pytest.mark.asyncio
async def test_translate_batch_async_pads_on_count_mismatch():
    client = _FakeAsyncClient([json.dumps({"translations": ["only-one"]})])
    out = await translate.translate_batch_async(["a", "b"], None, "Hebrew", client)
    assert len(out) == 2
    assert out[0] == "only-one"
    assert out[1] == "b"  # padded from source text


@pytest.mark.asyncio
async def test_translate_batch_async_empty_input():
    client = _FakeAsyncClient([])
    out = await translate.translate_batch_async([], "female", "Hebrew", client)
    assert out == []


@pytest.mark.asyncio
async def test_translate_batch_async_returns_source_on_unparseable_response():
    client = _FakeAsyncClient(["this is not json at all"])
    out = await translate.translate_batch_async(["a", "b"], "male", "Hebrew", client)
    assert out == ["a", "b"]  # JSON parse fails -> falls back to source text


def test_system_prompt_includes_addressee_sentence_when_set():
    prompt = translate._system_prompt("Hebrew", "female", addressee_gender="male")
    assert "male" in prompt
    assert "addressee" in prompt.lower()


def test_system_prompt_omits_addressee_sentence_when_unset():
    prompt = translate._system_prompt("Hebrew", "female")
    assert "addressee" not in prompt.lower()


def test_system_prompt_addresses_number_when_target_is_gender_aware():
    # Number guidance should appear for any gender-marked target language.
    prompt = translate._system_prompt("Hebrew", "male", addressee_gender=None)
    # The "you" / number guidance should be present regardless of addressee_gender.
    assert "plural" in prompt.lower()
