import json

import pytest

import src.backends.claude as translate


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]
        self.usage = None


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


class _FakeSettings:
    CLAUDE_MODEL = "claude-test"
    CLAUDE_MAX_RETRIES = 0

    def require_anthropic_key(self):
        return "fake-key"


def _make_backend(client):
    """Create a ClaudeBackend whose _get_client() returns the given fake client."""
    backend = translate.ClaudeBackend(_FakeSettings())
    backend._client = client  # bypass lazy init
    return backend


@pytest.mark.asyncio
async def test_translate_batch_async_returns_translations():
    client = _FakeAsyncClient([json.dumps({"translations": ["a-he", "b-he"]})])
    backend = _make_backend(client)
    out = await backend.translate_batch_async(["a", "b"], "male", "Hebrew")
    assert out == ["a-he", "b-he"]


@pytest.mark.asyncio
async def test_translate_batch_async_pads_on_count_mismatch():
    client = _FakeAsyncClient([json.dumps({"translations": ["only-one"]})])
    backend = _make_backend(client)
    out = await backend.translate_batch_async(["a", "b"], None, "Hebrew")
    assert len(out) == 2
    assert out[0] == "only-one"
    assert out[1] == "b"  # padded from source text


@pytest.mark.asyncio
async def test_translate_batch_async_empty_input():
    client = _FakeAsyncClient([])
    backend = _make_backend(client)
    out = await backend.translate_batch_async([], "female", "Hebrew")
    assert out == []


@pytest.mark.asyncio
async def test_translate_batch_async_returns_source_on_unparseable_response():
    client = _FakeAsyncClient(["this is not json at all"])
    backend = _make_backend(client)
    out = await backend.translate_batch_async(["a", "b"], "male", "Hebrew")
    assert out == ["a", "b"]  # JSON parse fails -> falls back to source text


def test_system_prompt_includes_addressee_sentence_when_set():
    prompt = translate._system_prompt("Hebrew", "female", addressee_gender="male")
    assert "male" in prompt
    assert "addressee" in prompt.lower()


def test_system_prompt_omits_addressee_sentence_when_unset():
    # When no specific addressee_gender hint is provided, the generic "matching
    # the addressee's number and gender" guidance still appears, but the specific
    # "most likely addressee" sentence does not.
    prompt = translate._system_prompt("Hebrew", "female")
    assert "matching the addressee" in prompt.lower()
    assert "most likely addressee" not in prompt.lower()


def test_system_prompt_addresses_number_when_target_is_gender_aware():
    # Number guidance should appear for any gender-marked target language.
    prompt = translate._system_prompt("Hebrew", "male", addressee_gender=None)
    # The "you" / number guidance should be present regardless of addressee_gender.
    assert "plural" in prompt.lower()


def test_system_prompt_includes_you_form_guidance_even_without_addressee_hint():
    # The generic guidance to choose the right "you" form must appear whenever
    # the speaker's gender is set, regardless of addressee_gender. The spec
    # requires this so Claude considers second-person form selection in
    # ambiguous group scenes too, not only when a specific hint is provided.
    # We check for the *contract* (mentioning "you" + matching the addressee)
    # rather than a specific phrasing, so future minimisation passes don't
    # break the test just because the wording got tighter.
    prompt = translate._system_prompt("Hebrew", "male", addressee_gender=None)
    assert '"you"' in prompt
    assert "matching the addressee" in prompt


class _RecordingMessages:
    """Captures the system prompt (and full kwargs) of every create() call."""

    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = 0
        self.systems: list[str] = []
        self.payloads: list[dict] = []

    async def create(self, **kwargs):
        # ``system`` may be a plain string (legacy) or a list of content
        # blocks (when cache_control is attached). Normalize to a string so
        # tests can keep using simple substring assertions.
        system = kwargs.get("system", "")
        if isinstance(system, list):
            system = "".join(b.get("text", "") for b in system if isinstance(b, dict))
        self.systems.append(system)
        self.payloads.append(kwargs)
        text = self._payloads[self.calls]
        self.calls += 1
        return _FakeResponse(text)


class _RecordingAsyncClient:
    def __init__(self, payloads):
        self.messages = _RecordingMessages(payloads)


@pytest.mark.asyncio
async def test_translate_batch_async_forwards_addressee_into_prompt():
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    backend = _make_backend(client)
    await backend.translate_batch_async(
        ["hello"], "female", "Hebrew", addressee_gender="male",
    )
    assert "male" in client.messages.systems[0]
    assert "addressee" in client.messages.systems[0].lower()


@pytest.mark.asyncio
async def test_translate_batch_async_no_addressee_when_unset():
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    backend = _make_backend(client)
    await backend.translate_batch_async(["hello"], "female", "Hebrew")
    # Generic "matching the addressee" guidance is always-on when gender is set,
    # but the specific "most likely addressee" hint must be absent.
    assert "most likely addressee" not in client.messages.systems[0].lower()


@pytest.mark.asyncio
async def test_translate_batch_async_source_language_reaches_prompt():
    # Default: prompt says "from English".
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    backend = _make_backend(client)
    await backend.translate_batch_async(["hello"], None, "Hebrew")
    assert "from English into Hebrew" in client.messages.systems[0]

    # Overridden: prompt reflects the caller's source_language.
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    backend = _make_backend(client)
    await backend.translate_batch_async(
        ["hello"], None, "Hebrew", source_language="French",
    )
    assert "from French into Hebrew" in client.messages.systems[0]
    assert "from English into Hebrew" not in client.messages.systems[0]


def test_system_prompt_source_language_default_and_override():
    # Direct unit-level check on _system_prompt: default + override behaviour.
    default = translate._system_prompt("Hebrew", None)
    assert "from English into Hebrew" in default
    overridden = translate._system_prompt(
        "Hebrew", None, source_language="Spanish",
    )
    assert "from Spanish into Hebrew" in overridden
    assert "from English" not in overridden


# --- Task 1 (subtitle-quality-improvements plan): prompt contract -------- #

def test_system_prompt_asks_for_transliterated_names():
    """User reported: 'T' / 'Mondo' (S04E05 @ 05:25) left as Latin in the
    Hebrew SRT. The old prompt explicitly banned transliteration; the new
    one must require it for proper nouns.
    """
    sp = translate._system_prompt("Hebrew", None)
    assert "transliterat" in sp.lower()
    # Must not still flatly forbid transliteration anywhere.
    forbidding = [
        line for line in sp.split(".")
        if "do not" in line.lower() and "transliterat" in line.lower()
    ]
    assert forbidding == [], f"prompt still forbids transliteration: {forbidding}"


def test_system_prompt_asks_for_idiomatic_slang():
    """User reported: slang translated literally rather than idiomatically.

    The pre-existing prompt already says ``"natural, idiomatic, concise"``
    in a general styling sentence, which isn't enough — we need explicit
    slang/idiom guidance directed at how to render them.
    """
    sp = translate._system_prompt("Hebrew", None).lower()
    # Require both 'slang' AND (idiom* OR equivalent*) — meaning a
    # dedicated sentence about rendering slang/idioms as their target-
    # language equivalents, not just the generic "idiomatic" descriptor.
    assert "slang" in sp, "prompt must explicitly mention 'slang'"
    assert "idiom" in sp or "equivalent" in sp, (
        "prompt must instruct rendering idioms/equivalents, not "
        "just descibe the output as 'idiomatic'"
    )


def test_system_prompt_prefers_natural_prepositions_for_hebrew():
    """User reported: at 05:06, ``את`` used where ``ב`` or ``של`` would be
    natural. The prompt should explicitly guide preposition choice.
    """
    sp = translate._system_prompt("Hebrew", None)
    # Hebrew-specific guidance must mention the natural-preposition rule.
    # Either the word "preposition" or the Hebrew את token is acceptable
    # to keep the test stable against minor re-wordings.
    assert "preposition" in sp.lower() or "את" in sp


# --- Language-specific prompt gating (PR #1 review feedback) -------------- #

def test_system_prompt_skips_transliteration_for_latin_script_target():
    """Transliteration guidance is meaningless when the target uses Latin
    letters (French, Spanish, German, etc.) — proper nouns stay as-is.
    The block must NOT appear in the system prompt for such targets.
    """
    for target in ("French", "Spanish", "German", "Italian", "Portuguese"):
        sp = translate._system_prompt(target, None)
        assert "transliterat" not in sp.lower(), (
            f"transliteration guidance leaked into {target} prompt: {sp[:200]}"
        )


def test_system_prompt_keeps_transliteration_for_non_latin_targets():
    """Non-Latin-script targets (Hebrew, Arabic, Russian, etc.) must still
    receive transliteration guidance — proper nouns need to be re-spelled
    in the target script.
    """
    for target in ("Hebrew", "Arabic", "Russian", "Hindi", "Japanese"):
        sp = translate._system_prompt(target, None)
        assert "transliterat" in sp.lower(), (
            f"transliteration guidance missing for {target}: {sp[:200]}"
        )


def test_system_prompt_skips_hebrew_preposition_block_for_non_hebrew():
    """The Hebrew-specific preposition rule (about ב, של, ל, מ, על, את)
    must NOT appear when the target is anything other than Hebrew.
    """
    for target in ("Arabic", "Spanish", "French", "Russian", "German"):
        sp = translate._system_prompt(target, None)
        assert "את" not in sp, (
            f"Hebrew-specific preposition block leaked into {target} prompt"
        )
        assert "ב, של" not in sp, (
            f"Hebrew preposition list leaked into {target} prompt"
        )


def test_system_prompt_hints_max_chars_per_line():
    """The downstream formatter (Task 2) will split long lines, but the
    prompt should still steer Claude toward short subtitle-friendly output.
    """
    import re
    sp = translate._system_prompt("Hebrew", None)
    assert re.search(r"\b(42|45|48|50)\b", sp) or "two lines" in sp.lower()


# --- Task 6 (improve-gender-detection): previous-scene context window ------ #

@pytest.mark.asyncio
async def test_previous_context_appears_in_user_message():
    """When previous_context is non-empty, the user message must include
    a numbered 'Earlier in this scene:' block with each line prefixed by
    the speaker's gender label so Claude can reconstruct turn-taking.
    """
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    backend = _make_backend(client)
    await backend.translate_batch_async(
        ["new line"], None, "Hebrew",
        previous_context=[
            ("male", "הוא הגיע בצהריים."),
            ("female", "היא כבר הייתה שם."),
        ],
    )
    user_msg = client.messages.payloads[0]["messages"][0]["content"]
    assert "Earlier in this scene" in user_msg
    assert "[male]: הוא הגיע בצהריים." in user_msg
    assert "[female]: היא כבר הייתה שם." in user_msg
    assert "new line" in user_msg
    # The actual line to translate must be clearly separated from context.
    assert user_msg.index("new line") > user_msg.index("היא כבר הייתה שם.")


@pytest.mark.asyncio
async def test_previous_context_renders_without_prefix_when_gender_unknown():
    """Plain-translate path passes None for speaker gender. Those lines
    should render without the ``[gender]: `` prefix so the prompt doesn't
    lie about information it doesn't have.
    """
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-ja"]})])
    backend = _make_backend(client)
    await backend.translate_batch_async(
        ["next line"], None, "Japanese",
        previous_context=[(None, "earlier-line-1"), (None, "earlier-line-2")],
    )
    user_msg = client.messages.payloads[0]["messages"][0]["content"]
    assert "earlier-line-1" in user_msg
    assert "earlier-line-2" in user_msg
    # No [male]: / [female]: prefixes because gender wasn't carried.
    assert "[male]" not in user_msg
    assert "[female]" not in user_msg


@pytest.mark.asyncio
async def test_previous_context_absent_when_window_is_empty():
    """Default (empty) context should not add any preamble — the user
    message looks identical to the pre-feature output.
    """
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    backend = _make_backend(client)
    await backend.translate_batch_async(
        ["only line"], None, "Hebrew",
    )
    user_msg = client.messages.payloads[0]["messages"][0]["content"]
    assert "Earlier in this scene" not in user_msg


def test_system_prompt_mentions_context_use():
    """When previous_context is plumbed, the system prompt should tell
    Claude how to use it. We only check the directive sentence exists —
    not the exact wording — so future re-phrasings don't break the test.
    """
    sp = translate._system_prompt("Hebrew", None)
    # The directive can be present unconditionally (independent of the
    # current batch's gender) — it costs nothing when no context lines
    # are passed.
    assert (
        "earlier" in sp.lower() and "context" in sp.lower()
    ), "system prompt should explain how to use the 'Earlier' context block"
