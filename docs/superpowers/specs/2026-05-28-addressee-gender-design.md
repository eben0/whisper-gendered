# Addressee-Aware Translation — Design

**Date:** 2026-05-28
**Status:** Approved design (pending spec review)
**Branch:** `feature/addressee-gender`

## Problem

The current translation prompt only tells Claude the **speaker's** gender, so first-person
forms are correct (`I am going` → `אני הולך`/`הולכת`). But English `"you"` maps to four
Hebrew forms — `אתה / את / אתם / אתן` — driven by the **addressee's** number and gender.
The model has to guess. In 1-on-1 dialogue it often guesses well from context. In
multi-speaker scenes it has to commit to a form blindly, and there's no bias toward the
inclusive masculine plural that Hebrew uses for mixed groups.

The pipeline already detects every speaker's gender via pitch. We just don't tell the model.

## Goal

Improve second-person translation accuracy without speaker-addressee identity tracking, by:

1. Expanding the translation system prompt with explicit guidance on choosing the right
   "you" form (number + gender, with a default for ambiguous group scenes).
2. Passing an `addressee_gender` hint derived from **the previous different speaker** —
   a cheap heuristic that's correct for back-and-forth dialogue and harmless when wrong.

## Non-goals (YAGNI)

- Speaker-addressee identity tracking via audio/visual cues.
- Multi-addressee modeling (group A speaks to group B).
- Detecting plural-vs-singular from audio. Number stays Claude's job, guided by the prompt.
- Per-line addressee switches inside a same-speaker group. Addressee is computed per group,
  not per segment within a group.

## Design

### Signature additions

`pipeline/translate.py`:

- `_system_prompt(target_language, gender, addressee_gender: str | None = None)`
- `translate_batch_async(texts, gender, target_language, client, addressee_gender: str | None = None)`
- `_translate_one_batch_async(texts, gender, target_language, client, addressee_gender: str | None = None)`
- Same `addressee_gender` parameter added to the sync `translate_batch` /
  `_translate_one_batch` for symmetry (no current sync callers, but the parallel pair stays
  parallel).

Defaults to `None`, so all existing callers and tests continue to work unchanged.

### Prompt extension

`_system_prompt` keeps its existing speaker-gender block (first-person forms) and appends
this paragraph when the target language is gender-marked (which all members of
`GENDER_AWARE_LANGUAGES` are):

> When the speaker addresses another person (English "you"), choose the *{target_language}*
> form matching the addressee's number and gender.
> *{If addressee_gender is provided:}* The most likely addressee in this exchange is
> *{addressee_gender}*; prefer that form for singular "you" unless context clearly implies a
> different addressee.
> Infer number from context — collective cues like "you all", "you guys", or plural verbs
> imply plural. When number is ambiguous in a multi-person scene, prefer the inclusive
> plural form (e.g., אתם in Hebrew). Don't mix forms within a single line.

The exact wording goes into the prompt string. When `addressee_gender is None`, the middle
sentence is omitted; the rest still gives the model better guidance than today.

### Addressee derivation

Inside `_translate_chunk`, addressee is the **most recent different speaker's gender** seen
so far in the conversation. Initial state for a chunk comes from a new
`prev_speaker_gender: str | None` argument.

Algorithm per chunk — "addressee = previous group's speaker gender":

```
prev_group_gender = prev_speaker_gender   # from prior chunk's tail, or None for the very first chunk
for speaker, group in groups:
    spk_gender = genders.get(speaker, "male") if speaker else "male"
    addressee = prev_group_gender         # the gender of whoever just spoke before this group
    translated = await translate_batch_async(
        [s.text for s in group], spk_gender, target, client,
        addressee_gender=addressee,
    )
    prev_group_gender = spk_gender        # this group becomes the addressee context for the next
```

Two consecutive same-gender speakers (different labels) still correctly produce a
matching `addressee_gender` because we use the gender, not the label. The very first
group of the very first chunk gets `addressee_gender=None`, which the prompt handles by
omitting the hint sentence and letting Claude infer from context.

### Cross-chunk carry

`_run_gender_aware` threads `prev_speaker_gender` across the chunk loop. To know each
chunk's last speaker's gender *before* launching its translate task, the grouping step
(`_group_consecutive(assigned)`) moves from inside `_translate_chunk` up into the
orchestrator loop. The task takes pre-built `groups` and the `prev_speaker_gender` starting
value; the orchestrator computes the chunk's last-group gender from `groups` and threads it
forward as the next chunk's `prev_speaker_gender`.

Loop shape (in `_run_gender_aware`):

```
prev_speaker_gender = None
for idx, chunk in enumerate(chunks):
    annotation, genders = await run_in_thread(_diarize_and_gender, ...)
    assigned = build_assigned(chunk, annotation)         # chunk-local for assign_speaker
    groups = _group_consecutive(assigned)
    last_gender = derive_last_gender(groups, genders)    # for the next chunk's prev
    tasks.append(asyncio.create_task(
        _translate_chunk(idx, groups, genders, target, client, sem, prev_speaker_gender)
    ))
    prev_speaker_gender = last_gender
```

`_translate_chunk(idx, groups, genders, target, client, sem, prev_speaker_gender)` is the
refactored task body: it no longer builds groups; it just iterates them, computing
addressee per the algorithm above, and mutates `seg.text` in place.

### Backward compatibility

- Existing 24-test suite stays green: new params default to `None`, prompt is purely
  additive.
- The single-chunk smoke clip from yesterday continues to behave identically (sole speaker
  in chunk → no addressee to rotate; `prev_speaker_gender=None` → no hint sent; first-person
  forms unchanged).

## Tests

Unit:
- **Prompt content** (in `tests/test_translate_async.py`): assert the prompt sent to the
  fake `messages.create` contains the addressee sentence when `addressee_gender="female"`
  is passed, and does NOT contain it when `addressee_gender=None`.

Orchestration (in `tests/test_orchestration.py`):
- **Within-chunk addressee rotation**: a chunk with three consecutive groups
  M → F → M. Assert `translate_batch_async` was called with
  `addressee_gender=(None|prev, "male", "female")` in order, captured via a fake that
  records arguments per call.
- **Cross-chunk carry**: two chunks; chunk 1's last group speaker is "female", chunk 2's
  first group speaker is "male". Assert chunk 2's first translate call received
  `addressee_gender="female"`.

Both orchestration tests use the same fake-translate pattern as today and require no real
audio or models.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Wrong addressee in 3+-person scene | Prompt instructs plural fallback when ambiguous; the hint is one input among many for Claude |
| Prompt regression (Claude over-relies on hint) | "unless context clearly implies a different addressee" disclaimer; spot-check on the long Bazarr clip |
| Order-of-mutation bug in cross-chunk carry | The cross-chunk test (`prev_speaker_gender` for chunk 2 first group) catches this directly |
| Sync `translate_batch` drift from async sibling | Mirror the new param even though there are no current sync callers; keeps the pair maintainable |

## Migration / config

No config changes. No new env vars. No new dependencies. No schema changes to the
structured-output JSON.

## Out of scope, follow-ups

- Tightening diarization (passing `max_speakers` hint) is a separate idea.
- Using larger surrounding chunk context (including other speakers' lines) in the
  translation prompt could further help. Deferred until we see whether this simple hint is
  enough.
