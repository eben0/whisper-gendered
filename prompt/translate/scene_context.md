The 'Earlier in this scene:' block (when present, before the numbered translation lines) is background — already translated into {target_language}, do NOT re-translate it. Each context line is prefixed with the speaker's gender label, e.g. `[female]: ...`, `[male]: ...`.

Use this block to determine the current speaker's addressee:

1. Find the most recent context line spoken by someone of a DIFFERENT gender than the current speaker. That prior speaker is the addressee. Choose the {target_language} "you" form matching their gender and number.

2. If the most recent context speaker shares the current speaker's gender (e.g., the current speaker is continuing their own turn, or just made a brief aside), read upward — the addressee is the most recent different-gender speaker further back in the block.

3. If no different-gender speaker appears anywhere in the block (the scene is single-gender), default to the current speaker's own gender form for "you" — the safest fit when there's no signal to do otherwise.

4. When this block's signal conflicts with any per-call addressee hint in the instructions above, prefer this block. The hint is a heuristic that may be stale at scene boundaries; the block contains the actual dialogue.

Worked example. Imagine the block contains:
  1. [female]: <line in {target_language}>
  2. [male]: <line in {target_language}>
  3. [male]: <line in {target_language}>

For the current line from a [male] speaker: the most recent context line (line 3) is male — the speaker's own continuation. Read upward: line 1 is [female], the nearest different-gender speaker. So the addressee is female; use the feminine "you" form in {target_language}.

Also use this block for vocabulary consistency and to pick translation choices most consistent with what was just said.
