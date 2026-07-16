"""Pure transcript post-processing shared by local and cloud workflows."""

from __future__ import annotations

from typing import Any


def group_timed_words(
    words: Any,
    *,
    pause_ms: int = 1200,
) -> list[dict[str, Any]]:
    """Group AssemblyAI-style timed words into utterance-like segments."""
    if not isinstance(words, list):
        return []
    valid = [
        word
        for word in words
        if isinstance(word, dict)
        and isinstance(word.get("text"), str)
        and isinstance(word.get("start"), (int, float))
        and isinstance(word.get("end"), (int, float))
    ]
    if not valid:
        return []

    segments: list[dict[str, Any]] = []
    buffered: list[str] = []
    segment_start = float(valid[0]["start"]) / 1000.0
    for index, word in enumerate(valid):
        buffered.append(word["text"])
        is_last = index == len(valid) - 1
        gap = 0 if is_last else float(valid[index + 1]["start"]) - float(word["end"])
        if is_last or gap > pause_ms:
            segments.append({
                "start_s": segment_start,
                "end_s": float(word["end"]) / 1000.0,
                "text": " ".join(buffered),
            })
            buffered = []
            if not is_last:
                segment_start = float(valid[index + 1]["start"]) / 1000.0
    return segments
