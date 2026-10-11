#!/usr/bin/env python3
"""Find the funniest moments in a day's Pascribe transcripts.

Reads the same server/transcripts/*.json files the meme generator uses, but
instead of rendering memes it produces a ranked highlights list (quote,
speaker, timestamp, why it's funny) as JSON + a readable Markdown digest.
This is the "find funniest conversation moments" half of making highlights;
the meme generator (projects/daily-transcript-meme-generator) can consume the
JSON as its moment source.

Usage:
  python3 highlights.py                       # most recent data-rich date
  python3 highlights.py --date 2026-10-01
  python3 highlights.py --date 2026-10-01 --count 8
  python3 highlights.py --date 2026-10-01 --judge glm   # cheap judge model

Output:
  server/analyses/highlights-<date>.json
  server/analyses/highlights-<date>.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "transcripts"
OUT_DIR = Path(__file__).resolve().parent.parent / "analyses"
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "/home/nik/.local/bin/claude")
# Cheap volume route for the GLM judge (--judge glm); extraction stays on the
# user's Claude subscription. GLM is hosted outside — transcripts are the
# friend group's own voice chat, which Nik has designated OK for it.
GLM = "/home/nik/projects/multisite-audit/tools/glm.py"

SYSTEM_PROMPT = """You are a comedy editor cutting a highlights reel from one
evening of a friend group's voice chat (gaming, projects, Latvian/English mix).

Pick the funniest moments. Strong shapes: absurd-specificity, false-confidence,
anti-climax, accidental-revelation, dark-mundane, philosophical-tangent,
repetition-spiral, spectacular rage/tilt, translation chaos. The quote MUST be
a verbatim substring of one speaker line — never paraphrase, never merge lines.
Skip mundane coordination, politics, and cruelty toward people outside the group.

For each moment return:
{
  "quote": "<verbatim, <=40 words>",
  "speaker": "<name exactly as it prefixes the line>",
  "time": "<HH:MM from the segment header the line sits under, or '' if none>",
  "shape": "<one of the shapes above, or 'other'>",
  "why": "<one short sentence>",
  "score": <1-10 funniness>
}

Rules:
- Verbatim means character-for-character including the original language
  (Latvian quotes stay Latvian). Do not translate the quote.
- Prefer moments a stranger would laugh at with zero context.
- Return ONLY a JSON array, best first, no prose, no markdown fences."""


def pick_default_date() -> str:
    counts: dict[str, int] = {}
    for p in TRANSCRIPT_DIR.glob("????-??-??_*.json"):
        counts[p.name[:10]] = counts.get(p.name[:10], 0) + 1
    candidates = sorted((d for d, n in counts.items() if n >= 3), reverse=True)
    if not candidates:
        sys.exit("no date with >=3 transcripts found")
    return candidates[0]


def load_day_lines(date_str: str) -> list[str]:
    """One text block per transcript file, keeping the `HH:MM speaker:` shape."""
    files = sorted(TRANSCRIPT_DIR.glob(f"{date_str}_*.json"))
    if not files:
        sys.exit(f"no transcripts for {date_str}")
    blocks: list[str] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        t = (data.get("transcript") or "").strip()
        if t:
            blocks.append(t)
    text = "\n\n".join(blocks)
    print(f"[load] {len(files)} files, {len(text):,} chars from {date_str}", file=sys.stderr)
    # Keep only lines that look like `speaker: text` or segment markers, drop
    # huge whitespace runs so the LLM input stays compact.
    lines = [ln.rstrip() for ln in text.splitlines()]
    return [ln for ln in lines if ln.strip()]


def call_claude(system: str, user: str, timeout: int = 600) -> str:
    cmd = [
        CLAUDE_BIN, "-p",
        "--model", "sonnet",
        "--output-format", "json",
        "--tools", "",
        "--no-session-persistence",
    ]
    res = subprocess.run(cmd, input=f"{system}\n\n---\n\n{user}",
                         capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"claude rc={res.returncode}: {res.stderr[:300]}")
    data = json.loads(res.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude error: {data.get('result', '?')[:300]}")
    return data["result"]


def call_glm(system: str, user: str, timeout: int = 600) -> str:
    prompt = f"{system}\n\n---\n\n{user}"
    res = subprocess.run([sys.executable, GLM, prompt],
                         capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"glm rc={res.returncode}: {res.stderr[:300]}")
    return res.stdout.strip()


def extract_json_array(text: str) -> list:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("[")
    if start < 0:
        raise ValueError(f"no [ in: {text[:200]}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced array")


def verify_verbatim(items: list[dict], corpus: str) -> list[dict]:
    """Drop picks whose quote isn't a verbatim corpus substring (normalizing
    whitespace, since transcripts wrap lines unpredictably)."""
    flat = re.sub(r"\s+", " ", corpus)
    kept = []
    for it in items:
        q = re.sub(r"\s+", " ", str(it.get("quote", ""))).strip()
        if q and q in flat:
            kept.append({**it, "quote": q})
    return kept


def write_markdown(date_str: str, items: list[dict], path: Path) -> None:
    out = [f"# Highlights — {date_str}", ""]
    for i, it in enumerate(items, 1):
        speaker = it.get("speaker", "?")
        when = it.get("time") or ""
        stamp = f" ({when})" if when else ""
        out.append(f"## {i}. [{it.get('score', '?')}/10] {speaker}{stamp}")
        out.append(f"> {it.get('quote', '')}")
        out.append("")
        shape = it.get("shape", "")
        why = it.get("why", "")
        tag = f" *({shape})*" if shape else ""
        out.append(f"{why}{tag}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--judge", choices=["claude", "glm"], default="claude")
    a = ap.parse_args()
    date_str = a.date or pick_default_date()

    lines = load_day_lines(date_str)
    corpus = "\n".join(lines)
    # Hard cap the LLM input; tail keeps the evening's freshest material.
    MAX_CHARS = 300_000
    llm_input = corpus if len(corpus) <= MAX_CHARS else corpus[-MAX_CHARS:]

    call = call_glm if a.judge == "glm" else call_claude
    raw = call(SYSTEM_PROMPT, f"Transcript of {date_str}:\n\n{llm_input}")
    items = extract_json_array(raw)
    items = verify_verbatim(items if isinstance(items, list) else [], corpus)
    items.sort(key=lambda x: x.get("score", 0), reverse=True)
    items = items[: a.count]
    if not items:
        sys.exit("no verbatim-verified highlights survived")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"highlights-{date_str}.json").write_text(
        json.dumps({"date": date_str, "_generated_at": datetime.now(timezone.utc).isoformat(),
                    "count": len(items), "highlights": items},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    write_markdown(date_str, items, OUT_DIR / f"highlights-{date_str}.md")
    print(f"[done] {len(items)} highlights -> {OUT_DIR}/highlights-{date_str}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
