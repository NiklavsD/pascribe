"""Transcribe speech segments the live pipeline skipped (e.g. while the
AssemblyAI balance was negative) and file them the way the live pipeline does:
append to recordings/<day>/_transcript.txt, mark _processed.txt, POST to the
Pascribe server.

Unlike process_new(), this transcribes every discussion (not only the latest),
and marks a discussion processed only when every segment in it transcribed
without an API error, so a rerun picks up whatever failed.

    venv/bin/python backfill_transcripts.py --from 2026-09-23 --to 2026-09-30 [--dry-run] [--no-send]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from audio.storage import (
    MIN_TRANSCRIBE_S,
    audio_seconds,
    get_new_speech_segments,
    load_speaker_ids,
    mark_files_processed,
)
from config import RECORDINGS_DIR
from transcription.assemblyai import transcribe_file
from transcription.pipeline import (
    _build_conversation,
    _merge_consecutive_segments,
    _send_to_pascribe_safe,
    _split_into_discussions,
)

log = logging.getLogger("backfill")
CONCURRENCY = 5


class FatalAPIError(RuntimeError):
    pass


async def _transcribe(sem: asyncio.Semaphore, username: str, path: Path) -> dict | None:
    if audio_seconds(path) < MIN_TRANSCRIBE_S:  # same cutoff as the live pipeline
        return None
    async with sem:
        for attempt in range(3):
            try:
                t = await transcribe_file(path)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "balance" in msg.lower() or "401" in msg:
                    raise FatalAPIError(msg) from e
                if "too short" in msg.lower():
                    return None
                log.warning("retry %d for %s/%s: %s", attempt + 1, username, path.name, msg)
                if attempt == 2:
                    raise
                await asyncio.sleep(5 * (attempt + 1))
    text = (t.get("text") or "").strip()
    if not text:
        return None
    return {
        "username": username,
        "text": text,
        "duration_seconds": t.get("audio_duration", 0),
        "filepath": path,
        "_aai": {k: t.get(k) for k in ("utterances", "language_code", "confidence") if t.get(k) is not None},
    }


async def backfill_day(day: datetime, dry_run: bool, send: bool) -> tuple[int, int, float]:
    day_str = day.strftime("%Y-%m-%d")
    # Skip the live bot's temporary merge files; they vanish mid-run.
    segs = [(u, p) for u, p in get_new_speech_segments(day) if not p.name.startswith("_")]
    if not segs:
        log.info("%s: nothing pending", day_str)
        return 0, 0, 0.0
    discussions = _split_into_discussions(segs)
    secs = sum(s for s in (audio_seconds(p) for _, p in segs) if s >= MIN_TRANSCRIBE_S)
    log.info("%s: %d segments, %d discussions, %.1f min of audio", day_str, len(segs), len(discussions), secs / 60)
    if dry_run:
        return len(discussions), 0, secs

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    for disc in discussions:
        merged = _merge_consecutive_segments(disc)
        try:
            results = await asyncio.gather(*(_transcribe(sem, u, p) for u, p in merged), return_exceptions=True)
        finally:
            for _, p in merged:
                if p.name.startswith("_merged_"):
                    p.unlink(missing_ok=True)
        fatal = [r for r in results if isinstance(r, FatalAPIError)]
        if fatal:
            raise fatal[0]
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            log.error("%s: discussion at %s left pending (%d errors, first: %s)",
                      day_str, disc[0][1].name[:6], len(errors), errors[0])
            continue
        transcripts = [r for r in results if r]
        mark_files_processed([p.name for _, p in disc], day)
        done += 1
        if not transcripts:
            continue
        conversation = _build_conversation(transcripts)
        start = disc[0][1].name[:6]
        with open(RECORDINGS_DIR / day_str / "_transcript.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n--- {start[:2]}:{start[2:4]} (backfill) ---\n\n")
            f.write(conversation)
        if send:
            ids = load_speaker_ids(day)
            users = sorted({t["username"] for t in transcripts})
            await _send_to_pascribe_safe(
                conversation, day_str, [{"username": u, "user_id": ids.get(u)} for u in users],
                rich_fields={"language_code": transcripts[0]["_aai"].get("language_code")},
            )
        log.info("%s: %s discussion done (%d chars, %d speakers)", day_str, start, len(conversation), len({t['username'] for t in transcripts}))
    return len(discussions), done, secs


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-send", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    day = datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(a.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    total_disc = total_done = 0
    total_secs = 0.0
    while day <= end:
        try:
            n, d, s = await backfill_day(day, a.dry_run, not a.no_send)
        except FatalAPIError as e:
            log.error("stopping: AssemblyAI refused the key: %s", e)
            return 2
        total_disc += n
        total_done += d
        total_secs += s
        day += timedelta(days=1)
    log.info("TOTAL: %d discussions, %d transcribed, %.1f h of audio", total_disc, total_done, total_secs / 3600)
    return 0 if a.dry_run or total_done == total_disc else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
