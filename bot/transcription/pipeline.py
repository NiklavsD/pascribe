"""Orchestrate transcription and send results to the Pascribe analysis server."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from audio.storage import (
    get_all_users_for_date,
    get_user_recordings,
    get_new_speech_segments,
    get_all_speech_segments_chronological,
    mark_files_processed,
    concatenate_wav_files,
)
from config import PASCRIBE_URL, PASCRIBE_TOKEN, DISCUSSION_GAP_S
from transcription.assemblyai import transcribe_file

log = logging.getLogger(__name__)


async def send_to_pascribe(payload: dict) -> dict:
    """POST transcription data to the Pascribe analysis server."""
    url = f"{PASCRIBE_URL}/api/transcription"
    headers = {
        "Authorization": f"Bearer {PASCRIBE_TOKEN}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status >= 400:
                body = await resp.text()
                log.error("Pascribe returned %d: %s", resp.status, body)
                return {"error": body}
            result = await resp.json()
            log.info("Sent to Pascribe: %s", result.get("id", "ok"))
            return result


def _seconds_from_filename(filename: str) -> float:
    """Extract seconds since midnight from a filename like '020345_440505_speech.wav'."""
    parts = filename.split("_")
    if len(parts) >= 1 and len(parts[0]) == 6:
        t = parts[0]
        return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    return 0.0


# Gap threshold for inserting a pause marker in conversation
CONVERSATION_GAP_S = 120  # 2 minutes


def _split_into_discussions(segments: list[tuple[str, Path]]) -> list[list[tuple[str, Path]]]:
    """Split segments into separate discussions based on 30min+ gaps."""
    if not segments:
        return []

    discussions = []
    current = [segments[0]]

    for i in range(1, len(segments)):
        prev_time = _seconds_from_filename(segments[i - 1][1].name)
        curr_time = _seconds_from_filename(segments[i][1].name)
        gap = curr_time - prev_time

        if gap >= DISCUSSION_GAP_S:
            discussions.append(current)
            current = [segments[i]]
        else:
            current.append(segments[i])

    if current:
        discussions.append(current)

    return discussions


async def transcribe_segments_by_user(segments: list[tuple[str, Path]]) -> list[dict]:
    """Group segments by user, concatenate each user's WAVs, transcribe."""
    # Group by user
    user_files: dict[str, list[Path]] = {}
    for username, filepath in segments:
        user_files.setdefault(username, []).append(filepath)

    results = []
    for username, files in user_files.items():
        combined = concatenate_wav_files(files)
        if not combined:
            continue

        try:
            transcript = await transcribe_file(combined)
        except Exception:
            log.exception("Failed to transcribe audio for %s", username)
            continue
        finally:
            if combined and combined.exists() and combined.name == "_combined.wav":
                combined.unlink(missing_ok=True)

        text = transcript.get("text", "")
        if not text or not text.strip():
            continue

        results.append({
            "username": username,
            "text": text,
            "duration_seconds": transcript.get("audio_duration", 0),
        })

    return results


def _build_conversation(segments: list[tuple[str, Path]], transcripts: list[dict]) -> str:
    """Build conversation text from segments and transcripts."""
    # Build sentence pools per user
    user_sentences: dict[str, list[str]] = {}
    for t in transcripts:
        username = t["username"]
        text = t["text"].strip()
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ".!?" and len(current) > 10:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        user_sentences[username] = sentences

    # Count segments per user
    user_seg_counts: dict[str, int] = {}
    for username, _ in segments:
        user_seg_counts[username] = user_seg_counts.get(username, 0) + 1

    # Distribute sentences across segments
    user_idx: dict[str, float] = {u: 0.0 for u in user_sentences}
    user_step: dict[str, float] = {}
    for username, sentences in user_sentences.items():
        count = user_seg_counts.get(username, 1)
        user_step[username] = len(sentences) / count if count > 0 else 0

    lines = []
    last_speaker = None
    last_seg_time = None
    current_block = ""

    for username, filepath in segments:
        seg_time = _seconds_from_filename(filepath.name)
        sentences = user_sentences.get(username, [])
        if not sentences:
            continue

        idx = int(user_idx.get(username, 0))
        step = user_step.get(username, 1)
        end_idx = min(int(idx + max(step, 1)), len(sentences))
        chunk = " ".join(sentences[idx:end_idx])
        user_idx[username] = end_idx

        if not chunk:
            continue

        # Check for conversation gap
        if last_seg_time is not None:
            gap = seg_time - last_seg_time
            if gap >= CONVERSATION_GAP_S:
                if current_block and last_speaker:
                    lines.append(f"{last_speaker}: {current_block.strip()}")
                gap_min = int(gap / 60)
                lines.append(f"*— {gap_min} min gap —*")
                last_speaker = None
                current_block = ""

        last_seg_time = seg_time

        if username == last_speaker:
            current_block += " " + chunk
        else:
            if current_block and last_speaker:
                lines.append(f"{last_speaker}: {current_block.strip()}")
            last_speaker = username
            current_block = chunk

    if current_block and last_speaker:
        lines.append(f"{last_speaker}: {current_block.strip()}")

    return "\n\n".join(lines)


async def process_new(date: datetime | None = None) -> dict | None:
    """Process only new (untranscribed) segments. Returns the latest discussion report."""
    date = date or datetime.now(timezone.utc)
    day_str = date.strftime("%Y-%m-%d")

    new_segments = get_new_speech_segments(date)
    if not new_segments:
        log.info("No new segments to process for %s", day_str)
        return None

    log.info("Processing %d new segments", len(new_segments))

    # Split into discussions (30min gap = new discussion)
    discussions = _split_into_discussions(new_segments)
    log.info("Found %d new discussion(s) with %d total segments", len(discussions), len(new_segments))

    # Process only the latest discussion (most relevant)
    latest = discussions[-1]

    transcripts = await transcribe_segments_by_user(latest)
    if not transcripts:
        return None

    # Mark these segments as processed
    processed_names = [p.name for _, p in latest]
    mark_files_processed(processed_names, date)

    # Also mark older discussions as processed (don't re-transcribe them)
    for disc in discussions[:-1]:
        mark_files_processed([p.name for _, p in disc], date)

    conversation = _build_conversation(latest, transcripts)
    total_duration = sum(t.get("duration_seconds", 0) for t in transcripts)
    participants = list({t["username"] for t in transcripts})

    # Send to Pascribe
    try:
        await send_to_pascribe({
            "transcript": conversation,
            "date": day_str,
            "participants": participants,
            "source": "benjamin-discord-bot",
        })
    except Exception:
        log.exception("Failed to send to Pascribe server")

    header = f"📅 **{day_str}** — {len(participants)} participant(s) ({', '.join(participants)}), {total_duration/60:.0f} min\n"

    convo_preview = conversation
    if len(convo_preview) > 1500:
        convo_preview = convo_preview[:1500] + "\n\n*... (truncated, full transcript saved)*"

    return {
        "text": f"{header}\n{convo_preview}",
        "pings": [],
        "transcripts": transcripts,
        "conversation": conversation,
    }


async def generate_daily_report(date: datetime | None = None) -> dict | None:
    """Generate report from new segments only."""
    return await process_new(date)
