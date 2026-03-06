"""Orchestrate transcription and send results to the Pascribe analysis server."""

from __future__ import annotations

import asyncio
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


async def _send_to_pascribe_safe(conversation: str, day_str: str, participants: list):
    """Fire-and-forget wrapper for Pascribe server."""
    try:
        await send_to_pascribe({
            "transcript": conversation,
            "date": day_str,
            "participants": participants,
            "source": "benjamin-discord-bot",
        })
    except Exception:
        log.debug("Pascribe send failed (non-blocking)")


async def send_to_pascribe(payload: dict) -> dict:
    """POST transcription data to the Pascribe analysis server."""
    url = f"{PASCRIBE_URL}/api/transcription"
    headers = {
        "Authorization": f"Bearer {PASCRIBE_TOKEN}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
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


async def _transcribe_single_segment(username: str, filepath: Path) -> dict | None:
    """Transcribe a single speech segment. Returns dict or None if empty."""
    # Skip very small files (< 10KB likely no real speech)
    if filepath.stat().st_size < 10_000:
        log.debug("Skipping tiny segment %s (%d bytes)", filepath.name, filepath.stat().st_size)
        return None

    try:
        transcript = await transcribe_file(filepath)
    except Exception:
        log.exception("Failed to transcribe segment %s for %s", filepath.name, username)
        return None

    text = transcript.get("text", "")
    if not text or not text.strip():
        return None

    return {
        "username": username,
        "text": text.strip(),
        "duration_seconds": transcript.get("audio_duration", 0),
        "filepath": filepath,
    }


def _merge_consecutive_segments(segments: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """Merge consecutive segments from the same user into combined files."""
    if not segments:
        return []
    
    merged = []
    current_user = segments[0][0]
    current_files = [segments[0][1]]
    
    for username, filepath in segments[1:]:
        if username == current_user:
            current_files.append(filepath)
        else:
            # Different user — flush current group
            if len(current_files) == 1:
                merged.append((current_user, current_files[0]))
            else:
                # Concatenate multiple files
                combined = concatenate_wav_files(current_files)
                if combined:
                    # Rename to indicate it's merged
                    merged_path = combined.parent / f"_merged_{current_files[0].name}"
                    combined.rename(merged_path)
                    merged.append((current_user, merged_path))
                else:
                    # Fallback: just use first file
                    merged.append((current_user, current_files[0]))
            
            current_user = username
            current_files = [filepath]
    
    # Don't forget last group
    if len(current_files) == 1:
        merged.append((current_user, current_files[0]))
    else:
        combined = concatenate_wav_files(current_files)
        if combined:
            merged_path = combined.parent / f"_merged_{current_files[0].name}"
            combined.rename(merged_path)
            merged.append((current_user, merged_path))
        else:
            merged.append((current_user, current_files[0]))
    
    if len(merged) < len(segments):
        log.info("Merged %d segments into %d API calls", len(segments), len(merged))
    
    return merged


async def transcribe_segments_individually(segments: list[tuple[str, Path]]) -> list[dict]:
    """Transcribe segments with accurate speaker attribution.
    
    Merges consecutive same-user segments into single API calls to reduce cost,
    while keeping different-user segments separate for correct attribution.
    """
    # Step 1: Merge consecutive same-user segments
    merged = _merge_consecutive_segments(segments)
    
    # Step 2: Transcribe merged segments in parallel batches
    results = []
    batch_size = 5
    for i in range(0, len(merged), batch_size):
        batch = merged[i:i + batch_size]
        tasks = [_transcribe_single_segment(username, filepath) for username, filepath in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in batch_results:
            if isinstance(res, Exception):
                log.warning("Segment transcription failed: %s", res)
                continue
            if res:
                results.append(res)
    
    # Step 3: Clean up any temp merged files
    for username, filepath in merged:
        if filepath.name.startswith("_merged_"):
            filepath.unlink(missing_ok=True)
    
    return results


def _build_conversation(segment_transcripts: list[dict]) -> str:
    """Build conversation text from individually transcribed segments.

    Each entry has username, text, duration_seconds, filepath — already in
    chronological order with correct speaker attribution.
    """
    lines = []
    last_speaker = None
    last_seg_time = None
    current_block = ""

    for t in segment_transcripts:
        username = t["username"]
        text = t["text"]
        seg_time = _seconds_from_filename(t["filepath"].name) if t.get("filepath") else 0

        # Check for conversation gap
        if last_seg_time is not None and seg_time > 0:
            gap = seg_time - last_seg_time
            if gap >= CONVERSATION_GAP_S:
                if current_block and last_speaker:
                    lines.append(f"{last_speaker}: {current_block.strip()}")
                gap_min = int(gap / 60)
                lines.append(f"*— {gap_min} min gap —*")
                last_speaker = None
                current_block = ""

        if seg_time > 0:
            last_seg_time = seg_time

        if username == last_speaker:
            current_block += " " + text
        else:
            if current_block and last_speaker:
                lines.append(f"{last_speaker}: {current_block.strip()}")
            last_speaker = username
            current_block = text

    if current_block and last_speaker:
        lines.append(f"{last_speaker}: {current_block.strip()}")

    return "\n".join(lines)


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

    transcripts = await transcribe_segments_individually(latest)
    if not transcripts:
        return None

    # Mark these segments as processed
    processed_names = [p.name for _, p in latest]
    mark_files_processed(processed_names, date)

    # Also mark older discussions as processed (don't re-transcribe them)
    for disc in discussions[:-1]:
        mark_files_processed([p.name for _, p in disc], date)

    conversation = _build_conversation(transcripts)
    total_duration = sum(t.get("duration_seconds", 0) for t in transcripts)
    participants = list({t["username"] for t in transcripts})

    # Send to Pascribe (fire-and-forget, don't block pipeline)
    asyncio.create_task(_send_to_pascribe_safe(conversation, day_str, participants))

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
