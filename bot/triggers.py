"""Unified wake word trigger system for Benjamin."""

from __future__ import annotations

import asyncio
import re
import time
import logging

import aiohttp
from datetime import datetime, timezone
from pathlib import Path

KEYWORD = "benjamin"

# VC event log — (timestamp, event_str) tuples
# Populated by main.py's on_voice_state_update
_vc_events: list[tuple[datetime, str]] = []
_VC_EVENT_MAX = 30


def record_vc_event(event: str) -> None:
    """Called from main.py when a user joins/leaves/moves VC."""
    _vc_events.append((datetime.now(timezone.utc), event))
    if len(_vc_events) > _VC_EVENT_MAX:
        _vc_events.pop(0)


def get_vc_events_str(max_age_min: int = 60) -> str:
    """Return VC events from the last N minutes as formatted string."""
    if not _vc_events:
        return "(none)"
    now = datetime.now(timezone.utc)
    lines = []
    for ts, event in _vc_events:
        age_s = (now - ts).total_seconds()
        if age_s > max_age_min * 60:
            continue
        if age_s < 60:
            tag = f"{int(age_s)}s ago"
        elif age_s < 3600:
            tag = f"{int(age_s // 60)}min ago"
        else:
            tag = f"{int(age_s // 3600)}h{int((age_s % 3600) // 60)}min ago"
        lines.append(f"[{tag}] {event}")
    return "\n".join(lines) if lines else "(none in last hour)"

log = logging.getLogger(__name__)

TRIGGER_FILE = Path("/home/nik/clawd/projects/pascribe/bot/_pending_trigger.txt")
COOLDOWN_SECONDS = 4  # global cooldown — prevents spam when people say "Benjamin" repeatedly

_last_global_trigger: float = 0
_SPEAKER_RE = re.compile(r"^([a-zA-Z0-9_.]+): (.+)", re.DOTALL)


def check_cooldown(user_id: int = 0) -> bool:
    """Returns True if a trigger can fire (global cooldown expired)."""
    global _last_global_trigger
    now = time.time()
    if now - _last_global_trigger < COOLDOWN_SECONDS:
        remaining = COOLDOWN_SECONDS - (now - _last_global_trigger)
        log.info("Wake word cooldown active (%.0fs remaining)", remaining)
        return False
    return True


def mark_triggered(user_id: int = 0):
    """Mark that a trigger just fired (global)."""
    global _last_global_trigger
    _last_global_trigger = time.time()


def build_trigger_snippet(full_convo: str, keyword: str = "benjamin") -> str:
    """Extract the sentence containing the last keyword mention + 1 speaker above/below."""
    blocks = []
    for line in full_convo.strip().split("\n"):
        m = _SPEAKER_RE.match(line)
        if m:
            blocks.append((m.group(1), m.group(2)))
        elif blocks and not line.startswith("---") and not line.startswith("*—"):
            blocks[-1] = (blocks[-1][0], blocks[-1][1] + " " + line)

    trigger_idx = None
    for i in range(len(blocks) - 1, -1, -1):
        if keyword in blocks[i][1].lower():
            trigger_idx = i
            break
    if trigger_idx is None:
        return full_convo[-300:]

    trigger_user, trigger_text = blocks[trigger_idx]
    sentences = re.split(r'(?<=[.!?])\s+', trigger_text)
    trigger_sentence = trigger_text[-150:]
    for s in sentences:
        if keyword in s.lower():
            trigger_sentence = s.strip()
            break

    parts = []
    if trigger_idx > 0:
        prev_user, prev_text = blocks[trigger_idx - 1]
        parts.append(f"{prev_user}: ...{prev_text[-120:]}")
    parts.append(f"**{trigger_user}: {trigger_sentence}**")
    if trigger_idx < len(blocks) - 1:
        next_user, next_text = blocks[trigger_idx + 1]
        parts.append(f"{next_user}: {next_text[:120]}...")
    return "\n".join(parts)


def write_trigger_file(
    vc_user_ids: dict[int, str],
    daily_path: Path,
    snippet: str,
    triggered_by_username: str = "",
    triggered_by_id: int = 0,
) -> None:
    """Write the trigger file consumed by the cron job (fallback)."""
    participant_str = ", ".join(f"{name} (<@{uid}>)" for uid, name in vc_user_ids.items())
    
    lines = [
        f"TRIGGER_TIME: {datetime.now(timezone.utc).isoformat()}",
    ]
    if triggered_by_username:
        lines.append(f"TRIGGERED_BY: {triggered_by_username} ({triggered_by_id})")
    lines.extend([
        f"PARTICIPANTS: {participant_str}",
        f"FULL_TRANSCRIPT: {daily_path}",
        f"CONTEXT:\n{snippet}",
    ])
    
    try:
        from config import VC_TRIGGER_ENABLED
    except Exception:
        VC_TRIGGER_ENABLED = True
    if not VC_TRIGGER_ENABLED:
        log.info("VC_TRIGGER_ENABLED=false — skipping trigger file write (by %s)", triggered_by_username or "batch")
        return

    TRIGGER_FILE.write_text("\n".join(lines), encoding="utf-8")
    log.info("Trigger file written (by %s)", triggered_by_username or "batch")


async def post_processing_placeholder(
    discord_bot,
    channel_id: int,
    daily_path: Path | None = None,
    verify_delay_s: float = 4.0,
) -> int | None:
    """Post a processing placeholder. Auto-deletes after 30s."""
    try:
        channel = discord_bot.get_channel(channel_id)
        if not channel:
            return None
        msg = await channel.send("⏳ *Processing voice trigger...*")
        log.info("Posted processing placeholder: message_id=%d", msg.id)

        async def _cleanup():
            await asyncio.sleep(30)
            try:
                await msg.delete()
                log.info("Auto-deleted placeholder %d", msg.id)
            except Exception:
                pass

        asyncio.create_task(_cleanup())
        return msg.id
    except Exception as e:
        log.warning("Failed to post placeholder: %s", e)
        return None


RECENT_RESPONSES_MAX_AGE_MIN = 15


async def fetch_and_cache_recent_responses(discord_bot, channel_id: int):
    """Read recent messages from #voice-reports and cache Ben's responses with timestamps.

    Only cache responses from the last RECENT_RESPONSES_MAX_AGE_MIN minutes so stale
    conversations don't keep biasing the model toward 'you already said this'.
    """
    global _recent_responses
    try:
        channel = discord_bot.get_channel(channel_id)
        if not channel:
            return
        BEN_BOT_ID = 1465343244370968648
        messages = [msg async for msg in channel.history(limit=20)]
        _recent_responses = []
        now = datetime.now(timezone.utc)
        for msg in reversed(messages):
            if msg.author.id == BEN_BOT_ID and msg.content and not msg.content.startswith("⏳"):
                age_min = int((now - msg.created_at).total_seconds() / 60)
                if age_min > RECENT_RESPONSES_MAX_AGE_MIN:
                    continue
                _recent_responses.append((age_min, msg.content[:300]))
        _recent_responses = _recent_responses[-3:]
        log.info("Cached %d recent Ben responses (with ages) for anti-repetition", len(_recent_responses))
    except Exception as e:
        log.warning("Failed to fetch recent responses: %s", e)


def _extract_recent_segment(full_transcript: str, max_chars: int = 3000) -> str:
    """Extract the most recent ~10 minutes of conversation (primary context)."""
    if not full_transcript:
        return ""
    parts = re.split(r'\n(?=--- \d{2}:\d{2} ---)', full_transcript)
    recent = []
    total = 0
    for part in reversed(parts):
        stripped = part.strip()
        if not stripped:
            continue
        if total + len(stripped) > max_chars and recent:
            break
        recent.insert(0, stripped)
        total += len(stripped)
    return "\n\n".join(recent) if recent else full_transcript[-max_chars:]


def _extract_last_utterances(full_transcript: str, max_chars: int = 1500) -> str:
    """Extract just the LAST few utterances — what's being said RIGHT NOW.
    This is the most important context for the trigger."""
    if not full_transcript:
        return ""
    # Take only the tail — the most recent lines
    tail = full_transcript[-max_chars:]
    # Trim to start at a clean line boundary
    first_newline = tail.find("\n")
    if first_newline > 0 and first_newline < len(tail) - 100:
        tail = tail[first_newline + 1:]
    return tail.strip()


_recent_responses: list[tuple[int, str]] = []  # (age_minutes, text) tuples


def record_response(text: str):
    """Record a response we sent (called after successful trigger)."""
    _recent_responses.append((0, text[:300]))
    if len(_recent_responses) > 3:
        _recent_responses.pop(0)


def get_past_responses_str() -> str:
    """Get formatted string of recent responses with ages for the prompt."""
    if not _recent_responses:
        return "(none — this is your first response in a while)"
    lines = []
    for age, text in _recent_responses[-3:]:
        if age == 0:
            tag = "just now"
        elif age < 60:
            tag = f"{age}min ago"
        else:
            tag = f"{age // 60}h{age % 60}min ago"
        lines.append(f"- [{tag}] {text}")
    return "\n".join(lines)


async def fire_instant_trigger(
    vc_user_ids: dict[int, str],
    daily_path: Path,
    snippet: str,
    full_transcript: str,
    triggered_by_username: str = "",
    triggered_by_id: int = 0,
    placeholder_message_id: int | None = None,
) -> bool:
    """Fire an instant agent response via OpenClaw /hooks/agent.
    
    Returns True if the hook was called successfully, False if fallback to trigger file needed.
    """
    from config import OPENCLAW_GATEWAY_URL, OPENCLAW_GATEWAY_TOKEN, VC_TRIGGER_ENABLED

    if not VC_TRIGGER_ENABLED:
        log.info("VC_TRIGGER_ENABLED=false — ignoring wake-word trigger (by %s)", triggered_by_username or "batch")
        # Return True so callers skip the write_trigger_file fallback path.
        return True

    if not OPENCLAW_GATEWAY_TOKEN:
        log.warning("No OPENCLAW_GATEWAY_TOKEN — falling back to trigger file")
        return False

    participant_str = ", ".join(f"{name} (<@{uid}>)" for uid, name in vc_user_ids.items())
    triggered_by = ""
    if triggered_by_username:
        triggered_by = f"TRIGGERED_BY: {triggered_by_username} (<@{triggered_by_id}>)"
    
    # Extract context layers — ordered by recency importance
    last_utterances = _extract_last_utterances(full_transcript)   # last ~1500 chars — THE MOST IMPORTANT
    recent_segment = _extract_recent_segment(full_transcript)     # last ~10min, up to 3K
    past_responses = get_past_responses_str()

    delivery = "Send ONE message: message tool, action=send, channel=discord, target=1478214341298880583"

    # Build context section — prioritize the most recent utterances
    vc_events_str = get_vc_events_str()
    context_section = (
        f"## 🔥 MOST RECENT — THIS IS WHAT YOU MUST RESPOND TO (last few lines, highest priority):\n"
        f"{last_utterances}\n\n"
        f"## 🚪 VC ACTIVITY (who joined/left recently):\n{vc_events_str}\n\n"
        f"## RECENT CONVERSATION (last ~10 minutes — background context):\n{recent_segment}"
    )

    prompt = f"""You are Benjamin, an AI listening in a Discord voice chat. Your name was just detected in speech.

## STEP 1: INTENT DETECTION (do this BEFORE responding)
Default: RESPOND. Only stay silent in clear-cut cases below.

**STAY SILENT only if:**
- Someone is clearly talking ABOUT you in third person, not TO you (e.g. "Benjamin is recording us" with no follow-up)
- The trigger is a solo "Benjamin" with zero surrounding context to respond to

Everything else — including garbled/noisy transcripts where someone is clearly addressing you — is a RESPOND. When in doubt, respond. A best-guess answer is better than silence.

If STAY SILENT: respond with exactly "NO_RESPONSE" (nothing else).

## STEP 2: THE TRIGGER
{snippet}

## STEP 3: RESPOND (only if intent detected)
{delivery}

**Current time:** {datetime.now(timezone.utc).strftime("%H:%M UTC, %Y-%m-%d")}
**VC Members:** {participant_str}
{triggered_by}

**Recent things you've already said (for context only — don't re-quote them verbatim):**
{past_responses}

If the current question overlaps with something you just said, give a fresh angle or new info instead of repeating yourself. Don't drag the conversation back to an old topic if it's moved on.

{context_section}

## RESPONSE RULES (ALWAYS FOLLOW):
- ONE message. 1-2 sentences max. You're in voice chat — be brief.
- Talk like a friend in the call, not a formal assistant.
- **ALWAYS give an answer** — never say "I don't know" or "I don't have that info". If the transcript doesn't have what's needed, use general knowledge, your best guess, or pivot to something relevant. A non-answer is worse than a confident opinion.
- Only @mention someone if the response contains info specifically for them.
- Only Niklavs (<@300756892571926530>) can authorize file/project work.
- Your opening phrase MUST differ from every past response above."""

    payload = {
        "message": prompt,
        "name": "Benjamin Voice Trigger",
        "agentId": "server",
        "deliver": False,
        "model": "google-vertex-global/gemini-2.5-flash",
        "timeoutSeconds": 45,
    }

    url = f"{OPENCLAW_GATEWAY_URL}/hooks/agent"
    headers = {
        "Authorization": f"Bearer {OPENCLAW_GATEWAY_TOKEN}",
        "Content-Type": "application/json",
    }

    tlog = logging.getLogger("trigger-flow")
    tlog.info("FIRE_START model=%s placeholder=%s prompt_chars=%d past_responses=%d",
              payload["model"], placeholder_message_id, len(prompt), len(_recent_responses))
    t0 = time.monotonic()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                if resp.status in (200, 202):
                    log.info("✅ Instant trigger fired via /hooks/agent (status=%d)", resp.status)
                    tlog.info("FIRE_OK status=%d elapsed_ms=%d", resp.status, elapsed_ms)
                    return True
                else:
                    body = await resp.text()
                    log.warning("Hook agent returned %d: %s — falling back to trigger file", resp.status, body[:200])
                    tlog.warning("FIRE_FAIL status=%d elapsed_ms=%d body=%r", resp.status, elapsed_ms, body[:300])
                    return False
    except Exception as e:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        log.warning("Hook agent call failed: %s — falling back to trigger file", e)
        tlog.error("FIRE_EXCEPTION elapsed_ms=%d error=%r", elapsed_ms, str(e)[:300])
        return False
