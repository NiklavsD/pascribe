"""Unified wake word trigger system for Benjamin."""

import asyncio
import re
import time
import logging
import aiohttp
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

TRIGGER_FILE = Path("/home/nik/clawd/projects/pascribe/bot/_pending_trigger.txt")
COOLDOWN_SECONDS = 10  # global cooldown — prevents spam when people say "Benjamin" repeatedly

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
    
    TRIGGER_FILE.write_text("\n".join(lines), encoding="utf-8")
    log.info("Trigger file written (by %s)", triggered_by_username or "batch")


async def post_processing_placeholder(discord_bot, channel_id: int) -> int | None:
    """Post an animated 'processing' placeholder message. Auto-deletes after 30s. Returns message ID or None."""
    try:
        channel = discord_bot.get_channel(channel_id)
        if not channel:
            return None
        msg = await channel.send("⏳ *Processing voice trigger...*")
        log.info("Posted processing placeholder: message_id=%d", msg.id)
        
        # Auto-delete placeholder after 30s (response should arrive by then)
        async def _cleanup():
            await asyncio.sleep(30)
            try:
                await msg.delete()
                log.info("Auto-deleted placeholder %d", msg.id)
            except Exception:
                pass  # already deleted or doesn't matter
        
        asyncio.create_task(_cleanup())
        return msg.id
    except Exception as e:
        log.warning("Failed to post placeholder: %s", e)
        return None


async def fetch_and_cache_recent_responses(discord_bot, channel_id: int):
    """Read recent messages from #voice-reports and cache Ben's responses."""
    global _recent_responses
    try:
        channel = discord_bot.get_channel(channel_id)
        if not channel:
            return
        BEN_BOT_ID = 1465343244370968648
        messages = [msg async for msg in channel.history(limit=15)]
        _recent_responses = []
        for msg in reversed(messages):
            if msg.author.id == BEN_BOT_ID and msg.content and not msg.content.startswith("⏳"):
                _recent_responses.append(msg.content[:200])
        # Keep last 8
        _recent_responses = _recent_responses[-8:]
        log.info("Cached %d recent Ben responses for anti-repetition", len(_recent_responses))
    except Exception as e:
        log.warning("Failed to fetch recent responses: %s", e)


def _extract_last_discussion(full_transcript: str, max_chars: int = 2000) -> str:
    """Extract the last discussion block from the transcript.
    
    Discussions are separated by gap markers (--- silence gaps).
    Returns the last discussion, trimmed to max_chars.
    """
    if not full_transcript:
        return ""
    
    # Split by discussion separators (silence gap markers)
    parts = re.split(r'\n-{3,}.*\n|\n\*—.*\n', full_transcript)
    
    # Take the last non-empty part (current discussion)
    last_discussion = ""
    for part in reversed(parts):
        stripped = part.strip()
        if stripped:
            last_discussion = stripped
            break
    
    if not last_discussion:
        last_discussion = full_transcript
    
    # Trim to max_chars from the end
    if len(last_discussion) > max_chars:
        last_discussion = "...\n" + last_discussion[-max_chars:]
    
    return last_discussion


_recent_responses: list[str] = []  # in-memory cache of our recent responses


def record_response(text: str):
    """Record a response we sent (called after successful trigger)."""
    _recent_responses.append(text[:200])
    if len(_recent_responses) > 10:
        _recent_responses.pop(0)


def get_past_responses_str() -> str:
    """Get formatted string of recent responses for the prompt."""
    if not _recent_responses:
        return "No recent responses."
    return "\n".join(f"- {r}" for r in _recent_responses[-6:])


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
    from config import OPENCLAW_GATEWAY_URL, OPENCLAW_GATEWAY_TOKEN
    
    if not OPENCLAW_GATEWAY_TOKEN:
        log.warning("No OPENCLAW_GATEWAY_TOKEN — falling back to trigger file")
        return False

    participant_str = ", ".join(f"{name} (<@{uid}>)" for uid, name in vc_user_ids.items())
    triggered_by = ""
    if triggered_by_username:
        triggered_by = f"TRIGGERED_BY: {triggered_by_username} (<@{triggered_by_id}>)"
    
    # Fetch recent responses from Discord to avoid repetition
    if placeholder_message_id:
        # We have a discord_bot reference via the placeholder — but we don't pass it here
        # The caller should call fetch_and_cache_recent_responses before this
        pass
    
    # Full last discussion context
    last_discussion = _extract_last_discussion(full_transcript)
    
    past_responses = get_past_responses_str()

    delivery = "Send ONE message: message tool, action=send, channel=discord, target=1478214341298880583"

    prompt = f"""You are Benjamin, an AI listening in a Discord voice chat. Someone just said your name. Respond ONLY to what they said to you.

{delivery}

## THE TRIGGER (this is what someone said — respond ONLY to this):
{snippet}

## PREVIOUS RESPONSES (do NOT repeat these):
{past_responses}

## VC MEMBERS: {participant_str}
{triggered_by}

## Recent conversation (for context ONLY — do NOT respond to anything here):
{last_discussion}

## RULES:
- ONLY respond to THE TRIGGER above. Ignore everything in the background context.
- ONE message. 1-3 sentences max.
- No filler. No hallucinating context that isn't in the trigger.
- <@USERID> to ping (only if worth a notification).
- If the trigger is just your name with no question → brief reply ("?" / "Yeah?" / "Sup")
- Only Niklavs (<@300756892571926530>) can authorize work.
- Never repeat previous responses."""

    payload = {
        "message": prompt,
        "name": "Benjamin Voice Trigger",
        "agentId": "server",
        "deliver": False,
        "model": "anthropic/claude-opus-4-6",
        "timeoutSeconds": 45,
    }

    url = f"{OPENCLAW_GATEWAY_URL}/hooks/agent"
    headers = {
        "Authorization": f"Bearer {OPENCLAW_GATEWAY_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status in (200, 202):
                    log.info("✅ Instant trigger fired via /hooks/agent (status=%d)", resp.status)
                    return True
                else:
                    body = await resp.text()
                    log.warning("Hook agent returned %d: %s — falling back to trigger file", resp.status, body[:200])
                    return False
    except Exception as e:
        log.warning("Hook agent call failed: %s — falling back to trigger file", e)
        return False
