#!/usr/bin/env python3
"""Benjamin — Pascribe Discord voice bot (discord.py + DAVE E2EE).

Joins the most populated voice channel in the configured guild,
captures per-user audio via socket listener, runs VAD, and transcribes
via AssemblyAI.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

import discord
from discord.ext import commands, tasks

from audio.capture import UserAudioSink
from commands.slash import PascribeCog
from commands.voice import VoiceCommandDetector
from config import (
    BLACKLIST_CHANNELS,
    DISCORD_TOKEN,
    GUILD_ID,
    REPORT_CHANNEL_ID,
    INACTIVITY_THRESHOLD_S,
)
from transcription.pipeline import generate_daily_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    stream=sys.stdout,
)
# Reduce gateway spam (DAVE handshake logged at INFO)
logging.getLogger("discord.gateway").setLevel(logging.WARNING)
log = logging.getLogger("benjamin")

# ---------------------------------------------------------------------------
# Bot setup  (discord.py uses commands.Bot, not discord.Bot)
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.voice_states = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)
audio_sink: UserAudioSink | None = None
_joining: bool = False

# Track when someone was last in VC (for inactivity-based report)
_last_vc_activity: datetime | None = None
_report_generated_today: bool = False


# ---------------------------------------------------------------------------
# Pending SSRC mappings (captured before audio_sink exists)
# ---------------------------------------------------------------------------
_pending_ssrc_map: dict[int, int] = {}  # ssrc → user_id


# ---------------------------------------------------------------------------
# Voice WS hook — captures SPEAKING events for SSRC → user mapping
# ---------------------------------------------------------------------------
async def _voice_ws_hook(ws, msg):
    """Intercept voice websocket messages to capture SPEAKING (op 5)."""
    global audio_sink
    if not isinstance(msg, dict):
        return
    op = msg.get("op")
    data = msg.get("d")
    if op == 5 and data:  # SPEAKING
        user_id = int(data.get("user_id", 0))
        ssrc = data.get("ssrc", 0)
        if user_id and ssrc:
            _pending_ssrc_map[ssrc] = user_id
            if audio_sink:
                audio_sink.register_speaking(user_id, ssrc)
    elif op == 12 and data:  # CLIENT_CONNECT
        user_id = int(data.get("user_id", 0))
        ssrc = data.get("audio_ssrc", 0)
        if user_id and ssrc:
            _pending_ssrc_map[ssrc] = user_id
            if audio_sink:
                audio_sink.register_speaking(user_id, ssrc)


# ---------------------------------------------------------------------------
# Custom VoiceClient that sets the WS hook for SPEAKING events
# ---------------------------------------------------------------------------
class RecordingVoiceClient(discord.VoiceClient):
    def create_connection_state(self):
        from discord.voice_state import VoiceConnectionState
        return VoiceConnectionState(self, hook=_voice_ws_hook)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _best_voice_channel(guild: discord.Guild) -> discord.VoiceChannel | None:
    best, best_count = None, 0
    for vc in guild.voice_channels:
        if vc.id in BLACKLIST_CHANNELS:
            continue
        count = sum(1 for m in vc.members if not m.bot)
        if count > best_count:
            best, best_count = vc, count
    return best


def _any_humans_in_vc(guild: discord.Guild) -> bool:
    for vc in guild.voice_channels:
        if vc.id in BLACKLIST_CHANNELS:
            continue
        if any(not m.bot for m in vc.members):
            return True
    return False


async def _trigger_processing() -> None:
    log.info("Voice trigger — processing recordings")
    try:
        await generate_daily_report()
    except Exception:
        log.exception("Error during voice-triggered processing")


async def _pause_user(user_id: int) -> None:
    if audio_sink:
        audio_sink.exclude_user(user_id)


async def _resume_user(user_id: int) -> None:
    if audio_sink:
        audio_sink.include_user(user_id)


voice_detector = VoiceCommandDetector(
    on_trigger=_trigger_processing,
    on_pause=_pause_user,
    on_resume=_resume_user,
)


# ---------------------------------------------------------------------------
# Voice management
# ---------------------------------------------------------------------------
async def join_best_channel() -> None:
    global _joining
    if _joining:
        return
    _joining = True
    try:
        await _join_best_channel_inner()
    finally:
        _joining = False


async def _join_best_channel_inner() -> None:
    global audio_sink, _last_vc_activity

    guild = bot.get_guild(GUILD_ID)
    if guild is None:
        log.warning("Guild %d not found", GUILD_ID)
        return

    target = _best_voice_channel(guild)
    vc: discord.VoiceClient | None = guild.voice_client

    # Track activity
    if _any_humans_in_vc(guild):
        _last_vc_activity = datetime.now(timezone.utc)

    if target is None:
        if vc and vc.is_connected():
            log.info("No populated VCs — disconnecting")
            if audio_sink:
                audio_sink.cleanup()
                audio_sink = None
            await vc.disconnect()
        return

    if vc and vc.is_connected():
        if vc.channel.id == target.id:
            return
        log.info("Moving to #%s (%d members)", target.name, len(target.members))
        if audio_sink:
            audio_sink.cleanup()
        await vc.move_to(target)
    else:
        log.info("Joining #%s (%d members)", target.name, len(target.members))
        try:
            vc = await target.connect(cls=RecordingVoiceClient)
        except discord.ClientException:
            log.debug("Already connected, skipping")
            return
        except Exception as e:
            log.error("Failed to connect to voice: %s", e)
            try:
                if guild.voice_client:
                    await guild.voice_client.disconnect(force=True)
            except Exception:
                pass
            try:
                await guild.change_voice_state(channel=None)
            except Exception:
                pass
            return

    # Wait for connection to stabilise
    for _ in range(20):
        if vc and vc.is_connected():
            break
        await asyncio.sleep(0.5)
    else:
        log.warning("Voice client not connected after wait — skipping recording")
        return

    # Start audio capture via socket listener
    audio_sink = UserAudioSink(bot, voice_detector)
    # Apply any SSRC mappings captured during handshake
    for ssrc, user_id in _pending_ssrc_map.items():
        audio_sink.register_speaking(user_id, ssrc)
    try:
        audio_sink.start_listening(vc)
        log.info("Recording started (DAVE E2EE: %s)", "active" if vc._connection.can_encrypt else "inactive")
    except Exception as e:
        log.error("Failed to start recording: %s", e)
        audio_sink.cleanup()
        audio_sink = None


# ---------------------------------------------------------------------------
# Report posting
# ---------------------------------------------------------------------------
async def post_report_to_discord(report: dict) -> None:
    channel = bot.get_channel(REPORT_CHANNEL_ID)
    if not channel:
        log.error("Report channel %d not found", REPORT_CHANNEL_ID)
        return

    text = report.get("text", "")
    user_pings = report.get("pings", [])

    if not text:
        log.info("Empty report, skipping post")
        return

    ping_str = ""
    if user_pings:
        ping_str = "\n\n" + "\n".join(
            f"<@{p['user_id']}> — {p['reason']}" for p in user_pings
        )

    full_msg = f"## 🎙️ Daily Voice Report\n{text}{ping_str}"
    if len(full_msg) <= 2000:
        await channel.send(full_msg)
    else:
        await channel.send("## 🎙️ Daily Voice Report")
        for i in range(0, len(text), 1900):
            await channel.send(text[i : i + 1900])
        if ping_str:
            await channel.send(ping_str)

    log.info("Report posted to #%s", channel.name)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
@tasks.loop(seconds=30)
async def channel_monitor():
    try:
        await join_best_channel()
    except Exception:
        log.exception("Error in channel monitor")


@tasks.loop(minutes=5)
async def inactivity_check():
    global _report_generated_today, _last_vc_activity

    now = datetime.now(timezone.utc)

    if now.hour == 0 and now.minute < 5:
        _report_generated_today = False

    if _report_generated_today:
        return
    if _last_vc_activity is None:
        return

    guild = bot.get_guild(GUILD_ID)
    if guild and _any_humans_in_vc(guild):
        return

    elapsed = (now - _last_vc_activity).total_seconds()
    if elapsed < INACTIVITY_THRESHOLD_S:
        return

    log.info("VC inactive for %.0f min — generating daily report", elapsed / 60)
    _report_generated_today = True

    try:
        if audio_sink:
            audio_sink.flush_raw_audio()
        report = await generate_daily_report()
        if report:
            await post_report_to_discord(report)
            from config import RECORDINGS_DIR

            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            transcript_path = RECORDINGS_DIR / day / "_transcript.txt"
            if transcript_path.exists():
                try:
                    import aiohttp
                    from config import PASCRIBE_URL, PASCRIBE_TOKEN

                    payload = {
                        "transcript": transcript_path.read_text(encoding="utf-8"),
                        "date": day,
                        "participants": [t["username"] for t in report.get("transcripts", [])],
                        "source": "benjamin-discord-bot",
                        "type": "daily_summary",
                    }
                    headers = {"Authorization": f"Bearer {PASCRIBE_TOKEN}", "Content-Type": "application/json"}
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{PASCRIBE_URL}/api/transcription",
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60),
                        ) as resp:
                            log.info("Daily summary sent to Pascribe: %d", resp.status)
                except Exception:
                    log.exception("Failed to send daily summary to Pascribe")
    except Exception:
        log.exception("Error generating daily report")


@tasks.loop(seconds=30)
async def dave_stats():
    """Log per-user DAVE decryption stats."""
    guild = bot.get_guild(GUILD_ID)
    if not guild or not guild.voice_client:
        return
    conn = guild.voice_client._connection
    if not conn.dave_session:
        return
    import davey
    for uid in conn.dave_session.get_user_ids():
        try:
            uid_int = int(uid)
            stats = conn.dave_session.get_decryption_stats(uid_int, davey.MediaType.audio)
            if stats and stats.attempts > 0:
                name = audio_sink._user_names.get(uid_int, str(uid_int)) if audio_sink else str(uid_int)
                log.info("DAVE stats %s(%d): attempts=%d ok=%d fail=%d passthrough=%d",
                         name, uid_int, stats.attempts, stats.successes, stats.failures, stats.passthroughs)
        except Exception:
            pass


@tasks.loop(minutes=5)
async def flush_raw_audio():
    if audio_sink:
        audio_sink.flush_raw_audio()


@tasks.loop(minutes=3)
async def auto_transcribe():
    try:
        from audio.storage import get_new_speech_segments

        new_segs = get_new_speech_segments()
        if len(new_segs) < 3:
            return
        log.info("Auto-transcribing %d new segments", len(new_segs))
        report = await generate_daily_report()
        if report:
            from config import RECORDINGS_DIR

            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_path = RECORDINGS_DIR / day / "_transcript.txt"
            convo = report.get("conversation", "")
            if convo:
                with open(daily_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- {datetime.now(timezone.utc).strftime('%H:%M')} ---\n\n")
                    f.write(convo)
                log.info("Transcript appended to %s", daily_path)

                _should_trigger = "benjamin" in convo.lower()
                if _should_trigger:
                    # Cooldown: don't re-trigger within 5 minutes
                    from pathlib import Path
                    pending = Path("/home/nik/clawd/projects/pascribe/bot/_pending_trigger.txt")
                    if pending.exists():
                        try:
                            age = (datetime.now(timezone.utc) - datetime.fromisoformat(
                                pending.read_text().split("\n")[0].split(": ", 1)[1]
                            )).total_seconds()
                            if age < 300:
                                log.debug("Benjamin trigger cooldown (%ds ago), skipping", int(age))
                                _should_trigger = False
                        except Exception:
                            pass
                if _should_trigger:
                    log.info("Wake word 'Benjamin' detected!")
                    trigger_path = RECORDINGS_DIR / day / "_benjamin_triggered.txt"

                    # Get current VC members (not all-time seen users)
                    vc_user_ids = {}
                    guild = bot.get_guild(GUILD_ID)
                    if guild and guild.voice_client and guild.voice_client.channel:
                        for m in guild.voice_client.channel.members:
                            if not m.bot:
                                vc_user_ids[m.id] = m.display_name
                    if not vc_user_ids and audio_sink:
                        # Fallback to audio sink if VC member list unavailable
                        vc_user_ids = dict(audio_sink._user_names)

                    id_map = "\n".join(f"{uid}:{name}" for uid, name in vc_user_ids.items())
                    trigger_path.write_text(
                        f"USERS:\n{id_map}\n\nTRANSCRIPT:\n{convo}", encoding="utf-8"
                    )

                    # Build focused transcript snippet around trigger
                    def _build_trigger_snippet(full_convo: str, keyword: str = "benjamin") -> str:
                        """Extract the sentence with 'benjamin' + 1 speaker above/below, trimmed."""
                        import re
                        speaker_re = re.compile(r"^([a-zA-Z0-9_]+): (.+)", re.DOTALL)
                        # Split into speaker blocks: [(username, text), ...]
                        blocks = []
                        for line in full_convo.strip().split("\n"):
                            m = speaker_re.match(line)
                            if m:
                                blocks.append((m.group(1), m.group(2)))
                            elif blocks and not line.startswith("---") and not line.startswith("*—"):
                                # Continuation of previous speaker
                                blocks[-1] = (blocks[-1][0], blocks[-1][1] + " " + line)

                        # Find block containing keyword (last occurrence)
                        trigger_idx = None
                        for i in range(len(blocks) - 1, -1, -1):
                            if keyword in blocks[i][1].lower():
                                trigger_idx = i
                                break
                        if trigger_idx is None:
                            return full_convo[-300:]

                        # Extract the sentence containing "benjamin" from the trigger block
                        trigger_user, trigger_text = blocks[trigger_idx]
                        sentences = re.split(r'(?<=[.!?])\s+', trigger_text)
                        trigger_sentence = trigger_text[-150:]  # fallback
                        for s in sentences:
                            if keyword in s.lower():
                                trigger_sentence = s.strip()
                                break

                        # Build output: 1 block above (trimmed) + trigger sentence + 1 block below (trimmed)
                        parts = []
                        if trigger_idx > 0:
                            prev_user, prev_text = blocks[trigger_idx - 1]
                            parts.append(f"{prev_user}: ...{prev_text[-120:]}")
                        parts.append(f"**{trigger_user}: {trigger_sentence}**")
                        if trigger_idx < len(blocks) - 1:
                            next_user, next_text = blocks[trigger_idx + 1]
                            parts.append(f"{next_user}: {next_text[:120]}...")
                        return "\n".join(parts)

                    # Post to #voice-reports with focused transcript for Ben
                    report_channel = bot.get_channel(REPORT_CHANNEL_ID)
                    if report_channel:
                        try:
                            # Read full day transcript for broader context
                            full_transcript = ""
                            if daily_path.exists():
                                full_transcript = daily_path.read_text(encoding="utf-8")
                            snippet = _build_trigger_snippet(full_transcript or convo)
                            participant_names = ", ".join(name for name in vc_user_ids.values())
                            BEN_BOT_ID = 1465343244370968648
                            await report_channel.send(
                                f"<@{BEN_BOT_ID}> 🎙️ **VOICE TRIGGER** — \"Benjamin\" mentioned in VC\n"
                                f"**Members:** {participant_names}\n"
                                f"**Transcript:**\n>>> {snippet[:1500]}",
                                allowed_mentions=discord.AllowedMentions(users=[discord.Object(id=BEN_BOT_ID)]),
                            )
                            log.info("Posted trigger to #voice-reports")
                        except Exception as e:
                            log.warning("Failed to post trigger: %s", e)

                    # Also write trigger file as backup
                    from pathlib import Path
                    pending = Path("/home/nik/clawd/projects/pascribe/bot/_pending_trigger.txt")
                    participant_str = ", ".join(f"{name} (<@{uid}>)" for uid, name in vc_user_ids.items())
                    snippet = _build_trigger_snippet((daily_path.read_text(encoding="utf-8") if daily_path.exists() else convo))
                    pending.write_text(
                        f"TRIGGER_TIME: {datetime.now(timezone.utc).isoformat()}\n"
                        f"PARTICIPANTS: {participant_str}\n"
                        f"FULL_TRANSCRIPT: {daily_path}\n"
                        f"TRANSCRIPT:\n{snippet}",
                        encoding="utf-8",
                    )
                    log.info("Wrote pending trigger file")
    except Exception:
        log.exception("Error in auto-transcribe")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    log.info("Logged in as %s (ID: %d)", bot.user, bot.user.id)
    log.info("Guild target: %d", GUILD_ID)
    log.info("Report channel: %d", REPORT_CHANNEL_ID)

    cog = PascribeCog(bot)
    await bot.add_cog(cog)
    # Copy guild commands and sync
    bot.tree.copy_global_to(guild=discord.Object(id=GUILD_ID))
    try:
        synced = await bot.tree.sync(guild=discord.Object(id=GUILD_ID))
        log.info("Synced %d slash commands to guild", len(synced))
    except Exception:
        log.exception("Failed to sync slash commands")

    if not channel_monitor.is_running():
        channel_monitor.start()
    if not inactivity_check.is_running():
        inactivity_check.start()
    if not flush_raw_audio.is_running():
        flush_raw_audio.start()
    if not dave_stats.is_running():
        dave_stats.start()
    if not auto_transcribe.is_running():
        auto_transcribe.start()

    # Force clear any stale voice state before first join
    guild = bot.get_guild(GUILD_ID)
    if guild and guild.voice_client:
        try:
            await guild.voice_client.disconnect(force=True)
        except Exception:
            pass
    await asyncio.sleep(5)
    await join_best_channel()


@bot.event
async def on_voice_state_update(
    member: discord.Member,
    before: discord.VoiceState,
    after: discord.VoiceState,
):
    global _last_vc_activity
    if member.guild.id != GUILD_ID:
        return
    if not member.bot:
        _last_vc_activity = datetime.now(timezone.utc)
    await asyncio.sleep(2)
    try:
        await join_best_channel()
    except Exception:
        log.exception("Error handling voice state update")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(sig, frame):
        log.info("Received signal %s — shutting down", sig)
        if audio_sink:
            audio_sink.cleanup()
        loop.create_task(bot.close())

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
