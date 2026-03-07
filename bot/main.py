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
from pathlib import Path

import discord
from discord.ext import commands, tasks

from audio.capture import UserAudioSink
from commands.slash import PascribeCog
from config import (
    BLACKLIST_CHANNELS,
    DISCORD_TOKEN,
    GUILD_ID,
    RECORDINGS_DIR,
    REPORT_CHANNEL_ID,
    INACTIVITY_THRESHOLD_S,
    OPENROUTER_API_KEY,
)
from transcription.pipeline import generate_daily_report
from transcription.streaming import PerUserStreamManager
from triggers import check_cooldown, mark_triggered, build_trigger_snippet, write_trigger_file
from wakeword import LocalWakeWordDetector
import analysis

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
stream_transcriber: PerUserStreamManager | None = None
wake_detector: LocalWakeWordDetector | None = None
_joining: bool = False
_main_loop: asyncio.AbstractEventLoop | None = None

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


COOLDOWN_CHIME_PATH = Path(__file__).parent / "assets" / "cooldown_chime.wav"

async def _play_cooldown_chime() -> None:
    """Play a short low-volume descending chime to signal cooldown denial."""
    guild = bot.get_guild(GUILD_ID)
    if not guild or not guild.voice_client or not guild.voice_client.is_connected():
        return
    vc = guild.voice_client
    if vc.is_playing():
        return  # don't interrupt active audio
    try:
        source = discord.FFmpegPCMAudio(str(COOLDOWN_CHIME_PATH))
        source = discord.PCMVolumeTransformer(source, volume=0.15)
        vc.play(source)
    except Exception:
        pass


def _vosk_wake_callback(user_id: int, username: str, text: str) -> None:
    """Called from audio thread when Vosk detects 'benjamin'. Bridge to async."""
    if not _main_loop:
        return
    if not check_cooldown(user_id):
        asyncio.run_coroutine_threadsafe(_play_cooldown_chime(), _main_loop)
        return
    mark_triggered(user_id)
    log.info("🔊 VOSK TRIGGER by %s: %s", username, text)
    asyncio.run_coroutine_threadsafe(_vosk_trigger_async(user_id, username, text), _main_loop)


async def _vosk_trigger_async(user_id: int, username: str, text: str) -> None:
    """Async handler for Vosk wake word detection."""
    try:
        from config import RECORDINGS_DIR
        await _play_trigger_chime()
        
        # Get VC members
        vc_user_ids = {}
        guild = bot.get_guild(GUILD_ID)
        if guild and guild.voice_client and guild.voice_client.channel:
            for m in guild.voice_client.channel.members:
                if not m.bot:
                    vc_user_ids[m.id] = m.name

        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_path = RECORDINGS_DIR / day / "_transcript.txt"
        
        # Use the real transcript for context, not Vosk's garbled text
        full_transcript = ""
        if daily_path.exists():
            try:
                full_transcript = daily_path.read_text(encoding="utf-8")
            except Exception:
                pass
        
        # Build snippet from the real transcript if available
        if full_transcript:
            snippet = build_trigger_snippet(full_transcript)
        else:
            snippet = f"**{username}: {text}**"

        from triggers import fire_instant_trigger, fetch_and_cache_recent_responses
        await fetch_and_cache_recent_responses(bot, REPORT_CHANNEL_ID)
        ok = await fire_instant_trigger(
            vc_user_ids, daily_path, snippet, full_transcript,
            username, user_id, 
        )
        if not ok:
            write_trigger_file(vc_user_ids, daily_path, snippet, username, user_id)
        log.info("Vosk trigger async complete (hook_ok=%s)", ok)
    except Exception:
        log.exception("Error in Vosk trigger async")


CHIME_PATH = Path(__file__).parent / "assets" / "trigger_chime.wav"

async def _play_trigger_chime() -> None:
    """Play a short notification chime in the voice channel."""
    guild = bot.get_guild(GUILD_ID)
    if not guild or not guild.voice_client or not guild.voice_client.is_connected():
        return
    vc = guild.voice_client
    if vc.is_playing():
        vc.stop()  # stop previous chime so new trigger can play
    try:
        source = discord.FFmpegPCMAudio(str(CHIME_PATH))
        # Lower volume to 20%
        source = discord.PCMVolumeTransformer(source, volume=0.5)
        vc.play(source)
        log.info("🔔 Playing trigger chime in VC")
    except Exception as e:
        log.warning("Failed to play trigger chime: %s", e)


async def _stream_wake_word(transcript: str, recent_context: str, trigger_user_id: int = 0, trigger_username: str = "unknown") -> None:
    """Called by per-user streaming transcriber when 'benjamin' detected in real-time."""
    from pathlib import Path
    from config import RECORDINGS_DIR

    # Check cooldown
    if not check_cooldown(trigger_user_id):
        await _play_cooldown_chime()
        return

    log.info("🎤 STREAM TRIGGER by %s", trigger_username)
    mark_triggered(trigger_user_id)
    
    await _play_trigger_chime()
    
    # Get current VC members
    vc_user_ids = {}
    guild = bot.get_guild(GUILD_ID)
    if guild and guild.voice_client and guild.voice_client.channel:
        for m in guild.voice_client.channel.members:
            if not m.bot:
                vc_user_ids[m.id] = m.name

    # Build snippet
    snippet = f"**{trigger_username}**: {transcript[:300]}"
    if recent_context:
        lines = recent_context.strip().split("\n")
        context_before = "\n".join(lines[-3:])
        snippet = f"{context_before}\n**{trigger_username}: {transcript[:200]}**"

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_path = RECORDINGS_DIR / day / "_transcript.txt"
    
    full_transcript = ""
    if daily_path.exists():
        try:
            full_transcript = daily_path.read_text(encoding="utf-8")
        except Exception:
            pass

    # Fire instant trigger (agent posts its own message)
    from triggers import fire_instant_trigger, fetch_and_cache_recent_responses
    await fetch_and_cache_recent_responses(bot, REPORT_CHANNEL_ID)
    instant_ok = await fire_instant_trigger(
        vc_user_ids, daily_path, snippet, full_transcript,
        trigger_username, trigger_user_id, 
    )
    
    if not instant_ok:
        write_trigger_file(vc_user_ids, daily_path, snippet, trigger_username, trigger_user_id)


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
    global audio_sink, _last_vc_activity, stream_transcriber

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
            if stream_transcriber:
                await stream_transcriber.disconnect_all()
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

    # Start per-user streaming manager
    if stream_transcriber is None:
        stream_transcriber = PerUserStreamManager(on_wake_word=_stream_wake_word)
        log.info("Per-user stream manager created for real-time wake word detection")

    # Initialize local wake word detector (Vosk, offline)
    global wake_detector
    if wake_detector is None:
        wake_detector = LocalWakeWordDetector(on_wake=_vosk_wake_callback)

    # Start audio capture via socket listener
    audio_sink = UserAudioSink(bot, stream_transcriber=stream_transcriber, wake_detector=wake_detector)
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


# Wake word handling moved to triggers.py


async def _check_wake_word(convo: str, daily_path, day: str):
    """Check for 'benjamin' wake word and trigger response if found."""
    from config import RECORDINGS_DIR
    from pathlib import Path

    # Count-based trigger: only fire if total "benjamin" mentions in full transcript
    # exceed what we've already triggered on. This prevents re-triggering on old
    # audio segments that get transcribed late (e.g. rifqn's old audio from hours ago).
    count_path = RECORDINGS_DIR / day / "_benjamin_trigger_count.txt"
    prev_count = 0
    if count_path.exists():
        try:
            prev_count = int(count_path.read_text().strip())
        except Exception:
            pass

    if daily_path.exists():
        full_text = daily_path.read_text(encoding="utf-8")
        current_count = full_text.lower().count("benjamin")
    else:
        current_count = convo.lower().count("benjamin")

    should_trigger = current_count > prev_count
    log.info("Wake word: current_count=%d prev_count=%d trigger=%s (convo=%d chars)",
             current_count, prev_count, should_trigger, len(convo))

    if not should_trigger:
        return

    # Check cooldown using shared function (batch doesn't know specific user)
    if not check_cooldown(0):
        await _play_cooldown_chime()
        return

    log.info("Wake word 'Benjamin' detected — triggering!")
    mark_triggered(0)
    await _play_trigger_chime()

    # Get current VC members
    vc_user_ids = {}
    guild = bot.get_guild(GUILD_ID)
    if guild and guild.voice_client and guild.voice_client.channel:
        for m in guild.voice_client.channel.members:
            if not m.bot:
                vc_user_ids[m.id] = m.name
    if not vc_user_ids and audio_sink:
        vc_user_ids = dict(audio_sink._user_names)

    # Build focused snippet from latest transcript
    full_transcript = daily_path.read_text(encoding="utf-8") if daily_path.exists() else convo
    snippet = build_trigger_snippet(full_transcript)

    # Extract who said Benjamin from the transcript for TRIGGERED_BY field
    triggered_by_username = ""
    triggered_by_id = 0
    # Find the last mention and extract the speaker
    if snippet and "**" in snippet:
        import re
        match = re.search(r'\*\*([^:]+):', snippet)
        if match:
            triggered_by_username = match.group(1)
            # Try to find the user ID
            for uid, name in vc_user_ids.items():
                if name == triggered_by_username:
                    triggered_by_id = uid
                    break

    # Fire instant trigger (agent posts its own message)
    from triggers import fire_instant_trigger, fetch_and_cache_recent_responses
    await fetch_and_cache_recent_responses(bot, REPORT_CHANNEL_ID)
    instant_ok = await fire_instant_trigger(
        vc_user_ids, daily_path, snippet, full_transcript,
        triggered_by_username, triggered_by_id, 
    )
    
    if not instant_ok:
        write_trigger_file(vc_user_ids, daily_path, snippet, triggered_by_username, triggered_by_id)

    # Update mention count so we don't re-trigger for same mentions
    if daily_path.exists():
        count_path = RECORDINGS_DIR / day / "_benjamin_trigger_count.txt"
        count_path.write_text(str(daily_path.read_text(encoding="utf-8").lower().count("benjamin")))

    # Legacy trigger file
    trigger_path = RECORDINGS_DIR / day / "_benjamin_triggered.txt"
    id_map = "\n".join(f"{uid}:{name}" for uid, name in vc_user_ids.items())
    trigger_path.write_text(f"USERS:\n{id_map}\n\nTRANSCRIPT:\n{convo}", encoding="utf-8")


@tasks.loop(seconds=30)
async def auto_transcribe():
    try:
        from audio.storage import get_new_speech_segments

        new_segs = get_new_speech_segments()
        if len(new_segs) < 1:
            return
        log.info("Auto-transcribing %d new segments", len(new_segs))
        report = await generate_daily_report()
        log.info("Report result: %s", "has conversation" if report and report.get("conversation") else f"empty (report={'present' if report else 'None'})")
        if report:
            from config import RECORDINGS_DIR

            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_path = RECORDINGS_DIR / day / "_transcript.txt"
            convo = report.get("conversation", "")
            log.info("Conversation length: %d chars", len(convo))
            if convo:
                with open(daily_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- {datetime.now(timezone.utc).strftime('%H:%M')} ---\n\n")
                    f.write(convo)
                log.info("Transcript appended to %s", daily_path)

                # Batch wake word: check ONLY the newly transcribed text (not full transcript)
                if "benjamin" in convo.lower():
                    log.info("Batch wake word found in new transcription (%d chars)", len(convo))
                    if not check_cooldown(0):
                        await _play_cooldown_chime()
                    else:
                        mark_triggered(0)
                        await _play_trigger_chime()
                        vc_ids = {}
                        g = bot.get_guild(GUILD_ID)
                        if g and g.voice_client and g.voice_client.channel:
                            for m in g.voice_client.channel.members:
                                if not m.bot:
                                    vc_ids[m.id] = m.name
                        snippet = build_trigger_snippet(convo)
                        full_text = daily_path.read_text(encoding="utf-8") if daily_path.exists() else convo
                        from triggers import fire_instant_trigger, post_processing_placeholder, fetch_and_cache_recent_responses
                        await fetch_and_cache_recent_responses(bot, REPORT_CHANNEL_ID)
                        ph_id = await post_processing_placeholder(bot, REPORT_CHANNEL_ID)
                        ok = await fire_instant_trigger(vc_ids, daily_path, snippet, full_text, placeholder_message_id=ph_id)
                        if not ok:
                            write_trigger_file(vc_ids, daily_path, snippet)
    except Exception:
        log.exception("Error in auto-transcribe")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    global _main_loop
    _main_loop = asyncio.get_running_loop()
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
    if not dave_stats.is_running():
        dave_stats.start()
    if not auto_transcribe.is_running():
        auto_transcribe.start()

    # Initialize and start transcript analysis
    if OPENROUTER_API_KEY:
        analysis.init_analyzer(bot, OPENROUTER_API_KEY)
        analysis.start_analysis()
        log.info("Transcript analysis initialized and started")
    else:
        log.warning("No OpenRouter API key configured - transcript analysis disabled")

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

    async def _do_shutdown():
        if audio_sink:
            audio_sink.cleanup()
        if stream_transcriber:
            await stream_transcriber.disconnect_all()
        await bot.close()

    def _shutdown(sig, frame):
        log.info("Received signal %s — shutting down", sig)
        # Schedule cleanup on the running loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_do_shutdown(), loop)
        else:
            sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
