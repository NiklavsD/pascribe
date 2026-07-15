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
    GCP_PROJECT_ID,
    TTS_ENABLED,
    TTS_VOICE,
    TTS_SPEAKING_RATE,
    TTS_PITCH,
    TTS_VOLUME_GAIN_DB,
    TTS_VOICE_POOL,
)
import random
from transcription.pipeline import generate_daily_report
from transcription.gcp_streaming import GCPPerUserStreamManager
from triggers import check_cooldown, mark_triggered, build_trigger_snippet, write_trigger_file
from wakeword import LocalWakeWordDetector
import analysis
import tts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
# Reduce gateway spam (DAVE handshake logged at INFO)
logging.getLogger("discord.gateway").setLevel(logging.WARNING)
log = logging.getLogger("benjamin")

# Dedicated trigger-flow logger writing to ./logs/trigger-flow.log
# Captures the full lifecycle of each wake word event with high-res timestamps.
import os
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_trigger_handler = logging.FileHandler(_log_dir / "trigger-flow.log", mode="a", encoding="utf-8")
_trigger_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
trigger_log = logging.getLogger("trigger-flow")
trigger_log.setLevel(logging.INFO)
# Idempotent — only add handler if not already present (prevents duplicates on re-import)
if not any(isinstance(h, logging.FileHandler) and "trigger-flow.log" in getattr(h, "baseFilename", "")
           for h in trigger_log.handlers):
    trigger_log.addHandler(_trigger_handler)
trigger_log.propagate = False  # don't double-log to stdout

# ---------------------------------------------------------------------------
# Bot setup  (discord.py uses commands.Bot, not discord.Bot)
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.voice_states = True
intents.guilds = True
intents.members = True
intents.message_content = True  # Needed to read Ben's response messages for TTS

bot = commands.Bot(command_prefix="!", intents=intents)
audio_sink: UserAudioSink | None = None
stream_transcriber: GCPPerUserStreamManager | None = None
wake_detector: LocalWakeWordDetector | None = None
_joining: bool = False
_main_loop: asyncio.AbstractEventLoop | None = None

# Track when someone was last in VC (for inactivity-based report)
_last_vc_activity: datetime | None = None
_report_generated_today: bool = False

# Voice-connection watchdog: when humans are in a VC but we have no live
# recording, this marks the moment the unhealthy state began. If it persists
# past VOICE_STALE_S the watchdog force-tears-down the (zombie) voice client
# and rejoins. Guards against discord.py voice sockets that die (e.g. WS close
# 1006) and never auto-recover — the failure mode that caused a ~19h silent
# outage on 2026-07-03/04.
_voice_unhealthy_since: datetime | None = None
VOICE_STALE_S: int = 90


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
THINKING_LOOP_PATH = Path(__file__).parent / "assets" / "thinking_loop.wav"

# State for the thinking-loop playback
_thinking_task: asyncio.Task | None = None
_thinking_stop: asyncio.Event | None = None


async def start_thinking_loop(start_delay_s: float = 2.0, max_duration_s: float = 25.0) -> None:
    """Play a subtle heartbeat in VC while Ben is processing.
    - Waits start_delay_s before starting (skips false triggers that clean up fast)
    - Auto-stops after max_duration_s regardless (safety timeout if no response comes)
    - Stopped explicitly by stop_thinking_loop() when a response lands
    """
    global _thinking_task, _thinking_stop
    await stop_thinking_loop()

    guild = bot.get_guild(GUILD_ID)
    if not guild or not guild.voice_client or not guild.voice_client.is_connected():
        return
    if not THINKING_LOOP_PATH.exists():
        return

    _thinking_stop = asyncio.Event()

    async def _loop_runner():
        import time as _time
        started_at = _time.monotonic()
        try:
            await asyncio.wait_for(_thinking_stop.wait(), timeout=start_delay_s)
            return
        except asyncio.TimeoutError:
            pass

        vc = guild.voice_client
        try:
            while not _thinking_stop.is_set():
                # Safety timeout — never play longer than max_duration_s total
                if _time.monotonic() - started_at > max_duration_s:
                    log.warning("Thinking loop hit max_duration_s (%.0fs) — auto-stopping", max_duration_s)
                    try:
                        if vc.is_connected() and vc.is_playing():
                            vc.stop()
                    except Exception:
                        pass
                    return
                if not vc.is_connected():
                    return
                if vc.is_playing():
                    await asyncio.sleep(0.2)
                    continue
                try:
                    source = discord.FFmpegPCMAudio(str(THINKING_LOOP_PATH))
                    source = discord.PCMVolumeTransformer(source, volume=0.18)
                    vc.play(source)
                except Exception:
                    return
                while vc.is_playing() and not _thinking_stop.is_set():
                    if _time.monotonic() - started_at > max_duration_s:
                        try:
                            vc.stop()
                        except Exception:
                            pass
                        return
                    await asyncio.sleep(0.15)
                if _thinking_stop.is_set() and vc.is_playing():
                    try:
                        vc.stop()
                    except Exception:
                        pass
                    return
        except asyncio.CancelledError:
            pass

    _thinking_task = asyncio.create_task(_loop_runner())


async def stop_thinking_loop() -> None:
    """Stop the thinking loop if running."""
    global _thinking_task, _thinking_stop
    if _thinking_stop is not None:
        _thinking_stop.set()
    if _thinking_task is not None:
        try:
            await asyncio.wait_for(_thinking_task, timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            _thinking_task.cancel()
    _thinking_task = None
    _thinking_stop = None
    # Stop any current playback so it doesn't block TTS
    guild = bot.get_guild(GUILD_ID)
    if guild and guild.voice_client and guild.voice_client.is_playing():
        try:
            guild.voice_client.stop()
        except Exception:
            pass

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
    import duplex_mode
    if duplex_mode.is_active():
        _dx = duplex_mode.get()
        if _dx and _dx.standby and _main_loop and \
                (not _dx.allowed_users or user_id in _dx.allowed_users):
            # Standby duplex: wake word brings the mic back instead of cascading.
            # Works regardless of VC_TRIGGER_ENABLED (that gates the cascade only).
            trigger_log.info("ACCEPT path=vosk reason=duplex_wake user=%s", username)
            asyncio.run_coroutine_threadsafe(_dx.wake("wake word"), _main_loop)
        else:
            # Live duplex session owns the conversation — skip the cascade.
            trigger_log.info("REJECT path=vosk reason=duplex_active user=%s", username)
        return
    # No active session, but a recent one may have idle-closed — let any VC
    # member's wake word resurrect it from its saved state (collaborative). Set
    # DUPLEX_ALLOWED_IDS to restrict who can revive it.
    _allowed = {int(x) for x in os.getenv("DUPLEX_ALLOWED_IDS", "")
                .replace(",", " ").split()}
    if _main_loop and (not _allowed or user_id in _allowed):
        import duplex_mode as _dm
        async def _try_resurrect():
            if await _dm.wake_from_state(bot):
                trigger_log.info("ACCEPT path=vosk reason=duplex_resurrect user=%s", username)
        asyncio.run_coroutine_threadsafe(_try_resurrect(), _main_loop)
    from config import VC_TRIGGER_ENABLED as _VC_ON
    if not _VC_ON:
        trigger_log.info("REJECT path=vosk reason=trigger_disabled user=%s", username)
        return
    if not _main_loop:
        return
    trigger_log.info("DETECT path=vosk user=%s text=%r", username, text[:200])
    if not check_cooldown(user_id):
        trigger_log.info("REJECT path=vosk reason=cooldown user=%s", username)
        asyncio.run_coroutine_threadsafe(_play_cooldown_chime(), _main_loop)
        return
    mark_triggered(user_id)
    trigger_log.info("ACCEPT path=vosk user=%s", username)
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

        from triggers import fire_instant_trigger, post_processing_placeholder, fetch_and_cache_recent_responses
        await fetch_and_cache_recent_responses(bot, REPORT_CHANNEL_ID)
        ph_id = await post_processing_placeholder(bot, REPORT_CHANNEL_ID, daily_path=daily_path)
        await start_thinking_loop()
        ok = await fire_instant_trigger(
            vc_user_ids, daily_path, snippet, full_transcript,
            username, user_id, placeholder_message_id=ph_id,
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
    from config import RECORDINGS_DIR, VC_TRIGGER_ENABLED as _VC_ON

    if not _VC_ON:
        return
    trigger_log.info("DETECT path=gcp_stream user=%s text=%r", trigger_username, transcript[:200])
    # Check cooldown
    if not check_cooldown(trigger_user_id):
        trigger_log.info("REJECT path=gcp_stream reason=cooldown user=%s", trigger_username)
        await _play_cooldown_chime()
        return

    trigger_log.info("ACCEPT path=gcp_stream user=%s", trigger_username)
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
    from triggers import fire_instant_trigger, post_processing_placeholder, fetch_and_cache_recent_responses
    await fetch_and_cache_recent_responses(bot, REPORT_CHANNEL_ID)
    ph_id = await post_processing_placeholder(bot, REPORT_CHANNEL_ID, daily_path=daily_path)
    await start_thinking_loop()
    instant_ok = await fire_instant_trigger(
        vc_user_ids, daily_path, snippet, full_transcript,
        trigger_username, trigger_user_id, placeholder_message_id=ph_id,
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

    # GCP STT disabled — credentials restricted (Account Restricted error)
    # Bot runs AssemblyAI-only mode for transcription
    stream_transcriber = None

    # Initialize local wake word detector (Vosk, offline). Always loaded: the
    # duplex standby→wake path needs it even when the legacy trigger cascade is
    # disabled (VC_TRIGGER_ENABLED only gates the cascade, in the callback).
    global wake_detector
    from config import VC_TRIGGER_ENABLED
    if wake_detector is None:
        wake_detector = LocalWakeWordDetector(on_wake=_vosk_wake_callback)
        if not VC_TRIGGER_ENABLED:
            log.info("VC_TRIGGER_ENABLED=false — Vosk loaded for duplex wake only "
                     "(legacy cascade stays off)")

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


async def _force_voice_reset(guild: discord.Guild) -> None:
    """Hard-teardown a dead/zombie voice client so the next join starts clean.

    A plain reconnect isn't enough: after the socket dies, discord.py can leave
    guild.voice_client in a half-state where target.connect() keeps raising
    'Already connected', so the normal 30s join loop spins forever. Forcing a
    disconnect + clearing our own voice state on the gateway clears it."""
    global audio_sink, stream_transcriber
    try:
        if audio_sink:
            audio_sink.cleanup()
            audio_sink = None
    except Exception:
        log.exception("watchdog: audio_sink cleanup failed")
    try:
        if stream_transcriber:
            await stream_transcriber.disconnect_all()
    except Exception:
        log.exception("watchdog: stream_transcriber teardown failed")
    try:
        if guild.voice_client:
            await guild.voice_client.disconnect(force=True)
    except Exception:
        log.exception("watchdog: force disconnect failed")
    # Tell the gateway we've left, in case discord.py's client object is stale.
    try:
        await guild.change_voice_state(channel=None)
    except Exception:
        log.exception("watchdog: change_voice_state(None) failed")
    # Give Discord a moment to tear the old session down before rejoining.
    await asyncio.sleep(3)


@tasks.loop(seconds=30)
async def voice_watchdog():
    """Detect and recover a dead voice connection.

    Healthy = there are no humans to record, OR we have a connected voice
    client with an active audio sink. If humans ARE present but we have no
    live recording for longer than VOICE_STALE_S, assume the voice socket is a
    zombie and force a clean reconnect."""
    global _voice_unhealthy_since
    try:
        guild = bot.get_guild(GUILD_ID)
        if guild is None:
            return
        target = _best_voice_channel(guild)
        vc = guild.voice_client
        healthy = (target is None) or bool(
            vc and vc.is_connected() and audio_sink is not None
        )
        if healthy:
            _voice_unhealthy_since = None
            return

        now = datetime.now(timezone.utc)
        if _voice_unhealthy_since is None:
            _voice_unhealthy_since = now
            log.warning(
                "watchdog: humans in VC but no live recording — starting stale timer"
            )
            return

        elapsed = (now - _voice_unhealthy_since).total_seconds()
        if elapsed < VOICE_STALE_S:
            return

        log.error(
            "watchdog: voice unhealthy for %.0fs — forcing reset + rejoin", elapsed
        )
        await _force_voice_reset(guild)
        await join_best_channel()
        _voice_unhealthy_since = None
    except Exception:
        log.exception("Error in voice watchdog")


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
                            f"{PASCRIBE_URL}/transcript",
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
                    from config import VC_TRIGGER_ENABLED as _VC_ON
                    if not _VC_ON:
                        log.info("VC_TRIGGER_ENABLED=false — skipping batch wake-word handler (no chime, no placeholder, no TTS)")
                    elif not check_cooldown(0):
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
                        ph_id = await post_processing_placeholder(bot, REPORT_CHANNEL_ID, daily_path=daily_path)
                        await start_thinking_loop()
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
    import duplex_mode
    await duplex_mode.setup_http(bot)

    async def _resume_duplex() -> None:
        # Wait for the VC to reconnect after a restart, then resume any
        # in-flight duplex session so a pending response callback still reaches
        # the user by voice (a restart to deploy a fix shouldn't drop the call).
        for _ in range(60):
            g = bot.get_guild(GUILD_ID)
            if g and g.voice_client and g.voice_client.is_connected():
                await duplex_mode.resume_if_active(bot)
                return
            await asyncio.sleep(1)
    asyncio.create_task(_resume_duplex())

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
    if not voice_watchdog.is_running():
        voice_watchdog.start()
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
async def on_message(message: discord.Message):
    """When Ben (OpenClaw) posts in the report channel, synthesize TTS and play in VC."""
    import duplex_mode

    # --- realtime duplex mode commands (any channel, humans only) ---
    if not message.author.bot and message.content.startswith("!duplex"):
        arg = message.content[len("!duplex"):].strip()
        try:
            if arg == "off":
                stopped = await duplex_mode.stop()
                await message.channel.send("🔇 duplex stopped" if stopped else "no active duplex session")
            elif arg.startswith("test"):
                s = duplex_mode.get()
                if not s:
                    await message.channel.send("no session — `!duplex` first")
                else:
                    await s.speak_update(arg[4:].strip() or
                                         "The tts project finished a build. All tests pass. "
                                         "It asks whether to deploy now or wait.")
                    await message.channel.send("📣 update injected")
            elif arg == "status":
                s = duplex_mode.get()
                if not s:
                    await message.channel.send("no active duplex session")
                else:
                    cost = s.in_tokens / 1e6 * 10 + s.out_tokens / 1e6 * 20
                    await message.channel.send(
                        f"live — {s.in_tokens}/{s.out_tokens} audio tok in/out (~${cost:.3f})")
            else:
                await duplex_mode.start(bot, message.channel)
                await message.channel.send("🎙️ duplex live — talk to me. `!duplex off` to stop, "
                                           "`!duplex test` to fake an update.")
        except Exception as e:
            log.exception("duplex command failed")
            await message.channel.send(f"⚠️ duplex error: {e}")
        return

    from config import VC_TRIGGER_ENABLED as _VC_ON
    if not _VC_ON:
        return
    if not TTS_ENABLED:
        return
    if message.channel.id != REPORT_CHANNEL_ID:
        return
    if not message.author.bot:
        return
    # Skip our own bot's messages (Benjamin) — only play OpenClaw Ben's responses
    if bot.user and message.author.id == bot.user.id:
        return
    text = message.content
    if not text or len(text.strip()) < 3:
        return

    trigger_log.info("RESPONSE_RECEIVED author=%s msg_id=%d chars=%d", message.author.name, message.id, len(text))

    # Stop the thinking loop — response has arrived
    await stop_thinking_loop()

    # Duplex mode owns the voice: speak the response through the realtime
    # session instead of GCP TTS (same voice, interruptible).
    _dx = duplex_mode.get()
    if _dx is not None:
        trigger_log.info("TTS_ROUTE=duplex msg_id=%d", message.id)
        await _dx.speak_update(text, source="Ben response")
        return

    guild = bot.get_guild(GUILD_ID)
    if not guild or not guild.voice_client or not guild.voice_client.is_connected():
        trigger_log.warning("TTS_SKIP reason=not_connected_to_vc msg_id=%d", message.id)
        log.debug("TTS skipped: not connected to VC")
        return

    vc = guild.voice_client
    log.info("🗣️ TTS: synthesizing %d chars from %s", len(text), message.author.name)
    trigger_log.info("TTS_START msg_id=%d chars=%d", message.id, len(text))
    _tts_t0 = asyncio.get_running_loop().time()

    # Pick a voice from the pool for variety (different voice each response)
    chosen_voice = random.choice(TTS_VOICE_POOL) if TTS_VOICE_POOL else TTS_VOICE

    # Synthesize off the event loop to avoid blocking
    loop = asyncio.get_running_loop()
    audio_path = await loop.run_in_executor(
        None,
        lambda: tts.synthesize(
            text,
            voice_name=chosen_voice,
            speaking_rate=TTS_SPEAKING_RATE,
            pitch=TTS_PITCH,
            volume_gain_db=TTS_VOLUME_GAIN_DB,
        ),
    )
    if not audio_path:
        log.warning("TTS synthesis returned None")
        trigger_log.error("TTS_FAIL msg_id=%d voice=%s", message.id, chosen_voice)
        return
    _synth_ms = int((asyncio.get_running_loop().time() - _tts_t0) * 1000)
    trigger_log.info("TTS_SYNTH_OK msg_id=%d voice=%s synth_ms=%d file=%s",
                     message.id, chosen_voice, _synth_ms, audio_path.name)

    try:
        # Wait for any in-flight chime to finish
        for _ in range(20):
            if not vc.is_playing():
                break
            await asyncio.sleep(0.1)
        if vc.is_playing():
            vc.stop()
            await asyncio.sleep(0.1)

        source = discord.FFmpegPCMAudio(str(audio_path))
        source = discord.PCMVolumeTransformer(source, volume=0.45)
        vc.play(source)
        log.info("🗣️ TTS playing in VC (%s)", audio_path.name)
        trigger_log.info("TTS_PLAY_OK msg_id=%d", message.id)

        # Periodic cache cleanup
        tts.cleanup_cache()
    except Exception as e:
        log.exception("Failed to play TTS audio")
        trigger_log.error("TTS_PLAY_FAIL msg_id=%d error=%r", message.id, str(e)[:200])


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
        # Record VC join/leave/move events for the trigger prompt
        try:
            from triggers import record_vc_event
            name = member.display_name or member.name
            before_ch = before.channel.name if before.channel else None
            after_ch = after.channel.name if after.channel else None
            if before_ch is None and after_ch is not None:
                record_vc_event(f"{name} joined #{after_ch}")
            elif before_ch is not None and after_ch is None:
                record_vc_event(f"{name} left #{before_ch}")
            elif before_ch != after_ch:
                record_vc_event(f"{name} moved #{before_ch} → #{after_ch}")
            # Mute/deafen state changes (optional, less useful)
            elif before.self_mute != after.self_mute:
                record_vc_event(f"{name} {'muted' if after.self_mute else 'unmuted'}")
        except Exception:
            log.exception("Failed to record VC event")
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
