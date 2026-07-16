"""
Pascribe — Rolling audio buffer → hotkey → Whisper transcription → clipboard
Windows tray app. Requires: Python 3.11+, CUDA-capable GPU recommended.
"""

import threading
import time
import sys
import os
import io
import json
import wave
import logging
import subprocess
import http.client
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
from PIL import Image, ImageDraw
from pystray import Icon, MenuItem, Menu

from pascribe_app import __version__
from pascribe_app.audio_processing import (
    ProcessingCancelled,
    build_segment_map,
    energy_vad_from_reader,
    format_ssmd as format_transcript,
    remap_to_original,
    resample as resample_audio,
    write_wav_segments,
)
from pascribe_app.diagnostics import collect_diagnostics, format_diagnostics
from pascribe_app.jobs import ExclusiveJobRunner, LatestJobRunner
from pascribe_app.paths import (
    build_app_paths,
    prepare_app_paths,
    resolve_recording_path,
)
from pascribe_app.recordings import assign_speaker, open_daily_snapshot
from pascribe_app.secrets import config_for_disk, hydrate_config_secrets
from pascribe_app.storage import atomic_write_json, load_json
from pascribe_app.transcription import group_timed_words
from pascribe_app.validation import normalize_config, validate_settings


APP_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
APP_PATHS = build_app_paths(APP_DIR)
_MIGRATED_FILES = prepare_app_paths(APP_PATHS)

# ─── CUDA DLL fix (Windows) ──────────────────────────────────────────────────
# pip-installed nvidia-cublas-cu12 / nvidia-cudnn-cu12 put DLLs in site-packages
# but Windows doesn't know to look there. Add them to PATH before CTranslate2 loads.
if sys.platform == "win32":
    _site_pkgs = Path(np.__file__).resolve().parent.parent
    _nvidia_bins = list((_site_pkgs / "nvidia").glob("*/bin"))
    if _nvidia_bins:
        os.environ["PATH"] = os.pathsep.join(str(p) for p in _nvidia_bins) + os.pathsep + os.environ["PATH"]

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_PATH = APP_PATHS.log

def setup_logging():
    """Configure logging. Routes to file under pythonw, console otherwise."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    if sys.stdout is None:  # pythonw.exe — no console
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        # Redirect stdout/stderr so library print() calls don't crash
        sys.stdout = open(LOG_PATH, "a", encoding="utf-8")
        sys.stderr = sys.stdout
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

setup_logging()
log = logging.getLogger("pascribe")
for _migrated_file in _MIGRATED_FILES:
    log.info(f"Migrated legacy {_migrated_file} to {APP_PATHS.data_dir}")

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG_PATH = APP_PATHS.config
DEFAULT_CONFIG = {
    "mic_device": None,
    "mic_device_name": None,
    "system_device": None,
    "system_device_name": None,
    "sample_rate": 16000,
    "buffer_minutes": 60,
    "whisper_model": "large-v3",
    "whisper_device": "cuda",
    "hotkeys": {
        "1": 10, "2": 20, "3": 30, "4": 40, "5": 50, "6": 60,
        "7": 5, "8": 3, "9": 1,
    },
    "hotkey_prefix": "ctrl+alt",
    "history_max_entries": 100,
    # Daily recording
    "daily_recording": False,
    "recording_path": str(APP_PATHS.recordings),
    "assemblyai_key": "",
    "homelab_url": None,
    "delete_after_transcribe": False,
}

def load_config() -> dict:
    result = load_json(CONFIG_PATH, dict)
    stored, had_plaintext_secret = hydrate_config_secrets(result.data)
    cfg, warnings = normalize_config(stored, DEFAULT_CONFIG)
    if result.recovered:
        suffix = f" Backup: {result.backup_path}" if result.backup_path else ""
        log.warning(f"Invalid config recovered with defaults.{suffix}")
    for warning in warnings:
        log.warning(warning)
    if had_plaintext_secret:
        save_config(cfg)
        log.info("Protected the AssemblyAI key with Windows DPAPI")
    return cfg

def save_config(cfg: dict):
    with _storage_lock:
        atomic_write_json(CONFIG_PATH, config_for_disk(cfg))

_storage_lock = threading.RLock()
config = load_config()

# ─── History Storage ──────────────────────────────────────────────────────────

HISTORY_PATH = APP_PATHS.history

def load_history() -> list:
    with _storage_lock:
        result = load_json(HISTORY_PATH, list)
    if result.recovered:
        log.warning(f"Invalid history file preserved at {result.backup_path}")
    return result.data if isinstance(result.data, list) else []

def save_history(history: list):
    with _storage_lock:
        atomic_write_json(HISTORY_PATH, history)

def add_history_entry(minutes: int, word_count: int, elapsed: float, text: str):
    with _storage_lock:
        history = load_history()
        history.append({
            "timestamp": datetime.now().isoformat(),
            "minutes": minutes,
            "word_count": word_count,
            "elapsed_seconds": round(elapsed, 1),
            "text": text,
        })
        max_entries = config.get("history_max_entries", 100)
        if len(history) > max_entries:
            history = history[-max_entries:]
        save_history(history)


def get_recording_path() -> Path:
    return resolve_recording_path(config.get("recording_path"), APP_PATHS)

# ─── Device Listing ──────────────────────────────────────────────────────────

def list_input_devices() -> list[tuple[int, str, int]]:
    """Return (index, name, max_input_channels) for input devices.
    Prefers WASAPI host API to avoid duplicate entries across APIs."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    wasapi_idx = next(
        (i for i, h in enumerate(hostapis) if 'WASAPI' in h['name']), None
    )

    results = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            if wasapi_idx is not None and d['hostapi'] != wasapi_idx:
                continue
            results.append((i, d['name'], d['max_input_channels']))

    if not results:
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                results.append((i, d['name'], d['max_input_channels']))

    return results

def resolve_device(device_id: int | None, device_name: str | None) -> int | None:
    """Resolve a device by name first (handles index changes across reboots),
    falling back to stored index if the name isn't found."""
    if device_id is None and device_name is None:
        return None

    devices = sd.query_devices()

    # If we have a stored name, try to find it by name first
    if device_name:
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and d['name'] == device_name:
                if i != device_id:
                    log.info(f"Device '{device_name}' moved: index {device_id} -> {i}")
                return i

    # Fall back to stored index, but validate it exists and is an input device
    if device_id is not None:
        try:
            d = sd.query_devices(device_id)
            if d['max_input_channels'] > 0:
                return device_id
        except Exception:
            pass

    if device_name:
        log.warning(f"Device '{device_name}' (index {device_id}) not found")
    return None


def categorize_device(name: str) -> str:
    """Return category label for a device name."""
    nl = name.lower()
    if 'voicemeeter' in nl:
        return "Voicemeeter"
    elif 'cable' in nl:
        return "Virtual Cable"
    elif any(k in nl for k in ('microphone', 'mic', 'analogue', 'stereo mix')):
        return "Microphone"
    return "Other"


def measure_input_device(device_id: int, duration_seconds: float = 0.75) -> tuple[float, int]:
    """Capture a short sample and return (normalized level, sample rate)."""
    info = sd.query_devices(device_id)
    sample_rate = int(info["default_samplerate"])
    channels = max(1, min(2, int(info["max_input_channels"])))
    frames = max(1, int(sample_rate * duration_seconds))
    with sd.InputStream(
        device=device_id,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
    ) as stream:
        audio, overflowed = stream.read(frames)
    if overflowed:
        log.warning(f"Overflow while testing input device {device_id}")
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    return min(1.0, rms * 12), sample_rate

# ─── Audio Buffer ─────────────────────────────────────────────────────────────

CHUNK_SECONDS = 1

@dataclass
class AudioBuffer:
    """Thread-safe rolling audio buffer storing chunks of audio."""
    max_chunks: int = 3600  # default 60min
    chunks: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    capture_rate: int = 16000  # actual sample rate used for capture

    def add_chunk(self, data: np.ndarray):
        with self.lock:
            self.chunks.append(data.copy())
            while len(self.chunks) > self.max_chunks:
                self.chunks.popleft()

    def get_last_n_minutes(self, minutes: int) -> np.ndarray | None:
        chunks_needed = minutes * 60 // CHUNK_SECONDS
        with self.lock:
            if not self.chunks:
                return None
            selected = list(self.chunks)[-chunks_needed:]
        return np.concatenate(selected) if selected else None

    @property
    def duration_seconds(self) -> float:
        with self.lock:
            return len(self.chunks) * CHUNK_SECONDS

    def resize(self, max_chunks: int) -> None:
        with self.lock:
            self.max_chunks = max(1, int(max_chunks))
            while len(self.chunks) > self.max_chunks:
                self.chunks.popleft()

# ─── Globals ──────────────────────────────────────────────────────────────────

max_chunks = config["buffer_minutes"] * 60 // CHUNK_SECONDS
mic_buffer = AudioBuffer(max_chunks=max_chunks)
loopback_buffer = AudioBuffer(max_chunks=max_chunks)
transcriber = None
_whisper_lock = threading.Lock()
_whisper_inference_lock = threading.Lock()
tray_icon = None
last_transcript = ""
_main_panel_open = False
_main_panel_window = None   # tk.Tk reference for the open panel
_main_panel_notebook = None  # ttk.Notebook reference for tab switching
_paused = False
daily_recorder = None
_instance_lock = None  # socket held to prevent multiple instances
_quick_jobs = None
_quick_jobs_init_lock = threading.Lock()
_daily_jobs = ExclusiveJobRunner(name="pascribe-daily-transcription")
_activity_lock = threading.Lock()
_activity_states: dict[str, tuple[str, float | None, int]] = {}
_activity_sequence = 0
_tray_error_until = 0.0
_mic_level = 0.0
_system_level = 0.0


def set_activity(
    message: str,
    progress: float | None = None,
    *,
    source: str,
) -> None:
    global _activity_sequence
    with _activity_lock:
        _activity_sequence += 1
        _activity_states[source] = (message, progress, _activity_sequence)


def clear_activity(source: str) -> None:
    with _activity_lock:
        _activity_states.pop(source, None)


def get_activity() -> tuple[str, float | None]:
    with _activity_lock:
        if not _activity_states:
            return "Ready", None
        message, progress, _sequence = max(
            _activity_states.values(),
            key=lambda state: state[2],
        )
        return message, progress


def _progress_activity(value: float | None, message: str) -> None:
    set_activity(message, value, source="daily")

# ─── Audio Capture ────────────────────────────────────────────────────────────

def mic_callback(indata, frames, time_info, status):
    global _mic_level
    if status:
        log.warning(f"Mic status: {status}")
    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    _mic_level = min(1.0, float(np.sqrt(np.mean(np.square(mono)))) * 12) if len(mono) else 0.0
    mic_buffer.add_chunk(mono)
    if daily_recorder:
        daily_recorder.write_mic(mono, mic_buffer.capture_rate)

def loopback_callback(indata, frames, time_info, status):
    global _system_level
    if status:
        log.warning(f"Loopback status: {status}")
    mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
    _system_level = min(1.0, float(np.sqrt(np.mean(np.square(mono)))) * 12) if len(mono) else 0.0
    loopback_buffer.add_chunk(mono)
    if daily_recorder:
        daily_recorder.write_sys(mono, loopback_buffer.capture_rate)

def _open_stream(device_id: int, channels: int, callback, preferred_rate: int):
    """Try to open an InputStream at preferred_rate, falling back to the device's default rate."""
    dev_info = sd.query_devices(device_id)
    rates_to_try = [preferred_rate]
    default_rate = int(dev_info['default_samplerate'])
    if default_rate != preferred_rate:
        rates_to_try.append(default_rate)

    last_err = None
    for sr in rates_to_try:
        try:
            stream = sd.InputStream(
                device=device_id,
                samplerate=sr,
                channels=channels,
                dtype='float32',
                blocksize=sr * CHUNK_SECONDS,
                callback=callback,
            )
            stream.start()
            return stream, sr
        except Exception as e:
            last_err = e
            if sr == preferred_rate:
                log.warning(f"Device {device_id} rejected {sr} Hz, trying {default_rate} Hz")
    raise last_err


def start_audio_streams():
    """Start mic and loopback audio capture streams."""
    global daily_recorder
    streams = []
    sr = config["sample_rate"]

    mic_dev = resolve_device(config["mic_device"], config.get("mic_device_name"))
    if mic_dev is not None:
        try:
            dev_info = sd.query_devices(mic_dev)
            stream, actual_rate = _open_stream(mic_dev, 1, mic_callback, sr)
            mic_buffer.capture_rate = actual_rate
            streams.append(stream)
            log.info(f"Mic: {dev_info['name']} @ {actual_rate} Hz")
            # Persist resolved index and name
            config["mic_device"] = mic_dev
            config["mic_device_name"] = dev_info['name']
        except Exception as e:
            log.error(f"Mic capture failed: {e}")
    else:
        log.info("Mic: skipped (none selected)")

    sys_dev = resolve_device(config["system_device"], config.get("system_device_name"))
    if sys_dev is not None:
        try:
            dev_info = sd.query_devices(sys_dev)
            channels = min(2, dev_info['max_input_channels'])
            stream, actual_rate = _open_stream(sys_dev, channels, loopback_callback, sr)
            loopback_buffer.capture_rate = actual_rate
            streams.append(stream)
            log.info(f"System audio: {dev_info['name']} @ {actual_rate} Hz")
            # Persist resolved index and name
            config["system_device"] = sys_dev
            config["system_device_name"] = dev_info['name']
        except Exception as e:
            log.error(f"System audio capture failed: {e}")
    else:
        log.info("System audio: skipped (none selected)")

    # Save back any resolved device changes
    save_config(config)

    if config.get("daily_recording"):
        recording_path = get_recording_path()
        daily_recorder = DailyAudioRecorder(str(recording_path))
        log.info(f"Daily recording -> {recording_path}")

    return streams

# ─── Transcription ────────────────────────────────────────────────────────────

WHISPER_VRAM_MB = {
    "tiny": 400, "base": 500, "small": 1000, "medium": 2000,
    "large-v1": 3200, "large-v2": 3200, "large-v3": 3200,
}
VRAM_SAFETY_MARGIN_MB = 512

def _check_vram(model_name: str) -> tuple[bool, str]:
    """Check if enough VRAM is available to load the model. Returns (ok, reason)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return True, ""
        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        needed_mb = WHISPER_VRAM_MB.get(model_name, 3200) + VRAM_SAFETY_MARGIN_MB
        if free_mb < needed_mb:
            return False, (
                f"Not enough VRAM: {free_mb:.0f} MB free, "
                f"~{needed_mb:.0f} MB needed for {model_name}"
            )
        return True, ""
    except ImportError:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            free_values = [
                float(line.strip())
                for line in result.stdout.splitlines()
                if line.strip()
            ]
            if free_values:
                free_mb = max(free_values)
                needed_mb = WHISPER_VRAM_MB.get(model_name, 3200) + VRAM_SAFETY_MARGIN_MB
                if free_mb < needed_mb:
                    return False, (
                        f"Not enough VRAM: {free_mb:.0f} MB free, "
                        f"~{needed_mb:.0f} MB needed for {model_name}"
                    )
        except Exception:
            pass
        return True, ""
    except Exception as e:
        log.warning(f"VRAM check failed: {e}")
        return True, ""  # Don't block on check failure

def init_whisper():
    """Initialize faster-whisper model (lazy load on first use)."""
    global transcriber
    if transcriber is not None:
        return
    with _whisper_lock:
        if transcriber is not None:
            return
        model = config["whisper_model"]
        device = config["whisper_device"]
        if device == "cuda":
            ok, reason = _check_vram(model)
            if not ok:
                log.warning(f"Skipping model load: {reason}")
                raise RuntimeError(reason)
        log.info(f"Loading Whisper {model} on {device}...")
        from faster_whisper import WhisperModel
        transcriber = WhisperModel(
            model,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        log.info("Whisper model loaded")

def unload_whisper():
    """Free Whisper model from VRAM."""
    global transcriber
    with _whisper_lock:
        transcriber = None
    try:
        import gc
        gc.collect()
    except Exception:
        pass  # native CUDA cleanup can segfault — don't let it kill us
    log.info("Whisper model unloaded")

WHISPER_HALLUCINATIONS = {
    "thank you", "thanks for watching", "thanks for listening",
    "subscribe", "like and subscribe", "bye", "goodbye",
    "the end", "you", "thanks",
}

def transcribe_audio(
    audio: np.ndarray | str | Path,
    cancel_event: threading.Event | None = None,
) -> list[tuple[float, float, str]]:
    """Run one isolated Whisper lifecycle and return timestamped segments."""
    with _whisper_inference_lock:
        if cancel_event and cancel_event.is_set():
            return []
        try:
            init_whisper()
            segments, _info = transcriber.transcribe(
                str(audio) if isinstance(audio, Path) else audio,
                beam_size=5,
                language=None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            result = []
            for segment in segments:
                if cancel_event and cancel_event.is_set():
                    log.info("Transcription cancelled")
                    return []
                text = segment.text.strip()
                if text:
                    result.append((segment.start, segment.end, text))

            if result:
                all_text = " ".join(text for _, _, text in result).strip().lower()
                all_text = all_text.strip(".,!?")
                if all_text in WHISPER_HALLUCINATIONS:
                    log.info(f"Filtered hallucination: '{all_text}'")
                    return []
            return result
        finally:
            if transcriber is not None:
                unload_whisper()


# ─── Daily Audio Recording ────────────────────────────────────────────────────

class DailyAudioRecorder:
    """Writes continuous audio to daily raw PCM files at 16 kHz.

    Stores mic and system as separate streams (mic.raw / sys.raw) under
    recordings/YYYY-MM-DD/. A meta.json records the session start time
    and sample rate so timestamps can be reconstructed later.
    """

    STORE_RATE = 16000

    def __init__(self, recording_path: str):
        self.recording_path = Path(recording_path)
        self._lock = threading.Lock()
        self._mic_fh = None
        self._sys_fh = None
        self._current_date: str | None = None
        self._write_dtype: str = "int16"  # format for new recordings

    def write_mic(self, data: np.ndarray, capture_rate: int):
        self._write("mic", data, capture_rate)

    def write_sys(self, data: np.ndarray, capture_rate: int):
        self._write("sys", data, capture_rate)

    def _write(self, stream: str, data: np.ndarray, capture_rate: int):
        today = date.today().isoformat()
        if capture_rate != self.STORE_RATE:
            data = resample_audio(data, capture_rate, self.STORE_RATE)
        with self._lock:
            if today != self._current_date:
                self._rotate(today)
            fh = self._mic_fh if stream == "mic" else self._sys_fh
            if fh:
                if self._write_dtype == "int16":
                    fh.write((data * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
                else:
                    fh.write(data.astype(np.float32).tobytes())

    def _rotate(self, today: str):
        self._close()
        self._current_date = today
        day_dir = self.recording_path / today
        day_dir.mkdir(parents=True, exist_ok=True)

        meta_path = day_dir / "meta.json"
        if not meta_path.exists():
            # New day: write int16 (half the storage of float32)
            self._write_dtype = "int16"
            meta = {"started": datetime.now().isoformat(), "sample_rate": self.STORE_RATE, "dtype": "int16"}
            atomic_write_json(meta_path, meta)
            log.info(f"Daily recording started: {day_dir}")
        else:
            # Resume: match whatever dtype the existing file uses
            with open(meta_path) as f:
                meta = json.load(f)
            self._write_dtype = meta.get("dtype", "float32")
            log.info(f"Daily recording resumed: {day_dir} ({self._write_dtype})")

        self._mic_fh = open(day_dir / "mic.raw", "ab")
        self._sys_fh = open(day_dir / "sys.raw", "ab")

    def _close(self):
        for fh in (self._mic_fh, self._sys_fh):
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
        self._mic_fh = None
        self._sys_fh = None

    def close(self):
        with self._lock:
            self._close()

    def flush_and_get_sample_limits(self, day: str) -> dict[str, int]:
        """Flush active handles and return stable sample counts for a snapshot."""
        with self._lock:
            for handle in (self._mic_fh, self._sys_fh):
                if handle:
                    handle.flush()
            day_dir = self.recording_path / day
            dtype = np.dtype(self._write_dtype)
            limits: dict[str, int] = {}
            for stream, name in (("mic", "mic.raw"), ("sys", "sys.raw")):
                path = day_dir / name
                if path.exists():
                    limits[stream] = path.stat().st_size // dtype.itemsize
            return limits


# ─── AssemblyAI ───────────────────────────────────────────────────────────────

ASSEMBLYAI_BASE = "https://api.assemblyai.com/v2"


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy audio to WAV bytes (16-bit PCM)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _upload_wav_path(
    path: Path,
    api_key: str,
    cancel_event: threading.Event | None = None,
    progress=None,
) -> str:
    """Stream a WAV from disk so multi-hour audio is never duplicated in RAM."""
    file_size = path.stat().st_size
    connection = http.client.HTTPSConnection("api.assemblyai.com", timeout=60)
    sent = 0
    try:
        connection.putrequest("POST", "/v2/upload")
        connection.putheader("authorization", api_key)
        connection.putheader("content-type", "application/octet-stream")
        connection.putheader("content-length", str(file_size))
        connection.endheaders()
        with path.open("rb") as handle:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise ProcessingCancelled("Upload cancelled")
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
                sent += len(chunk)
                if progress is not None:
                    progress(sent / file_size if file_size else 1.0, "Uploading speech audio")
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"AssemblyAI upload {response.status}: {body[:400]}")
        return json.loads(body)["upload_url"]
    finally:
        connection.close()


def transcribe_with_assemblyai(
    audio: np.ndarray | str | Path,
    sample_rate: int,
    cancel_event: threading.Event | None = None,
    progress=None,
) -> list[dict]:
    """Upload VAD-stripped audio to AssemblyAI, poll until done, return segments.
    Each segment: {start_s, end_s, text}.
    """
    api_key = config.get("assemblyai_key", "")
    if not api_key:
        raise RuntimeError("AssemblyAI key is not configured in Settings")

    if isinstance(audio, (str, Path)):
        wav_path = Path(audio)
        log.info(f"Uploading {wav_path.stat().st_size / 1024 / 1024:.1f} MB to AssemblyAI...")
        upload_url = _upload_wav_path(wav_path, api_key, cancel_event, progress)
    else:
        wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
        log.info(f"Uploading {len(wav_bytes) / 1024 / 1024:.1f} MB to AssemblyAI...")

        request = urllib.request.Request(
            f"{ASSEMBLYAI_BASE}/upload",
            data=wav_bytes,
            headers={"authorization": api_key, "content-type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                upload_url = json.loads(response.read())["upload_url"]
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AssemblyAI upload {error.code}: {body[:400]}")

    hdrs_json = {"authorization": api_key, "content-type": "application/json"}

    # 2. Submit transcription job
    body = json.dumps({"audio_url": upload_url, "language_detection": True}).encode()
    req = urllib.request.Request(
        f"{ASSEMBLYAI_BASE}/transcript", data=body, headers=hdrs_json, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            transcript_id = json.loads(r.read())["id"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AssemblyAI submit {e.code}: {body[:400]}")
    log.info(f"AssemblyAI job: {transcript_id}")

    # 3. Poll
    poll_url = f"{ASSEMBLYAI_BASE}/transcript/{transcript_id}"
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise ProcessingCancelled("Daily transcription cancelled")
        req = urllib.request.Request(poll_url, headers={"authorization": api_key})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                result = json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AssemblyAI poll {e.code}: {body[:400]}")
        status = result["status"]
        if status == "completed":
            break
        elif status == "error":
            raise RuntimeError(f"AssemblyAI error: {result.get('error', 'unknown')}")
        log.info(f"AssemblyAI: {status}...")
        if progress is not None:
            progress(None, f"AssemblyAI: {status}")
        if cancel_event is not None:
            if cancel_event.wait(5):
                raise ProcessingCancelled("Daily transcription cancelled")
        else:
            time.sleep(5)

    return group_timed_words(result.get("words", []))


def post_to_homelab(data: dict, url: str):
    """POST transcript JSON to homelab server."""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"content-type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        log.info(f"Homelab response: {r.status}")

# ─── Daily Transcription ──────────────────────────────────────────────────────

def run_daily_transcription():
    """Start one cancellable daily job and reject accidental duplicates."""
    started = _daily_jobs.start(_daily_transcription_worker)
    if not started:
        notify("Daily transcription is already running")
        log.info("Ignored duplicate daily transcription request")
    return started


def cancel_daily_transcription():
    cancelled = _daily_jobs.cancel()
    if cancelled:
        set_activity("Cancelling daily transcription…", source="daily")
        notify("Cancelling daily transcription…")
    return cancelled


def _daily_transcription_worker(cancel_event: threading.Event):
    snapshot = None
    speech_wav: Path | None = None
    try:
        set_activity("Loading today's recordings…", 0.0, source="daily")
        refresh_tray_icon()
        notify("Loading today's recordings...")

        today = date.today().isoformat()
        recording_path = get_recording_path()
        sample_limits = (
            daily_recorder.flush_and_get_sample_limits(today)
            if daily_recorder is not None
            else None
        )
        snapshot = open_daily_snapshot(
            recording_path,
            today,
            sample_limits=sample_limits,
        )
        if snapshot is None or snapshot.total_samples == 0:
            notify("No recordings found — enable Daily Recording in Settings")
            return

        sample_rate = snapshot.sample_rate
        duration_s = snapshot.total_samples / sample_rate
        reader = snapshot.mixed_reader()
        log.info(
            f"Daily audio snapshot: {duration_s / 3600:.1f}h "
            f"from {snapshot.started.strftime('%H:%M')}"
        )

        notify("VAD pre-processing...")
        vad_segs = energy_vad_from_reader(
            reader,
            snapshot.total_samples,
            sample_rate,
            cancel_event=cancel_event,
            progress=lambda value, message: _progress_activity(value * 0.2, message),
        )

        if not vad_segs:
            notify("No speech detected in today's recordings")
            return

        total_speech_s = sum(e - s for s, e in vad_segs)
        pct = 100 * total_speech_s / duration_s if duration_s > 0 else 0
        log.info(
            f"VAD: {total_speech_s / 60:.1f} min speech / "
            f"{duration_s / 60:.1f} min total ({pct:.0f}%)"
        )

        speech_wav = snapshot.day_dir / f".pascribe-speech-{os.getpid()}.wav"
        set_activity("Preparing speech audio…", 0.2, source="daily")
        write_wav_segments(
            speech_wav,
            reader,
            vad_segs,
            sample_rate,
            cancel_event=cancel_event,
            progress=lambda value, message: _progress_activity(0.2 + value * 0.25, message),
        )
        seg_map = build_segment_map(vad_segs)

        if cancel_event.is_set():
            raise ProcessingCancelled("Daily transcription cancelled")

        api_key = config.get("assemblyai_key", "")
        if api_key:
            notify(f"Transcribing {total_speech_s / 60:.0f} min via AssemblyAI...")
            set_activity("Uploading speech audio…", 0.45, source="daily")
            raw_segs = transcribe_with_assemblyai(
                speech_wav,
                sample_rate,
                cancel_event,
                progress=lambda value, message: _progress_activity(
                    None if value is None else 0.45 + value * 0.3,
                    message,
                ),
            )
        else:
            notify(f"No API key — using local GPU ({total_speech_s / 60:.0f} min)...")
            set_activity("Running local Whisper transcription…", 0.5, source="daily")
            local = transcribe_audio(speech_wav, cancel_event)
            raw_segs = [{"start_s": s, "end_s": e, "text": t} for s, e, t in local]

        if cancel_event.is_set():
            raise ProcessingCancelled("Daily transcription cancelled")
        if not raw_segs:
            notify("No speech transcribed")
            return

        # Remap timestamps from stripped audio back to original audio time
        for seg in raw_segs:
            seg["start_s"] = remap_to_original(
                seg["start_s"],
                seg_map,
                prefer_next_at_boundary=True,
            )
            seg["end_s"] = remap_to_original(seg["end_s"], seg_map)

        set_activity("Assigning speakers…", 0.82, source="daily")
        for index, seg in enumerate(raw_segs):
            if cancel_event.is_set():
                raise ProcessingCancelled("Daily transcription cancelled")
            if snapshot.has_both:
                seg["speaker"] = assign_speaker(
                    seg["start_s"],
                    seg["end_s"],
                    snapshot.mic,
                    snapshot.system,
                    sample_rate,
                )
            else:
                seg["speaker"] = "you" if snapshot.system is None else "discord"
            if index % 25 == 0:
                set_activity(
                    "Assigning speakers…",
                    0.82 + 0.08 * (index + 1) / len(raw_segs),
                    source="daily",
                )

        for seg in raw_segs:
            wall = snapshot.started + timedelta(seconds=seg["start_s"])
            seg["wall_time"] = wall.strftime("%H:%M:%S")

        word_count = sum(len(s["text"].split()) for s in raw_segs)
        log.info(f"Daily transcript: {word_count} words, {len(raw_segs)} segments")

        payload = {
            "date": today,
            "recorded_from": snapshot.started.isoformat(),
            "duration_minutes": round(duration_s / 60, 1),
            "speech_minutes": round(total_speech_s / 60, 1),
            "word_count": word_count,
            "segments": [
                {"time": s["wall_time"], "speaker": s["speaker"], "text": s["text"]}
                for s in raw_segs
            ],
        }

        set_activity("Saving transcript…", 0.92, source="daily")
        day_dir = snapshot.day_dir
        out_path = day_dir / "transcript.json"
        atomic_write_json(out_path, payload)
        log.info(f"Transcript saved: {out_path}")

        # POST to homelab if configured
        homelab_url = config.get("homelab_url")
        if homelab_url:
            notify("Sending to homelab...")
            try:
                post_to_homelab(payload, homelab_url)
            except Exception as e:
                log.error(f"Homelab POST failed: {e}")
                notify(f"Homelab error: {e}")

        if config.get("delete_after_transcribe"):
            if daily_recorder is not None:
                log.info("Preserved active raw audio; it cannot be safely deleted while recording")
            else:
                snapshot.close()
                snapshot = None
                for name in ("mic.raw", "sys.raw"):
                    path = day_dir / name
                    if path.exists():
                        path.unlink()
                log.info("Raw audio deleted after transcription")

        set_activity("Complete", 1.0, source="daily")
        notify(f"Done — {word_count} words, {total_speech_s / 60:.0f} min of speech")

    except ProcessingCancelled:
        log.info("Daily transcription cancelled")
        notify("Daily transcription cancelled")
    except Exception as e:
        log.error(f"Daily transcription failed: {e}", exc_info=True)
        notify(f"Daily transcription failed: {e}")
        show_tray_error()
    finally:
        if snapshot is not None:
            snapshot.close()
        if speech_wav is not None:
            try:
                speech_wav.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                log.warning(f"Could not remove temporary speech WAV: {error}")
        clear_activity("daily")
        refresh_tray_icon()

# ─── Hotkey Handling ──────────────────────────────────────────────────────────

def on_transcribe(minutes: int):
    """Queue a quick transcription; the latest request replaces older pending work."""
    if _paused:
        return
    runner = _get_quick_jobs()
    submission = runner.submit(minutes)
    if not submission.started_immediately:
        message = f"Queued latest request: last {minutes} min"
        set_activity(message, source="quick")
        notify(message)
        log.info(message)


def cancel_quick_transcription():
    runner = _quick_jobs
    if runner is not None and runner.cancel():
        set_activity("Cancelling quick transcription…", source="quick")
        notify("Cancelling quick transcription…")
        return True
    return False


def _get_quick_jobs():
    global _quick_jobs
    if _quick_jobs is None:
        with _quick_jobs_init_lock:
            if _quick_jobs is None:
                _quick_jobs = LatestJobRunner(
                    _on_transcribe_inner,
                    name="pascribe-quick-transcription",
                    on_error=lambda error: log.error(
                        f"Unhandled transcription worker error: {error}", exc_info=True
                    ),
                )
    return _quick_jobs


def _on_transcribe_inner(minutes: int, cancel: threading.Event):
    try:
        _run_quick_transcription(minutes, cancel)
    except Exception as error:
        if not cancel.is_set():
            log.error(f"Transcription error: {error}", exc_info=True)
            show_tray_error()
            notify(f"Error: {error}")
    finally:
        clear_activity("quick")
        refresh_tray_icon()


def _run_quick_transcription(minutes: int, cancel: threading.Event):
    global last_transcript

    log.info(f"Transcribing last {minutes} minute(s)...")
    set_activity(f"Transcribing last {minutes} min…", source="quick")
    refresh_tray_icon()
    notify(f"Transcribing last {minutes} min...")

    # Get audio from buffers and resample to 16 kHz for Whisper
    whisper_rate = 16000

    mic_audio = mic_buffer.get_last_n_minutes(minutes)
    if mic_audio is not None and mic_buffer.capture_rate != whisper_rate:
        mic_audio = resample_audio(mic_audio, mic_buffer.capture_rate, whisper_rate)

    loopback_audio = loopback_buffer.get_last_n_minutes(minutes)
    if loopback_audio is not None and loopback_buffer.capture_rate != whisper_rate:
        loopback_audio = resample_audio(loopback_audio, loopback_buffer.capture_rate, whisper_rate)

    if mic_audio is None and loopback_audio is None:
        log.warning("No audio in buffer")
        show_tray_error()
        notify("No audio in buffer")
        return

    # Mix streams
    if mic_audio is not None and loopback_audio is not None:
        min_len = min(len(mic_audio), len(loopback_audio))
        mixed = mic_audio[:min_len] * 0.5 + loopback_audio[:min_len] * 0.5
    elif mic_audio is not None:
        mixed = mic_audio
    else:
        mixed = loopback_audio

    # Normalize
    peak = np.abs(mixed).max()
    if peak > 0:
        mixed = mixed / peak * 0.95

    try:
        start = time.time()
        segments = transcribe_audio(mixed, cancel_event=cancel)
        elapsed = time.time() - start

        # If cancelled during transcription, discard silently
        if cancel.is_set():
            log.info("Transcription was cancelled, discarding result")
            return

        if segments:
            transcript = format_transcript(segments)
            pyperclip.copy(transcript)
            last_transcript = transcript
            word_count = sum(len(t.split()) for _, _, t in segments)
            log.info(f"{word_count} words in {elapsed:.1f}s -> clipboard")
            notify(f"{word_count} words in {elapsed:.1f}s")
            add_history_entry(minutes, word_count, elapsed, transcript)
        else:
            log.info("No speech detected")
            notify("No speech detected")
    except Exception as e:
        if cancel.is_set():
            return
        log.error(f"Transcription error: {e}")
        show_tray_error()
        notify(f"Error: {e}")

def register_hotkeys():
    """Register hotkeys from config."""
    prefix = config["hotkey_prefix"]
    hotkeys = config["hotkeys"]

    log.info(f"Hotkeys ({prefix} + key):")
    for key in sorted(hotkeys.keys(), key=int):
        minutes = hotkeys[key]
        keyboard.add_hotkey(
            f"{prefix}+{key}",
            lambda m=minutes: on_transcribe(m),
            suppress=True,
        )
        log.info(f"  {prefix} + {key} -> {minutes} min")

# ─── Tray Icon ────────────────────────────────────────────────────────────────

def create_tray_image(color: str = "green") -> Image.Image:
    """Create a 64×64 RGBA tray icon: rounded rect + 5 waveform bars."""
    COLOR_MAP = {
        "green":  "#22c55e",
        "yellow": "#eab308",
        "red":    "#ef4444",
        "gray":   "#6b7280",
    }
    bg = COLOR_MAP.get(color, "#22c55e")
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Rounded square background
    draw.rounded_rectangle([2, 2, 62, 62], radius=14, fill=bg)
    # 5 waveform bars (white), varying heights, centered
    bar_w, gap = 7, 3
    heights = [18, 32, 46, 28, 14]
    total_w = len(heights) * bar_w + (len(heights) - 1) * gap
    x = (64 - total_w) // 2
    cy = 32
    for h in heights:
        top = cy - h // 2
        draw.rounded_rectangle([x, top, x + bar_w, top + h], radius=3, fill="white")
        x += bar_w + gap
    return img

def update_tray_icon(color: str):
    """Change the tray icon color (green/yellow/red/gray)."""
    if tray_icon:
        tray_icon.icon = create_tray_image(color)


def refresh_tray_icon() -> None:
    """Derive the steady tray color from error, pause, and active job state."""
    if time.monotonic() < _tray_error_until:
        update_tray_icon("red")
    elif _paused:
        update_tray_icon("gray")
    else:
        with _activity_lock:
            busy = bool(_activity_states)
        update_tray_icon("yellow" if busy else "green")


def show_tray_error(seconds: float = 3.0) -> None:
    global _tray_error_until
    _tray_error_until = max(_tray_error_until, time.monotonic() + seconds)
    update_tray_icon("red")
    threading.Timer(seconds, refresh_tray_icon).start()

def notify(message: str):
    """Show a balloon notification and update the tray tooltip."""
    if tray_icon:
        tray_icon.title = f"Pascribe — {message}"
        try:
            tray_icon.notify(message, "Pascribe")
        except Exception:
            pass

# ─── Panel Tab Builders ────────────────────────────────────────────────────────

def _build_dashboard_tab(frame, root):
    """Dashboard: live status, quick-transcribe buttons, actions, last transcript."""
    import tkinter as tk
    from tkinter import ttk

    # ── Status ────────────────────────────────────────────────────────────────
    status_lf = ttk.LabelFrame(frame, text="Status", padding=10)
    status_lf.pack(fill=tk.X, padx=14, pady=(14, 6))

    dot = tk.Label(status_lf, text="●", font=("Segoe UI", 18), fg="#22c55e")
    dot.grid(row=0, column=0, padx=(0, 8))
    status_txt = ttk.Label(status_lf, text="Ready", font=("Segoe UI", 11, "bold"))
    status_txt.grid(row=0, column=1, sticky="w")
    info_txt = ttk.Label(status_lf, text="", foreground="#6b7280", font=("Segoe UI", 9))
    info_txt.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
    meters = ttk.Frame(status_lf)
    meters.grid(row=0, column=2, rowspan=2, sticky="e", padx=(24, 0))
    ttk.Label(meters, text="Mic", width=7).grid(row=0, column=0, sticky="w")
    mic_meter = ttk.Progressbar(meters, maximum=100, length=150, mode="determinate")
    mic_meter.grid(row=0, column=1, padx=(4, 0))
    ttk.Label(meters, text="System", width=7).grid(row=1, column=0, sticky="w")
    system_meter = ttk.Progressbar(meters, maximum=100, length=150, mode="determinate")
    system_meter.grid(row=1, column=1, padx=(4, 0))
    status_lf.columnconfigure(2, weight=1)

    progress = ttk.Progressbar(status_lf, maximum=100, mode="determinate")
    progress.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
    progress.grid_remove()

    # ── Quick Transcribe ──────────────────────────────────────────────────────
    qt_lf = ttk.LabelFrame(frame, text="Quick Transcribe", padding=10)
    qt_lf.pack(fill=tk.X, padx=14, pady=(0, 6))
    ttk.Label(
        qt_lf,
        text="Transcribe the last N minutes and copy to clipboard:",
        foreground="#6b7280",
    ).pack(anchor="w", pady=(0, 8))
    btn_row = ttk.Frame(qt_lf)
    btn_row.pack(anchor="w")
    for m in sorted(set(config.get("hotkeys", {}).values())):
        def _make_cmd(mins=m):
            return lambda: on_transcribe(mins)
        ttk.Button(btn_row, text=f"{m} min", width=7, command=_make_cmd()).pack(
            side=tk.LEFT, padx=2, pady=1
        )

    # ── Actions ───────────────────────────────────────────────────────────────
    act_lf = ttk.LabelFrame(frame, text="Actions", padding=10)
    act_lf.pack(fill=tk.X, padx=14, pady=(0, 6))
    act_row = ttk.Frame(act_lf)
    act_row.pack(anchor="w")

    pause_lbl = tk.StringVar(value="▶ Resume Hotkeys" if _paused else "⏸ Pause Hotkeys")
    def _toggle_pause():
        on_pause_toggle(None, None)
        pause_lbl.set("▶ Resume Hotkeys" if _paused else "⏸ Pause Hotkeys")
    ttk.Button(act_row, textvariable=pause_lbl, width=18, command=_toggle_pause).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    ttk.Button(
        act_row,
        text="🎙 Transcribe Today",
        command=run_daily_transcription,
    ).pack(side=tk.LEFT, padx=(0, 6))
    def _cancel_active():
        cancelled = cancel_quick_transcription()
        cancelled = cancel_daily_transcription() or cancelled
        if not cancelled:
            notify("No active transcription to cancel")
    ttk.Button(act_row, text="Cancel Active", command=_cancel_active).pack(
        side=tk.LEFT, padx=(0, 6)
    )
    startup_var = tk.BooleanVar(value=is_startup_enabled())
    def _toggle_startup():
        set_startup(startup_var.get())
    ttk.Checkbutton(
        act_row, text="Run on startup", variable=startup_var, command=_toggle_startup
    ).pack(side=tk.LEFT, padx=(4, 0))

    # ── Last transcript ───────────────────────────────────────────────────────
    lt_lf = ttk.LabelFrame(frame, text="Last Quick Transcript", padding=10)
    lt_lf.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
    lt_txt = tk.Text(
        lt_lf, wrap=tk.WORD, font=("Segoe UI", 9), relief=tk.FLAT,
        state=tk.DISABLED, fg="#1f2937", padx=4, pady=4,
    )
    lt_sb = ttk.Scrollbar(lt_lf, command=lt_txt.yview)
    lt_txt.configure(yscrollcommand=lt_sb.set)
    lt_sb.pack(side=tk.RIGHT, fill=tk.Y)
    lt_txt.pack(fill=tk.BOTH, expand=True)
    lt_copy_row = ttk.Frame(lt_lf)
    lt_copy_row.pack(fill=tk.X, pady=(6, 0))
    lt_copy_status = ttk.Label(lt_copy_row, text="", foreground="#6b7280")
    lt_copy_status.pack(side=tk.LEFT)
    def _copy_lt():
        if last_transcript:
            pyperclip.copy(last_transcript)
            lt_copy_status.config(text="Copied!")
            root.after(2000, lambda: lt_copy_status.config(text=""))
    ttk.Button(lt_copy_row, text="Copy to Clipboard", command=_copy_lt).pack(side=tk.RIGHT)

    # ── Live status poll ──────────────────────────────────────────────────────
    def _refresh():
        try:
            if not root.winfo_exists():
                return
        except Exception:
            return
        activity, activity_progress = get_activity()
        quick_active = _quick_jobs is not None and _quick_jobs.is_active
        if _paused:
            dot.config(fg="#6b7280")
            status_txt.config(text="Paused")
        elif quick_active or _daily_jobs.is_active:
            dot.config(fg="#eab308")
            status_txt.config(text=activity)
        else:
            dot.config(fg="#22c55e")
            status_txt.config(text="Recording daily" if config.get("daily_recording") else "Ready")
        if activity_progress is None:
            progress.grid_remove()
        else:
            progress["value"] = max(0, min(100, activity_progress * 100))
            progress.grid()
        mic_meter["value"] = _mic_level * 100
        system_meter["value"] = _system_level * 100
        rec_path = get_recording_path()
        mic_f = rec_path / date.today().isoformat() / "mic.raw"
        info_txt.config(
            text=f"Today's recording: {mic_f.stat().st_size / 1024 / 1024:.1f} MB"
            if mic_f.exists() else ""
        )
        lt_txt.configure(state=tk.NORMAL)
        lt_txt.delete("1.0", tk.END)
        preview = last_transcript
        if len(preview) > 4000:
            preview = preview[:4000] + "\n\n… Open History to view the complete transcript."
        lt_txt.insert(
            tk.END,
            preview or "(No transcription yet — use a hotkey or click a button above)",
        )
        lt_txt.configure(state=tk.DISABLED)
        root.after(1000, _refresh)

    _refresh()


def _build_history_tab(frame, root):
    """History: list of past transcriptions with full-text viewer below."""
    import tkinter as tk
    from tkinter import ttk

    # Vertical split: top = list, bottom = full text viewer
    paned = tk.PanedWindow(frame, orient=tk.VERTICAL, sashwidth=5,
                           sashrelief=tk.FLAT, bg="#e5e7eb")
    paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    # ── Top: history list ─────────────────────────────────────────────────────
    top_f = ttk.Frame(paned)
    paned.add(top_f, minsize=120)

    tree_f = ttk.Frame(top_f)
    tree_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))

    columns = ("time", "duration", "words", "preview")
    tree = ttk.Treeview(tree_f, columns=columns, show="headings", selectmode="browse")
    tree.heading("time",     text="Time")
    tree.heading("duration", text="Duration")
    tree.heading("words",    text="Words")
    tree.heading("preview",  text="Preview")
    tree.column("time",     width=140, minwidth=120)
    tree.column("duration", width=70,  minwidth=60)
    tree.column("words",    width=55,  minwidth=50)
    tree.column("preview",  width=380, minwidth=150)
    sb = ttk.Scrollbar(tree_f, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    top_bot = ttk.Frame(top_f)
    top_bot.pack(fill=tk.X, padx=10, pady=(0, 4))
    list_status = ttk.Label(top_bot, text="Select a row to view", foreground="#6b7280")
    list_status.pack(side=tk.LEFT)
    texts: dict[str, str] = {}

    def _load_entries():
        for iid in tree.get_children():
            tree.delete(iid)
        texts.clear()
        for entry in reversed(load_history()):
            ts = entry.get("timestamp", "")
            try:
                time_str = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_str = ts[:16]
            mins    = entry.get("minutes", "?")
            words   = entry.get("word_count", "?")
            text    = entry.get("text", "")
            preview = text.replace("\n", " ")[:120]
            iid = tree.insert("", "end", values=(time_str, f"{mins} min", words, preview))
            texts[iid] = text
        list_status.config(text=f"{len(texts)} entries — select a row to view")

    _load_entries()
    ttk.Button(top_bot, text="Refresh", command=_load_entries).pack(side=tk.RIGHT)

    # ── Bottom: full transcript viewer ────────────────────────────────────────
    bot_f = ttk.Frame(paned)
    paned.add(bot_f, minsize=120)

    view_header = ttk.Frame(bot_f)
    view_header.pack(fill=tk.X, padx=10, pady=(6, 2))
    view_title = ttk.Label(view_header, text="Transcript", font=("Segoe UI", 9, "bold"),
                           foreground="#374151")
    view_title.pack(side=tk.LEFT)
    copy_status = ttk.Label(view_header, text="", foreground="#6b7280")
    copy_status.pack(side=tk.LEFT, padx=(10, 0))

    txt_f = ttk.Frame(bot_f)
    txt_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

    view_txt = tk.Text(
        txt_f, wrap=tk.WORD, font=("Segoe UI", 9), relief=tk.FLAT,
        state=tk.DISABLED, fg="#1f2937", padx=6, pady=6, spacing3=2,
        bg="#fafafa",
    )
    view_sb = ttk.Scrollbar(txt_f, command=view_txt.yview)
    view_txt.configure(yscrollcommand=view_sb.set)
    view_sb.pack(side=tk.RIGHT, fill=tk.Y)
    view_txt.pack(fill=tk.BOTH, expand=True)

    # Tags for timestamp formatting
    view_txt.tag_configure("ts",   foreground="#9ca3af", font=("Segoe UI", 8))
    view_txt.tag_configure("body", font=("Segoe UI", 9))

    def _copy_current():
        sel = tree.selection()
        if sel:
            full_text = texts.get(sel[0], "")
            if full_text:
                pyperclip.copy(full_text)
                copy_status.config(text="Copied!")
                root.after(2000, lambda: copy_status.config(text=""))

    ttk.Button(view_header, text="Copy", command=_copy_current).pack(side=tk.RIGHT)

    def on_select(_event):
        sel = tree.selection()
        if not sel:
            return
        full_text = texts.get(sel[0], "")
        view_txt.configure(state=tk.NORMAL)
        view_txt.delete("1.0", tk.END)
        if full_text:
            for line in full_text.split("\n"):
                if line.startswith("[") and "]" in line:
                    # timestamp line: colour the [mm:ss] part
                    bracket_end = line.index("]") + 1
                    view_txt.insert(tk.END, line[:bracket_end] + " ", "ts")
                    view_txt.insert(tk.END, line[bracket_end:].strip() + "\n", "body")
                else:
                    view_txt.insert(tk.END, line + "\n", "body")
        view_txt.configure(state=tk.DISABLED)
        view_txt.yview_moveto(0)
        # update title
        vals = tree.item(sel[0], "values")
        view_title.config(text=f"Transcript — {vals[0]}  ({vals[1]}, {vals[2]} words)")

    tree.bind("<<TreeviewSelect>>", on_select)


def _build_transcripts_tab(frame, root):
    """Daily Transcripts: split-pane viewer with speaker colours, search, copy."""
    import tkinter as tk
    from tkinter import ttk

    paned = tk.PanedWindow(
        frame, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.FLAT, bg="#e5e7eb"
    )
    paned.pack(fill=tk.BOTH, expand=True)

    # Left: date list
    left = ttk.Frame(paned, width=140)
    paned.add(left, minsize=110)
    ttk.Label(left, text="Dates", font=("Segoe UI", 9, "bold")).pack(
        anchor="w", padx=8, pady=(8, 2)
    )
    date_lb = tk.Listbox(
        left, activestyle="dotbox", selectmode=tk.SINGLE, font=("Segoe UI", 9),
        relief=tk.FLAT, selectbackground="#3b82f6", selectforeground="white",
        exportselection=False,
    )
    date_scroll = ttk.Scrollbar(left, command=date_lb.yview)
    date_lb.configure(yscrollcommand=date_scroll.set)
    date_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    date_lb.pack(fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))

    # Right: search + transcript + bottom bar
    right = ttk.Frame(paned)
    paned.add(right, minsize=400)

    search_frame = ttk.Frame(right)
    search_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
    ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
    search_var = tk.StringVar()
    ttk.Entry(search_frame, textvariable=search_var, width=28).pack(side=tk.LEFT, padx=(4, 8))
    filter_var = tk.StringVar(value="all")
    for val, label in [("all", "All"), ("you", "You"), ("discord", "Discord")]:
        ttk.Radiobutton(
            search_frame, text=label, variable=filter_var, value=val
        ).pack(side=tk.LEFT, padx=(0 if val == "all" else 4, 0))

    ttk.Separator(search_frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)
    plain_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(search_frame, text="Plain text", variable=plain_var,
                    command=lambda: _on_filter_change()).pack(side=tk.LEFT)

    txt_frame = ttk.Frame(right)
    txt_frame.pack(fill=tk.BOTH, expand=True, padx=8)
    txt = tk.Text(
        txt_frame, wrap=tk.WORD, font=("Segoe UI", 9), relief=tk.FLAT,
        state=tk.DISABLED, cursor="arrow", padx=6, pady=4, spacing1=1, spacing3=2,
    )
    txt_sb = ttk.Scrollbar(txt_frame, command=txt.yview)
    txt.configure(yscrollcommand=txt_sb.set)
    txt_sb.pack(side=tk.RIGHT, fill=tk.Y)
    txt.pack(fill=tk.BOTH, expand=True)
    txt.tag_configure("you",      foreground="#2563eb", font=("Segoe UI", 9, "bold"))
    txt.tag_configure("discord",  foreground="#059669", font=("Segoe UI", 9, "bold"))
    txt.tag_configure("ts",       foreground="#9ca3af", font=("Segoe UI", 8))
    txt.tag_configure("body",     font=("Segoe UI", 9))
    txt.tag_configure("search",   background="#fef08a")
    txt.tag_configure("selected", background="#dbeafe")

    bot = ttk.Frame(right)
    bot.pack(fill=tk.X, padx=8, pady=(4, 8))
    status_var = tk.StringVar(value="Select a date")
    ttk.Label(bot, textvariable=status_var, foreground="gray").pack(side=tk.LEFT)

    _current_segments: list[dict] = []

    def _copy_filtered(speaker: str):
        segs = _current_segments if speaker == "all" else [
            s for s in _current_segments if s.get("speaker") == speaker
        ]
        if not segs:
            return
        text = "\n".join(f"[{s['time']}] {s['text']}" for s in segs)
        root.clipboard_clear()
        root.clipboard_append(text)
        status_var.set(f"Copied {len(segs)} segments")
        root.after(2000, lambda: status_var.set(f"{len(_current_segments)} segments"))

    ttk.Button(bot, text="Copy All",     command=lambda: _copy_filtered("all")    ).pack(side=tk.RIGHT, padx=(4, 0))
    ttk.Button(bot, text="Copy Discord", command=lambda: _copy_filtered("discord")).pack(side=tk.RIGHT, padx=(4, 0))
    ttk.Button(bot, text="Copy You",     command=lambda: _copy_filtered("you")    ).pack(side=tk.RIGHT, padx=(4, 0))

    def _render(segs: list[dict]):
        txt.configure(state=tk.NORMAL)
        txt.delete("1.0", tk.END)
        q    = search_var.get().strip().lower()
        filt = filter_var.get()
        plain = plain_var.get()
        shown = 0
        for seg in segs:
            spk  = seg.get("speaker", "unknown")
            if filt != "all" and spk != filt:
                continue
            t    = seg.get("time", "")
            body = seg.get("text", "")
            if q and q not in body.lower() and q not in spk:
                continue
            if plain:
                # plain text: just the text, no speaker labels or highlights
                txt.insert(tk.END, body + "\n", "body")
            else:
                tag   = "you" if spk == "you" else "discord"
                label = "YOU" if spk == "you" else "DISC"
                txt.insert(tk.END, f"[{t}] ", "ts")
                txt.insert(tk.END, f"[{label}] ", tag)
                if q:
                    low = body.lower()
                    idx = 0
                    while True:
                        pos = low.find(q, idx)
                        if pos == -1:
                            txt.insert(tk.END, body[idx:], "body")
                            break
                        txt.insert(tk.END, body[idx:pos], "body")
                        txt.insert(tk.END, body[pos:pos + len(q)], "search")
                        idx = pos + len(q)
                else:
                    txt.insert(tk.END, body, "body")
                txt.insert(tk.END, "\n")
            shown += 1
        txt.configure(state=tk.DISABLED)
        status_var.set(f"{shown} of {len(segs)} segments")

    def _click_copy(event):
        idx        = txt.index(f"@{event.x},{event.y}")
        line_start = txt.index(f"{idx} linestart")
        line_end   = txt.index(f"{idx} lineend")
        line_text  = txt.get(line_start, line_end).strip()
        if line_text:
            root.clipboard_clear()
            root.clipboard_append(line_text)
            txt.tag_remove("selected", "1.0", tk.END)
            txt.tag_add("selected", line_start, line_end + "+1c")
            status_var.set("Line copied")
            root.after(1500, lambda: status_var.set(f"{len(_current_segments)} segments"))

    txt.bind("<Button-1>", _click_copy)

    def _load_date(_event=None):
        nonlocal _current_segments
        sel = date_lb.curselection()
        if not sel:
            return
        chosen   = date_lb.get(sel[0])
        rec_path = get_recording_path()
        tf       = rec_path / chosen / "transcript.json"
        if not tf.exists():
            txt.configure(state=tk.NORMAL)
            txt.delete("1.0", tk.END)
            txt.insert(
                tk.END,
                "No transcript yet for this date.\n"
                "Use 'Transcribe Today' in the Dashboard tab.",
            )
            txt.configure(state=tk.DISABLED)
            status_var.set("No transcript")
            _current_segments = []
            return
        with open(tf, encoding="utf-8") as f:
            data = json.load(f)
        _current_segments = data.get("segments", [])
        mins  = data.get("speech_minutes", 0)
        words = data.get("word_count", 0)
        root.title(f"Pascribe — {chosen}  ({words} words, {mins:.0f} min speech)")
        _render(_current_segments)

    def _on_filter_change(*_):
        if _current_segments:
            _render(_current_segments)

    search_var.trace_add("write", _on_filter_change)
    filter_var.trace_add("write", _on_filter_change)
    date_lb.bind("<<ListboxSelect>>", _load_date)

    rec_path = get_recording_path()
    if rec_path.exists():
        dates = sorted([d.name for d in rec_path.iterdir() if d.is_dir()], reverse=True)
        for d in dates:
            date_lb.insert(tk.END, d)
            if not (rec_path / d / "transcript.json").exists():
                date_lb.itemconfig(tk.END, foreground="#9ca3af")
        if dates:
            date_lb.selection_set(0)
            date_lb.event_generate("<<ListboxSelect>>")
    else:
        status_var.set("No recordings folder found")


def _build_settings_tab(frame, root):
    """Settings: scrollable form covering all config options."""
    import tkinter as tk
    from tkinter import ttk

    # Scrollable canvas wrapper
    canvas = tk.Canvas(frame, highlightthickness=0)
    vsb    = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    inner = ttk.Frame(canvas)
    cwin  = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(cwin, width=e.width))
    canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

    PAD = dict(padx=14, pady=(0, 8))

    def _section(title: str):
        ttk.Separator(inner, orient="horizontal").pack(fill=tk.X, padx=14, pady=(8, 6))
        ttk.Label(inner, text=title, font=("Segoe UI", 9, "bold")).pack(
            anchor="w", padx=14, pady=(0, 6)
        )

    def _field_row(label_text: str) -> ttk.Frame:
        r = ttk.Frame(inner)
        r.pack(fill=tk.X, **PAD)
        ttk.Label(r, text=label_text, width=24, anchor="w").pack(side=tk.LEFT)
        return r

    # ── Audio Devices ─────────────────────────────────────────────────────────
    ttk.Label(inner, text="Audio Devices", font=("Segoe UI", 9, "bold")).pack(
        anchor="w", padx=14, pady=(14, 6)
    )
    devices       = list_input_devices()
    device_labels = ["(None)"]
    device_ids    = [None]
    device_names  = [None]
    for dev_id, name, _ in devices:
        device_labels.append(f"[{categorize_device(name)}] {name}")
        device_ids.append(dev_id)
        device_names.append(name)

    def _find_label(dev_id, dev_name=None):
        # Prefer name-based match (reliable across reboots)
        if dev_name:
            for i, name in enumerate(device_names):
                if name == dev_name:
                    return device_labels[i]
        for i, did in enumerate(device_ids):
            if did == dev_id:
                return device_labels[i]
        return "(None)"

    mic_var = tk.StringVar(value=_find_label(config["mic_device"], config.get("mic_device_name")))
    r = _field_row("Microphone:")
    ttk.Combobox(r, textvariable=mic_var, values=device_labels,
                 state="readonly", width=55).pack(side=tk.LEFT, fill=tk.X, expand=True)

    sys_var = tk.StringVar(value=_find_label(config["system_device"], config.get("system_device_name")))
    r = _field_row("System Audio:")
    ttk.Combobox(r, textvariable=sys_var, values=device_labels,
                 state="readonly", width=55).pack(side=tk.LEFT, fill=tk.X, expand=True)

    device_test_var = tk.StringVar(value="")
    device_test_row = ttk.Frame(inner)
    device_test_row.pack(fill=tk.X, padx=14, pady=(0, 4))
    ttk.Label(device_test_row, textvariable=device_test_var, foreground="#6b7280").pack(
        side=tk.LEFT
    )

    def _test_selected_devices():
        try:
            selected = []
            for label, value in (("Mic", mic_var.get()), ("System", sys_var.get())):
                index = device_labels.index(value)
                device_id = device_ids[index]
                if device_id is not None:
                    selected.append((label, device_id))
        except ValueError:
            device_test_var.set("Select valid devices first.")
            return
        if not selected:
            device_test_var.set("No devices selected.")
            return
        device_test_var.set("Testing selected devices…")
        test_button.state(["disabled"])

        def worker():
            messages = []
            for label, device_id in selected:
                try:
                    level, sample_rate = measure_input_device(device_id)
                    messages.append(f"{label}: {level * 100:.0f}% level at {sample_rate} Hz")
                except Exception as error:
                    messages.append(f"{label}: failed — {error}")
            root.after(0, lambda: device_test_var.set("  |  ".join(messages)))
            root.after(0, lambda: test_button.state(["!disabled"]))

        threading.Thread(target=worker, name="pascribe-device-test", daemon=True).start()

    test_button = ttk.Button(
        device_test_row,
        text="Test selected devices",
        command=_test_selected_devices,
    )
    test_button.pack(side=tk.RIGHT)

    def _hint(text: str):
        ttk.Label(inner, text=text, foreground="#9ca3af", font=("Segoe UI", 8)).pack(
            anchor="w", padx=14, pady=(0, 6)
        )

    _hint("Test devices now; changing active capture devices still requires a restart.")

    _section("Transcription")

    model_var = tk.StringVar(value=config.get("whisper_model", "large-v3"))
    r = _field_row("Whisper Model:")
    ttk.Combobox(r, textvariable=model_var,
                 values=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                 state="readonly", width=18).pack(side=tk.LEFT)
    _hint("larger = more accurate, more VRAM. large-v3 needs ~3.5 GB VRAM")

    wdev_var = tk.StringVar(value=config.get("whisper_device", "cuda"))
    r = _field_row("Whisper Device:")
    ttk.Radiobutton(r, text="CUDA (GPU)", variable=wdev_var, value="cuda").pack(side=tk.LEFT)
    ttk.Radiobutton(r, text="CPU",        variable=wdev_var, value="cpu" ).pack(side=tk.LEFT, padx=(12, 0))
    _hint("CUDA requires an NVIDIA GPU. CPU works everywhere but is much slower.")

    buf_var = tk.IntVar(value=config.get("buffer_minutes", 60))
    r = _field_row("Buffer (minutes):")
    ttk.Spinbox(r, from_=5, to=240, increment=5, textvariable=buf_var, width=8).pack(side=tk.LEFT)
    _hint("How much audio is kept in RAM for quick hotkey transcription.")

    prefix_var = tk.StringVar(value=config.get("hotkey_prefix", "ctrl+alt"))
    r = _field_row("Hotkey Prefix:")
    ttk.Entry(r, textvariable=prefix_var, width=22).pack(side=tk.LEFT)
    ttk.Label(r, text="e.g. ctrl+alt, right shift", foreground="#9ca3af").pack(
        side=tk.LEFT, padx=(8, 0)
    )
    _hint("ctrl+alt is safe for gaming. Change requires a restart.")

    _section("Daily Recording")

    daily_var = tk.BooleanVar(value=config.get("daily_recording", False))
    ttk.Checkbutton(
        inner,
        text="Enable daily recording — saves audio to disk for end-of-day transcription",
        variable=daily_var,
    ).pack(anchor="w", padx=14, pady=(0, 4))
    _hint("Stores ~32 KB/s per stream (int16 PCM). Two streams = ~1.75 GB per 8-hour day.")

    path_var = tk.StringVar(value=str(get_recording_path()))
    r = _field_row("Recording Path:")
    ttk.Entry(r, textvariable=path_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
    def _browse_path():
        from tkinter.filedialog import askdirectory
        d = askdirectory(title="Select Recording Folder",
                         initialdir=path_var.get() if Path(path_var.get()).exists() else ".")
        if d:
            path_var.set(d)
    ttk.Button(r, text="Browse…", command=_browse_path, width=8).pack(side=tk.LEFT, padx=(4, 0))

    key_var = tk.StringVar(value=config.get("assemblyai_key", ""))
    r = _field_row("AssemblyAI Key:")
    key_entry = ttk.Entry(r, textvariable=key_var, show="*", width=38)
    key_entry.pack(side=tk.LEFT)
    show_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        r, text="Show", variable=show_var,
        command=lambda: key_entry.config(show="" if show_var.get() else "*"),
    ).pack(side=tk.LEFT, padx=(6, 0))
    _hint("Required for cloud transcription. Protected with Windows DPAPI when saved.")

    url_var = tk.StringVar(value=config.get("homelab_url") or "")
    r = _field_row("Homelab URL:")
    ttk.Entry(r, textvariable=url_var, width=45).pack(side=tk.LEFT, fill=tk.X, expand=True)
    _hint("Optional. Daily transcripts will be HTTP POSTed as JSON to this endpoint.")

    del_var = tk.BooleanVar(value=config.get("delete_after_transcribe", False))
    ttk.Checkbutton(
        inner, text="Delete raw audio after successful transcription", variable=del_var
    ).pack(anchor="w", padx=14, pady=(0, 4))
    _hint("Raw audio is deleted only when it is not actively being recorded and transcription succeeded.")

    ttk.Separator(inner, orient="horizontal").pack(fill=tk.X, padx=14, pady=(8, 6))
    status_var = tk.StringVar(value="")
    ttk.Label(inner, textvariable=status_var, foreground="#6b7280").pack(anchor="w", padx=14)
    btn_r = ttk.Frame(inner)
    btn_r.pack(anchor="e", padx=14, pady=(4, 14))

    def _save():
        try:
            mic_i = device_labels.index(mic_var.get())
            sys_i = device_labels.index(sys_var.get())
            buffer_minutes = int(buf_var.get())
        except (ValueError, tk.TclError):
            status_var.set("Select valid devices and enter a numeric buffer duration.")
            return

        updates = {
            "mic_device":              device_ids[mic_i],
            "mic_device_name":         device_names[mic_i],
            "system_device":           device_ids[sys_i],
            "system_device_name":      device_names[sys_i],
            "whisper_model":           model_var.get(),
            "whisper_device":          wdev_var.get(),
            "buffer_minutes":          buffer_minutes,
            "hotkey_prefix":           prefix_var.get().strip(),
            "daily_recording":         daily_var.get(),
            "recording_path":          path_var.get().strip(),
            "assemblyai_key":          key_var.get().strip(),
            "homelab_url":             url_var.get().strip() or None,
            "delete_after_transcribe": del_var.get(),
        }
        errors = validate_settings(updates)
        if errors:
            status_var.set(errors[0])
            return

        global daily_recorder
        old_recording_path = get_recording_path()
        old_daily_recording = config.get("daily_recording", False)
        config.update(updates)
        save_config(config)
        mic_buffer.resize(buffer_minutes * 60 // CHUNK_SECONDS)
        loopback_buffer.resize(buffer_minutes * 60 // CHUNK_SECONDS)

        new_recording_path = get_recording_path()
        if old_daily_recording != daily_var.get() or old_recording_path != new_recording_path:
            if daily_recorder is not None:
                daily_recorder.close()
                daily_recorder = None
            if daily_var.get():
                daily_recorder = DailyAudioRecorder(str(new_recording_path))

        status_var.set(
            "Saved. Recording, storage, model, and buffer settings are active; "
            "restart for device/hotkey changes."
        )
        log.info("Settings saved via panel")

    ttk.Button(btn_r, text="Save Settings", command=_save).pack(side=tk.RIGHT)


def _build_diagnostics_tab(frame, root):
    """Diagnostics: copyable checks for installation, audio, storage, and CUDA."""
    import tkinter as tk
    from tkinter import ttk

    header = ttk.Frame(frame)
    header.pack(fill=tk.X, padx=14, pady=(14, 8))
    ttk.Label(
        header,
        text="Desktop Diagnostics",
        font=("Segoe UI", 11, "bold"),
    ).pack(side=tk.LEFT)
    status = ttk.Label(header, text="", foreground="#6b7280")
    status.pack(side=tk.LEFT, padx=(12, 0))

    text_frame = ttk.Frame(frame)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))
    output = tk.Text(
        text_frame,
        wrap=tk.WORD,
        font=("Consolas", 9),
        relief=tk.FLAT,
        padx=8,
        pady=8,
        state=tk.DISABLED,
    )
    scrollbar = ttk.Scrollbar(text_frame, command=output.yview)
    output.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    output.pack(fill=tk.BOTH, expand=True)
    output.tag_configure("PASS", foreground="#059669")
    output.tag_configure("WARN", foreground="#b45309")
    output.tag_configure("FAIL", foreground="#dc2626")
    current_text = {"value": ""}

    def refresh():
        status.config(text="Running checks…")
        try:
            results = collect_diagnostics(
                data_dir=APP_PATHS.data_dir,
                recording_dir=get_recording_path(),
                whisper_device=config.get("whisper_device", "cuda"),
                input_devices=list_input_devices(),
            )
            report = format_diagnostics(results)
            report += (
                f"\n\nData directory: {APP_PATHS.data_dir}"
                f"\nConfig file: {CONFIG_PATH}"
                f"\nLog file: {LOG_PATH}"
            )
            current_text["value"] = report
            output.configure(state=tk.NORMAL)
            output.delete("1.0", tk.END)
            for line in report.splitlines(keepends=True):
                tag = line[1:5].rstrip("]") if line.startswith("[") else ""
                output.insert(tk.END, line, tag if tag in {"PASS", "WARN", "FAIL"} else None)
            output.configure(state=tk.DISABLED)
            failures = sum(1 for item in results if item.status == "FAIL")
            status.config(
                text=(
                    "All essential checks passed"
                    if not failures
                    else f"{failures} check(s) need attention"
                )
            )
        except Exception as error:
            status.config(text=f"Diagnostics failed: {error}")

    def copy_report():
        if current_text["value"]:
            pyperclip.copy(current_text["value"])
            status.config(text="Copied diagnostics")

    def open_data_folder():
        APP_PATHS.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(APP_PATHS.data_dir)  # type: ignore[attr-defined]
        except Exception as error:
            status.config(text=f"Could not open data folder: {error}")

    buttons = ttk.Frame(frame)
    buttons.pack(fill=tk.X, padx=14, pady=(0, 14))
    ttk.Button(buttons, text="Refresh", command=refresh).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Open Data Folder", command=open_data_folder).pack(side=tk.LEFT, padx=(6, 0))
    ttk.Button(buttons, text="Copy Report", command=copy_report).pack(side=tk.RIGHT)
    refresh()


# ─── Main Control Panel ───────────────────────────────────────────────────────

def open_main_panel(start_tab: int = 0):
    """Open (or focus) the unified Pascribe control panel."""
    global _main_panel_open, _main_panel_window, _main_panel_notebook

    if _main_panel_open and _main_panel_window:
        # Panel already open — bring to front and switch tab
        try:
            def _activate():
                _main_panel_window.deiconify()
                _main_panel_window.lift()
                _main_panel_window.focus_force()
                if _main_panel_notebook:
                    _main_panel_notebook.select(start_tab)
            _main_panel_window.after(0, _activate)
        except Exception:
            pass
        return

    _main_panel_open = True

    def run_panel():
        global _main_panel_open, _main_panel_window, _main_panel_notebook
        try:
            import tkinter as tk
            from tkinter import ttk

            root = tk.Tk()
            _main_panel_window = root
            root.title("Pascribe")
            root.geometry("1020x680")
            root.minsize(720, 500)
            try:
                ttk.Style().theme_use("vista")
            except Exception:
                pass

            # Set window icon using the same waveform image
            try:
                from PIL import ImageTk as _ImageTk
                _icon = create_tray_image("green").resize((32, 32), Image.LANCZOS)
                _photo = _ImageTk.PhotoImage(_icon)
                root.iconphoto(True, _photo)
                root._icon_ref = _photo  # prevent GC
            except Exception:
                pass

            # Footer status bar
            footer = ttk.Frame(root)
            footer.pack(fill=tk.X, side=tk.BOTTOM)
            ttk.Separator(footer, orient="horizontal").pack(fill=tk.X)
            footer_inner = ttk.Frame(footer)
            footer_inner.pack(fill=tk.X, padx=10, pady=3)
            _ver_lbl = ttk.Label(footer_inner, text=f"Pascribe v{__version__}",
                                 foreground="#9ca3af", font=("Segoe UI", 8))
            _ver_lbl.pack(side=tk.LEFT)
            _footer_status = ttk.Label(footer_inner, text="", foreground="#6b7280",
                                       font=("Segoe UI", 8))
            _footer_status.pack(side=tk.RIGHT)

            def _update_footer():
                try:
                    if not root.winfo_exists():
                        return
                except Exception:
                    return
                if _paused:
                    _footer_status.config(text="Hotkeys paused")
                elif (_quick_jobs is not None and _quick_jobs.is_active) or _daily_jobs.is_active:
                    activity, _progress = get_activity()
                    _footer_status.config(text=activity)
                elif config.get("daily_recording"):
                    rec_path = get_recording_path()
                    mic_f = rec_path / date.today().isoformat() / "mic.raw"
                    mb = mic_f.stat().st_size / 1024 / 1024 if mic_f.exists() else 0
                    _footer_status.config(text=f"Recording — {mb:.0f} MB today")
                else:
                    _footer_status.config(text="Ready")
                root.after(2000, _update_footer)

            _update_footer()

            nb = ttk.Notebook(root)
            nb.pack(fill=tk.BOTH, expand=True)
            _main_panel_notebook = nb

            for title, builder in [
                ("  Dashboard  ",         _build_dashboard_tab),
                ("  History  ",           _build_history_tab),
                ("  Daily Transcripts  ", _build_transcripts_tab),
                ("  Settings  ",          _build_settings_tab),
                ("  Diagnostics  ",       _build_diagnostics_tab),
            ]:
                f = ttk.Frame(nb)
                nb.add(f, text=title)
                builder(f, root)

            nb.select(min(start_tab, 4))
            root.protocol("WM_DELETE_WINDOW", root.destroy)
            root.mainloop()
        finally:
            _main_panel_open = False
            _main_panel_window = None
            _main_panel_notebook = None

    threading.Thread(target=run_panel, daemon=True).start()


def open_settings():
    """Open the control panel at the Settings tab."""
    open_main_panel(start_tab=3)


def open_history():
    """Open the control panel at the History tab."""
    open_main_panel(start_tab=1)


def open_daily_transcripts():
    """Open the control panel at the Daily Transcripts tab."""
    open_main_panel(start_tab=2)


def open_diagnostics():
    """Open the control panel at the Diagnostics tab."""
    open_main_panel(start_tab=4)

# ─── Windows Startup Toggle ──────────────────────────────────────────────────

STARTUP_DIR = (
    Path(os.environ.get("APPDATA", ""))
    / "Microsoft"
    / "Windows"
    / "Start Menu"
    / "Programs"
    / "Startup"
)
STARTUP_LNK = STARTUP_DIR / "Pascribe.lnk"

def is_startup_enabled() -> bool:
    """Check if the startup shortcut exists."""
    return STARTUP_LNK.exists()

def set_startup(enabled: bool):
    """Create or remove the Windows startup shortcut."""
    if enabled:
        STARTUP_DIR.mkdir(parents=True, exist_ok=True)
        frozen = getattr(sys, "frozen", False)
        target = str(
            Path(sys.executable).resolve()
            if frozen
            else (APP_DIR / "venv" / "Scripts" / "pythonw.exe").resolve()
        )
        script = "" if frozen else str(Path(__file__).resolve())
        working_dir = str(APP_DIR)
        lnk_path = str(STARTUP_LNK)

        def ps_quote(value: str) -> str:
            return value.replace("'", "''")

        # Create .lnk shortcut via PowerShell WScript.Shell COM object
        arguments = "" if frozen else f'"{ps_quote(script)}"'
        ps_cmd = (
            f"$ws = New-Object -ComObject WScript.Shell; "
            f"$s = $ws.CreateShortcut('{ps_quote(lnk_path)}'); "
            f"$s.TargetPath = '{ps_quote(target)}'; "
            f"$s.Arguments = '{arguments}'; "
            f"$s.WorkingDirectory = '{ps_quote(working_dir)}'; "
            f"$s.Save()"
        )

        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                creationflags=subprocess.CREATE_NO_WINDOW,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "PowerShell returned an error")
            log.info("Startup shortcut created")
        except Exception as e:
            log.error(f"Failed to create startup shortcut: {e}")
    else:
        try:
            STARTUP_LNK.unlink()
            log.info("Startup shortcut removed")
        except FileNotFoundError:
            pass

def on_startup_toggle(icon, item):
    set_startup(not is_startup_enabled())

def on_pause_toggle(icon, item):
    global _paused
    _paused = not _paused
    if _paused:
        update_tray_icon("gray")
        notify("Paused — hotkeys disabled")
        log.info("Hotkeys paused by user")
    else:
        refresh_tray_icon()
        notify("Resumed — hotkeys active")
        log.info("Hotkeys resumed by user")

# ─── Tray Setup ──────────────────────────────────────────────────────────────

def on_quit(icon, item):
    cancel_quick_transcription()
    cancel_daily_transcription()
    try:
        keyboard.unhook_all_hotkeys()
    except Exception:
        pass
    if daily_recorder:
        daily_recorder.close()
    icon.stop()
    os._exit(0)

def _format_prefix(prefix: str) -> str:
    """Abbreviate hotkey prefix for display."""
    mapping = {
        "right shift": "RShift",
        "left shift": "LShift",
        "right ctrl": "RCtrl",
        "left ctrl": "LCtrl",
        "right alt": "RAlt",
        "left alt": "LAlt",
        "ctrl+alt": "Ctrl+Alt",
    }
    return mapping.get(prefix.lower(), prefix.title())

def setup_tray():
    global tray_icon
    hotkeys = config["hotkeys"]
    prefix = config["hotkey_prefix"]
    prefix_short = _format_prefix(prefix)
    hotkey_info = ", ".join(
        f"{k}={v}min"
        for k, v in sorted(hotkeys.items(), key=lambda x: int(x[0]))
    )

    daily_on = config.get("daily_recording", False)
    daily_label = "● Recording daily" if daily_on else "○ Daily recording off"

    menu = Menu(
        MenuItem("Open Panel…", lambda icon, item: open_main_panel(), default=True),
        Menu.SEPARATOR,
        MenuItem(f"Pascribe v{__version__}", lambda: None, enabled=False),
        MenuItem(daily_label, lambda: None, enabled=False),
        Menu.SEPARATOR,
        MenuItem("Settings…",              lambda icon, item: open_settings()),
        MenuItem("Transcription History…", lambda icon, item: open_history()),
        MenuItem("Daily Transcripts…",     lambda icon, item: open_daily_transcripts()),
        MenuItem("Diagnostics…",           lambda icon, item: open_diagnostics()),
        Menu.SEPARATOR,
        MenuItem("Transcribe today…",      lambda icon, item: run_daily_transcription()),
        MenuItem("Pause hotkeys",          on_pause_toggle, checked=lambda item: _paused),
        MenuItem("Run on startup",         on_startup_toggle, checked=lambda item: is_startup_enabled()),
        Menu.SEPARATOR,
        MenuItem("Quit", on_quit),
    )

    tray_icon = Icon(
        "Pascribe",
        create_tray_image("green"),
        title="Pascribe — Ready",
        menu=menu,
    )
    return tray_icon

# ─── Main ─────────────────────────────────────────────────────────────────────

def _acquire_instance_lock() -> bool:
    """Bind a local socket to prevent duplicate instances. Returns True if acquired."""
    global _instance_lock
    import socket as _sock
    s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 0)
    try:
        s.bind(("127.0.0.1", 47823))
        _instance_lock = s  # keep alive
        return True
    except OSError:
        return False


def main():
    if not _acquire_instance_lock():
        log.warning("Pascribe is already running — exiting duplicate")
        sys.exit(0)

    log.info(f"Pascribe v{__version__} starting")

    # Start audio capture — app continues even if no streams start
    streams = start_audio_streams()

    # Register hotkeys
    register_hotkeys()

    # Build tray
    icon = setup_tray()

    no_devices = config["mic_device"] is None and config["system_device"] is None

    def on_ready():
        time.sleep(2)  # Wait for tray icon to initialize
        if not streams:
            if no_devices:
                notify("No audio devices configured. Right-click -> Settings.")
            else:
                notify("Audio device error. Right-click -> Settings to reconfigure.")
        else:
            log.info("Pascribe running — recording audio")

    threading.Thread(target=on_ready, daemon=True).start()
    icon.run()

if __name__ == "__main__":
    main()
