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

# ─── CUDA DLL fix (Windows) ──────────────────────────────────────────────────
# pip-installed nvidia-cublas-cu12 / nvidia-cudnn-cu12 put DLLs in site-packages
# but Windows doesn't know to look there. Add them to PATH before CTranslate2 loads.
if sys.platform == "win32":
    _site_pkgs = Path(np.__file__).resolve().parent.parent
    _nvidia_bins = list((_site_pkgs / "nvidia").glob("*/bin"))
    if _nvidia_bins:
        os.environ["PATH"] = os.pathsep.join(str(p) for p in _nvidia_bins) + os.pathsep + os.environ["PATH"]

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_PATH = Path(__file__).parent / "pascribe.log"

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

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.json"
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
    "recording_path": "recordings",
    "assemblyai_key": "",
    "homelab_url": None,
    "delete_after_transcribe": False,
}

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

config = load_config()

# ─── History Storage ──────────────────────────────────────────────────────────

HISTORY_PATH = Path(__file__).parent / "history.json"

def load_history() -> list:
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_history(history: list):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def add_history_entry(minutes: int, word_count: int, elapsed: float, text: str):
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

# ─── Globals ──────────────────────────────────────────────────────────────────

max_chunks = config["buffer_minutes"] * 60 // CHUNK_SECONDS
mic_buffer = AudioBuffer(max_chunks=max_chunks)
loopback_buffer = AudioBuffer(max_chunks=max_chunks)
transcriber = None
_whisper_lock = threading.Lock()
tray_icon = None
last_transcript = ""
_current_cancel = threading.Event()
_cancel_lock = threading.Lock()
_main_panel_open = False
_main_panel_window = None   # tk.Tk reference for the open panel
_main_panel_notebook = None  # ttk.Notebook reference for tab switching
_paused = False
_transcribing = False
_transcribing_lock = threading.Lock()
daily_recorder = None
_instance_lock = None  # socket held to prevent multiple instances

# ─── Audio Capture ────────────────────────────────────────────────────────────

def mic_callback(indata, frames, time_info, status):
    if status:
        log.warning(f"Mic status: {status}")
    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    mic_buffer.add_chunk(mono)
    if daily_recorder:
        daily_recorder.write_mic(mono, mic_buffer.capture_rate)

def loopback_callback(indata, frames, time_info, status):
    if status:
        log.warning(f"Loopback status: {status}")
    mono = indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
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
        daily_recorder = DailyAudioRecorder(config.get("recording_path", "recordings"))
        log.info(f"Daily recording -> {config.get('recording_path', 'recordings')}/")

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
            return True, ""  # CPU mode, no VRAM concern
        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        needed_mb = WHISPER_VRAM_MB.get(model_name, 3200) + VRAM_SAFETY_MARGIN_MB
        if free_mb < needed_mb:
            return False, (
                f"Not enough VRAM: {free_mb:.0f} MB free, "
                f"~{needed_mb:.0f} MB needed for {model_name}"
            )
        return True, ""
    except ImportError:
        return True, ""  # No torch = likely CPU mode
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
    audio: np.ndarray, cancel_event: threading.Event | None = None
) -> list[tuple[float, float, str]]:
    """Transcribe numpy audio array. Returns list of (start, end, text) tuples."""
    init_whisper()

    segments, info = transcriber.transcribe(
        audio,
        beam_size=5,
        language=None,  # Auto-detect
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

    # Filter hallucinations: if the entire transcription is just a known
    # hallucination phrase, discard it
    if result:
        all_text = " ".join(t for _, _, t in result).strip().lower()
        all_text = all_text.strip(".,!?")
        if all_text in WHISPER_HALLUCINATIONS:
            log.info(f"Filtered hallucination: '{all_text}'")
            return []

    return result


def format_ssmd(segments: list[tuple[float, float, str]], pause_threshold: float = 2.0) -> str:
    """Format transcription segments as timestamped text with paragraph breaks on pauses."""
    if not segments:
        return ""

    lines = []
    prev_end = None

    for start, end, text in segments:
        # Insert paragraph break on significant pause
        if prev_end is not None and (start - prev_end) > pause_threshold:
            lines.append("")

        mins = int(start) // 60
        secs = int(start) % 60
        lines.append(f"[{mins:02d}:{secs:02d}] {text}")
        prev_end = end

    return "\n".join(lines)

# ─── Resampling ──────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio from src_rate to dst_rate using linear interpolation."""
    if src_rate == dst_rate:
        return audio
    ratio = dst_rate / src_rate
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

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
            data = _resample(data, capture_rate, self.STORE_RATE)
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
            with open(meta_path, "w") as f:
                json.dump(meta, f)
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


def load_todays_recordings() -> tuple[np.ndarray | None, np.ndarray | None, datetime | None, int]:
    """Load today's mic and system audio from disk.
    Returns (mic_audio, sys_audio, start_time, sample_rate).
    """
    rec_path = Path(config.get("recording_path", "recordings"))
    today = date.today().isoformat()
    day_dir = rec_path / today

    if not day_dir.exists():
        return None, None, None, 16000

    meta_path = day_dir / "meta.json"
    if not meta_path.exists():
        return None, None, None, 16000

    with open(meta_path) as f:
        meta = json.load(f)

    sample_rate = meta.get("sample_rate", 16000)
    started = datetime.fromisoformat(meta["started"])

    raw_dtype = np.dtype(meta.get("dtype", "float32"))

    def load_raw(name: str) -> np.ndarray | None:
        p = day_dir / name
        if not p.exists() or p.stat().st_size == 0:
            return None
        audio = np.frombuffer(p.read_bytes(), dtype=raw_dtype).copy()
        if raw_dtype == np.dtype("int16"):
            audio = audio.astype(np.float32) / 32767.0
        return audio

    return load_raw("mic.raw"), load_raw("sys.raw"), started, sample_rate


def _energy_vad(audio: np.ndarray, sample_rate: int,
                frame_ms: int = 30,
                min_speech_s: float = 0.4,
                pad_s: float = 0.3) -> list[tuple[float, float]]:
    """Energy-based VAD. Returns list of (start_s, end_s) speech segments.

    Uses an adaptive threshold (fraction of top-energy frames) so it works
    across quiet and loud environments without manual tuning.
    """
    frame_samples = int(sample_rate * frame_ms / 1000)
    if len(audio) < frame_samples:
        return []

    n_frames = len(audio) // frame_samples
    frames_rms = np.array([
        np.sqrt(np.mean(audio[i * frame_samples:(i + 1) * frame_samples] ** 2))
        for i in range(n_frames)
    ])

    if frames_rms.max() == 0:
        return []

    # Adaptive: threshold is 15% of the median of the loudest 20% of frames,
    # floored at a tiny noise floor so silence-only recordings don't hallucinate speech.
    top_20 = np.sort(frames_rms)[int(n_frames * 0.8):]
    threshold = max(np.median(top_20) * 0.15, 1e-4)

    is_speech = frames_rms > threshold

    # Fill short gaps (< 500 ms) so words don't get split
    gap_fill = int(500 / frame_ms)
    for i in range(len(is_speech) - gap_fill):
        if is_speech[i] and is_speech[i + gap_fill]:
            is_speech[i:i + gap_fill] = True

    # Extract contiguous regions, add padding
    pad_frames = int(pad_s * 1000 / frame_ms)
    segments: list[list[float]] = []
    in_speech = False
    seg_start = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            seg_start = max(0, i - pad_frames)
            in_speech = True
        elif not speech and in_speech:
            seg_end = min(n_frames - 1, i + pad_frames)
            dur = (seg_end - seg_start) * frame_ms / 1000
            if dur >= min_speech_s:
                segments.append([seg_start * frame_ms / 1000, seg_end * frame_ms / 1000])
            in_speech = False

    if in_speech:
        seg_end = n_frames - 1
        dur = (seg_end - seg_start) * frame_ms / 1000
        if dur >= min_speech_s:
            segments.append([seg_start * frame_ms / 1000, seg_end * frame_ms / 1000])

    # Merge overlapping segments
    merged: list[list[float]] = []
    for seg in segments:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], seg[1])
        else:
            merged.append(seg)

    max_s = len(audio) / sample_rate
    return [(float(s[0]), float(min(s[1], max_s))) for s in merged]


def _build_segment_map(vad_segs: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    """Build a map from stripped-audio time → original-audio time.
    Returns list of (stripped_start, stripped_end, orig_start).
    """
    seg_map = []
    t = 0.0
    for orig_start, orig_end in vad_segs:
        dur = orig_end - orig_start
        seg_map.append((t, t + dur, orig_start))
        t += dur
    return seg_map


def _remap_to_original(t: float, seg_map: list[tuple[float, float, float]]) -> float:
    """Convert a stripped-audio timestamp back to original-audio timestamp."""
    for strip_start, strip_end, orig_start in seg_map:
        if strip_start <= t <= strip_end:
            return orig_start + (t - strip_start)
    return t


def _assign_speaker(start_s: float, end_s: float,
                    mic_audio: np.ndarray, sys_audio: np.ndarray,
                    sample_rate: int) -> str:
    """Determine speaker by comparing RMS energy in mic vs system streams."""
    s = int(start_s * sample_rate)
    e = int(end_s * sample_rate)
    mic_rms = float(np.sqrt(np.mean(mic_audio[s:min(e, len(mic_audio))] ** 2))) if s < len(mic_audio) else 0.0
    sys_rms = float(np.sqrt(np.mean(sys_audio[s:min(e, len(sys_audio))] ** 2))) if s < len(sys_audio) else 0.0
    return "you" if mic_rms >= sys_rms else "discord"

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


def transcribe_with_assemblyai(audio: np.ndarray, sample_rate: int) -> list[dict]:
    """Upload VAD-stripped audio to AssemblyAI, poll until done, return segments.
    Each segment: {start_s, end_s, text}.
    """
    api_key = config.get("assemblyai_key", "")
    if not api_key:
        raise RuntimeError("assemblyai_key not configured in config.json")

    wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
    log.info(f"Uploading {len(wav_bytes) / 1024 / 1024:.1f} MB to AssemblyAI...")

    hdrs_json = {"authorization": api_key, "content-type": "application/json"}
    hdrs_bin  = {"authorization": api_key, "content-type": "application/octet-stream"}

    # 1. Upload
    req = urllib.request.Request(
        f"{ASSEMBLYAI_BASE}/upload", data=wav_bytes, headers=hdrs_bin, method="POST"
    )
    try:
        with urllib.request.urlopen(req) as r:
            upload_url = json.loads(r.read())["upload_url"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AssemblyAI upload {e.code}: {body[:400]}")

    # 2. Submit transcription job
    body = json.dumps({"audio_url": upload_url, "language_detection": True}).encode()
    req = urllib.request.Request(
        f"{ASSEMBLYAI_BASE}/transcript", data=body, headers=hdrs_json, method="POST"
    )
    try:
        with urllib.request.urlopen(req) as r:
            transcript_id = json.loads(r.read())["id"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AssemblyAI submit {e.code}: {body[:400]}")
    log.info(f"AssemblyAI job: {transcript_id}")

    # 3. Poll
    poll_url = f"{ASSEMBLYAI_BASE}/transcript/{transcript_id}"
    while True:
        req = urllib.request.Request(poll_url, headers={"authorization": api_key})
        try:
            with urllib.request.urlopen(req) as r:
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
        time.sleep(5)

    # 4. Group words into utterances by pause threshold
    words = result.get("words", [])
    if not words:
        return []

    PAUSE_MS = 1200
    segments: list[dict] = []
    buf: list[str] = []
    seg_start_s = words[0]["start"] / 1000.0

    for i, w in enumerate(words):
        buf.append(w["text"])
        is_last = i == len(words) - 1
        gap = 0 if is_last else words[i + 1]["start"] - w["end"]
        if is_last or gap > PAUSE_MS:
            segments.append({
                "start_s": seg_start_s,
                "end_s": w["end"] / 1000.0,
                "text": " ".join(buf),
            })
            buf = []
            if not is_last:
                seg_start_s = words[i + 1]["start"] / 1000.0

    return segments


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
    """Trigger daily transcription in a background thread (called from tray)."""
    threading.Thread(target=_daily_transcription_worker, daemon=True).start()


def _daily_transcription_worker():
    try:
        update_tray_icon("yellow")
        notify("Loading today's recordings...")

        mic_audio, sys_audio, started, sample_rate = load_todays_recordings()

        if mic_audio is None and sys_audio is None:
            notify("No recordings found — enable Daily Recording in Settings")
            update_tray_icon("green")
            return

        has_both = mic_audio is not None and sys_audio is not None

        if has_both:
            min_len = min(len(mic_audio), len(sys_audio))
            mix = mic_audio[:min_len] * 0.5 + sys_audio[:min_len] * 0.5
        else:
            mix = mic_audio if mic_audio is not None else sys_audio

        duration_s = len(mix) / sample_rate
        log.info(f"Daily audio: {duration_s / 3600:.1f}h from {started.strftime('%H:%M')}")

        # VAD pre-strip to cut API costs
        notify("VAD pre-processing...")
        vad_segs = _energy_vad(mix, sample_rate)

        if not vad_segs:
            notify("No speech detected in today's recordings")
            update_tray_icon("green")
            return

        total_speech_s = sum(e - s for s, e in vad_segs)
        pct = 100 * total_speech_s / duration_s if duration_s > 0 else 0
        log.info(
            f"VAD: {total_speech_s / 60:.1f} min speech / "
            f"{duration_s / 60:.1f} min total ({pct:.0f}%)"
        )

        # Build stripped audio + segment map for timestamp remapping
        chunks = [mix[int(s * sample_rate):int(e * sample_rate)] for s, e in vad_segs]
        stripped = np.concatenate(chunks)
        peak = np.abs(stripped).max()
        if peak > 0:
            stripped = stripped / peak * 0.95
        seg_map = _build_segment_map(vad_segs)

        # Transcribe via AssemblyAI (or fall back to local GPU)
        api_key = config.get("assemblyai_key", "")
        if api_key:
            notify(f"Transcribing {total_speech_s / 60:.0f} min via AssemblyAI...")
            raw_segs = transcribe_with_assemblyai(stripped, sample_rate)
        else:
            notify(f"No API key — using local GPU ({total_speech_s / 60:.0f} min)...")
            local = transcribe_audio(stripped)
            raw_segs = [{"start_s": s, "end_s": e, "text": t} for s, e, t in local]

        if not raw_segs:
            notify("No speech transcribed")
            update_tray_icon("green")
            return

        # Remap timestamps from stripped audio back to original audio time
        for seg in raw_segs:
            seg["start_s"] = _remap_to_original(seg["start_s"], seg_map)
            seg["end_s"]   = _remap_to_original(seg["end_s"],   seg_map)

        # Assign speaker labels using per-segment energy comparison
        for seg in raw_segs:
            if has_both:
                seg["speaker"] = _assign_speaker(
                    seg["start_s"], seg["end_s"], mic_audio, sys_audio, sample_rate
                )
            else:
                seg["speaker"] = "you" if sys_audio is None else "discord"

        # Add wall-clock timestamps
        for seg in raw_segs:
            wall = started + timedelta(seconds=seg["start_s"])
            seg["wall_time"] = wall.strftime("%H:%M:%S")

        word_count = sum(len(s["text"].split()) for s in raw_segs)
        log.info(f"Daily transcript: {word_count} words, {len(raw_segs)} segments")

        payload = {
            "date": date.today().isoformat(),
            "recorded_from": started.isoformat(),
            "duration_minutes": round(duration_s / 60, 1),
            "speech_minutes": round(total_speech_s / 60, 1),
            "word_count": word_count,
            "segments": [
                {"time": s["wall_time"], "speaker": s["speaker"], "text": s["text"]}
                for s in raw_segs
            ],
        }

        # Save local copy alongside the raw audio
        rec_path = Path(config.get("recording_path", "recordings"))
        day_dir = rec_path / date.today().isoformat()
        out_path = day_dir / "transcript.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
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

        # Optionally delete raw audio after successful transcription
        if config.get("delete_after_transcribe"):
            for name in ("mic.raw", "sys.raw"):
                p = day_dir / name
                if p.exists():
                    p.unlink()
            log.info("Raw audio deleted after transcription")

        notify(f"Done — {word_count} words, {total_speech_s / 60:.0f} min of speech")
        update_tray_icon("green")

    except Exception as e:
        log.error(f"Daily transcription failed: {e}", exc_info=True)
        notify(f"Daily transcription failed: {e}")
        update_tray_icon("red")
        threading.Timer(3, lambda: update_tray_icon("green")).start()

# ─── Hotkey Handling ──────────────────────────────────────────────────────────

def on_transcribe(minutes: int):
    """Handle hotkey press: cancel previous, grab audio, transcribe, clipboard."""
    global last_transcript, _current_cancel, _transcribing

    if _paused:
        return

    # Prevent overlapping transcriptions — only one at a time
    with _transcribing_lock:
        if _transcribing:
            log.info("Transcription already in progress, cancelling previous")
        _transcribing = True

    try:
        _on_transcribe_inner(minutes)
    except Exception as e:
        log.error(f"Unhandled transcription thread error: {e}")
        try:
            update_tray_icon("red")
            threading.Timer(3, lambda: update_tray_icon("green")).start()
        except Exception:
            pass
    finally:
        with _transcribing_lock:
            _transcribing = False

def _on_transcribe_inner(minutes: int):
    global last_transcript, _current_cancel

    # Cancel any in-progress transcription and create a fresh event
    with _cancel_lock:
        _current_cancel.set()
        cancel = threading.Event()
        _current_cancel = cancel

    log.info(f"Transcribing last {minutes} minute(s)...")
    update_tray_icon("yellow")
    notify(f"Transcribing last {minutes} min...")

    # Get audio from buffers and resample to 16 kHz for Whisper
    whisper_rate = 16000

    mic_audio = mic_buffer.get_last_n_minutes(minutes)
    if mic_audio is not None and mic_buffer.capture_rate != whisper_rate:
        mic_audio = _resample(mic_audio, mic_buffer.capture_rate, whisper_rate)

    loopback_audio = loopback_buffer.get_last_n_minutes(minutes)
    if loopback_audio is not None and loopback_buffer.capture_rate != whisper_rate:
        loopback_audio = _resample(loopback_audio, loopback_buffer.capture_rate, whisper_rate)

    if mic_audio is None and loopback_audio is None:
        log.warning("No audio in buffer")
        update_tray_icon("red")
        notify("No audio in buffer")
        threading.Timer(3, lambda: update_tray_icon("green")).start()
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
            transcript = format_ssmd(segments)
            pyperclip.copy(transcript)
            last_transcript = transcript[:200]
            word_count = sum(len(t.split()) for _, _, t in segments)
            log.info(f"{word_count} words in {elapsed:.1f}s -> clipboard")
            update_tray_icon("green")
            notify(f"{word_count} words in {elapsed:.1f}s")
            add_history_entry(minutes, word_count, elapsed, transcript)
        else:
            log.info("No speech detected")
            update_tray_icon("green")
            notify("No speech detected")
    except Exception as e:
        if cancel.is_set():
            return
        log.error(f"Transcription error: {e}")
        update_tray_icon("red")
        notify(f"Error: {e}")
        threading.Timer(3, lambda: update_tray_icon("green")).start()
    finally:
        try:
            unload_whisper()
        except Exception as e:
            log.error(f"Error unloading Whisper: {e}")

def register_hotkeys():
    """Register hotkeys from config."""
    prefix = config["hotkey_prefix"]
    hotkeys = config["hotkeys"]

    log.info(f"Hotkeys ({prefix} + key):")
    for key in sorted(hotkeys.keys(), key=int):
        minutes = hotkeys[key]
        keyboard.add_hotkey(
            f"{prefix}+{key}",
            lambda m=minutes: threading.Thread(
                target=on_transcribe, args=(m,), daemon=True
            ).start(),
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
            def _cmd():
                threading.Thread(target=on_transcribe, args=(mins,), daemon=True).start()
            return _cmd
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
        command=lambda: threading.Thread(target=run_daily_transcription, daemon=True).start(),
    ).pack(side=tk.LEFT, padx=(0, 6))
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
        if _paused:
            dot.config(fg="#6b7280")
            status_txt.config(text="Paused")
        elif _transcribing:
            dot.config(fg="#eab308")
            status_txt.config(text="Transcribing…")
        else:
            dot.config(fg="#22c55e")
            status_txt.config(text="Recording daily" if config.get("daily_recording") else "Ready")
        rec_path = Path(config.get("recording_path", "recordings"))
        mic_f = rec_path / date.today().isoformat() / "mic.raw"
        info_txt.config(
            text=f"Today's recording: {mic_f.stat().st_size / 1024 / 1024:.1f} MB"
            if mic_f.exists() else ""
        )
        lt_txt.configure(state=tk.NORMAL)
        lt_txt.delete("1.0", tk.END)
        lt_txt.insert(
            tk.END,
            last_transcript or "(No transcription yet — use a hotkey or click a button above)",
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
        rec_path = Path(config.get("recording_path", "recordings"))
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

    rec_path = Path(config.get("recording_path", "recordings"))
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

    def _hint(text: str):
        ttk.Label(inner, text=text, foreground="#9ca3af", font=("Segoe UI", 8)).pack(
            anchor="w", padx=14, pady=(0, 6)
        )

    _hint("Device changes require a restart to take effect.")

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

    path_var = tk.StringVar(value=config.get("recording_path", "recordings"))
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
    _hint("Required for daily transcription. ~$0.0035/min billed after VAD silence stripping.")

    url_var = tk.StringVar(value=config.get("homelab_url") or "")
    r = _field_row("Homelab URL:")
    ttk.Entry(r, textvariable=url_var, width=45).pack(side=tk.LEFT, fill=tk.X, expand=True)
    _hint("Optional. Daily transcripts will be HTTP POSTed as JSON to this endpoint.")

    del_var = tk.BooleanVar(value=config.get("delete_after_transcribe", False))
    ttk.Checkbutton(
        inner, text="Delete raw audio after successful transcription", variable=del_var
    ).pack(anchor="w", padx=14, pady=(0, 4))
    _hint("Frees disk space. Transcription must succeed first — raw files are never deleted on error.")

    ttk.Separator(inner, orient="horizontal").pack(fill=tk.X, padx=14, pady=(8, 6))
    status_var = tk.StringVar(value="")
    ttk.Label(inner, textvariable=status_var, foreground="#6b7280").pack(anchor="w", padx=14)
    btn_r = ttk.Frame(inner)
    btn_r.pack(anchor="e", padx=14, pady=(4, 14))

    def _save():
        mic_i = device_labels.index(mic_var.get())
        sys_i = device_labels.index(sys_var.get())
        config.update({
            "mic_device":              device_ids[mic_i],
            "mic_device_name":         device_names[mic_i],
            "system_device":           device_ids[sys_i],
            "system_device_name":      device_names[sys_i],
            "whisper_model":           model_var.get(),
            "whisper_device":          wdev_var.get(),
            "buffer_minutes":          buf_var.get(),
            "hotkey_prefix":           prefix_var.get().strip(),
            "daily_recording":         daily_var.get(),
            "recording_path":          path_var.get().strip() or "recordings",
            "assemblyai_key":          key_var.get().strip(),
            "homelab_url":             url_var.get().strip() or None,
            "delete_after_transcribe": del_var.get(),
        })
        save_config(config)
        status_var.set("Saved! Restart Pascribe for device and hotkey changes to take effect.")
        log.info("Settings saved via panel")

    ttk.Button(btn_r, text="Save Settings", command=_save).pack(side=tk.RIGHT)


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
            _ver_lbl = ttk.Label(footer_inner, text="Pascribe v0.5",
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
                elif _transcribing:
                    _footer_status.config(text="Transcribing…")
                elif config.get("daily_recording"):
                    rec_path = Path(config.get("recording_path", "recordings"))
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
            ]:
                f = ttk.Frame(nb)
                nb.add(f, text=title)
                builder(f, root)

            nb.select(min(start_tab, 3))
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
        target = str(
            (Path(__file__).parent / "venv" / "Scripts" / "pythonw.exe").resolve()
        )
        script = str(Path(__file__).resolve())
        working_dir = str(Path(__file__).parent.resolve())
        lnk_path = str(STARTUP_LNK)

        # Create .lnk shortcut via PowerShell WScript.Shell COM object
        ps_cmd = (
            f"$ws = New-Object -ComObject WScript.Shell; "
            f"$s = $ws.CreateShortcut('{lnk_path}'); "
            f"$s.TargetPath = '{target}'; "
            f"$s.Arguments = '\"{script}\"'; "
            f"$s.WorkingDirectory = '{working_dir}'; "
            f"$s.Save()"
        )

        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                creationflags=subprocess.CREATE_NO_WINDOW,
                capture_output=True,
            )
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
        update_tray_icon("green")
        notify("Resumed — hotkeys active")
        log.info("Hotkeys resumed by user")

# ─── Tray Setup ──────────────────────────────────────────────────────────────

def on_quit(icon, item):
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
        MenuItem("Pascribe v0.5", lambda: None, enabled=False),
        MenuItem(daily_label, lambda: None, enabled=False),
        Menu.SEPARATOR,
        MenuItem("Settings…",              lambda icon, item: open_settings()),
        MenuItem("Transcription History…", lambda icon, item: open_history()),
        MenuItem("Daily Transcripts…",     lambda icon, item: open_daily_transcripts()),
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

    log.info("Pascribe v0.5 starting")

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
