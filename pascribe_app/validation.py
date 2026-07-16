"""Configuration normalization and settings validation."""

from __future__ import annotations

import copy
from typing import Any
from urllib.parse import urlparse


WHISPER_MODELS = {"tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def normalize_config(raw: Any, defaults: dict) -> tuple[dict, list[str]]:
    """Merge stored values with defaults and replace unsafe invalid values."""
    warnings: list[str] = []
    cfg = copy.deepcopy(defaults)
    if not isinstance(raw, dict):
        return cfg, ["Configuration root was not an object; defaults were restored."]
    cfg.update(raw)

    def restore(key: str, predicate, message: str) -> None:
        if not predicate(cfg.get(key)):
            cfg[key] = copy.deepcopy(defaults[key])
            warnings.append(message)

    restore("sample_rate", lambda v: _is_int(v) and 8000 <= v <= 192000, "Invalid sample rate was reset.")
    restore("buffer_minutes", lambda v: _is_int(v) and 5 <= v <= 240, "Buffer duration was reset.")
    restore("history_max_entries", lambda v: _is_int(v) and 1 <= v <= 5000, "History limit was reset.")
    restore("whisper_model", lambda v: isinstance(v, str) and v in WHISPER_MODELS, "Whisper model was reset.")
    restore("whisper_device", lambda v: v in {"cpu", "cuda"}, "Whisper device was reset.")
    restore("hotkey_prefix", lambda v: isinstance(v, str) and bool(v.strip()), "Hotkey prefix was reset.")
    restore("recording_path", lambda v: isinstance(v, str) and bool(v.strip()), "Recording path was reset.")

    for key in ("daily_recording", "delete_after_transcribe"):
        restore(key, lambda v: isinstance(v, bool), f"{key} was reset.")

    for key in ("mic_device", "system_device"):
        restore(key, lambda v: v is None or _is_int(v), f"{key} was reset.")
    for key in ("mic_device_name", "system_device_name"):
        restore(key, lambda v: v is None or isinstance(v, str), f"{key} was reset.")

    hotkeys = cfg.get("hotkeys")
    hotkeys_ok = isinstance(hotkeys, dict) and bool(hotkeys)
    if hotkeys_ok:
        hotkeys_ok = all(
            isinstance(key, str)
            and bool(key.strip())
            and _is_int(minutes)
            and 1 <= minutes <= 240
            for key, minutes in hotkeys.items()
        )
    if not hotkeys_ok:
        cfg["hotkeys"] = copy.deepcopy(defaults["hotkeys"])
        warnings.append("Hotkey mappings were reset.")

    url = cfg.get("homelab_url")
    if url not in (None, ""):
        parsed = urlparse(str(url))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            cfg["homelab_url"] = None
            warnings.append("Invalid homelab URL was removed.")
    else:
        cfg["homelab_url"] = None

    key = cfg.get("assemblyai_key", "")
    if not isinstance(key, str):
        cfg["assemblyai_key"] = ""
        warnings.append("Invalid AssemblyAI key was removed.")

    return cfg, warnings


def validate_settings(values: dict) -> list[str]:
    errors: list[str] = []
    buffer_minutes = values.get("buffer_minutes")
    if not _is_int(buffer_minutes) or not 5 <= buffer_minutes <= 240:
        errors.append("Buffer duration must be between 5 and 240 minutes.")

    prefix = values.get("hotkey_prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        errors.append("Hotkey prefix cannot be empty.")

    recording_path = values.get("recording_path")
    if not isinstance(recording_path, str) or not recording_path.strip():
        errors.append("Recording path cannot be empty.")

    url = values.get("homelab_url")
    if url:
        parsed = urlparse(str(url))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append("Homelab URL must start with http:// or https:// and include a host.")

    if values.get("whisper_model") not in WHISPER_MODELS:
        errors.append("Select a supported Whisper model.")
    if values.get("whisper_device") not in {"cpu", "cuda"}:
        errors.append("Select CPU or CUDA for Whisper.")
    return errors
