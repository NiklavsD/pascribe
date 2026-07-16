"""User-facing, non-destructive desktop diagnostics."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Diagnostic:
    status: str
    name: str
    detail: str


def _nearest_existing_parent(path: Path) -> Path:
    current = Path(path)
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def collect_diagnostics(
    *,
    data_dir: Path,
    recording_dir: Path,
    whisper_device: str,
    input_devices: Iterable[tuple[int, str, int]],
) -> list[Diagnostic]:
    results: list[Diagnostic] = []
    version = sys.version_info
    python_ok = (version.major, version.minor) >= (3, 11)
    results.append(Diagnostic(
        "PASS" if python_ok else "FAIL",
        "Python",
        f"{version.major}.{version.minor}.{version.micro}"
        + ("" if python_ok else " — Python 3.11 or newer is required"),
    ))
    results.append(Diagnostic(
        "PASS" if sys.platform == "win32" else "WARN",
        "Platform",
        "Windows" if sys.platform == "win32" else f"{sys.platform} — Pascribe desktop is Windows-only",
    ))

    data_parent = _nearest_existing_parent(data_dir)
    data_writable = data_parent.exists() and os.access(data_parent, os.W_OK)
    results.append(Diagnostic(
        "PASS" if data_writable else "FAIL",
        "Application data",
        f"{data_dir}" + ("" if data_writable else " — parent directory is not writable"),
    ))

    recording_parent = _nearest_existing_parent(recording_dir)
    recording_writable = recording_parent.exists() and os.access(recording_parent, os.W_OK)
    try:
        free_gb = shutil.disk_usage(recording_parent).free / (1024 ** 3)
        disk_detail = f"{recording_dir} — {free_gb:.1f} GB free"
        if not recording_writable:
            disk_status = "FAIL"
            disk_detail += " — parent directory is not writable"
        else:
            disk_status = "PASS" if free_gb >= 2 else "WARN"
    except OSError as exc:
        disk_detail = f"{recording_dir} — {exc}"
        disk_status = "FAIL"
    results.append(Diagnostic(disk_status, "Recording storage", disk_detail))

    devices = list(input_devices)
    results.append(Diagnostic(
        "PASS" if devices else "FAIL",
        "Audio inputs",
        f"{len(devices)} input device(s) detected" if devices else "No input devices detected",
    ))

    packages = ["numpy", "sounddevice", "keyboard", "pystray", "PIL", "pyperclip", "faster_whisper"]
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    results.append(Diagnostic(
        "PASS" if not missing else "FAIL",
        "Dependencies",
        "All required packages found" if not missing else "Missing: " + ", ".join(missing),
    ))

    if whisper_device == "cuda":
        cuda_status = "WARN"
        cuda_detail = "CUDA selected; availability is verified when Whisper loads"
        try:
            import ctranslate2
            count = ctranslate2.get_cuda_device_count()
            cuda_status = "PASS" if count > 0 else "FAIL"
            cuda_detail = f"{count} CUDA device(s) reported by CTranslate2"
        except Exception as exc:
            cuda_detail = f"Could not query CUDA: {exc}"
        results.append(Diagnostic(cuda_status, "Whisper acceleration", cuda_detail))
    else:
        results.append(Diagnostic("PASS", "Whisper acceleration", "CPU mode selected"))
    return results


def format_diagnostics(results: Iterable[Diagnostic]) -> str:
    return "\n".join(f"[{item.status}] {item.name}: {item.detail}" for item in results)
