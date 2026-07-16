"""Application and user-data path handling for the Windows desktop app."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .storage import atomic_write_json


@dataclass(frozen=True)
class AppPaths:
    app_dir: Path
    data_dir: Path
    config: Path
    history: Path
    log: Path
    recordings: Path


def build_app_paths(
    app_dir: Path,
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> AppPaths:
    env = os.environ if environ is None else environ
    current_platform = sys.platform if platform is None else platform
    app_dir = Path(app_dir).resolve()

    override = env.get("PASCRIBE_DATA_DIR")
    if override:
        data_dir = Path(override).expanduser().resolve()
    elif current_platform == "win32":
        local = env.get("LOCALAPPDATA") or env.get("APPDATA")
        data_dir = Path(local).expanduser() / "Pascribe" if local else app_dir / "data"
    else:
        # Pascribe only runs on Windows. Keeping non-Windows development data
        # inside the checkout avoids touching an unrelated home directory.
        data_dir = app_dir / ".pascribe-data"
    data_dir = data_dir.resolve()

    return AppPaths(
        app_dir=app_dir,
        data_dir=data_dir,
        config=data_dir / "config.json",
        history=data_dir / "history.json",
        log=data_dir / "pascribe.log",
        recordings=data_dir / "recordings",
    )


def prepare_app_paths(paths: AppPaths) -> list[str]:
    """Create the data directory and copy legacy files without data loss."""
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    migrated: list[str] = []
    if paths.data_dir == paths.app_dir:
        return migrated

    legacy_config = paths.app_dir / "config.json"
    if not paths.config.exists() and legacy_config.exists():
        migration_succeeded = False
        try:
            with legacy_config.open(encoding="utf-8") as handle:
                config = json.load(handle)
            recording_path = config.get("recording_path")
            if isinstance(recording_path, str) and recording_path and not Path(recording_path).is_absolute():
                config["recording_path"] = str((paths.app_dir / recording_path).resolve())
            atomic_write_json(paths.config, config)
            migration_succeeded = True
        except (OSError, json.JSONDecodeError, TypeError):
            try:
                shutil.copy2(legacy_config, paths.config)
                migration_succeeded = True
            except OSError:
                pass
        if migration_succeeded:
            backup = paths.app_dir / "config.json.migrated.bak"
            try:
                os.replace(legacy_config, backup)
            except OSError:
                pass
            migrated.append("config.json")

    legacy_history = paths.app_dir / "history.json"
    if not paths.history.exists() and legacy_history.exists():
        shutil.copy2(legacy_history, paths.history)
        migrated.append("history.json")

    return migrated


def resolve_recording_path(value: str | Path | None, paths: AppPaths) -> Path:
    if value is None or not str(value).strip():
        return paths.recordings
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = paths.data_dir / candidate
    return candidate.resolve()
