"""Crash-safe JSON persistence used by the desktop application."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class JsonLoadResult:
    data: Any
    recovered: bool = False
    backup_path: Path | None = None


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON beside the target and atomically replace the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def load_json(
    path: Path,
    default_factory: Callable[[], Any],
    *,
    backup_corrupt: bool = True,
) -> JsonLoadResult:
    """Load a JSON file and preserve malformed data before recovering."""
    path = Path(path)
    if not path.exists():
        return JsonLoadResult(copy.deepcopy(default_factory()))

    try:
        with path.open(encoding="utf-8") as handle:
            return JsonLoadResult(json.load(handle))
    except (json.JSONDecodeError, OSError, UnicodeError):
        backup_path = None
        if backup_corrupt:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            backup_path = path.with_name(f"{path.name}.corrupt-{stamp}")
            try:
                os.replace(path, backup_path)
            except OSError:
                backup_path = None
        return JsonLoadResult(
            copy.deepcopy(default_factory()),
            recovered=True,
            backup_path=backup_path,
        )
