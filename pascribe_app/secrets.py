"""Windows DPAPI protection for locally stored desktop credentials."""

from __future__ import annotations

import base64
import copy
import ctypes
import sys
from ctypes import wintypes


_PREFIX = "dpapi:"


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _blob_from_bytes(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(data)
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    return blob, buffer


def protect_secret(secret: str) -> str | None:
    if not secret or sys.platform != "win32":
        return None
    raw = secret.encode("utf-8")
    input_blob, input_buffer = _blob_from_bytes(raw)
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    # CRYPTPROTECT_UI_FORBIDDEN prevents unexpected desktop prompts.
    if not crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(output_blob),
    ):
        return None
    try:
        protected = ctypes.string_at(output_blob.pbData, output_blob.cbData)
        return _PREFIX + base64.b64encode(protected).decode("ascii")
    finally:
        kernel32.LocalFree(output_blob.pbData)
        del input_buffer


def unprotect_secret(value: str) -> str | None:
    if not value.startswith(_PREFIX) or sys.platform != "win32":
        return None
    try:
        protected = base64.b64decode(value[len(_PREFIX):], validate=True)
    except (ValueError, TypeError):
        return None
    input_blob, input_buffer = _blob_from_bytes(protected)
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(output_blob),
    ):
        return None
    try:
        raw = ctypes.string_at(output_blob.pbData, output_blob.cbData)
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    finally:
        kernel32.LocalFree(output_blob.pbData)
        del input_buffer


def hydrate_config_secrets(stored: dict) -> tuple[dict, bool]:
    """Return an in-memory config containing a plain key and migration flag."""
    config = copy.deepcopy(stored)
    protected = config.get("assemblyai_key_protected")
    migrated_plaintext = bool(config.get("assemblyai_key")) and sys.platform == "win32"
    if isinstance(protected, str):
        decrypted = unprotect_secret(protected)
        if decrypted is not None:
            config["assemblyai_key"] = decrypted
    config.pop("assemblyai_key_protected", None)
    return config, migrated_plaintext


def config_for_disk(config: dict) -> dict:
    """Protect the API key with the current Windows account when possible."""
    stored = copy.deepcopy(config)
    secret = stored.pop("assemblyai_key", "")
    if isinstance(secret, str) and secret:
        protected = protect_secret(secret)
        if protected is not None:
            stored["assemblyai_key_protected"] = protected
        else:
            # Non-Windows development and unusual DPAPI failures retain
            # compatibility instead of silently losing the user's key.
            stored["assemblyai_key"] = secret
    else:
        stored.pop("assemblyai_key_protected", None)
    return stored
