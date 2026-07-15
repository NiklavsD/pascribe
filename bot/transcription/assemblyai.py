"""AssemblyAI API client for audio transcription."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiohttp

from config import ASSEMBLYAI_API_KEY

log = logging.getLogger(__name__)

BASE_URL = "https://api.assemblyai.com/v2"
HEADERS = {"authorization": ASSEMBLYAI_API_KEY}


async def _raise_with_body(resp: aiohttp.ClientResponse) -> None:
    """Like resp.raise_for_status() but logs the response body first.

    AssemblyAI's 4xx responses carry a JSON `{"error": "..."}` that
    explains *why* — without this, all we see in the journal is a generic
    `400 Bad Request` stack trace and the actual cause (e.g. negative
    account balance) is invisible.
    """
    if resp.status < 400:
        return
    try:
        body = await resp.text()
    except Exception:
        body = "<unreadable>"
    log.error("AssemblyAI %s %s -> HTTP %d body=%s",
              resp.method, resp.url.path, resp.status, body[:500])
    resp.raise_for_status()


async def upload_file(filepath: Path) -> str:
    """Upload a local audio file to AssemblyAI. Returns the upload URL."""
    async with aiohttp.ClientSession() as session:
        with open(filepath, "rb") as f:
            data = f.read()
        async with session.post(
            f"{BASE_URL}/upload",
            headers={**HEADERS, "content-type": "application/octet-stream"},
            data=data,
        ) as resp:
            await _raise_with_body(resp)
            result = await resp.json()
            return result["upload_url"]


async def transcribe(audio_url: str) -> dict:
    """Submit a transcription job and poll until complete. Returns the transcript object."""
    async with aiohttp.ClientSession() as session:
        # Submit
        payload = {
            "audio_url": audio_url,
            "speech_models": ["universal-3-5-pro", "universal-2"],
            "language_detection": True,
            "speaker_labels": True,
            # auto_chapters → AAI segments into topic-coherent chapters with
            # per-chapter `summary` / `headline` / `gist` + start/end ms.
            # Used downstream by idea-gen prepare_inputs.py to chunk by topic
            # instead of arbitrary 80K-char windows. (Mutually exclusive with
            # `summarization`, which is why we don't enable both.)
            "auto_chapters": True,
        }
        async with session.post(
            f"{BASE_URL}/transcript",
            headers={**HEADERS, "content-type": "application/json"},
            json=payload,
        ) as resp:
            await _raise_with_body(resp)
            result = await resp.json()
            transcript_id = result["id"]

        log.info("Transcription submitted: %s", transcript_id)

        # Poll
        poll_url = f"{BASE_URL}/transcript/{transcript_id}"
        while True:
            await asyncio.sleep(3)
            async with session.get(poll_url, headers=HEADERS) as resp:
                await _raise_with_body(resp)
                result = await resp.json()

            status = result["status"]
            if status == "completed":
                log.info("Transcription complete: %s", transcript_id)
                return result
            elif status == "error":
                error = result.get("error", "Unknown error")
                log.error("Transcription failed: %s — %s", transcript_id, error)
                raise RuntimeError(f"AssemblyAI transcription error: {error}")
            else:
                log.debug("Transcription %s: %s", transcript_id, status)


async def transcribe_file(filepath: Path, **kwargs) -> dict:
    """Upload and transcribe a local audio file. Returns transcript dict."""
    url = await upload_file(filepath)
    return await transcribe(url, **kwargs)
