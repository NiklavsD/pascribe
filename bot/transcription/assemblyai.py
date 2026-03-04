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
            resp.raise_for_status()
            result = await resp.json()
            return result["upload_url"]


async def transcribe(audio_url: str) -> dict:
    """Submit a transcription job and poll until complete. Returns the transcript object."""
    async with aiohttp.ClientSession() as session:
        # Submit
        payload = {
            "audio_url": audio_url,
            "speech_models": ["universal-3-pro"],
            "language_detection": True,
            "speaker_labels": True,
        }
        async with session.post(
            f"{BASE_URL}/transcript",
            headers={**HEADERS, "content-type": "application/json"},
            json=payload,
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()
            transcript_id = result["id"]

        log.info("Transcription submitted: %s", transcript_id)

        # Poll
        poll_url = f"{BASE_URL}/transcript/{transcript_id}"
        while True:
            await asyncio.sleep(3)
            async with session.get(poll_url, headers=HEADERS) as resp:
                resp.raise_for_status()
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
