"""Voice command detection — listens for keywords in transcribed speech."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Callable, Awaitable

log = logging.getLogger(__name__)

# Patterns that trigger processing
TRIGGER_PATTERNS = [
    re.compile(r"\bbenjamin\b", re.IGNORECASE),
    re.compile(r"\bben[,.]?\s*process\s+this\b", re.IGNORECASE),
]

# Patterns that trigger privacy exclusion
PAUSE_PATTERNS = [
    re.compile(r"\bpause\s+recording\b", re.IGNORECASE),
    re.compile(r"\bstop\s+listening\b", re.IGNORECASE),
]

RESUME_PATTERNS = [
    re.compile(r"\bresume\s+recording\b", re.IGNORECASE),
    re.compile(r"\bstart\s+listening\b", re.IGNORECASE),
]


class VoiceCommandDetector:
    """Detects voice commands from transcribed text snippets."""

    def __init__(
        self,
        on_trigger: Callable[[], Awaitable[None]] | None = None,
        on_pause: Callable[[int], Awaitable[None]] | None = None,
        on_resume: Callable[[int], Awaitable[None]] | None = None,
    ):
        self.on_trigger = on_trigger
        self.on_pause = on_pause
        self.on_resume = on_resume

    async def check_text(self, text: str, user_id: int) -> None:
        """Check transcribed text for voice commands."""
        if not text:
            return

        # Check privacy commands first
        for pattern in PAUSE_PATTERNS:
            if pattern.search(text):
                log.info("Pause recording command from user %d", user_id)
                if self.on_pause:
                    await self.on_pause(user_id)
                return

        for pattern in RESUME_PATTERNS:
            if pattern.search(text):
                log.info("Resume recording command from user %d", user_id)
                if self.on_resume:
                    await self.on_resume(user_id)
                return

        # Check trigger commands
        for pattern in TRIGGER_PATTERNS:
            if pattern.search(text):
                log.info("Trigger command detected from user %d: %s", user_id, text[:80])
                if self.on_trigger:
                    await self.on_trigger()
                return
