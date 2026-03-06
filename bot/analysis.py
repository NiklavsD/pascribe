#!/usr/bin/env python3
"""Transcript analysis pipeline using free OpenRouter models.

Monitors daily transcript files for new content and performs AI analysis
to extract insights, questions, and key discussions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import aiohttp
import discord

from config import RECORDINGS_DIR, REPORT_CHANNEL_ID

log = logging.getLogger(__name__)

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen3-235b-a22b-thinking"  # Best free model for reasoning
ANALYSIS_TRIGGER_LINES = 50  # Trigger analysis every N new lines

class TranscriptAnalyzer:
    """Monitors transcripts and performs AI analysis on new content."""
    
    def __init__(self, bot: discord.Client, openrouter_api_key: str):
        self.bot = bot
        self.openrouter_api_key = openrouter_api_key
        self._line_counts: Dict[str, int] = {}  # date -> processed_line_count
        self._state_file = RECORDINGS_DIR / "analysis_state.json"
        self._load_state()
        
    def _load_state(self) -> None:
        """Load processing state from disk."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    self._line_counts = json.load(f)
                log.info(f"Loaded analysis state: {len(self._line_counts)} dates tracked")
        except Exception as e:
            log.warning(f"Failed to load analysis state: {e}")
            self._line_counts = {}
    
    def _save_state(self) -> None:
        """Save processing state to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(self._line_counts, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save analysis state: {e}")
    
    def _count_transcript_lines(self, transcript_path: Path) -> int:
        """Count non-empty lines in transcript file."""
        try:
            if not transcript_path.exists():
                return 0
            content = transcript_path.read_text(encoding='utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            return len(lines)
        except Exception as e:
            log.warning(f"Failed to count lines in {transcript_path}: {e}")
            return 0
    
    def check_for_new_content(self, date_str: str) -> bool:
        """Check if transcript has enough new content to trigger analysis."""
        transcript_path = RECORDINGS_DIR / date_str / "_transcript.txt"
        current_lines = self._count_transcript_lines(transcript_path)
        processed_lines = self._line_counts.get(date_str, 0)
        
        new_lines = current_lines - processed_lines
        log.debug(f"{date_str}: {current_lines} total, {processed_lines} processed, {new_lines} new")
        
        if new_lines >= ANALYSIS_TRIGGER_LINES:
            log.info(f"Triggering analysis for {date_str}: {new_lines} new lines")
            return True
        
        return False
    
    def _extract_latest_discussion(self, transcript: str, focus_lines: int = 20) -> str:
        """Extract the latest discussion from transcript for focused analysis."""
        lines = [line.strip() for line in transcript.split('\n') if line.strip()]
        if len(lines) <= focus_lines:
            return transcript
        
        # Get last N lines for focus, but include full transcript for context
        latest_lines = lines[-focus_lines:]
        latest_discussion = '\n'.join(latest_lines)
        
        return f"=== LATEST DISCUSSION (FOCUS HERE) ===\n{latest_discussion}\n\n=== FULL TRANSCRIPT FOR CONTEXT ===\n{transcript}"
    
    async def _get_past_reports(self, limit: int = 5) -> str:
        """Fetch recent Benjamin reports from Discord for context continuity."""
        try:
            channel = self.bot.get_channel(REPORT_CHANNEL_ID)
            if not channel:
                return ""
            
            reports = []
            async for message in channel.history(limit=50):  # Check more messages
                if message.author.bot and ("🎙️" in message.content or "📋" in message.content):
                    reports.append(f"[{message.created_at.strftime('%H:%M')}] {message.content[:200]}...")
                    if len(reports) >= limit:
                        break
            
            if reports:
                return "\n".join(reversed(reports))  # Chronological order
            
        except Exception as e:
            log.warning(f"Failed to fetch past reports: {e}")
        
        return ""
    
    async def _call_openrouter(self, prompt: str) -> Optional[str]:
        """Send prompt to OpenRouter and get analysis response."""
        if not self.openrouter_api_key:
            log.warning("No OpenRouter API key configured")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nikdotcrm/pascribe-bot",  # Required for free tier
        }
        
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Benjamin's analysis assistant. Analyze voice chat transcripts and provide CONCISE insights. Only respond if there's something genuinely valuable - no spam. Focus on: key topics, questions that could be answered, action items, technical discussions worth noting. Keep responses SHORT and actionable."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 300,  # Keep responses compact
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{OPENROUTER_BASE_URL}/chat/completions", 
                                      headers=headers, json=payload) as response:
                    
                    if response.status == 429:
                        log.warning("OpenRouter rate limit hit")
                        return None
                    elif response.status >= 400:
                        error_text = await response.text()
                        log.error(f"OpenRouter API error {response.status}: {error_text}")
                        return None
                    
                    result = await response.json()
                    
                    if 'choices' in result and result['choices']:
                        return result['choices'][0]['message']['content'].strip()
                    else:
                        log.warning(f"Unexpected OpenRouter response format: {result}")
                        return None
                        
        except asyncio.TimeoutError:
            log.warning("OpenRouter API call timed out")
        except Exception as e:
            log.error(f"OpenRouter API call failed: {e}")
        
        return None
    
    def _should_post_analysis(self, analysis: str) -> bool:
        """Determine if analysis contains valuable insights worth posting."""
        if not analysis or len(analysis.strip()) < 50:
            return False
        
        # Filter out generic or low-value responses
        low_value_indicators = [
            "nothing significant",
            "general conversation",
            "casual chat",
            "no specific topics",
            "brief exchange",
            "social interaction",
            "not much to report"
        ]
        
        analysis_lower = analysis.lower()
        for indicator in low_value_indicators:
            if indicator in analysis_lower:
                log.debug(f"Skipping analysis - low value indicator: {indicator}")
                return False
        
        return True
    
    async def analyze_transcript(self, date_str: str) -> bool:
        """Analyze transcript for the given date and post insights if valuable."""
        try:
            transcript_path = RECORDINGS_DIR / date_str / "_transcript.txt"
            if not transcript_path.exists():
                log.debug(f"No transcript file for {date_str}")
                return False
            
            # Read full transcript
            transcript = transcript_path.read_text(encoding='utf-8')
            if len(transcript.strip()) < 100:  # Skip very short transcripts
                log.debug(f"Transcript too short for analysis: {len(transcript)} chars")
                return False
            
            # Prepare context with latest discussion focus
            focused_transcript = self._extract_latest_discussion(transcript)
            past_reports = await self._get_past_reports()
            
            # Build analysis prompt
            prompt = f"""
TRANSCRIPT TO ANALYZE ({date_str}):
{focused_transcript}

RECENT BENJAMIN REPORTS (for context continuity):
{past_reports}

INSTRUCTIONS:
- Focus on the LATEST DISCUSSION section
- Extract only VALUABLE insights: key topics, actionable questions, decisions, technical discussions
- Be CONCISE - max 2-3 sentences
- If nothing significant happened, respond with "SKIP" (don't waste Discord space)
- Format as a brief summary, not bullet points
"""

            # Get AI analysis
            analysis = await self._call_openrouter(prompt)
            if not analysis:
                log.warning(f"Failed to get analysis for {date_str}")
                return False
            
            # Check if analysis is valuable enough to post
            if not self._should_post_analysis(analysis) or analysis.upper().startswith("SKIP"):
                log.info(f"Skipping analysis for {date_str} - not valuable enough")
                self._update_processed_lines(date_str)
                return False
            
            # Post to Discord
            await self._post_analysis(date_str, analysis)
            
            # Update state
            self._update_processed_lines(date_str)
            return True
            
        except Exception as e:
            log.error(f"Analysis failed for {date_str}: {e}")
            return False
    
    def _update_processed_lines(self, date_str: str) -> None:
        """Update the processed line count for a date."""
        transcript_path = RECORDINGS_DIR / date_str / "_transcript.txt"
        current_lines = self._count_transcript_lines(transcript_path)
        self._line_counts[date_str] = current_lines
        self._save_state()
        log.debug(f"Updated processed lines for {date_str}: {current_lines}")
    
    async def _post_analysis(self, date_str: str, analysis: str) -> None:
        """Post analysis to Discord voice reports channel."""
        try:
            channel = self.bot.get_channel(REPORT_CHANNEL_ID)
            if not channel:
                log.error("Could not find voice reports channel")
                return
            
            # Format message compactly
            timestamp = datetime.now(timezone.utc).strftime("%H:%M UTC")
            message = f"📋 **TRANSCRIPT ANALYSIS** ({date_str} • {timestamp})\n>>> {analysis}"
            
            await channel.send(message)
            log.info(f"Posted analysis for {date_str} to #voice-reports")
            
        except Exception as e:
            log.error(f"Failed to post analysis to Discord: {e}")
    
    async def run_analysis_loop(self) -> None:
        """Main analysis loop - checks for new content periodically."""
        log.info("Starting transcript analysis loop")
        
        while True:
            try:
                # Check today's transcript
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                
                if self.check_for_new_content(today):
                    await self.analyze_transcript(today)
                
                # Sleep for 2 minutes before next check
                await asyncio.sleep(120)
                
            except Exception as e:
                log.error(f"Analysis loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def start_background_analysis(self) -> None:
        """Start the analysis loop as a background task."""
        if not self.openrouter_api_key:
            log.warning("No OpenRouter API key - analysis disabled")
            return
        
        asyncio.create_task(self.run_analysis_loop())
        log.info("Background transcript analysis started")


# Global analyzer instance
_analyzer: Optional[TranscriptAnalyzer] = None

def init_analyzer(bot: discord.Client, openrouter_api_key: str) -> None:
    """Initialize the global analyzer instance."""
    global _analyzer
    _analyzer = TranscriptAnalyzer(bot, openrouter_api_key)

def start_analysis() -> None:
    """Start background analysis if analyzer is initialized."""
    if _analyzer:
        _analyzer.start_background_analysis()
    else:
        log.warning("Analyzer not initialized - call init_analyzer() first")

def manual_analyze(date_str: str) -> asyncio.Task:
    """Manually trigger analysis for a specific date."""
    if not _analyzer:
        raise RuntimeError("Analyzer not initialized")
    return asyncio.create_task(_analyzer.analyze_transcript(date_str))