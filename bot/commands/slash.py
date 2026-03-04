"""Slash commands for the Pascribe bot (discord.py app_commands)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import discord
from discord import app_commands
from discord.ext import commands

from config import GUILD_ID, REPORT_CHANNEL_ID
from transcription.pipeline import generate_daily_report

log = logging.getLogger(__name__)


class PascribeCog(commands.Cog):
    """Slash commands under /pascribe."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._last_report_msg: discord.Message | None = None

    pascribe = app_commands.Group(name="pascribe", description="Pascribe voice transcription commands")

    @pascribe.command(name="process", description="Process and transcribe recent voice recordings")
    async def process(self, interaction: discord.Interaction):
        await interaction.response.defer()

        try:
            report = await generate_daily_report()
            if not report:
                await interaction.followup.send("ℹ️ No new recordings to process.")
                return

            channel = self.bot.get_channel(REPORT_CHANNEL_ID)
            if not channel:
                await interaction.followup.send(f"❌ Report channel <#{REPORT_CHANNEL_ID}> not found.")
                return

            if self._last_report_msg:
                try:
                    await self._last_report_msg.delete()
                except Exception:
                    pass

            text = f"## 🎙️ Voice Transcript\n{report['text']}"
            if len(text) <= 2000:
                self._last_report_msg = await channel.send(text)
            else:
                self._last_report_msg = await channel.send("## 🎙️ Voice Transcript")
                parts = report["text"]
                for i in range(0, len(parts), 1900):
                    msg = await channel.send(parts[i : i + 1900])
                    self._last_report_msg = msg

            n_users = len(report.get("transcripts", []))
            await interaction.followup.send(
                f"✅ Processed {n_users} user(s). Report posted to <#{REPORT_CHANNEL_ID}>."
            )

        except Exception as e:
            log.exception("Error processing recordings")
            await interaction.followup.send(f"❌ Error: {e}")

    @pascribe.command(name="status", description="Show bot status and current recording info")
    async def status(self, interaction: discord.Interaction):
        voice_client = interaction.guild.voice_client if interaction.guild else None

        embed = discord.Embed(
            title="🎙️ Benjamin — Status",
            color=discord.Color.green() if voice_client else discord.Color.default(),
            timestamp=datetime.now(timezone.utc),
        )

        if voice_client and voice_client.is_connected():
            channel = voice_client.channel
            members = [m for m in channel.members if not m.bot]
            embed.add_field(name="Channel", value=channel.name, inline=True)
            embed.add_field(name="Listeners", value=str(len(members)), inline=True)
            embed.add_field(name="Recording", value="✅", inline=True)

            # DAVE status
            conn = voice_client._connection
            dave_status = "✅ Active" if conn.can_encrypt else "❌ Inactive"
            embed.add_field(name="DAVE E2EE", value=dave_status, inline=True)
        else:
            embed.add_field(name="Status", value="Not in VC", inline=False)

        from audio.storage import get_all_users_for_date, get_user_recordings, get_new_speech_segments

        users = get_all_users_for_date()
        new_segs = get_new_speech_segments()
        total_files = sum(len(get_user_recordings(u)) for u in users)
        embed.add_field(name="Users Today", value=str(len(users)), inline=True)
        embed.add_field(name="Total Files", value=str(total_files), inline=True)
        embed.add_field(name="New (unprocessed)", value=str(len(new_segs)), inline=True)

        await interaction.response.send_message(embed=embed)
