"""
Pascribe Analyzer — Called by OpenClaw to process transcripts.
Reads a transcript file and outputs structured analysis for sub-agents.
"""

import json
import sys
from pathlib import Path


def load_transcript(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


def build_full_text(data: dict) -> str:
    """Build readable transcript from either flat text or segments."""
    if data.get("transcript"):
        return data["transcript"]
    
    segments = data.get("segments", [])
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        time = seg.get("time", "")
        text = seg.get("text", "")
        prefix = f"[{time}] {speaker}:" if time else f"{speaker}:"
        lines.append(f"{prefix} {text}")
    return "\n".join(lines)


def extract_analysis_prompts(data: dict, full_text: str) -> list[dict]:
    """Generate analysis task prompts for sub-agents."""
    date = data.get("date", "unknown")
    participants = ", ".join(data.get("participants", ["unknown"]))
    source = data.get("source", "unknown")
    
    context = f"Date: {date}\nParticipants: {participants}\nSource: {source}\n\nTranscript:\n{full_text}"
    
    tasks = [
        {
            "name": "topics_and_ideas",
            "prompt": f"""Analyze this Discord voice transcript and extract:

1. **Key Topics Discussed** — List every distinct topic/subject covered
2. **Ideas & Projects** — Any business ideas, project proposals, or creative concepts mentioned
3. **Action Items** — Things people said they'd do or should do
4. **Interesting Links/References** — Any tools, websites, people, or resources mentioned

For each idea/project, provide:
- Brief description
- Who mentioned it
- Potential value/viability (your assessment)
- Suggested next steps

Be thorough — don't miss any topic, even brief ones.

{context}"""
        },
        {
            "name": "knowledge_extraction",
            "prompt": f"""Extract all knowledge and learnable content from this Discord voice transcript:

1. **Technical Knowledge** — Programming, tools, frameworks, techniques discussed
2. **Business Insights** — Market observations, strategies, lessons shared
3. **Life Lessons / Wisdom** — Personal insights, advice, experiences shared
4. **Interesting Facts** — Anything factual or informative mentioned
5. **Recommendations** — Books, tools, services, approaches recommended by participants

For each item, note who shared it and provide additional context/research if relevant.

{context}"""
        },
        {
            "name": "research_deep_dive",
            "prompt": f"""From this Discord voice transcript, identify the 2-3 most promising or interesting topics/ideas discussed, then:

1. **Research each topic** — Search the web for current state, competitors, market size
2. **Expand the idea** — How could it be developed further? What's the MVP?
3. **Provide resources** — Links, tools, frameworks relevant to pursuing it
4. **Reality check** — Is this feasible? What are the challenges?

Focus on actionable, concrete research that turns casual conversation into potential projects.

{context}"""
        },
    ]
    
    return tasks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <transcript.json>")
        sys.exit(1)
    
    data = load_transcript(sys.argv[1])
    full_text = build_full_text(data)
    tasks = extract_analysis_prompts(data, full_text)
    
    print(json.dumps(tasks, indent=2))
