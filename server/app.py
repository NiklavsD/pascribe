"""
Pascribe Server — Receives transcripts and triggers analysis via OpenClaw.
Public at talk.benbox.dev via Cloudflare tunnel.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

# Simple token in URL path (no auth header needed from client)
API_TOKEN = os.environ.get("PASCRIBE_TOKEN", "8f338b6c289a4b9898a221bfa3081c64")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "pascribe"})


@app.route("/transcript", methods=["POST"])
def receive_transcript():
    """
    Receive a transcript. Accepts both JSON and plain text.
    
    Auth: Bearer token in header OR ?token= query param.
    
    JSON format:
    {
        "date": "2026-02-28",
        "duration_minutes": 120,
        "participants": ["Niklavs", ...],
        "source": "discord",
        "transcript": "full text here..."
    }
    
    Plain text: just the transcript in the request body.
    """
    if not _check_auth():
        return jsonify({"error": "unauthorized"}), 401

    content_type = request.content_type or ""
    
    if "json" in content_type:
        data = request.get_json(silent=True) or {}
    else:
        # Plain text body
        text = request.get_data(as_text=True)
        data = {"transcript": text}
    
    transcript_text = data.get("transcript", "")
    segments = data.get("segments", [])
    
    if not transcript_text and not segments:
        return jsonify({"error": "empty transcript"}), 400

    # Build full text from segments if needed
    if not transcript_text and segments:
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "")
            time_s = seg.get("time", "")
            text = seg.get("text", "")
            prefix = f"[{time_s}] {speaker}:" if time_s and speaker else (speaker + ":" if speaker else "")
            lines.append(f"{prefix} {text}" if prefix else text)
        transcript_text = "\n".join(lines)
        data["transcript"] = transcript_text

    date_str = data.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{date_str}_{timestamp}.json"
    filepath = TRANSCRIPTS_DIR / filename
    
    # Enrich
    data.setdefault("date", date_str)
    data["_received_at"] = datetime.now(timezone.utc).isoformat()
    data["_filename"] = filename
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Write trigger for cron pickup
    trigger_dir = Path(__file__).parent / "pending"
    trigger_dir.mkdir(exist_ok=True)
    (trigger_dir / f"{Path(filepath).stem}.trigger").write_text(str(filepath))

    word_count = len(transcript_text.split())

    return jsonify({
        "status": "received",
        "filename": filename,
        "word_count": word_count,
        "analysis": "queued"
    }), 200


def _check_auth():
    """Accept Bearer header OR ?token= query param."""
    # Query param
    if request.args.get("token") == API_TOKEN:
        return True
    # Bearer header
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {API_TOKEN}" or auth == API_TOKEN:
        return True
    return False


if __name__ == "__main__":
    port = int(os.environ.get("PASCRIBE_PORT", 3089))
    print(f"🎙️ Pascribe Server listening on :{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
