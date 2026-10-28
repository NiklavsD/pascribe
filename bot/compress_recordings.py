"""Convert recorded WAV speech segments to Ogg Opus, verify each one, and
optionally delete the WAV once its Opus copy checks out.

Uses the same encoder settings as audio/storage.py, so converted days look
like days recorded after the switch. A WAV is only deleted when its Opus copy
decodes to the same number of samples, at the same level (±1 dB), with the
same loudness envelope (20 ms RMS correlation ≥ 0.97). Anything that fails
stays a WAV and is listed in the manifest.

Skips today's UTC day and files touched in the last 10 minutes, so a bot still
writing WAVs is left alone. Safe to rerun: existing .opus copies are
re-verified rather than re-encoded.

Needs numpy (system python3 has it; the bot venv doesn't):

    python3 compress_recordings.py                 # convert + verify, keep WAVs
    python3 compress_recordings.py --delete        # ...and delete verified WAVs
    python3 compress_recordings.py --days 2026-03-03 2026-03-04
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Keep in sync with OPUS_ENCODE_ARGS in audio/storage.py.
OPUS_ENCODE_ARGS = ["-c:a", "libopus", "-b:a", "64k", "-application", "audio", "-f", "ogg"]
MAX_SAMPLE_DRIFT_S = 0.001
MAX_LEVEL_DIFF_DB = 1.0  # real segments land within ±0.2 dB
MIN_ENVELOPE_CORR = 0.97  # real segments land at ≥ 0.99
SILENT_DBFS = -70.0  # below this, level/envelope checks are meaningless
FRAME_S = 0.02


def _ffmpeg(*args: str) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", *args],
        capture_output=True, check=True, timeout=3600,
    ).stdout


def _dbfs(x: np.ndarray) -> float:
    if not len(x):
        return -200.0
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) / 32768 + 1e-12)


def _envelope(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    return np.sqrt(np.mean(x[: n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1))


def verify(wav: Path, opus: Path) -> tuple[bool, str, float]:
    """Compare an Opus copy against its WAV. Returns (ok, detail, seconds)."""
    with wave.open(str(wav), "rb") as w:
        rate, chans, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if width != 2:
            return False, f"sample width {width}", 0.0
        src = np.frombuffer(w.readframes(frames), dtype=np.int16)
    seconds = frames / rate
    dec = np.frombuffer(
        _ffmpeg("-i", str(opus), "-f", "s16le", "-ar", str(rate), "-ac", str(chans), "pipe:1"),
        dtype=np.int16,
    )
    drift = (len(dec) - len(src)) / (rate * chans)
    if abs(drift) > MAX_SAMPLE_DRIFT_S:
        return False, f"length differs by {drift:+.3f}s", seconds
    n = min(len(src), len(dec))
    src, dec = src[:n], dec[:n]
    src_db, dec_db = _dbfs(src), _dbfs(dec)
    if src_db < SILENT_DBFS:
        return True, f"silent ({src_db:.0f} dBFS)", seconds
    level = dec_db - src_db
    if abs(level) > MAX_LEVEL_DIFF_DB:
        return False, f"level differs by {level:+.2f} dB", seconds
    frame = max(1, int(rate * FRAME_S)) * chans
    ea, eb = _envelope(src, frame), _envelope(dec, frame)
    if len(ea) >= 3 and ea.std() > 0 and eb.std() > 0:
        corr = float(np.corrcoef(ea, eb)[0, 1])
        if corr < MIN_ENVELOPE_CORR:
            return False, f"envelope correlation {corr:.3f}", seconds
    else:
        corr = float("nan")
    return True, f"level {level:+.2f} dB, envelope {corr:.4f}", seconds


def convert(wav: Path, delete: bool) -> tuple[str, Path, int, int, float, str]:
    """Returns (status, wav, wav_bytes, opus_bytes, seconds, detail)."""
    opus = wav.with_suffix(".opus")
    wav_bytes = wav.stat().st_size
    try:
        if not opus.exists():
            tmp = opus.with_name(opus.name + ".part")
            try:
                _ffmpeg("-y", "-i", str(wav), *OPUS_ENCODE_ARGS, str(tmp))
                st = wav.stat()
                os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
                os.replace(tmp, opus)
            finally:
                tmp.unlink(missing_ok=True)
        ok, detail, seconds = verify(wav, opus)
    except subprocess.CalledProcessError as e:
        return "error", wav, wav_bytes, 0, 0.0, (e.stderr or b"").decode(errors="replace").strip()[:200]
    except Exception as e:  # noqa: BLE001
        return "error", wav, wav_bytes, 0, 0.0, repr(e)[:200]
    opus_bytes = opus.stat().st_size
    if not ok:
        return "failed", wav, wav_bytes, opus_bytes, seconds, detail
    if delete:
        wav.unlink()
        return "deleted", wav, wav_bytes, opus_bytes, seconds, detail
    return "verified", wav, wav_bytes, opus_bytes, seconds, detail


def find_wavs(root: Path, days: list[str] | None, min_age_s: float) -> list[Path]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cutoff = time.time() - min_age_s
    out = []
    for day_dir in sorted(root.iterdir()):
        if not day_dir.is_dir() or day_dir.name == today:
            continue
        if days and day_dir.name not in days:
            continue
        for wav in sorted(day_dir.glob("*/*.wav")):
            if wav.stat().st_mtime < cutoff:
                out.append(wav)
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(os.getenv("RECORDINGS_DIR", here / "recordings")))
    ap.add_argument("--days", nargs="*", help="only these YYYY-MM-DD day folders")
    ap.add_argument("--delete", action="store_true", help="delete each WAV once its Opus copy verifies")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--min-age", type=float, default=600, help="skip WAVs modified this recently (s)")
    ap.add_argument("--manifest", type=Path,
                    default=here / "logs" / f"compress-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.tsv")
    a = ap.parse_args()

    root = a.root.resolve()
    wavs = find_wavs(root, a.days, a.min_age)
    total_bytes = sum(w.stat().st_size for w in wavs)
    print(f"{len(wavs)} WAVs, {total_bytes / 1e9:.1f} GB under {root}; "
          f"{'deleting' if a.delete else 'keeping'} verified WAVs; manifest {a.manifest}", flush=True)

    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    wav_sum = opus_sum = 0
    secs_sum = 0.0
    lock = threading.Lock()
    started = time.time()
    with open(a.manifest, "a", encoding="utf-8") as man, ThreadPoolExecutor(a.workers) as pool:
        man.write("status\twav\twav_bytes\topus_bytes\tseconds\tdetail\n")
        futures = [pool.submit(convert, w, a.delete) for w in wavs]
        for i, fut in enumerate(as_completed(futures), 1):
            status, wav, wb, ob, secs, detail = fut.result()
            with lock:
                counts[status] = counts.get(status, 0) + 1
                if status in ("verified", "deleted"):
                    wav_sum += wb
                    opus_sum += ob
                    secs_sum += secs
                man.write(f"{status}\t{wav.relative_to(root)}\t{wb}\t{ob}\t{secs:.3f}\t{detail}\n")
                if status in ("failed", "error"):
                    print(f"{status}: {wav.relative_to(root)}: {detail}", flush=True)
            if i % 500 == 0 or i == len(futures):
                man.flush()
                rate = i / max(1e-9, time.time() - started)
                eta = (len(futures) - i) / rate if rate else 0
                print(f"[{i}/{len(futures)}] {counts} {wav_sum / 1e9:.1f} GB → {opus_sum / 1e9:.2f} GB, "
                      f"eta {eta / 60:.0f} min", flush=True)

    ratio = wav_sum / opus_sum if opus_sum else 0
    print(f"DONE in {(time.time() - started) / 60:.1f} min: {counts}; {secs_sum / 3600:.1f} h of audio, "
          f"{wav_sum / 1e9:.1f} GB WAV → {opus_sum / 1e9:.2f} GB Opus ({ratio:.1f}x)", flush=True)
    return 1 if counts.get("failed") or counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
