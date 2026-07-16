import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pascribe_app.recordings import assign_speaker, open_daily_snapshot


class RecordingSnapshotTests(unittest.TestCase):
    def test_disk_backed_snapshot_mixes_and_assigns_speakers(self):
        with tempfile.TemporaryDirectory() as directory:
            day = "2026-07-16"
            day_dir = Path(directory) / day
            day_dir.mkdir()
            (day_dir / "meta.json").write_text(json.dumps({
                "started": "2026-07-16T09:00:00",
                "sample_rate": 100,
                "dtype": "int16",
            }), encoding="utf-8")
            mic = np.concatenate([
                np.full(100, 20000, dtype=np.int16),
                np.full(100, 1000, dtype=np.int16),
            ])
            system = np.concatenate([
                np.full(100, 1000, dtype=np.int16),
                np.full(100, 20000, dtype=np.int16),
            ])
            mic.tofile(day_dir / "mic.raw")
            system.tofile(day_dir / "sys.raw")

            snapshot = open_daily_snapshot(Path(directory), day)
            self.assertIsNotNone(snapshot)
            try:
                self.assertEqual(snapshot.total_samples, 200)
                mixed = snapshot.mixed_reader()(0, 50)
                self.assertEqual(len(mixed), 50)
                self.assertEqual(
                    assign_speaker(0, 1, snapshot.mic, snapshot.system, 100),
                    "you",
                )
                self.assertEqual(
                    assign_speaker(1, 2, snapshot.mic, snapshot.system, 100),
                    "discord",
                )
            finally:
                snapshot.close()

    def test_sample_limit_creates_stable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            day = "2026-07-16"
            day_dir = Path(directory) / day
            day_dir.mkdir()
            (day_dir / "meta.json").write_text(json.dumps({
                "started": "2026-07-16T09:00:00",
                "sample_rate": 10,
                "dtype": "int16",
            }), encoding="utf-8")
            np.arange(100, dtype=np.int16).tofile(day_dir / "mic.raw")
            snapshot = open_daily_snapshot(
                Path(directory),
                day,
                sample_limits={"mic": 40},
            )
            self.assertEqual(snapshot.total_samples, 40)
            snapshot.close()


if __name__ == "__main__":
    unittest.main()
