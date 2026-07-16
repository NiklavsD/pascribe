import tempfile
import threading
import unittest
import wave
from pathlib import Path

import numpy as np

from pascribe_app.audio_processing import (
    ProcessingCancelled,
    build_segment_map,
    energy_vad,
    energy_vad_from_reader,
    format_ssmd,
    remap_to_original,
    resample,
    write_wav_segments,
)


class AudioProcessingTests(unittest.TestCase):
    def test_resample_preserves_duration_and_handles_empty_audio(self):
        audio = np.linspace(-1, 1, 8000, dtype=np.float32)
        result = resample(audio, 8000, 16000)
        self.assertEqual(len(result), 16000)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(len(resample(np.array([], dtype=np.float32), 8000, 16000)), 0)

    def test_vad_array_and_streaming_reader_agree(self):
        sample_rate = 16000
        audio = np.zeros(sample_rate * 3, dtype=np.float32)
        tone_time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio[sample_rate:sample_rate * 2] = 0.2 * np.sin(2 * np.pi * 220 * tone_time)

        largest_read = 0

        def reader(start, end):
            nonlocal largest_read
            largest_read = max(largest_read, end - start)
            return audio[start:end]

        direct = energy_vad(audio, sample_rate)
        streamed = energy_vad_from_reader(
            reader,
            len(audio),
            sample_rate,
            chunk_seconds=1,
        )
        self.assertTrue(direct)
        self.assertEqual(len(direct), len(streamed))
        self.assertAlmostEqual(direct[0][0], streamed[0][0], places=2)
        self.assertAlmostEqual(direct[0][1], streamed[0][1], places=2)
        self.assertLess(direct[0][0], 1.0)
        self.assertGreater(direct[0][1], 2.0)
        self.assertLessEqual(largest_read, sample_rate)

    def test_vad_rejects_flat_background_noise(self):
        noise = np.full(16000 * 2, 0.002, dtype=np.float32)
        self.assertEqual(energy_vad(noise, 16000), [])

    def test_segment_map_remaps_stripped_timestamps(self):
        mapping = build_segment_map([(10.0, 12.0), (20.0, 23.0)])
        self.assertEqual(mapping, [(0.0, 2.0, 10.0), (2.0, 5.0, 20.0)])
        self.assertAlmostEqual(remap_to_original(1.5, mapping), 11.5)
        self.assertAlmostEqual(remap_to_original(3.0, mapping), 21.0)
        self.assertAlmostEqual(remap_to_original(2.0, mapping), 12.0)
        self.assertAlmostEqual(
            remap_to_original(2.0, mapping, prefer_next_at_boundary=True),
            20.0,
        )
        self.assertAlmostEqual(remap_to_original(10.0, mapping), 23.0)

    def test_format_ssmd_inserts_paragraph_for_long_pause(self):
        text = format_ssmd([(0.0, 1.0, "Hello"), (4.0, 5.0, "World")])
        self.assertEqual(text, "[00:00] Hello\n\n[00:04] World")

    def test_wav_writer_uses_bounded_reads_and_can_cancel(self):
        sample_rate = 1000
        audio = np.linspace(-0.5, 0.5, sample_rate * 4, dtype=np.float32)
        largest_read = 0

        def reader(start, end):
            nonlocal largest_read
            largest_read = max(largest_read, end - start)
            return audio[start:end]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "speech.wav"
            seconds = write_wav_segments(
                output,
                reader,
                [(0.5, 1.5), (2.0, 3.0)],
                sample_rate,
                chunk_seconds=1,
            )
            self.assertAlmostEqual(seconds, 2.0)
            self.assertLessEqual(largest_read, sample_rate)
            with wave.open(str(output), "rb") as handle:
                self.assertEqual(handle.getnframes(), sample_rate * 2)

            cancelled = threading.Event()
            cancelled.set()
            with self.assertRaises(ProcessingCancelled):
                write_wav_segments(
                    Path(directory) / "cancelled.wav",
                    reader,
                    [(0.0, 1.0)],
                    sample_rate,
                    cancel_event=cancelled,
                )


if __name__ == "__main__":
    unittest.main()
