import unittest

from pascribe_app.transcription import group_timed_words


class TranscriptionPostProcessingTests(unittest.TestCase):
    def test_group_timed_words_splits_at_long_pause(self):
        words = [
            {"text": "hello", "start": 0, "end": 400},
            {"text": "there", "start": 500, "end": 900},
            {"text": "next", "start": 2300, "end": 2700},
        ]
        self.assertEqual(group_timed_words(words), [
            {"start_s": 0.0, "end_s": 0.9, "text": "hello there"},
            {"start_s": 2.3, "end_s": 2.7, "text": "next"},
        ])

    def test_group_timed_words_ignores_invalid_entries(self):
        words = [
            {"text": "valid", "start": 0, "end": 100},
            {"text": None, "start": 200, "end": 300},
        ]
        self.assertEqual(group_timed_words(words), [
            {"start_s": 0.0, "end_s": 0.1, "text": "valid"},
        ])


if __name__ == "__main__":
    unittest.main()
