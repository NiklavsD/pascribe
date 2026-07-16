import threading
import unittest

from pascribe_app.jobs import ExclusiveJobRunner, LatestJobRunner


class JobRunnerTests(unittest.TestCase):
    def test_latest_runner_serializes_and_keeps_only_newest_pending_request(self):
        first_started = threading.Event()
        release_first = threading.Event()
        calls = []
        cancellations = []
        active = 0
        max_active = 0
        lock = threading.Lock()

        def handler(value, cancel_event):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                calls.append(value)
            if value == 1:
                first_started.set()
                self.assertTrue(release_first.wait(2))
                cancellations.append(cancel_event.is_set())
            with lock:
                active -= 1

        runner = LatestJobRunner(handler)
        self.assertTrue(runner.submit(1).started_immediately)
        self.assertTrue(first_started.wait(1))
        self.assertFalse(runner.submit(2).started_immediately)
        third = runner.submit(3)
        self.assertFalse(third.started_immediately)
        self.assertTrue(third.replaced_pending)
        release_first.set()
        self.assertTrue(runner.wait(2))
        self.assertEqual(calls, [1, 3])
        self.assertEqual(cancellations, [True])
        self.assertEqual(max_active, 1)

    def test_latest_runner_cancel_clears_pending_work(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def handler(value, cancel_event):
            calls.append(value)
            started.set()
            release.wait(2)

        runner = LatestJobRunner(handler)
        runner.submit("current")
        self.assertTrue(started.wait(1))
        runner.submit("pending")
        self.assertTrue(runner.cancel(clear_pending=True))
        release.set()
        self.assertTrue(runner.wait(2))
        self.assertEqual(calls, ["current"])

    def test_exclusive_runner_rejects_duplicates_and_exposes_cancellation(self):
        started = threading.Event()
        observed_cancel = threading.Event()

        def handler(cancel_event):
            started.set()
            if cancel_event.wait(2):
                observed_cancel.set()

        runner = ExclusiveJobRunner()
        self.assertTrue(runner.start(handler))
        self.assertTrue(started.wait(1))
        self.assertFalse(runner.start(handler))
        self.assertTrue(runner.cancel())
        self.assertTrue(runner.wait(2))
        self.assertTrue(observed_cancel.is_set())
        self.assertFalse(runner.is_active)


if __name__ == "__main__":
    unittest.main()
