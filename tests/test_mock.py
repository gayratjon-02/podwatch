import threading
import unittest

from podwatch.mock import KINDS, MAX_PODS, MockStream, mock_streams
from podwatch.render import pod_state


def make(kind: str) -> MockStream:
    return MockStream("p", kind, lines=8, seed=1, halt=threading.Event())


class MockTest(unittest.TestCase):
    def test_train_has_gpu_log_and_falling_records(self):
        stream = make("train")
        for _ in range(30):
            snap = stream.tick()
        self.assertEqual(len(snap["gpus"][0]), 3)
        self.assertGreater(len(snap["train"]["records"]), 5)
        self.assertLessEqual(len(snap["log"]["lines"]), 8)
        self.assertEqual(pod_state(snap)[0], "running")

    def test_eval_has_progress_lines_and_no_metrics(self):
        stream = make("eval")
        for _ in range(5):
            snap = stream.tick()
        self.assertNotIn("train", snap)
        self.assertIn("generate_until", snap["log"]["lines"][-1])

    def test_idle_looks_idle(self):
        self.assertEqual(pod_state(make("idle").tick())[0], "idle")

    def test_gpu_history_is_prefilled(self):
        for kind in ("train", "eval", "idle"):
            self.assertEqual(len(make(kind).gpu_history), 120)
        self.assertEqual(set(make("idle").gpu_history), {0.0})

    def test_pod_count_is_clamped(self):
        halt = threading.Event()
        self.assertEqual(len(mock_streams(0, 8, 1, halt)), 1)
        self.assertEqual(len(mock_streams(99, 8, 1, halt)), MAX_PODS)
        self.assertEqual(len(KINDS), MAX_PODS)


if __name__ == "__main__":
    unittest.main()
