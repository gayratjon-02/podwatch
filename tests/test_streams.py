import threading
import time
import unittest

from podwatch.streams import GPU_HISTORY, ProcessStream, Stream


class ReceiveTest(unittest.TestCase):
    def test_history_is_the_mean_over_gpus(self):
        stream = Stream("p", threading.Event())
        stream.receive({"gpus": [[100.0, 1.0, 2.0], [50.0, 1.0, 2.0]]})
        self.assertEqual(stream.gpu_history, (75.0,))

    def test_snapshots_without_gpus_leave_the_history_alone(self):
        stream = Stream("p", threading.Event())
        stream.receive({"gpus": [[10.0, 1.0, 2.0]]})
        stream.receive({"error": "not reachable"})
        stream.receive({"log": {"name": "a.log", "age": 1.0, "lines": []}})
        self.assertEqual(stream.gpu_history, (10.0,))

    def test_history_is_capped(self):
        stream = Stream("p", threading.Event())
        for i in range(GPU_HISTORY + 30):
            stream.remember(float(i))
        self.assertEqual(len(stream.gpu_history), GPU_HISTORY)
        self.assertEqual(stream.gpu_history[-1], GPU_HISTORY + 29.0)


class ProcessStreamTest(unittest.TestCase):
    def test_noise_and_non_object_json_are_ignored(self):
        halt = threading.Event()
        command = ["bash", "-c", "cat >/dev/null; printf 'banner\\n123\\n{\"gpus\": [[50.0, 1, 2]]}\\n'"]
        stream = ProcessStream("t", command, "", halt)
        stream.start()
        try:
            deadline = time.time() + 5
            while time.time() < deadline and not stream.gpu_history:
                time.sleep(0.05)
            self.assertEqual(stream.gpu_history, (50.0,))
        finally:
            halt.set()
            stream.close()
            stream.join(timeout=5)

    def test_missing_program_is_reported(self):
        halt = threading.Event()
        stream = ProcessStream("t", ["definitely-not-a-program-xyz"], "", halt)
        stream.start()
        try:
            deadline = time.time() + 5
            while time.time() < deadline and "cannot start" not in stream.data.get("error", ""):
                time.sleep(0.05)
            self.assertIn("cannot start", stream.data["error"])
        finally:
            halt.set()
            stream.close()
            stream.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
