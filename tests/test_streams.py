import threading
import time
import unittest

from pathlib import Path

from podwatch.config import PodConfig
from podwatch.streams import GPU_HISTORY, ProcessStream, Stream, ssh_command


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

    def test_unchanged_files_are_filled_in_from_the_previous_snapshot(self):
        stream = Stream("p", threading.Event())
        stream.receive({"log": {"name": "a.log", "age": 1.0, "lines": ["x", "y"]},
                        "train": {"run": "r", "age": 1.0, "records": [{"step": 1}]}})
        stream.receive({"log": {"name": "a.log", "age": 2.0, "same": True},
                        "train": {"run": "r", "age": 2.0, "same": True}})
        self.assertEqual(stream.data["log"]["lines"], ["x", "y"])
        self.assertEqual(stream.data["train"]["records"], [{"step": 1}])
        self.assertNotIn("same", stream.data["log"])
        self.assertEqual(stream.data["log"]["age"], 2.0)  # the live fields still update

    def test_history_is_capped(self):
        stream = Stream("p", threading.Event())
        for i in range(GPU_HISTORY + 30):
            stream.remember(float(i))
        self.assertEqual(len(stream.gpu_history), GPU_HISTORY)
        self.assertEqual(stream.gpu_history[-1], GPU_HISTORY + 29.0)


class SshCommandTest(unittest.TestCase):
    def test_bare_host_leaves_user_port_and_keys_to_ssh_config(self):
        command = ssh_command(PodConfig(name="a", host="prod-gpu-1"))
        self.assertEqual(command[-3:], ["prod-gpu-1", "bash", "-s"])
        for flag in ("-p", "-J", "-i"):
            self.assertNotIn(flag, command)

    def test_everything_set(self):
        command = ssh_command(PodConfig(name="a", host="10.0.0.5", user="ops", port=2222, key="~/id_x",
                                        jump="ops@bastion"))
        self.assertEqual(command[command.index("-J") + 1], "ops@bastion")
        self.assertEqual(command[command.index("-p") + 1], "2222")
        self.assertEqual(command[command.index("-i") + 1], str(Path("~/id_x").expanduser()))
        self.assertEqual(command[-3:], ["ops@10.0.0.5", "bash", "-s"])

    def test_always_non_interactive(self):
        command = ssh_command(PodConfig(name="a", host="h"))
        self.assertIn("BatchMode=yes", command)


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
