import io
import threading
import unittest

from rich.console import Console

from podwatch.mock import mock_streams
from podwatch.render import chart_height, frame, pod_state


def text_of(streams, size, mock=True) -> str:
    console = Console(file=io.StringIO(), width=size[0], height=size[1], color_system=None, force_terminal=True)
    console.print(frame(streams, "loss", mock, size))
    return console.file.getvalue()


def ready_streams(count: int):
    streams = mock_streams(count, 30, 1, threading.Event())
    for stream in streams:
        stream.receive(stream.tick())
    return streams


class PodStateTest(unittest.TestCase):
    def test_error_states(self):
        self.assertEqual(pod_state({"error": "connecting…"})[0], "connecting")
        self.assertEqual(pod_state({"error": "not reachable"})[0], "unreachable")

    def test_fresh_log_means_running(self):
        self.assertEqual(pod_state({"log": {"name": "a.log", "age": 3.0, "lines": []}})[0], "running")

    def test_old_log_means_idle(self):
        self.assertEqual(pod_state({"log": {"name": "a.log", "age": 600.0, "lines": []}})[0], "idle")

    def test_no_files(self):
        self.assertEqual(pod_state({"gpus": [[0.0, 1.0, 2.0]]})[0], "no recent logs")


class ChartHeightTest(unittest.TestCase):
    def test_log_keeps_its_lines(self):
        self.assertEqual([chart_height(room) for room in (30, 7, 6, 5, 4, 0)], [3, 3, 2, 1, 0, 0])


class FrameTest(unittest.TestCase):
    def test_wide_terminal_has_every_pod(self):
        out = text_of(ready_streams(4), (200, 50))
        for name in ("pod1", "pod2", "pod3", "pod4", "MOCK DATA", "GPU", "VRAM", "loss"):
            self.assertIn(name, out)

    def test_progress_line_becomes_a_bar(self):
        out = text_of(ready_streams(2), (200, 40))
        self.assertIn("Running generate_until requests", out)
        self.assertIn("eta", out)

    def test_narrow_and_short_terminals_do_not_crash(self):
        for size in ((80, 24), (60, 14), (120, 12)):
            self.assertIn("pod1", text_of(ready_streams(3), size))

    def test_unreachable_pod_is_drawn(self):
        streams = ready_streams(2)
        streams[1].data = {"error": "not reachable (stopped or being recreated), retrying"}
        self.assertIn("unreachable", text_of(streams, (160, 30)))

    def test_gpu_history_fills_panels_without_metrics(self):
        out = text_of(ready_streams(4), (200, 50))
        self.assertEqual(out.count("GPU util"), 1)  # only the evaluating pod has no loss to chart
        self.assertIn("min", out)

    def test_odd_number_of_pods_leaves_the_last_cell_empty(self):
        out = text_of(ready_streams(5), (200, 60))
        self.assertIn("pod5", out)
        self.assertNotIn("Layout", out)  # Rich's placeholder box
        self.assertNotIn("(100 x", out)


if __name__ == "__main__":
    unittest.main()
