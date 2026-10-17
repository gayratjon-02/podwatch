"""Runs the real agent through a local `bash -s`, the same way a pod would, against a throwaway directory."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from podwatch.agent import build_agent


def first_snapshot(root: Path, train_log: str = "trainer_log.jsonl") -> dict:
    proc = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    try:
        proc.stdin.write(build_agent(str(root), 3, train_log))
        proc.stdin.close()
        return json.loads(proc.stdout.readline())
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


class AgentTest(unittest.TestCase):
    def test_reads_newest_log_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run1").mkdir()
            (root / "run1" / "driver.log").write_text("one\ntwo\nthree\nfour\n")
            (root / "run1" / "trainer_log.jsonl").write_text(
                "\n".join(json.dumps({"step": i, "loss": 2.0 - i / 10}) for i in range(5)) + "\nnot json\n")
            snap = first_snapshot(root)
        self.assertEqual(snap["log"]["name"], "driver.log")
        self.assertEqual(snap["log"]["lines"], ["two", "three", "four"])
        self.assertEqual(snap["train"]["run"], "run1/trainer_log.jsonl")
        self.assertEqual(snap["train"]["records"][-1]["step"], 4)

    def test_progress_bar_updates_collapse_to_the_newest_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bars = "\r".join(f"Scoring:  {n}%|####| {n}/100 [00:0{n % 10}<00:09, 1.00it/s]" for n in (10, 20, 30))
            (root / "run.log").write_text(f"INFO start\n{bars}\nINFO middle\n{bars}\r")
            snap = first_snapshot(root)
        lines = snap["log"]["lines"]
        self.assertEqual(lines[:2], ["INFO start", "INFO middle"])
        self.assertEqual(len(lines), 3)
        self.assertIn("30%|", lines[-1])

    def test_skips_old_files_and_vendored_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".venv").mkdir()
            (root / ".venv" / "noise.log").write_text("noise\n")
            snap = first_snapshot(root)
        self.assertNotIn("log", snap)
        self.assertNotIn("train", snap)

    def test_odd_names_are_not_interpreted_by_a_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "it's $(touch pwned) dir"
            root.mkdir()
            (root / "a.log").write_text("hello\n")
            snap = first_snapshot(root)
            self.assertFalse((Path.cwd() / "pwned").exists())
        self.assertEqual(snap["log"]["lines"], ["hello"])


if __name__ == "__main__":
    unittest.main()
