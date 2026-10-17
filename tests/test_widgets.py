import unittest

from podwatch import theme
from podwatch.widgets import (chart, chart_rows, lerp_color, metrics_text, parse_progress, resample, smooth,
                              style_log_line, trend)


class ChartTest(unittest.TestCase):
    def test_shape(self):
        rows = chart([float(i) for i in range(50)], width=30, height=3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(r) == 30 for r in rows))

    def test_falling_series_starts_high_and_ends_low(self):
        rows = chart([3.0, 2.0, 1.0], width=3, height=2)
        self.assertEqual(rows[0][0], "█")  # the first value reaches the top row
        self.assertEqual(rows[0][-1], " ")  # the last one does not
        self.assertEqual(rows[1][-1], "▁")

    def test_fixed_scale(self):
        rows = chart([50.0, 50.0], 4, 2, low=0, high=100)  # half way up a 0 to 100 scale
        self.assertEqual(rows, ["▁▁▁▁", "████"])
        self.assertEqual(chart([500.0, -9.0], 2, 1, low=0, high=100), ["█▁"])  # values outside the scale are clamped

    def test_needs_two_points(self):
        self.assertEqual(chart([1.0], 10, 3), [])
        self.assertEqual(chart([], 10, 3), [])

    def test_flat_series_does_not_divide_by_zero(self):
        self.assertEqual(len(chart([2.0] * 10, 10, 2)), 2)


class SmoothTest(unittest.TestCase):
    def test_same_length_and_flattens_noise(self):
        noisy = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0]
        out = smooth(noisy)
        self.assertEqual(len(out), len(noisy))
        self.assertLess(max(out) - min(out), 2.0)

    def test_keeps_a_straight_line(self):
        self.assertEqual(smooth([2.0] * 6), [2.0] * 6)


class ColorTest(unittest.TestCase):
    def test_lerp_ends_and_middle(self):
        self.assertEqual(lerp_color("#000000", "#ffffff", 0), "#000000")
        self.assertEqual(lerp_color("#000000", "#ffffff", 1), "#ffffff")
        self.assertEqual(lerp_color("#000000", "#fefefe", 0.5), "#7f7f7f")

    def test_chart_rows_keep_the_text_and_colour_the_top_cell_differently(self):
        rows = ["  ▅", "▃██"]
        texts = chart_rows(rows)
        self.assertEqual([t.plain for t in texts], rows)
        top, below = str(texts[0].spans[0].style), str(texts[1].spans[-1].style)
        self.assertNotEqual(top, below)


class ResampleTest(unittest.TestCase):
    def test_exact_width_both_ways(self):
        self.assertEqual(len(resample(list(map(float, range(100))), 20)), 20)
        self.assertEqual(len(resample([1.0, 2.0, 3.0], 20)), 20)

    def test_buckets_are_means(self):
        self.assertEqual(resample([1.0, 3.0, 5.0, 7.0], 2), [2.0, 6.0])


class TrendTest(unittest.TestCase):
    def test_signs(self):
        self.assertLess(trend([3.0, 3.0, 2.5, 2.0, 1.5, 1.0]), 0)
        self.assertGreater(trend([1.0, 1.0, 1.5, 2.0, 2.5, 3.0]), 0)

    def test_too_short(self):
        self.assertIsNone(trend([1.0, 2.0]))


class ProgressTest(unittest.TestCase):
    def test_tqdm_line_with_timestamp(self):
        line = "2026-10-17 17:00:46,349 Running generate_until requests:  76%|███████▒  | 1009/1319 [18:23<05:03, 1.02it/s]"
        p = parse_progress(line)
        self.assertEqual((p.label, p.percent, p.done, p.total, p.eta, p.rate),
                         ("Running generate_until requests", 76, 1009, 1319, "05:03", "1.02it/s"))

    def test_line_without_timing(self):
        p = parse_progress("Fetching 8 files: 100%|██████████| 8/8")
        self.assertEqual((p.label, p.percent, p.eta), ("Fetching 8 files", 100, "?"))

    def test_plain_lines_are_not_progress(self):
        self.assertIsNone(parse_progress("2026-10-17 17:00:46 INFO train: step 5 loss 1.2"))
        self.assertIsNone(parse_progress("saved 50% of the cache"))


class TextTest(unittest.TestCase):
    def test_metrics_text(self):
        text = metrics_text({"step": 12, "loss": 1.23456, "note": "x", "ok": True, "lr": 0.0001})
        self.assertEqual(text.plain, "step 12 │ loss 1.235 │ lr 1.0e-04")

    def test_metrics_are_limited(self):
        text = metrics_text({"step": 1, **{f"m{i}": float(i) for i in range(10)}})
        self.assertEqual(text.plain.count("│"), 5)

    def test_errors_stand_out(self):
        text = style_log_line("2026-10-17 17:00:46 ERROR train: CUDA out of memory")
        self.assertTrue(any(theme.BAD in str(span.style) for span in text.spans))

    def test_plain_line_keeps_its_text(self):
        line = "2026-10-17 17:00:46 INFO train: step 5 loss 1.2"
        self.assertEqual(style_log_line(line).plain, line)


if __name__ == "__main__":
    unittest.main()
