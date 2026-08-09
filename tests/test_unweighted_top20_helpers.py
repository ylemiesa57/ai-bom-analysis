"""Tests for the small pure helper functions in ai_unweighted_top20.py.

None of these had any test coverage before, including
_get_node_color()'s crc32-based determinism fix (see
tests/test_squash_params_default.py's sibling history for the
hash()-randomization bug this guards against) -- nothing was actually
verifying that fix stays in place. Only exercises pure helpers; no
graph generation, plotting, or file I/O beyond a tmp directory for
_find_latest_csv.
"""
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_bom.ai_unweighted_top20 import (
    _find_latest_csv,
    _format_node_label,
    _get_metric_acronym,
    _get_metric_full_name,
    _get_node_color,
)


class TestFormatNodeLabel(unittest.TestCase):
    def test_ai_node_prefix_is_converted(self):
        self.assertEqual(_format_node_label("ai_node_42"), "N-42")

    def test_non_matching_name_is_returned_unchanged(self):
        self.assertEqual(_format_node_label("some_other_node"), "some_other_node")


class TestGetNodeColor(unittest.TestCase):
    def test_same_node_name_gives_same_color_within_process(self):
        self.assertEqual(_get_node_color("ai_node_7"), _get_node_color("ai_node_7"))

    def test_lighter_flag_changes_the_color(self):
        # Same node, but lighter=True should shift saturation/value, so the
        # two hex colors must differ.
        normal = _get_node_color("ai_node_7", lighter=False)
        lighter = _get_node_color("ai_node_7", lighter=True)
        self.assertNotEqual(normal, lighter)

    def test_returns_a_valid_hex_color(self):
        color = _get_node_color("ai_node_3")
        self.assertRegex(color, r"^#[0-9a-f]{6}$")

    def test_non_ai_node_name_is_deterministic_across_processes(self):
        # This is the actual regression case: node names that don't match
        # the "ai_node_<id>" pattern fall back to zlib.crc32 specifically
        # because Python's built-in hash() for str is randomized per
        # process (PYTHONHASHSEED). Spawn two fresh interpreters with
        # different hash seeds and confirm the color is still identical --
        # this is the scenario the old hash()-based code would have failed.
        script = (
            "import sys; sys.path.insert(0, %r); "
            "from ai_bom.ai_unweighted_top20 import _get_node_color; "
            "print(_get_node_color('custom-node-name'))"
        ) % str(Path(__file__).resolve().parents[1])

        def run_with_seed(seed):
            env = {"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"}
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip()

        color_seed_1 = run_with_seed(1)
        color_seed_2 = run_with_seed(2)
        self.assertEqual(color_seed_1, color_seed_2)


class TestMetricNameHelpers(unittest.TestCase):
    def test_acronym_for_pagerank(self):
        self.assertEqual(_get_metric_acronym("PageRank"), "PRC")

    def test_acronym_for_betweenness(self):
        self.assertEqual(_get_metric_acronym("Betweenness"), "BC")

    def test_acronym_is_case_insensitive(self):
        self.assertEqual(_get_metric_acronym("pagerank"), "PRC")

    def test_acronym_falls_back_to_input_for_unknown_metric(self):
        self.assertEqual(_get_metric_acronym("Closeness"), "Closeness")

    def test_full_name_for_pagerank(self):
        self.assertEqual(_get_metric_full_name("pagerank"), "PageRank")

    def test_full_name_for_betweenness(self):
        self.assertEqual(_get_metric_full_name("BETWEENNESS"), "Betweenness")

    def test_full_name_falls_back_to_input_for_unknown_metric(self):
        self.assertEqual(_get_metric_full_name("Closeness"), "Closeness")


class TestFindLatestCsv(unittest.TestCase):
    def test_returns_none_when_no_files_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_find_latest_csv(Path(tmp), "betweenness"))

    def test_returns_most_recently_modified_matching_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            older = tmp_path / "top20_betweenness_20250101_000000.csv"
            newer = tmp_path / "top20_betweenness_20250102_000000.csv"
            older.write_text("node,Betweenness\n")
            time.sleep(0.01)
            newer.write_text("node,Betweenness\n")

            result = _find_latest_csv(tmp_path, "betweenness")
            self.assertEqual(result, newer)

    def test_ignores_files_for_a_different_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "top20_pagerank_20250101_000000.csv").write_text("x")
            self.assertIsNone(_find_latest_csv(tmp_path, "betweenness"))


if __name__ == "__main__":
    unittest.main()
