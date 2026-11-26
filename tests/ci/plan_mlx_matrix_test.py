"""Unit tests for .github/ci/plan_mlx_matrix.py: the CI matrix and its
agreement with the tested MLX releases. Standard library only."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "ci"))
import plan_mlx_matrix  # noqa: E402


def write_repository(repository_root, tested_releases_text, ci_releases):
    (repository_root / "cmake").mkdir()
    (repository_root / ".github" / "ci").mkdir(parents=True)
    (repository_root / "cmake" / "mlx_perf_mlx_prefix.cmake").write_text(
        f'set(MLX_PERF_TESTED_MLX_VERSIONS "{tested_releases_text}")\n', encoding="utf-8")
    (repository_root / ".github" / "ci" / "mlx-releases.json").write_text(
        json.dumps({"releases": ci_releases}), encoding="utf-8")


class PlanMlxMatrixTest(unittest.TestCase):
    # --- positive ---

    def test_should_plan_one_entry_per_tested_release_in_file_order(self):
        with tempfile.TemporaryDirectory() as directory_name:
            repository_root = Path(directory_name)
            write_repository(repository_root, "0.32.4;0.32.5",
                             [{"mlx": "0.32.4", "mlx_bridge_tag": "v0.32.4-alpha.6"},
                              {"mlx": "0.32.5", "mlx_bridge_tag": "v0.32.5-alpha.1"}])
            matrix = plan_mlx_matrix.plan_matrix(repository_root)
        self.assertEqual(matrix["include"][1], {"mlx": "0.32.5", "mlx_bridge_tag": "v0.32.5-alpha.1"})
        self.assertEqual(len(matrix["include"]), 2)

    def test_should_agree_with_this_repositorys_own_files(self):
        repository_root = Path(__file__).resolve().parents[2]
        matrix = plan_mlx_matrix.plan_matrix(repository_root)
        self.assertEqual([entry["mlx"] for entry in matrix["include"]],
                         plan_mlx_matrix.read_tested_mlx_releases(
                             (repository_root / "cmake" / "mlx_perf_mlx_prefix.cmake").read_text(encoding="utf-8")))

    # --- edge / boundary ---

    def test_should_ignore_an_empty_item_in_the_cmake_list(self):
        self.assertEqual(plan_mlx_matrix.read_tested_mlx_releases('set(MLX_PERF_TESTED_MLX_VERSIONS "0.32.4;")'),
                         ["0.32.4"])

    def test_should_agree_when_the_orders_differ(self):
        self.assertEqual(plan_mlx_matrix.find_disagreements(
            ["0.32.5", "0.32.4"], [{"mlx": "0.32.4", "mlx_bridge_tag": "a"}, {"mlx": "0.32.5", "mlx_bridge_tag": "b"}]),
            [])

    # --- negative ---

    def test_should_name_a_tested_release_ci_does_not_cover(self):
        disagreements = plan_mlx_matrix.find_disagreements(["0.32.4", "0.33.0"],
                                                           [{"mlx": "0.32.4", "mlx_bridge_tag": "a"}])
        self.assertEqual(len(disagreements), 1)
        self.assertIn("MLX 0.33.0 is tested", disagreements[0])

    def test_should_name_a_ci_release_that_is_not_tested_and_a_duplicate(self):
        disagreements = plan_mlx_matrix.find_disagreements(
            ["0.32.4"], [{"mlx": "0.32.4", "mlx_bridge_tag": "a"}, {"mlx": "0.32.4", "mlx_bridge_tag": "b"},
                         {"mlx": "0.31.0", "mlx_bridge_tag": "c"}])
        self.assertTrue(any("0.31.0 is in .github/ci/mlx-releases.json but not" in message for message in disagreements))
        self.assertTrue(any("0.32.4 appears more than once" in message for message in disagreements))

    def test_should_refuse_an_entry_without_a_bridge_tag(self):
        with self.assertRaises(ValueError):
            plan_mlx_matrix.read_ci_releases(json.dumps({"releases": [{"mlx": "0.32.4"}]}))

    def test_should_refuse_a_cmake_file_without_the_tested_list(self):
        with self.assertRaises(ValueError):
            plan_mlx_matrix.read_tested_mlx_releases("set(SOMETHING_ELSE 1)")

    def test_should_exit_with_an_error_when_the_lists_disagree(self):
        with tempfile.TemporaryDirectory() as directory_name:
            repository_root = Path(directory_name)
            write_repository(repository_root, "0.32.4;0.32.5", [{"mlx": "0.32.4", "mlx_bridge_tag": "a"}])
            self.assertEqual(plan_mlx_matrix.main([directory_name]), 1)


if __name__ == "__main__":
    unittest.main()
