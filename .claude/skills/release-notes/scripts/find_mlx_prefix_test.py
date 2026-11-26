"""Unit tests for find_mlx_prefix.py: finding the one MLX prefix of the
requested release in mlx-bridge's native cache. Standard library only."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import find_mlx_prefix  # noqa: E402


def make_prefix(native_cache_directory, identity, mlx_release):
    """<identity>/prefix with an MLX version header, or none when mlx_release is None."""
    header_directory = native_cache_directory / identity / "prefix" / "include" / "mlx"
    header_directory.mkdir(parents=True)
    if mlx_release is not None:
        major, minor, patch = mlx_release.split(".")
        (header_directory / "version.h").write_text(
            f"#define MLX_VERSION_MAJOR {major}\n#define MLX_VERSION_MINOR {minor}\n"
            f"#define MLX_VERSION_PATCH {patch}\n", encoding="utf-8")
    return native_cache_directory / identity / "prefix"


class FindMlxPrefixTest(unittest.TestCase):
    # --- positive ---

    def test_should_select_the_one_prefix_of_the_expected_release(self):
        with tempfile.TemporaryDirectory() as directory_name:
            native_cache_directory = Path(directory_name)
            make_prefix(native_cache_directory, "aaaa", "0.32.4")
            expected_prefix = make_prefix(native_cache_directory, "bbbb", "0.32.5")
            selected_prefix = find_mlx_prefix.select_prefix(
                find_mlx_prefix.find_prefix_releases(native_cache_directory), "0.32.5")
            self.assertEqual(selected_prefix, expected_prefix)

    def test_should_read_a_release_from_the_version_header(self):
        self.assertEqual(find_mlx_prefix.read_mlx_release(
            "#define MLX_VERSION_MAJOR 0\n#define MLX_VERSION_MINOR 32\n#define MLX_VERSION_PATCH 5\n"), "0.32.5")

    # --- edge / boundary ---

    def test_should_read_no_release_from_an_incomplete_header(self):
        self.assertIsNone(find_mlx_prefix.read_mlx_release("#define MLX_VERSION_MAJOR 0\n"))

    def test_should_list_a_prefix_without_a_version_header_as_unknown(self):
        with tempfile.TemporaryDirectory() as directory_name:
            native_cache_directory = Path(directory_name)
            make_prefix(native_cache_directory, "cccc", None)
            self.assertEqual(find_mlx_prefix.find_prefix_releases(native_cache_directory)[0][1], None)

    # --- negative ---

    def test_should_refuse_when_only_another_release_was_built(self):
        with tempfile.TemporaryDirectory() as directory_name:
            native_cache_directory = Path(directory_name)
            make_prefix(native_cache_directory, "aaaa", "0.32.4")
            with self.assertRaisesRegex(ValueError, "no prefix of MLX 0.32.5; found: .*MLX 0.32.4"):
                find_mlx_prefix.select_prefix(find_mlx_prefix.find_prefix_releases(native_cache_directory), "0.32.5")

    def test_should_refuse_two_prefixes_of_the_same_release(self):
        with tempfile.TemporaryDirectory() as directory_name:
            native_cache_directory = Path(directory_name)
            make_prefix(native_cache_directory, "aaaa", "0.32.5")
            make_prefix(native_cache_directory, "bbbb", "0.32.5")
            with self.assertRaisesRegex(ValueError, "2 prefixes of MLX 0.32.5"):
                find_mlx_prefix.select_prefix(find_mlx_prefix.find_prefix_releases(native_cache_directory), "0.32.5")

    def test_should_refuse_an_empty_cache(self):
        with tempfile.TemporaryDirectory() as directory_name:
            self.assertEqual(find_mlx_prefix.main(["0.32.5", directory_name]), 1)

    def test_should_refuse_a_call_without_a_release(self):
        self.assertEqual(find_mlx_prefix.main([]), 2)


if __name__ == "__main__":
    unittest.main()
