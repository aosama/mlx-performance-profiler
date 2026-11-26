#!/usr/bin/env python3
"""Find the MLX prefix mlx-bridge built, and refuse one of the wrong MLX release.

    python3 .github/ci/check_mlx_prefix.py <native cache directory> <expected MLX release>

mlx-bridge's `cargo xtask build-native` installs MLX and MLX-C into
<native cache directory>/<identity>/prefix, where the identity is a hash CI
does not know in advance. This finds every prefix there, keeps those whose
include/mlx/version.h names the expected release, and prints the single
match's path. It exits 1 naming what it found when there is no match or
more than one.

Why check the version: the mapping from an MLX release to an mlx-bridge tag
is maintained by hand (.github/ci/mlx-releases.json). A wrong tag would
build and test another MLX than the job's name claims, and every check
would still pass. Standard library only.
"""

import re
import sys
from pathlib import Path

VERSION_PART_PATTERN = re.compile(r"#define MLX_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)")


def read_mlx_release(version_header_text):
    """"MAJOR.MINOR.PATCH" from mlx/version.h, or None when a part is missing."""
    version_parts = dict(VERSION_PART_PATTERN.findall(version_header_text))
    if not all(part in version_parts for part in ("MAJOR", "MINOR", "PATCH")):
        return None
    return f"{version_parts['MAJOR']}.{version_parts['MINOR']}.{version_parts['PATCH']}"


def find_prefix_releases(native_cache_directory):
    """(prefix path, MLX release or None) for every <identity>/prefix in the cache."""
    prefix_releases = []
    for prefix_path in sorted(native_cache_directory.glob("*/prefix")):
        version_header = prefix_path / "include" / "mlx" / "version.h"
        mlx_release = read_mlx_release(version_header.read_text(encoding="utf-8")) if version_header.is_file() else None
        prefix_releases.append((prefix_path, mlx_release))
    return prefix_releases


def select_prefix(prefix_releases, expected_mlx_release):
    """The one prefix of the expected release, or a ValueError describing what was found."""
    matching_prefixes = [prefix_path for prefix_path, mlx_release in prefix_releases
                         if mlx_release == expected_mlx_release]
    if len(matching_prefixes) == 1:
        return matching_prefixes[0]
    found = ", ".join(f"{prefix_path} (MLX {mlx_release or 'unknown'})" for prefix_path, mlx_release in prefix_releases)
    if not matching_prefixes:
        raise ValueError(f"no prefix of MLX {expected_mlx_release}; found: {found or 'nothing'}")
    raise ValueError(f"{len(matching_prefixes)} prefixes of MLX {expected_mlx_release}: {found}")


def main(command_arguments):
    if len(command_arguments) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    native_cache_directory, expected_mlx_release = Path(command_arguments[0]).expanduser(), command_arguments[1]
    try:
        print(select_prefix(find_prefix_releases(native_cache_directory), expected_mlx_release))
    except ValueError as mismatch:
        print(f"error: {mismatch}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
