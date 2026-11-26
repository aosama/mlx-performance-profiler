#!/usr/bin/env python3
"""Plan continuous integration's MLX matrix, and refuse a tested set CI does not cover.

    python3 .github/ci/plan_mlx_matrix.py <repository root>

Reads the tested MLX releases from MLX_PERF_TESTED_MLX_VERSIONS in
cmake/mlx_perf_mlx_prefix.cmake and the mlx-bridge release for each from
.github/ci/mlx-releases.json. Prints the matrix as one line of JSON
({"include": [{"mlx": ..., "mlx_bridge_tag": ...}, ...]}) for GitHub's
fromJSON, and exits 1 with a message naming each difference when the two
lists disagree.

Why a check and not one list: the CMake list is what a build promises
(VERSIONING.md), and the JSON file is how CI keeps that promise. Adding a
release to one without the other must fail, so the tested set stays a
checked fact rather than a claim (#20). Standard library only.
"""

import json
import re
import sys
from pathlib import Path

TESTED_VERSIONS_PATTERN = re.compile(r'set\(MLX_PERF_TESTED_MLX_VERSIONS\s+"([^"]*)"\)')


def read_tested_mlx_releases(cmake_text):
    """The releases in MLX_PERF_TESTED_MLX_VERSIONS, in order."""
    match = TESTED_VERSIONS_PATTERN.search(cmake_text)
    if match is None:
        raise ValueError("MLX_PERF_TESTED_MLX_VERSIONS is not set in the CMake file")
    return [release for release in match.group(1).split(";") if release]


def read_ci_releases(releases_json_text):
    """The {"mlx", "mlx_bridge_tag"} entries of mlx-releases.json, checked for shape."""
    ci_releases = json.loads(releases_json_text).get("releases", [])
    for ci_release in ci_releases:
        if not ci_release.get("mlx") or not ci_release.get("mlx_bridge_tag"):
            raise ValueError(f"an entry in mlx-releases.json lacks mlx or mlx_bridge_tag: {ci_release}")
    return ci_releases


def find_disagreements(tested_mlx_releases, ci_releases):
    """One message per release CI misses, covers without it being tested, or lists twice."""
    ci_mlx_releases = [ci_release["mlx"] for ci_release in ci_releases]
    disagreements = [
        f"MLX {release} is tested (MLX_PERF_TESTED_MLX_VERSIONS) but has no entry in .github/ci/mlx-releases.json"
        for release in tested_mlx_releases if release not in ci_mlx_releases]
    disagreements += [
        f"MLX {release} is in .github/ci/mlx-releases.json but not in MLX_PERF_TESTED_MLX_VERSIONS"
        for release in ci_mlx_releases if release not in tested_mlx_releases]
    disagreements += [
        f"MLX {release} appears more than once in .github/ci/mlx-releases.json"
        for release in sorted(set(ci_mlx_releases)) if ci_mlx_releases.count(release) > 1]
    return disagreements


def plan_matrix(repository_root):
    """The matrix JSON, or a ValueError naming every disagreement."""
    tested_mlx_releases = read_tested_mlx_releases(
        (repository_root / "cmake" / "mlx_perf_mlx_prefix.cmake").read_text(encoding="utf-8"))
    ci_releases = read_ci_releases(
        (repository_root / ".github" / "ci" / "mlx-releases.json").read_text(encoding="utf-8"))
    disagreements = find_disagreements(tested_mlx_releases, ci_releases)
    if disagreements:
        raise ValueError("\n".join(disagreements))
    return {"include": [{"mlx": ci_release["mlx"], "mlx_bridge_tag": ci_release["mlx_bridge_tag"]}
                        for ci_release in ci_releases]}


def main(command_arguments):
    if len(command_arguments) != 1:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    try:
        matrix = plan_matrix(Path(command_arguments[0]))
    except ValueError as disagreement:
        print(f"error: {disagreement}", file=sys.stderr)
        return 1
    print(json.dumps(matrix, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
