# Release notes template

Copy this skeleton to `release-notes/vX.Y.Z.md` and fill every section from
the fact sheet and the issues and PRs behind it. The comments in `<!-- -->`
say what each section must contain; delete them when writing. Write each
paragraph and list item on a single line: GitHub renders every line break
in a release body as a visible break.

````markdown
# mlx-performance-profiler X.Y.Z: <the theme, in a few words>

<!-- One paragraph, three or four sentences. Who this release is for, what
they can do now that they couldn't before, and the one measured result
that best shows it. A reader who stops here knows whether to read on. -->

## Highlights

<!-- Two to four, the things worth celebrating. Each: a bold outcome
headline with its issue in parentheses, then a few sentences giving the
reader's problem, the entry point, one concrete measured example, and why
it can be trusted. Link the issue (#N) and, where it adds proof, the PR. -->

## Get started

<!-- Copy-paste blocks, pinned to vX.Y.Z, for each way in:
1. From source with CMake FetchContent (GIT_TAG vX.Y.Z), as in the README.
2. The installed package: `git clone --branch vX.Y.Z <repo URL>`, configure
   with the library-only options, build, `cmake --install`, then
   `find_package(mlx_perf X.Y REQUIRED)` in the consumer.
3. Pinning the source archive, with the digest the fact sheet reports as
   matching:
     curl -LO <archive URL>
     echo "<sha256>  <archive name>" | shasum -a 256 -c
Say which MLX MLX_PREFIX must point at. Then the smallest working example
of the headline feature (a few lines of C) and the files it writes, and
links to the full docs at the tag:
https://github.com/<repo>/blob/vX.Y.Z/README.md#<section> -->

## What's new

<!-- One short subsection per shipped issue, ordered by importance. Each:
what it does, the C API or tool that exposes it, what files it writes, and
links: issue #N, PR #M, docs at the tag. Smaller additions go in a list at
the end. -->

## Upgrading from A.B.C

<!-- What an existing user must change, step by step: version pins
(find_package version, GIT_TAG), new files the library now writes, changed
meanings of existing numbers, refused inputs that used to be accepted.
When nothing must change, say so in one sentence. Then say what did NOT
change that a reader might worry about, e.g. "0.1.0 C calls and structs are
unchanged." -->

## Fixes

<!-- Defects a user of an earlier release could have hit, most severe
first. Anything security-relevant leads, described plainly: what could
happen before, what happens now. Link the PR. Defects found and fixed in
new code before it shipped go under verification instead. In a first
release this section says there is nothing to list. -->

## Compatibility

<!-- - MLX: the declared tested releases (fact sheet), and the release the
  verifying test run actually used; backend and platform.
- C API: functions and structs added and removed.
- File formats: measurement and comparison schema versions.
- What VERSIONING.md says this release number promises. -->

## How this release was verified

<!-- The test count (and what kinds) and the MLX release it ran on,
real-model runs with their results and sizes, and reviews that ran,
including defects they caught before release. Facts only, each traceable
to a PR. -->

## Known limitations and what's next

<!-- Honest limits of this release, then the open roadmap issues (#N, one
line each). State what is tracked, not when it will land. -->

## For agents

<!-- A fenced YAML block an agent can parse. Every value comes from the fact
sheet; omit a key rather than guess it. -->

```yaml
release: X.Y.Z
tag: vX.Y.Z
previous_release: A.B.C
repository: <owner>/<repo>
source_archive:
  url: <from fact sheet>
  sha256: <from fact sheet>
cmake:
  find_package: "find_package(mlx_perf X.Y REQUIRED)"
  target: mlx_perf::mlx_perf
  requires: MLX_PREFIX pointing at a tested MLX release
tested_mlx_releases: [<declared, from fact sheet>]
verified_on_mlx: [<what the release's own test run used>]
platform: macOS, Apple silicon, Metal
c_api:
  header: include/mlx_perf/mlx_perf_c_api.h
  added: [<functions, from fact sheet>]
  structs_added: [<from fact sheet>]
  removed: [<from fact sheet>]
file_formats:
  measurement_file_schema_version: <n>
  comparison_record_schema_version: <n>
shipped_issues: [<numbers>]
docs:
  readme: https://github.com/<repo>/blob/vX.Y.Z/README.md
  changelog: https://github.com/<repo>/blob/vX.Y.Z/CHANGELOG.md
```

**Full changelog:** https://github.com/<repo>/compare/vA.B.C...vX.Y.Z
````
