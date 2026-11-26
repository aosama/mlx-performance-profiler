# Versioning

**Status: proposed, awaiting the maintainer's agreement.** Until it is
agreed, no release is tagged.

## In one paragraph

mlx-performance-profiler has its **own [Semantic Version](https://semver.org)**
(`0.1.0` first), independent of MLX's number. Which MLX releases a version
works with is **declared as data**: a tested list in the build, in the
README and in every release's notes. A source build accepts any tested MLX
release; an installed, prebuilt archive accepts only the exact MLX release
it was compiled against, and says so at configure time.

## Why not reuse MLX's number

MLX has been `0.x` since its first release. It ships a new minor about every
one to three months, with patch releases in between, and a minor may change
the C++ internals this library reads (the Metal command encoder in
`mlx/backend/metal/device.h`).

The sibling `mlx-bridge` crate takes the other approach: its version names
the MLX release it is verified against (`0.32.4-alpha.6` is MLX `0.32.4`).
That fits a crate that vendors and pins one MLX. It does not fit here, for
three reasons:

1. **This library supports several MLX releases at once.** It already builds
   and passes its tests on MLX `0.32.4` and `0.32.5`; a number naming one of
   them would misstate the other.
2. **Our own changes need numbers.** A fix or a new C API function would have
   no slot of its own; `mlx-bridge`'s release notes record this as an open
   problem ("stable fix slot") and make breaking changes wait for the next
   MLX release.
3. **Build metadata does not help.** A tag like `v0.1.0+mlx.0.32.5` is valid
   SemVer, but build metadata is ignored when versions are compared, and
   neither CMake's `find_package` nor Cargo can select on it.

## What carries a version

| Surface | Versioned by |
|---|---|
| The C API (`include/mlx_perf/*.h`) and the archive's ABI | This project's version |
| The installed CMake package (`find_package(mlx_perf)`) | This project's version, `SameMinorVersion` before 1.0 |
| The per-node JSON file (#11) | Its own `schema_version` field, an integer bumped on any incompatible change |
| The HTML reports and console output | Not versioned: for people, may change in any release |
| C++ headers in `src/` | Not versioned: internal, may change in any release |

## How a change moves the version

Before `1.0.0`, the minor digit plays the role of the major digit:

| Change | Before 1.0 | From 1.0 |
|---|---|---|
| Fix, docs, performance, internal refactor | patch | patch |
| Another MLX patch release tested, no code change | patch | patch |
| New C API function or struct (additive) | minor | minor |
| Support for a new MLX minor series | minor | minor |
| Changed or removed C API function, struct field or layout | minor | **major** |
| Dropping an MLX minor series | minor | **major** |
| New `schema_version` of the JSON file | minor | **major** |

`mlx_perf_version()` and `MLX_PERF_VERSION_*` let a caller detect a header
and an archive from different releases.

## How this reconciles with MLX

- **Tested set.** `MLX_PERF_TESTED_MLX_VERSIONS` in
  `cmake/mlx_perf_mlx_link.cmake` is the single source of truth for the MLX
  releases a version has been built and tested against. The README's
  compatibility table and each release's notes are copied from it.
- **Support window.** A release supports the MLX minor series that was
  current when it was cut, with every patch release in it we have tested.
  When MLX publishes a new minor, our next minor release adds it. The
  previous series stays supported for as long as no code has to branch on
  `MLX_VERSION_NUMERIC`; dropping it follows the table above.
- **Untested MLX releases.** Configure warns and continues, so a locally
  patched MLX or a release newer than our tested set can still be tried.
  The README lists the four MLX behaviours the library relies on; a patched
  MLX that keeps them works.
- **Prebuilt archives.** An installed `libmlx_perf.a` was compiled against
  one MLX's C++ headers, so its package records that release and refuses a
  consumer whose `MLX_PREFIX` holds another. Building from source (or with
  FetchContent) has no such restriction within the tested set.
- **`mlx-bridge`.** Because the bridge's version names its MLX, pairing is
  direct: `mlx-bridge 0.32.5-*` pairs with any mlx-performance-profiler
  release whose tested set includes MLX `0.32.5`.

## Tags and releases

- Tags are `vMAJOR.MINOR.PATCH` on `main`, and the tag must equal the
  `project(... VERSION ...)` in `CMakeLists.txt` (the release workflow
  checks this).
- Each release has notes in `CHANGELOG.md` listing its tested MLX
  releases, and a source archive attached to the GitHub release with its
  SHA-256, which consumers pin. An attached asset is byte-stable; GitHub's
  automatic source archives are not guaranteed to be.

## Reaching 1.0

Proposed criteria: the C API covers #11 (per-node data) and #12 (two-graph
comparison), and it has gone through one MLX minor-series bump without a
breaking change.
