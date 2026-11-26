---
name: release-notes
description: The release skill. Write the GitHub release notes for a mlx-performance-profiler release, as release-notes/vX.Y.Z.md, which the release workflow publishes as the release body, and make sure the Context7 documentation coding agents read covers everything the release adds or changes. Use this whenever the user asks to cut, tag, publish or prepare a release, to write, improve or rewrite release notes, to announce what shipped, or to edit a published GitHub release's description, even if they only say "make a release", "what's in this version" or "write it up for users". Also use it when an agent needs to explain what changed between two tags.
---

# Release notes for mlx-performance-profiler

Release notes are what most people will ever read about a version. They are
read by three kinds of reader at once, and each needs something different:

- **Someone deciding whether to upgrade or adopt.** What does this release
  let me do that I couldn't before, and is it worth it? They need the
  outcome first, with one concrete, measured example.
- **Someone trying it for the first time, today.** How do I start? They need
  commands they can copy and paste, pinned to this exact tag.
- **An agent integrating the library.** What exactly changed? It needs exact
  facts it can act on: function names, schema versions, tested MLX releases,
  the archive and its digest.

`CHANGELOG.md` is the terse, permanent record. The release notes tell the
story of the release and link every claim to its issue or pull request, so
a reader can always go one level deeper.

## Workflow

### 1. Gather the facts, never write from memory

```bash
python3 .claude/skills/release-notes/scripts/gather_release_facts.py \
    --previous-tag vA.B.C --target vX.Y.Z
# for a release not yet tagged: --target HEAD (or the release branch)
```

The script lists everything that shipped, read from git and GitHub:
- the merged pull requests;
- the issues they and their commits close, and the issues they only
  reference;
- every issue closed between the two tags;
- the C API functions and structs added and removed, and the JSON schema
  version;
- the declared tested MLX releases and the test count;
- the changelog entry, the release assets with their SHA-256 (checked
  against a fresh download of the archive), and the open roadmap.

Its parsers, and the capture script's, are unit-tested; after changing
either script, run
`python3 -m unittest discover -s .claude/skills/release-notes/scripts -p "*_test.py"`.

Treat its output as the checklist: every shipped issue appears in the notes,
and nothing appears that is not on it.

"Closed between the two tags" is not the same as "shipped in this release".
An issue may have been closed as not planned, or its work may have shipped
earlier and only been closed later. Check each against the merged pull
requests before claiming it.

### 2. Read the story behind each item

For each shipped issue and its pull request, read them in full
(`gh issue view N`, `gh pr view N`):

- **The issue** says why the work mattered: its "Need" and "Done when".
- **The pull request** says what was built and how it was verified, often
  with measured numbers.
- **The README section** for the feature has the commands to get started.

Take numbers from these sources only, and keep each number's context: what
was measured, on what, against what, at what size (a model's context
length, a matrix's shape). A number that can't be traced to a PR, a test or
a measurement doesn't go in.

Two distinctions the notes must get right:

- **Fixed, or never shipped?** "Fixes" lists only defects a user of an
  earlier release could have hit. A defect found and fixed in new code
  before it was released (for example, by a review of the release
  candidate) is not a fix to anyone; it belongs in "How this release was
  verified", where it shows the review worked.
- **Declared, or observed?** `MLX_PERF_TESTED_MLX_VERSIONS` is the list the
  project claims; step 7 runs the suite on each of them. The notes say
  which MLX releases this release's own test runs used. If that is narrower
  than the declared list, say so plainly; it is not a defect to hide.

### 3. Choose the headline

Ask what a user can do now that they couldn't before. That is the title's
theme and the first sentence. One release, one theme: if it has two big
things, name both in the title. Name things by what they do for a reader,
not by internal names.

### 4. Capture the opening visual

Every release page opens with one picture, directly under the title, that
the profiler itself drew. Most readers decide in the first screen whether
to read on, and a real report showing the headline outcome convinces faster
than any sentence. It is also what people share.

- **Real, from this release.** Build the release's code, run the tool whose
  report shows the headline (`hello_world`, `qwen35_profile`, or the C API
  example), and capture that report. Never a mock-up, never an edited
  screenshot, never an image from an older build.
- **One idea, readable at a glance.** Pick the section that shows the
  headline: the verdict for a comparison feature, the execution graph for a
  graph feature, the timeline for a timing feature. Hide dense notes
  (`--hide`). If the image needs a paragraph to decode, pick another view.
- **Capture it** with
  [`scripts/capture_report_visual.py`](scripts/capture_report_visual.py):

  ```bash
  python3 .claude/skills/release-notes/scripts/capture_report_visual.py \
      build/qwen35_decode.html release-notes/media/vX.Y.Z-verdict.png \
      --sections verdict --hide "nav,#verdict .note" --colours 256
  # a graph view, driven first, at 16:9 for social posts:
  #   --sections dag --page-script "mlxPerfGraphView.mountedViews[0].setColourMode('heat')" \
  #   --width 1600 --height 900 --no-trim
  ```

  It renders the report in headless Chrome, keeps only the sections named,
  and trims the empty background. `--colours 256` keeps a UI screenshot
  sharp at about a third of the bytes. Look at the PNG before using it.
- **Commit it** as `release-notes/media/vX.Y.Z-<what>.png` in the release
  pull request. A release body does not resolve relative paths, so link
  the committed file by its raw URL at the tag:
  `https://raw.githubusercontent.com/<owner>/<repo>/vX.Y.Z/release-notes/media/vX.Y.Z-<what>.png`.
  For a release already published, use `main` in place of the tag. That
  path lives on `main` but not at the old tag.
- **Alt text states the result**, not the picture ("compiled is 5.8%
  faster than eager, 12 of 12 rounds", not "a screenshot"). Under the
  image, one italic line says what it shows, that the profiler drew it,
  which job and size, and "one run on one Mac", since numbers vary run to
  run. If the picture's numbers differ from a number quoted in the text,
  the caption says they come from different runs.
- **Privacy:** check the image as you would the text. A report title or
  path in the picture must not show a workstation path or a private model
  location.

The same image can open the README section for the feature. For a social
post, capture a 16:9 version (`--width 1600 --height 900 --no-trim`).

### 5. Write `release-notes/vX.Y.Z.md`

Follow [`references/template.md`](references/template.md). It gives the
sections in order and what each must contain. Every section earns its
place; drop one that would be empty, except "Upgrading" (write "Nothing to
change" when that's true, because readers look for it) and "For agents".

### 6. Bring the Context7 documentation up to date

Coding agents learn this library from Context7, which indexes `docs/` and
the rules in `context7.json`, not from the release notes. A capability that
is missing from `docs/` does not exist for them, and a changed behaviour
that `docs/` still describes the old way is worse: agents will write wrong
code with confidence. So before the release pull request, go through the
fact sheet item by item. For every capability the release adds and every
behaviour it changes, confirm each of these, and fix any gap in the release
pull request (or in a pull request before it):

- **A page answers it.** A task guide in `docs/` (a new page for a new
  question, a section for an extension), indexed in `docs/README.md`'s
  "Which page answers your question" table when it is a new question.
- **A complete example shows it.** Every C, CMake or Python example is a
  file in `docs/examples/`, copied into the page between
  `<!-- example: ... -->` markers with `python3 tests/docs/docs_check.py sync .`,
  and built and run by `docs_examples_journey`.
- **The reference has it.** Every new or changed C function, struct and
  constant is in `docs/reference/c-api.md`, with its size for structs that
  other languages mirror.
- **The data formats have it.** Every new JSON field is in
  `docs/measurement-file.md` and in its JSON Schema in `docs/schema/`, and
  a real file from this release validates against that schema.
- **Errors are findable.** Every new error message has a troubleshooting
  entry quoting it, and a surprising new number has an entry explaining it.
- **Agents are told what they must not get wrong.** If the release changes
  how a caller must use the library (a new contract, a new default, a new
  trap), add or update a rule in `context7.json`. Rules stay under 255
  characters, or Context7 drops them all.
- **Nothing describes the old behaviour.** Search `docs/`, `README.md` and
  the rules for the old wording of anything the release changed (a renamed
  function, a limitation it removed, a figure it corrected) and update it.
- **The pins name this release.** `docs/README.md`'s "Versions" paragraph,
  the `GIT_TAG` and `--branch` pins and the reference's version macros say
  `X.Y.Z`.

Then run `python3 tests/docs/docs_check.py check .` and the `docs_*` CTest
tests (`ctest --test-dir build -R docs_ --output-on-failure`): they catch
drifted examples, broken links, a reference missing a declaration, a
troubleshooting quote the code cannot produce, a stale version pin,
`context7.json` over Context7's limits, and real output that disagrees with
a schema. They cannot tell whether a capability is documented at all; the
item-by-item pass above is what catches that.

**After the release pull request merges**, confirm Context7 actually serves
the new documentation:

1. The `context7-refresh` workflow ran for the merge and its log says
   `Refresh started successfully`, not the "skipping" notice it prints when
   the `CONTEXT7_API_KEY` secret is missing:
   `gh run view "$(gh run list --workflow context7-refresh.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --log | grep -E 'Refresh started|skipping'`.
   If it did not run, start it: `gh workflow run context7-refresh.yml`.
2. Ask Context7 for one new name from the release and see it come back:
   `curl -s "https://context7.com/api/v1/aosama/mlx-performance-profiler?type=txt&topic=<topic>&tokens=3000" | grep -c <new_function_name>`.
   Indexing takes a few minutes after a refresh; try again before
   concluding it is missing. If it stays missing, the page is probably
   excluded (`excludeFolders` in `context7.json`) or not linked from
   `docs/README.md`.

Report the result to the user in one line: what you checked and what
Context7 returned.

### 7. Build and test on every tested MLX release

`VERSIONING.md` promises that a release builds and passes on every MLX
release in `MLX_PERF_TESTED_MLX_VERSIONS` (`cmake/mlx_perf_mlx_prefix.cmake`).
Nothing outside this step checks that promise: this repository has no
continuous integration, by choice, because MLX comes from `mlx-bridge`'s
native build and the maintainer keeps the two repositories' automation
apart. So run the full suite on each release here, on the release's own
code, before the release pull request.

1. **Wait for a quiet GPU.** Check `ioreg -r -d 1 -c IOAccelerator` for
   "Device Utilization %": the maintainer's own GPU work makes timings
   noisy and can fail the timing and wired-memory tests. When it is busy,
   wait (a background loop that waits for 30 quiet seconds works) rather
   than reading a failure as the code's.
2. **Find each release's prefix.** For every release in the tested list:
   `python3 .claude/skills/release-notes/scripts/find_mlx_prefix.py <release>`
   prints the prefix `mlx-bridge` built for it under
   `~/.cache/mlx-bridge/native`, after checking its `mlx/version.h`. When a
   release has none, it says so: build it with `cargo xtask build-native` in
   a checkout of the `mlx-bridge` release that pins that MLX, or ask the
   user. Never test on a prefix of another MLX and call it this one.
3. **Build and test in a directory per release**, beside the main `build/`
   (the main one also feeds the Rust example and the release visual):
   ```bash
   cmake -S . -B build-mlx-<release> -DMLX_PREFIX=<prefix>
   cmake --build build-mlx-<release> --parallel
   ctest --test-dir build-mlx-<release> --output-on-failure
   ```
   The build prints its progress, and ctest prints each test as it ends.
   The Rust example's tests (`cargo test` in `rust/mlx_perf_rust_example`)
   run once, against the main `build/`.
4. **Record what passed, not what was declared.** The notes'
   `verified_on_mlx` lists exactly the releases whose full suite passed in
   this step, and the Compatibility section says which release each was.
   A release that fails is a finding to fix or to report to the user before
   publishing, never one to leave out of the notes quietly.

### 8. Check before publishing

- **The opening visual** renders on the page (open it after publishing),
  comes from this release's build, and agrees with the text or says why
  not.
- **Links:** every `#N` points at the issue or PR meant (`gh issue view N`).
  Docs links point at the tag, not `main`, so they keep matching this
  release.
- **Commands:** every command and snippet uses this release's tag and
  version, and is one you have seen work, in this repo's packaging tests,
  the README, or a run of your own. Run any you add (cloning the tag,
  checking the archive digest); they are quick.
- **The agent facts block** matches the fact sheet exactly: names, versions,
  and a digest the script reports as matching the downloaded archive.
- **Privacy:** no personal names or emails, no workstation paths, no local
  model locations, no hostnames. Public model names and public URLs are
  fine.
- **Honesty:** limitations and open issues are stated plainly, and nothing
  is promised for a future release.
- **Every tested MLX release:** step 7 is done, and `verified_on_mlx`
  names exactly the releases whose full suite passed on this release's code.
- **Context7:** step 6 is done: every new capability and changed behaviour
  is in `docs/`, the docs checks pass, and (after merge) Context7 returns
  the new names.

### 9. Publish

- **A new release:** commit `release-notes/vX.Y.Z.md` and its image in
  `release-notes/media/` in the release pull request, beside the version
  bump and the `CHANGELOG.md` entry. The release
  workflow refuses a tag without its notes file and publishes the file as
  the release body.
- **A release already published:** commit the file through a pull request,
  then `gh release edit vX.Y.Z --notes-file release-notes/vX.Y.Z.md`. That
  edits a public page, so confirm with the user first unless they asked for
  it.

## Style

- **Outcome first, mechanism second.** "Compare your Metal kernel against
  the stock primitive and get a verdict you can trust" before "paired
  rounds with a sign-test interval".
- **Celebrate with evidence, not adjectives.** A measured result ("compiled
  is 5.6% faster than eager on a Qwen3.5 decode step, in 12 of 12 rounds")
  is worth more than "blazing fast" or "huge improvement". Avoid hype words
  altogether.
- **Plain language, short sentences, defined terms.** A reader new to MLX
  should follow the highlights. Explain a statistical or GPU term in a few
  words the first time.
- **Link everything shipped:** issue and PR, as `#N` (GitHub links these
  automatically in a release body). Link docs by full URL at the tag.
- **One line per paragraph or list item; don't hard-wrap.** A release body
  is not rendered like a README: GitHub turns every line break in it into a
  visible break, so prose wrapped at 80 columns shows up ragged on the
  release page. (Code blocks are unaffected.)
- **Say what did not change** when readers would worry. For example: "0.1.0
  calls keep working unchanged."
- **Credit work that improved the release,** such as an independent review
  that found defects, without naming private individuals.

## Example: a highlight

Weak:
> Added comparison support and many improvements to measurement accuracy.

Strong:
> **Know whether your kernel is really faster (#12).** Hand
> `mlx_perf_compare_to_report` two graphs that compute the same result, a
> stock MLX primitive and your custom Metal kernel, say, and the report
> opens with a verdict: which is faster, by how much, and whether the
> difference exceeds run-to-run noise. On a Qwen3.5 decode step: *compiled
> is 5.6% faster than eager, faster in 12 of 12 rounds.* Two identical
> variants are called different under 4% of the time. That guarantee is
> tested, not assumed.

The strong version names the reader's problem, the entry point, a real
result and why it can be trusted, and links the issue.
