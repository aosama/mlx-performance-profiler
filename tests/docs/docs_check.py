#!/usr/bin/env python3
"""Keeps the documentation true to the code it describes.

Documentation that agents read (through Context7) is only useful while it is
right, and docs drift silently. Each check here turns one kind of drift into
a failing test:

- an embedded example differs from the file CTest compiles and runs;
- a link or #anchor points nowhere;
- the C API reference misses a declaration, or quotes one that changed;
- the troubleshooting page quotes an error message the library cannot return;
- context7.json breaks one of Context7's limits, so it would be rejected;
- a measurement file has a field the JSON Schema does not describe, or lacks
  one it requires.

Usage (from anywhere; standard library only):
    python3 tests/docs/docs_check.py check <repository root>
    python3 tests/docs/docs_check.py sync <repository root>     # rewrite embedded examples
    python3 tests/docs/docs_check.py schema <schema.json> <file.json>...
"""

import json
import re
import sys
from pathlib import Path

# <!-- example: docs/examples/profile_one_job.c --> directly above a fenced
# block marks that block as a verbatim copy of the named file.
# The body is matched line by line up to a line holding only the fence, so
# an empty placeholder block (opening fence, closing fence) ends at its own
# closing fence instead of running on to the next block's.
EXAMPLE_MARKER = re.compile(r"<!-- example: (?P<path>[^ \n]+) -->\n(?P<fence>```+)(?P<language>[^\n]*)\n"
                            r"(?P<body>(?:.*\n)*?)(?P=fence)\n")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)\s]+)\)")
# Context7's published limits (context7.com/schema/context7.json): a rule
# over 255 characters makes Context7 drop every rule.
CONTEXT7_MAX_RULES = 50
CONTEXT7_MAX_RULE_LENGTH = 255
CONTEXT7_MAX_FOLDERS = 50
CONTEXT7_MAX_EXCLUDED_FILES = 100


def heading_anchor(heading_text):
    """GitHub's anchor for a heading: lower case, punctuation dropped,
    spaces to hyphens."""
    without_code_marks = heading_text.strip().lower().replace("`", "")
    kept_characters = re.sub(r"[^\w\- ]", "", without_code_marks)
    return kept_characters.replace(" ", "-")


def find_heading_anchors(markdown_text):
    without_code_blocks = re.sub(r"```.*?```", "", markdown_text, flags=re.DOTALL)
    return {heading_anchor(heading) for heading in
            re.findall(r"^#{1,6} (.+)$", without_code_blocks, flags=re.MULTILINE)}


def find_example_embeds(markdown_text):
    """(example path, embedded body) for every marked block."""
    return [(match["path"], match["body"].rstrip("\n")) for match in EXAMPLE_MARKER.finditer(markdown_text)]


def find_drifted_examples(markdown_path, repository_root):
    problems = []
    for example_path, embedded_body in find_example_embeds(markdown_path.read_text(encoding="utf-8")):
        example_file = repository_root / example_path
        if not example_file.is_file():
            problems.append(f"{markdown_path}: embeds {example_path}, which does not exist")
        elif example_file.read_text(encoding="utf-8").rstrip("\n") != embedded_body:
            problems.append(f"{markdown_path}: the copy of {example_path} differs from the file "
                            "(run: python3 tests/docs/docs_check.py sync .)")
    return problems


def sync_examples(markdown_path, repository_root):
    """Rewrites every marked block from its file. Returns whether it changed.
    Refuses, changing nothing, when a marker names a missing file."""
    markdown_text = markdown_path.read_text(encoding="utf-8")
    missing_paths = [example_path for example_path, _ in find_example_embeds(markdown_text)
                     if not (repository_root / example_path).is_file()]
    if missing_paths:
        raise ValueError(f"{markdown_path} embeds missing files: {', '.join(missing_paths)}")

    def replace_body(match):
        example_text = (repository_root / match["path"]).read_text(encoding="utf-8").rstrip("\n")
        return (f"<!-- example: {match['path']} -->\n{match['fence']}{match['language']}\n"
                f"{example_text}\n{match['fence']}\n")

    synced_text = EXAMPLE_MARKER.sub(replace_body, markdown_text)
    if synced_text != markdown_text:
        markdown_path.write_text(synced_text, encoding="utf-8")
    return synced_text != markdown_text


def find_broken_links(markdown_path, repository_root):
    """Relative links must name an existing file, and #anchors a heading in it."""
    problems = []
    # Code can hold "[x](y)" that is not a link; only prose links count.
    prose_text = re.sub(r"```.*?```", "", markdown_path.read_text(encoding="utf-8"), flags=re.DOTALL)
    for target in (match["target"] for match in MARKDOWN_LINK.finditer(prose_text)):
        if re.match(r"^[a-z]+:", target):
            continue  # absolute URLs are checked by hand at release time
        target_path_text, _, anchor = target.partition("#")
        target_path = (markdown_path.parent / target_path_text).resolve() if target_path_text else markdown_path
        if not target_path.exists():
            problems.append(f"{markdown_path}: link to {target} names no file")
        elif anchor and target_path.suffix == ".md" and \
                anchor not in find_heading_anchors(target_path.read_text(encoding="utf-8")):
            problems.append(f"{markdown_path}: link to {target} names no heading")
    return problems


def normalise_whitespace(code_text):
    return " ".join(code_text.split())


def strip_c_comments(c_text):
    return re.sub(r"/\*.*?\*/", "", c_text, flags=re.DOTALL)


def find_c_api_declarations(header_text):
    """Every function prototype, struct name and verdict constant the public
    header declares."""
    without_comments = strip_c_comments(header_text)
    prototypes = [normalise_whitespace(match) for match in re.findall(
        r"^(?:const )?[a-z_]+\*? ?\**mlx_perf_[a-z_]+\([^;]*\);", without_comments, flags=re.MULTILINE)]
    struct_names = re.findall(r"typedef struct (mlx_perf_[a-z_]+)", without_comments)
    constant_names = re.findall(r"^\s*(MLX_PERF_VERDICT_[A-Z_]+)", without_comments, flags=re.MULTILINE)
    return prototypes, struct_names, constant_names


def find_quoted_definitions(header_text):
    """Struct definitions and the verdict enum, whitespace-normalised: the
    reference copies them whole, so a field added to the header must show up
    in the reference's copy too."""
    without_comments = strip_c_comments(header_text)
    definitions = re.findall(r"typedef struct mlx_perf_[a-z_]+ \{.*?\} mlx_perf_[a-z_]+;",
                             without_comments, flags=re.DOTALL)
    definitions += re.findall(r"enum \{[^}]*MLX_PERF_VERDICT_[^}]*\};", without_comments)
    return [normalise_whitespace(definition) for definition in definitions]


def find_undocumented_declarations(header_text, reference_text):
    prototypes, struct_names, constant_names = find_c_api_declarations(header_text)
    flattened_reference = normalise_whitespace(reference_text)
    problems = [f"C API reference does not quote the prototype: {prototype}"
                for prototype in prototypes if prototype not in flattened_reference]
    problems += [f"C API reference does not describe {name}"
                 for name in struct_names + constant_names if name not in reference_text]
    problems += [f"C API reference's copy differs from the header's definition: {definition[:70]}..."
                 for definition in find_quoted_definitions(header_text)
                 if definition not in flattened_reference]
    return problems


def find_unknown_messages(troubleshooting_text, message_kind, source_text):
    """Each `> <message_kind>: `...`` line quotes a message; its fixed text
    (the parts around any variable part, written as <...>) must appear in
    the source that produces it: the library's C++ for "error", the CMake
    files for "configure"."""
    problems = []
    quoted_messages = re.findall(rf"^> {message_kind}: `(.+)`$", troubleshooting_text, flags=re.MULTILINE)
    for quoted_message in quoted_messages:
        for fixed_part in (part.strip('" ') for part in re.split(r"<[^>]+>", quoted_message)):
            if fixed_part and fixed_part not in source_text:
                problems.append(f"troubleshooting quotes \"{quoted_message}\", but nothing "
                                f"produces \"{fixed_part}\"")
    return problems


def find_context7_problems(context7_config, repository_root):
    problems = []
    rules = context7_config.get("rules", [])
    if len(rules) > CONTEXT7_MAX_RULES:
        problems.append(f"context7.json has {len(rules)} rules; at most {CONTEXT7_MAX_RULES}")
    problems += [f"context7.json rule is {len(rule)} characters (max {CONTEXT7_MAX_RULE_LENGTH}): {rule[:60]}..."
                 for rule in rules if not 1 <= len(rule) <= CONTEXT7_MAX_RULE_LENGTH]
    for folder_name in context7_config.get("folders", []):
        if not (repository_root / folder_name).is_dir():
            problems.append(f"context7.json indexes folder {folder_name}, which does not exist")
    if len(context7_config.get("folders", [])) > CONTEXT7_MAX_FOLDERS:
        problems.append("context7.json lists too many folders")
    excluded_files = context7_config.get("excludeFiles", [])
    if len(excluded_files) > CONTEXT7_MAX_EXCLUDED_FILES:
        problems.append("context7.json excludes too many files")
    problems += [f"context7.json excludeFiles entry {name} must be a file name, without '/'"
                 for name in excluded_files if "/" in name or "\\" in name]
    if ("url" in context7_config) != ("public_key" in context7_config):
        problems.append("context7.json needs url and public_key together, or neither")
    return problems


def matches_schema_type(instance, type_names):
    type_checks = {"object": lambda candidate: isinstance(candidate, dict),
                   "array": lambda candidate: isinstance(candidate, list),
                   "string": lambda candidate: isinstance(candidate, str),
                   "boolean": lambda candidate: isinstance(candidate, bool),
                   "integer": lambda candidate: isinstance(candidate, int) and not isinstance(candidate, bool),
                   "number": lambda candidate: isinstance(candidate, (int, float)) and not isinstance(candidate, bool),
                   "null": lambda candidate: candidate is None}
    names = type_names if isinstance(type_names, list) else [type_names]
    return any(type_checks[name](instance) for name in names)


def find_schema_mismatches(schema, instance, location="$"):
    """The subset of JSON Schema these files use: type, const, enum, required,
    properties and items. A field the schema does not describe is reported
    too, because an undocumented field is a documentation bug."""
    problems = []
    if "const" in schema and instance != schema["const"]:
        return [f"{location}: {instance!r} is not {schema['const']!r}"]
    if "enum" in schema and instance not in schema["enum"]:
        return [f"{location}: {instance!r} is not one of {schema['enum']}"]
    if "type" in schema and not matches_schema_type(instance, schema["type"]):
        return [f"{location}: {type(instance).__name__} is not {schema['type']}"]
    if isinstance(instance, dict) and "properties" in schema:
        problems += [f"{location}: missing required field {name}"
                     for name in schema.get("required", []) if name not in instance]
        problems += [f"{location}: field {name} is not in the schema"
                     for name in instance if name not in schema["properties"]]
        for name, member_schema in schema["properties"].items():
            if name in instance:
                problems += find_schema_mismatches(member_schema, instance[name], f"{location}.{name}")
    if isinstance(instance, list) and "items" in schema:
        for item_index, element in enumerate(instance):
            problems += find_schema_mismatches(schema["items"], element, f"{location}[{item_index}]")
    return problems


def check_file_against_schema(schema, json_path):
    """Schema problems of one file, or why it could not be read: a missing or
    malformed file is a finding to report, not a crash."""
    try:
        instance = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as read_failure:
        return [f"{json_path}: cannot read it as JSON ({read_failure})"]
    return [f"{json_path}: {problem}" for problem in find_schema_mismatches(schema, instance)]


def read_project_version(cmake_lists_text):
    match = re.search(r"project\(\S+\s+VERSION\s+([0-9.]+)", cmake_lists_text)
    return match.group(1) if match else None


def find_stale_version_pins(markdown_text, project_version, source_name):
    """Release tags and version macros quoted in the docs must name the
    current release, or a reader copies a pin to an older version."""
    problems = []
    for pinned_version in re.findall(r"(?:--branch |GIT_TAG |releases/download/)v([0-9]+\.[0-9]+\.[0-9]+)", markdown_text):
        if pinned_version != project_version:
            problems.append(f"{source_name}: pins v{pinned_version}, but the project is {project_version}")
    for quoted_version in re.findall(r'MLX_PERF_VERSION_STRING "([0-9.]+)"|current release, ([0-9.]+[0-9])',
                                     markdown_text):
        quoted_version = next(version_text for version_text in quoted_version if version_text)
        if quoted_version != project_version:
            problems.append(f"{source_name}: quotes version {quoted_version}, but the project is {project_version}")
    return problems


def check_repository(repository_root):
    docs_directory = repository_root / "docs"
    markdown_paths = sorted(docs_directory.rglob("*.md")) + [repository_root / "README.md"]
    problems = []
    project_version = read_project_version((repository_root / "CMakeLists.txt").read_text(encoding="utf-8"))
    for markdown_path in markdown_paths:
        problems += find_drifted_examples(markdown_path, repository_root)
        problems += find_broken_links(markdown_path, repository_root)
        problems += find_stale_version_pins(markdown_path.read_text(encoding="utf-8"), project_version,
                                            str(markdown_path.relative_to(repository_root)))
    problems += find_stale_version_pins((docs_directory / "examples/CMakeLists.txt").read_text(encoding="utf-8"),
                                        project_version, "docs/examples/CMakeLists.txt")
    header_text = (repository_root / "include/mlx_perf/mlx_perf_c_api.h").read_text(encoding="utf-8")
    problems += find_undocumented_declarations(
        header_text, (docs_directory / "reference/c-api.md").read_text(encoding="utf-8"))
    troubleshooting_text = (docs_directory / "troubleshooting.md").read_text(encoding="utf-8")
    library_source_text = "\n".join(path.read_text(encoding="utf-8")
                                    for path in sorted((repository_root / "src").glob("*.cpp")))
    cmake_source_text = "\n".join(path.read_text(encoding="utf-8")
                                  for path in sorted((repository_root / "cmake").glob("*.cmake*")))
    problems += find_unknown_messages(troubleshooting_text, "error", library_source_text)
    problems += find_unknown_messages(troubleshooting_text, "configure", cmake_source_text)
    problems += find_context7_problems(
        json.loads((repository_root / "context7.json").read_text(encoding="utf-8")), repository_root)
    return problems


def main(command_arguments):
    if len(command_arguments) < 2 or command_arguments[0] not in ("check", "sync", "schema"):
        sys.exit(__doc__)
    if command_arguments[0] == "schema":
        schema = json.loads(Path(command_arguments[1]).read_text(encoding="utf-8"))
        problems = [problem for json_path in command_arguments[2:]
                    for problem in check_file_against_schema(schema, Path(json_path))]
    elif command_arguments[0] == "sync":
        repository_root = Path(command_arguments[1]).resolve()
        for markdown_path in sorted((repository_root / "docs").rglob("*.md")):
            try:
                if sync_examples(markdown_path, repository_root):
                    print(f"synced {markdown_path.relative_to(repository_root)}")
            except ValueError as missing_example:
                print(f"not synced: {missing_example}")
                return 1
        return 0
    else:
        problems = check_repository(Path(command_arguments[1]).resolve())
    for problem in problems:
        print(problem)
    print(f"{len(problems)} documentation problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
