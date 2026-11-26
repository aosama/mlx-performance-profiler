#!/usr/bin/env python3
"""Collect the facts release notes must not guess, as a Markdown fact sheet.

Release notes go wrong in two ways: they miss what shipped, or they invent
it. Everything here is read from git and GitHub (the version, the tested MLX
releases, the merged pull requests and the issues they close, the C API
functions added or removed, the changelog entry, the open roadmap), so the
writer starts from a complete, checkable list and spends its effort on
explaining it.

Usage (from the repository root):
    python3 .claude/skills/release-notes/scripts/gather_release_facts.py \\
        --previous-tag v0.1.0 --target v0.2.0
    # --target defaults to HEAD (notes for a release not yet tagged);
    # --previous-tag defaults to the newest tag before the target.

Needs git and an authenticated gh CLI. Read-only: it never writes, tags or
posts anything.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

CLOSING_KEYWORDS = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
REFERENCE_KEYWORDS = r"(?:refs?|part of|relates to|follows from)"
C_API_HEADER_PATH = "include/mlx_perf/mlx_perf_c_api.h"
TESTED_MLX_RECIPE_PATH = "cmake/mlx_perf_mlx_prefix.cmake"
SCHEMA_VERSION_HEADER_PATH = "src/measurement_document.h"


def run_command(command_arguments):
    """Runs a command and returns its stdout; a failure stops the script,
    because a fact sheet with silent gaps is worse than none."""
    completed = subprocess.run(command_arguments, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.exit(f"error: {' '.join(command_arguments)}\n{completed.stderr.strip()}")
    return completed.stdout


def read_file_at(git_ref, file_path):
    """A file's text at a ref, or "" when it did not exist there yet."""
    completed = subprocess.run(["git", "show", f"{git_ref}:{file_path}"],
                               capture_output=True, text=True)
    return completed.stdout if completed.returncode == 0 else ""


def read_project_version(git_ref):
    match = re.search(r"project\(\S+\s+VERSION\s+([0-9.]+)", read_file_at(git_ref, "CMakeLists.txt"))
    return match.group(1) if match else "unknown"


def read_tested_mlx_versions(git_ref):
    match = re.search(r'set\(MLX_PERF_TESTED_MLX_VERSIONS\s+"([^"]*)"\)',
                      read_file_at(git_ref, TESTED_MLX_RECIPE_PATH))
    return match.group(1).split(";") if match else []


def parse_c_api_function_names(header_text):
    """Function names declared in the public header's text. Declarations
    start at column 0 with a return type; comments, macros and the callback
    typedef (whose name sits inside "(*...)") do not match."""
    declaration_pattern = r"^[A-Za-z][\w\s\*]*?\b(mlx_perf_[a-z_]+)\s*\("
    return set(re.findall(declaration_pattern, header_text, flags=re.MULTILINE))


def read_c_api_function_names(git_ref):
    return parse_c_api_function_names(read_file_at(git_ref, C_API_HEADER_PATH))


def parse_c_api_struct_names(header_text):
    """Struct type names defined in the header's text: the closing
    "} name;" of each typedef struct."""
    return set(re.findall(r"^\}\s*(mlx_perf_[a-z_]+)\s*;", header_text, flags=re.MULTILINE))


def read_c_api_struct_names(git_ref):
    return parse_c_api_struct_names(read_file_at(git_ref, C_API_HEADER_PATH))


def parse_schema_version(header_text):
    """The measurement file's schema version; the comparison record writes
    the same constant. None before the files existed."""
    match = re.search(r"MEASUREMENT_SCHEMA_VERSION\s*=\s*(\d+)", header_text)
    return match.group(1) if match else None


def read_schema_version(git_ref):
    return parse_schema_version(read_file_at(git_ref, SCHEMA_VERSION_HEADER_PATH))


def count_registered_tests(git_ref):
    """Tests registered with CTest at a ref: unit, component, integration,
    acceptance and packaging tests alike."""
    listed_files = run_command(["git", "ls-tree", "-r", "--name-only", git_ref, "tests"])
    test_count = 0
    for file_path in listed_files.splitlines():
        if file_path.endswith("CMakeLists.txt"):
            file_text = read_file_at(git_ref, file_path)
            test_count += len(re.findall(r"^mlx_perf_add_test\(", file_text, flags=re.MULTILINE))
            test_count += len(re.findall(r"^add_test\(NAME", file_text, flags=re.MULTILINE))
    return test_count


def read_changelog_entry(git_ref, version):
    """The CHANGELOG section for `version`, or the Unreleased one."""
    changelog_text = read_file_at(git_ref, "CHANGELOG.md")
    for heading in (f"## {version}", "## Unreleased"):
        start = changelog_text.find(heading)
        if start != -1:
            end = changelog_text.find("\n## ", start + len(heading))
            return changelog_text[start:end if end != -1 else None].strip()
    return "(no CHANGELOG entry found)"


def list_merged_pull_requests(previous_tag, target_ref):
    """Pull requests merged into the target since the previous tag, oldest
    first, read from merge commits (this repository merges with merge
    commits, never squash, so each PR leaves one)."""
    merge_subjects = run_command(
        ["git", "log", "--merges", "--reverse", "--format=%s", f"{previous_tag}..{target_ref}"])
    pull_request_numbers = re.findall(r"^Merge pull request #(\d+)", merge_subjects,
                                      flags=re.MULTILINE)
    merged_pull_requests = []
    for pull_request_number in pull_request_numbers:
        merged_pull_requests.append(json.loads(run_command(
            ["gh", "pr", "view", pull_request_number, "--json",
             "number,title,url,body,closingIssuesReferences"])))
    return merged_pull_requests


def numbers_after_keywords(text, keyword_pattern):
    """Issue numbers that follow a keyword, including lists: "Closes #24 and
    #25", "Refs #11 #12", "Part of #13, #14"."""
    list_pattern = rf"\b{keyword_pattern}\s+((?:#\d+(?:\s*(?:,|and)?\s*))+)"
    found_numbers = set()
    for number_list in re.findall(list_pattern, text, flags=re.IGNORECASE):
        found_numbers |= {int(number) for number in re.findall(r"#(\d+)", number_list)}
    return found_numbers


def find_issue_numbers(pull_request):
    """Issues a PR closes (GitHub's own link, plus Closes/Fixes keywords) and
    issues it only references (Refs, Part of), kept apart: a referenced issue
    is not shipped."""
    pull_request_body = pull_request.get("body") or ""
    closed_numbers = {reference["number"] for reference in pull_request["closingIssuesReferences"]}
    closed_numbers |= numbers_after_keywords(pull_request_body, CLOSING_KEYWORDS)
    referenced_numbers = numbers_after_keywords(pull_request_body, REFERENCE_KEYWORDS)
    return closed_numbers, referenced_numbers - closed_numbers


def find_issues_closed_by_commits(previous_tag, target_ref):
    """Closing keywords in commit messages: GitHub closes an issue from a
    commit's "Closes #N" when it reaches the default branch, even when the
    PR body did not link it."""
    commit_messages = run_command(["git", "log", "--format=%B", f"{previous_tag}..{target_ref}"])
    return numbers_after_keywords(commit_messages, CLOSING_KEYWORDS)


def list_issues_closed_between(previous_ref, target_ref):
    """Issues closed (by any means, including by hand) between the two refs'
    commit dates. Closed is not the same as shipped (an issue can be closed
    as not planned), so the writer checks each against the merged work."""
    since_date = run_command(["git", "log", "-1", "--format=%cI", previous_ref]).strip()
    until_date = run_command(["git", "log", "-1", "--format=%cI", target_ref]).strip()
    return json.loads(run_command(
        ["gh", "issue", "list", "--state", "closed", "--limit", "100",
         "--search", f"closed:{since_date}..{until_date}",
         "--json", "number,title,url,stateReason"]))


def read_issue(issue_number):
    return json.loads(run_command(
        ["gh", "issue", "view", str(issue_number), "--json", "number,title,state,url"]))


def list_open_issues():
    return json.loads(run_command(
        ["gh", "issue", "list", "--state", "open", "--limit", "50",
         "--json", "number,title,url"]))


def read_release_assets(tag_name):
    """The published release's URL, assets and archive digest, when the tag
    has been released; None before that."""
    completed = subprocess.run(["gh", "release", "view", tag_name, "--json", "url,assets"],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    release = json.loads(completed.stdout)
    digest_completed = subprocess.run(
        ["gh", "release", "download", tag_name, "--pattern", "*.sha256", "--output", "-"],
        capture_output=True, text=True)
    release["sha256_line"] = digest_completed.stdout.strip()
    release["digest_check"] = verify_archive_digest(release)
    return release


def verify_archive_digest(release):
    """Downloads the release's source archive the way a consumer would
    (anonymously, by its public URL) and checks it against the published
    digest, so the notes never repeat a digest that does not match."""
    archive_assets = [asset for asset in release["assets"] if asset["name"].endswith(".tar.gz")]
    published_digest = release["sha256_line"].split()[0] if release["sha256_line"] else ""
    if not archive_assets or not published_digest:
        return "not checked: no archive or no published digest"
    with tempfile.TemporaryDirectory() as download_directory:
        archive_path = Path(download_directory) / archive_assets[0]["name"]
        urllib.request.urlretrieve(archive_assets[0]["url"], archive_path)
        computed_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if computed_digest == published_digest:
        return "matches the downloaded archive"
    return f"MISMATCH: downloaded archive hashes to {computed_digest}"


def repository_name():
    return json.loads(run_command(["gh", "repo", "view", "--json", "nameWithOwner"]))[
        "nameWithOwner"]


def format_pull_request_lines(merged_pull_requests):
    """One line per PR with the issues it closes and only references."""
    lines = []
    for pull_request in merged_pull_requests:
        closed_numbers, referenced_numbers = find_issue_numbers(pull_request)
        closes_text = ", ".join(f"#{number}" for number in sorted(closed_numbers)) or "none"
        refs_text = ", ".join(f"#{number}" for number in sorted(referenced_numbers)) or "none"
        lines.append(f"- #{pull_request['number']} {pull_request['title']} "
                     f"(closes: {closes_text}; references: {refs_text}) {pull_request['url']}")
    return lines


def format_issue_lines(merged_pull_requests, commit_closed_numbers):
    """Every issue a merged PR or a commit in the range closes, with its
    title: what shipped."""
    shipped_numbers = set(commit_closed_numbers)
    for pull_request in merged_pull_requests:
        shipped_numbers |= find_issue_numbers(pull_request)[0]
    lines = []
    for issue_number in sorted(shipped_numbers):
        issue = read_issue(issue_number)
        lines.append(f"- #{issue['number']} {issue['title']} [{issue['state'].lower()}] "
                     f"{issue['url']}")
    return lines or ["- none"]


def format_api_lines(previous_tag, target_ref):
    lines = []
    for kind, read_names in (("functions", read_c_api_function_names),
                             ("structs", read_c_api_struct_names)):
        previous_names = read_names(previous_tag)
        target_names = read_names(target_ref)
        lines.append(f"- {kind} added: {', '.join(sorted(target_names - previous_names)) or 'none'}")
        lines.append(f"- {kind} removed: "
                     f"{', '.join(sorted(previous_names - target_names)) or 'none'}")
    lines.append(f"- all functions now: {', '.join(sorted(read_c_api_function_names(target_ref)))}")
    schema_version = read_schema_version(target_ref)
    lines.append(f"- measurement file and comparison record schema_version: "
                 f"{schema_version or 'none (no JSON output at this ref)'}")
    return lines


def format_release_lines(tag_name):
    release = read_release_assets(tag_name)
    if release is None:
        return [f"- {tag_name} is not released yet: no assets or digest to cite"]
    lines = [f"- release page: {release['url']}"]
    lines += [f"- asset: {asset['name']} ({asset['size']} bytes) {asset['url']}"
              for asset in release["assets"]]
    lines.append(f"- sha256 file content: {release['sha256_line'] or 'unavailable'}")
    lines.append(f"- sha256 check: {release['digest_check']}")
    return lines


def build_fact_sheet(previous_tag, target_ref, tag_name):
    version = read_project_version(target_ref)
    merged_pull_requests = list_merged_pull_requests(previous_tag, target_ref)
    repository = repository_name()
    sections = [
        f"# Release facts: {tag_name} (previous: {previous_tag})",
        "",
        f"- repository: {repository}",
        f"- version in CMakeLists.txt at {target_ref}: {version}",
        f"- previous version: {read_project_version(previous_tag)}",
        f"- tested MLX releases: {', '.join(read_tested_mlx_versions(target_ref)) or 'unknown'}",
        f"- registered tests: {count_registered_tests(previous_tag)} at {previous_tag}, "
        f"{count_registered_tests(target_ref)} at {target_ref}",
        f"- compare: https://github.com/{repository}/compare/{previous_tag}...{tag_name}",
        "", "## Pull requests merged", *format_pull_request_lines(merged_pull_requests),
        "", "## Issues closed by them or their commits (shipped)",
        *format_issue_lines(merged_pull_requests,
                            find_issues_closed_by_commits(previous_tag, target_ref)),
        "", "## All issues closed between the two refs (check each: closed is not shipped)",
        *([f"- #{issue['number']} {issue['title']} [{(issue.get('stateReason') or '').lower()}] "
           f"{issue['url']}" for issue in list_issues_closed_between(previous_tag, target_ref)]
          or ["- none"]),
        "", "## C API (include/mlx_perf/mlx_perf_c_api.h)",
        *format_api_lines(previous_tag, target_ref),
        "", "## Published release", *format_release_lines(tag_name),
        "", "## Open issues (the roadmap, not shipped)",
        *[f"- #{issue['number']} {issue['title']} {issue['url']}" for issue in list_open_issues()],
        "", "## CHANGELOG entry", read_changelog_entry(target_ref, version),
    ]
    return "\n".join(sections) + "\n"


def parse_arguments():
    argument_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_parser.add_argument("--target", default="HEAD",
                                 help="tag or ref being released (default: HEAD)")
    argument_parser.add_argument("--previous-tag",
                                 help="the release before it (default: newest tag before target)")
    argument_parser.add_argument("--tag-name",
                                 help="the tag the notes are for (default: the target if it is "
                                      "a tag, else v<version>)")
    return argument_parser.parse_args()


def main():
    arguments = parse_arguments()
    previous_tag = arguments.previous_tag or run_command(
        ["git", "describe", "--tags", "--abbrev=0", f"{arguments.target}^"]).strip()
    target_is_tag = subprocess.run(["git", "rev-parse", "-q", "--verify",
                                    f"refs/tags/{arguments.target}"],
                                   capture_output=True).returncode == 0
    tag_name = arguments.tag_name or (arguments.target if target_is_tag
                                      else f"v{read_project_version(arguments.target)}")
    sys.stdout.write(build_fact_sheet(previous_tag, arguments.target, tag_name))


if __name__ == "__main__":
    main()
