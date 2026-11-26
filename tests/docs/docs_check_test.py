#!/usr/bin/env python3
"""Unit tests for docs_check: each check must catch its kind of drift and
leave correct documentation alone. Synthetic inputs; no build or GPU.

Run: python3 -m unittest discover -s tests/docs -p "*_test.py"
"""

import tempfile
import unittest
from pathlib import Path

import docs_check

HEADER_TEXT = """
/* Measures the job. */
int mlx_perf_profile_to_report(mlx_perf_build_graph_fn build_graph,
                               void* builder_payload);
const char* mlx_perf_version(void);
typedef struct mlx_perf_job_summary { double job_wall_median_us; } mlx_perf_job_summary;
enum {
  MLX_PERF_VERDICT_WITHIN_SPREAD = 0
};
"""
# The definitions a complete reference quotes whole, as in docs/reference.
QUOTED_DEFINITIONS = ("typedef struct mlx_perf_job_summary { double job_wall_median_us; } mlx_perf_job_summary;\n"
                      "enum { MLX_PERF_VERDICT_WITHIN_SPREAD = 0 };\n")
MEASUREMENT_SCHEMA = {
    "type": "object", "required": ["schema_version", "nodes"],
    "properties": {"schema_version": {"const": 1},
                   "nodes": {"type": "array", "items": {
                       "type": "object", "required": ["compute_us"],
                       "properties": {"compute_us": {"type": ["number", "null"]}}}}}}


class ExampleEmbedTest(unittest.TestCase):
    # --- positive ---

    def test_should_find_an_embedded_example_and_its_body(self):
        markdown_text = "<!-- example: docs/a.c -->\n```c\nint a;\nint b;\n```\n"
        self.assertEqual(docs_check.find_example_embeds(markdown_text), [("docs/a.c", "int a;\nint b;")])

    def test_should_sync_a_block_from_its_file_and_leave_the_rest(self):
        with tempfile.TemporaryDirectory() as repository_directory:
            repository_root = Path(repository_directory)
            (repository_root / "a.c").write_text("int fresh;\n", encoding="utf-8")
            markdown_path = repository_root / "page.md"
            markdown_path.write_text("Intro.\n\n<!-- example: a.c -->\n```c\nint stale;\n```\n\nOutro.\n",
                                     encoding="utf-8")
            self.assertTrue(docs_check.sync_examples(markdown_path, repository_root))
            self.assertEqual(markdown_path.read_text(encoding="utf-8"),
                             "Intro.\n\n<!-- example: a.c -->\n```c\nint fresh;\n```\n\nOutro.\n")
            self.assertEqual(docs_check.find_drifted_examples(markdown_path, repository_root), [])

    # --- edge / boundary ---

    def test_should_end_an_empty_placeholder_at_its_own_closing_fence(self):
        # An empty block once ran on to the next block's fence and the sync
        # deleted every line in between.
        markdown_text = "<!-- example: a.c -->\n```c\n```\n\nKeep this.\n\n```text\nother\n```\n"
        match = docs_check.EXAMPLE_MARKER.search(markdown_text)
        self.assertEqual(match.group(0), "<!-- example: a.c -->\n```c\n```\n")
        self.assertEqual(docs_check.find_example_embeds(markdown_text), [("a.c", "")])

    def test_should_ignore_a_block_without_a_marker(self):
        self.assertEqual(docs_check.find_example_embeds("```c\nint a;\n```\n"), [])

    # --- negative ---

    def test_should_report_a_copy_that_differs_from_its_file(self):
        with tempfile.TemporaryDirectory() as repository_directory:
            repository_root = Path(repository_directory)
            (repository_root / "a.c").write_text("int fresh;\n", encoding="utf-8")
            markdown_path = repository_root / "page.md"
            markdown_path.write_text("<!-- example: a.c -->\n```c\nint stale;\n```\n", encoding="utf-8")
            self.assertEqual(len(docs_check.find_drifted_examples(markdown_path, repository_root)), 1)

    def test_should_refuse_to_sync_a_page_that_embeds_a_missing_file(self):
        with tempfile.TemporaryDirectory() as repository_directory:
            markdown_path = Path(repository_directory) / "page.md"
            original_text = "<!-- example: gone.c -->\n```c\nint a;\n```\n"
            markdown_path.write_text(original_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                docs_check.sync_examples(markdown_path, Path(repository_directory))
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), original_text)

    def test_should_report_an_embed_of_a_missing_file(self):
        with tempfile.TemporaryDirectory() as repository_directory:
            markdown_path = Path(repository_directory) / "page.md"
            markdown_path.write_text("<!-- example: gone.c -->\n```c\n```\n", encoding="utf-8")
            problems = docs_check.find_drifted_examples(markdown_path, Path(repository_directory))
            self.assertIn("does not exist", problems[0])


class LinkTest(unittest.TestCase):
    def write_pages(self, repository_root, page_text):
        (repository_root / "other.md").write_text("# Other page\n\n## The `job` record\n", encoding="utf-8")
        markdown_path = repository_root / "page.md"
        markdown_path.write_text(page_text, encoding="utf-8")
        return markdown_path

    # --- positive ---

    def test_should_accept_links_to_existing_files_and_headings(self):
        with tempfile.TemporaryDirectory() as repository_directory:
            markdown_path = self.write_pages(Path(repository_directory),
                                             "[a](other.md) [b](other.md#the-job-record) [c](https://x.y/z)")
            self.assertEqual(docs_check.find_broken_links(markdown_path, Path(repository_directory)), [])

    # --- edge / boundary ---

    def test_should_make_github_anchors_from_headings_with_code_and_punctuation(self):
        self.assertEqual(docs_check.heading_anchor("The `job` record"), "the-job-record")
        self.assertEqual(docs_check.heading_anchor("Step 1: get an MLX prefix"), "step-1-get-an-mlx-prefix")
        self.assertEqual(docs_check.heading_anchor("mlx_perf_version"), "mlx_perf_version")

    def test_should_not_read_code_inside_a_block_as_a_link(self):
        with tempfile.TemporaryDirectory() as repository_directory:
            markdown_path = self.write_pages(Path(repository_directory),
                                             "```c\nvalues[index](argument);\n```\n")
            self.assertEqual(docs_check.find_broken_links(markdown_path, Path(repository_directory)), [])

    def test_should_not_take_headings_from_inside_code_blocks(self):
        self.assertEqual(docs_check.find_heading_anchors("```sh\n# not a heading\n```\n## Real\n"), {"real"})

    # --- negative ---

    def test_should_report_a_missing_file_and_a_missing_heading(self):
        with tempfile.TemporaryDirectory() as repository_directory:
            markdown_path = self.write_pages(Path(repository_directory),
                                             "[a](gone.md) [b](other.md#no-such-heading)")
            problems = docs_check.find_broken_links(markdown_path, Path(repository_directory))
            self.assertEqual(len(problems), 2)


class ReferenceAndMessageTest(unittest.TestCase):
    # --- positive ---

    def test_should_find_every_prototype_struct_and_constant(self):
        prototypes, struct_names, constant_names = docs_check.find_c_api_declarations(HEADER_TEXT)
        self.assertEqual(prototypes, [
            "int mlx_perf_profile_to_report(mlx_perf_build_graph_fn build_graph, void* builder_payload);",
            "const char* mlx_perf_version(void);"])
        self.assertEqual(struct_names, ["mlx_perf_job_summary"])
        self.assertEqual(constant_names, ["MLX_PERF_VERDICT_WITHIN_SPREAD"])

    def test_should_accept_a_quoted_message_with_variable_parts(self):
        library_source = 'throw std::runtime_error("cannot write the report into " + directory);'
        troubleshooting = "> error: `cannot write the report into <directory>`\n"
        self.assertEqual(docs_check.find_unknown_messages(troubleshooting, "error", library_source), [])

    # --- edge / boundary ---

    def test_should_match_a_prototype_however_the_reference_wraps_it(self):
        reference_text = ("int mlx_perf_profile_to_report(mlx_perf_build_graph_fn build_graph,\n"
                          "    void* builder_payload);\nconst char* mlx_perf_version(void);\n" + QUOTED_DEFINITIONS)
        self.assertEqual(docs_check.find_undocumented_declarations(HEADER_TEXT, reference_text), [])

    # --- negative ---

    def test_should_report_a_changed_prototype_and_a_missing_constant(self):
        reference_text = ("int mlx_perf_profile_to_report(void);\nconst char* mlx_perf_version(void);\n" +
                          QUOTED_DEFINITIONS.replace("MLX_PERF_VERDICT_WITHIN_SPREAD", "MLX_PERF_VERDICT_RENAMED"))
        problems = docs_check.find_undocumented_declarations(HEADER_TEXT, reference_text)
        # The changed prototype, the missing constant, and the enum's stale copy.
        self.assertEqual(len(problems), 3)

    def test_should_report_a_struct_whose_copy_lacks_a_new_field(self):
        header_with_new_field = HEADER_TEXT.replace("double job_wall_median_us;",
                                                    "double job_wall_median_us; int new_count;")
        reference_text = HEADER_TEXT  # the reference still quotes the old struct
        problems = docs_check.find_undocumented_declarations(header_with_new_field, reference_text)
        self.assertTrue(any("differs from the header's definition" in problem for problem in problems))

    def test_should_report_a_message_the_library_never_produces(self):
        troubleshooting = "> error: `the job exploded`\n"
        self.assertEqual(len(docs_check.find_unknown_messages(troubleshooting, "error", "other text")), 1)


class Context7AndVersionTest(unittest.TestCase):
    # --- positive ---

    def test_should_accept_a_valid_configuration(self):
        with tempfile.TemporaryDirectory() as repository_directory:
            (Path(repository_directory) / "docs").mkdir()
            context7_config = {"folders": ["docs"], "rules": ["Call x before y."], "excludeFiles": ["NOTES.md"]}
            self.assertEqual(docs_check.find_context7_problems(context7_config, Path(repository_directory)), [])

    # --- edge / boundary ---

    def test_should_accept_a_rule_of_exactly_the_maximum_length(self):
        context7_config = {"rules": ["r" * docs_check.CONTEXT7_MAX_RULE_LENGTH]}
        self.assertEqual(docs_check.find_context7_problems(context7_config, Path(".")), [])

    # --- negative ---

    def test_should_report_each_way_context7_would_reject_the_file(self):
        context7_config = {"folders": ["missing_folder"],
                           "rules": ["r" * (docs_check.CONTEXT7_MAX_RULE_LENGTH + 1)],
                           "excludeFiles": ["docs/NOTES.md"], "url": "https://context7.com/a/b"}
        problems = docs_check.find_context7_problems(context7_config, Path("/nonexistent"))
        self.assertEqual(len(problems), 4)

    def test_should_report_a_pin_to_another_release(self):
        markdown_text = 'GIT_TAG v0.2.0)\n--branch v0.3.0\n#define MLX_PERF_VERSION_STRING "0.2.0"\n'
        problems = docs_check.find_stale_version_pins(markdown_text, "0.3.0", "page.md")
        self.assertEqual(len(problems), 2)


class SchemaTest(unittest.TestCase):
    # --- positive ---

    def test_should_accept_a_file_that_matches_the_schema(self):
        measurements = {"schema_version": 1, "nodes": [{"compute_us": 1.5}, {"compute_us": None}]}
        self.assertEqual(docs_check.find_schema_mismatches(MEASUREMENT_SCHEMA, measurements), [])

    # --- edge / boundary ---

    def test_should_accept_an_empty_array_and_not_count_a_boolean_as_a_number(self):
        self.assertEqual(docs_check.find_schema_mismatches(MEASUREMENT_SCHEMA, {"schema_version": 1, "nodes": []}), [])
        self.assertEqual(len(docs_check.find_schema_mismatches(
            MEASUREMENT_SCHEMA, {"schema_version": 1, "nodes": [{"compute_us": True}]})), 1)

    # --- negative ---

    def test_should_report_an_unreadable_file_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as scratch_directory:
            broken_path = Path(scratch_directory) / "broken.json"
            broken_path.write_text("{not json", encoding="utf-8")
            for unreadable_path in (broken_path, Path(scratch_directory) / "missing.json"):
                problems = docs_check.check_file_against_schema(MEASUREMENT_SCHEMA, unreadable_path)
                self.assertEqual(len(problems), 1)
                self.assertIn("cannot read it as JSON", problems[0])

    def test_should_report_missing_undocumented_and_mistyped_fields(self):
        measurements = {"schema_version": 2, "nodes": [{"gpu_us": 1.0}]}
        problems = docs_check.find_schema_mismatches(MEASUREMENT_SCHEMA, measurements)
        self.assertEqual(len(problems), 3)
        self.assertTrue(any("is not in the schema" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
