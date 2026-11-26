#!/usr/bin/env python3
"""Unit tests for gather_release_facts's parsers: what decides "shipped".

A regression here silently drops an issue or an API change from release
notes, so each parser is pinned on positive, edge and negative inputs.
Pure text in, no git or GitHub needed.

Run: python3 -m unittest discover -s .claude/skills/release-notes/scripts -p "*_test.py"
"""

import unittest

import gather_release_facts as facts

SAMPLE_HEADER = """
typedef int (*mlx_perf_build_graph_fn)(mlx_vector_array fresh_outputs,
                                       void* builder_payload);
typedef struct mlx_perf_variant {
  const char* variant_name;
} mlx_perf_variant;
/* Measures the job: see mlx_perf_last_error(). */
int mlx_perf_profile_to_report(mlx_perf_build_graph_fn build_graph,
                               void* builder_payload);
const char* mlx_perf_version(void);
void mlx_perf_set_console_output(int is_enabled);
"""


def pull_request(body, linked_numbers=()):
    return {"body": body,
            "closingIssuesReferences": [{"number": number} for number in linked_numbers]}


class NumbersAfterKeywordsTest(unittest.TestCase):
    # --- positive ---
    def test_should_read_every_number_in_a_closing_list(self):
        self.assertEqual(facts.numbers_after_keywords("Closes #24 and #25", facts.CLOSING_KEYWORDS),
                         {24, 25})
        self.assertEqual(facts.numbers_after_keywords("Fixes #3, #4", facts.CLOSING_KEYWORDS),
                         {3, 4})

    # --- edge ---
    def test_should_read_keywords_in_any_case_and_tense(self):
        self.assertEqual(facts.numbers_after_keywords("RESOLVED #7", facts.CLOSING_KEYWORDS), {7})
        self.assertEqual(facts.numbers_after_keywords("closed #8", facts.CLOSING_KEYWORDS), {8})

    # --- negative ---
    def test_should_find_nothing_without_a_keyword(self):
        self.assertEqual(facts.numbers_after_keywords("Discussed in #9", facts.CLOSING_KEYWORDS),
                         set())


class FindIssueNumbersTest(unittest.TestCase):
    # --- positive ---
    def test_should_merge_githubs_links_with_body_keywords(self):
        closed, referenced = facts.find_issue_numbers(
            pull_request("Closes #24 and #25", linked_numbers=[24]))
        self.assertEqual(closed, {24, 25})
        self.assertEqual(referenced, set())

    # --- edge ---
    def test_should_not_count_an_issue_as_both_closed_and_referenced(self):
        closed, referenced = facts.find_issue_numbers(pull_request("Closes #11. Refs #11 #12"))
        self.assertEqual(closed, {11})
        self.assertEqual(referenced, {12})

    # --- negative: a reference is not a shipped issue ---
    def test_should_keep_part_of_and_refs_out_of_closed(self):
        closed, referenced = facts.find_issue_numbers(pull_request("Part of #13. Refs #11 #12"))
        self.assertEqual(closed, set())
        self.assertEqual(referenced, {11, 12, 13})

    def test_should_handle_a_pull_request_with_no_body(self):
        self.assertEqual(facts.find_issue_numbers(pull_request(None)), (set(), set()))


class HeaderParsingTest(unittest.TestCase):
    # --- positive ---
    def test_should_find_every_declared_function(self):
        self.assertEqual(facts.parse_c_api_function_names(SAMPLE_HEADER),
                         {"mlx_perf_profile_to_report", "mlx_perf_version",
                          "mlx_perf_set_console_output"})

    def test_should_find_every_typedef_struct(self):
        self.assertEqual(facts.parse_c_api_struct_names(SAMPLE_HEADER), {"mlx_perf_variant"})

    def test_should_read_the_schema_version(self):
        self.assertEqual(facts.parse_schema_version(
            "inline constexpr int MEASUREMENT_SCHEMA_VERSION = 1;"), "1")

    # --- edge ---
    def test_should_find_nothing_in_an_empty_header(self):
        self.assertEqual(facts.parse_c_api_function_names(""), set())
        self.assertIsNone(facts.parse_schema_version(""))

    # --- negative: neither the callback typedef nor a comment is a function ---
    def test_should_not_mistake_a_typedef_or_comment_for_a_function(self):
        function_names = facts.parse_c_api_function_names(SAMPLE_HEADER)
        self.assertNotIn("mlx_perf_build_graph_fn", function_names)
        self.assertNotIn("mlx_perf_last_error", function_names)


if __name__ == "__main__":
    unittest.main()
