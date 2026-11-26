#!/usr/bin/env python3
"""Unit tests for capture_report_visual's page preparation and trimming:
what ends up in the published picture. Pure, no Chrome needed.

Run: python3 -m unittest discover -s .claude/skills/release-notes/scripts -p "*_test.py"
"""

import unittest

from PIL import Image

import capture_report_visual as capture

SAMPLE_REPORT = ("<html><head><style>body{}</style></head><body><header>t</header>"
                 "<section id=\"job\">j</section><section id=\"dag\">d</section>"
                 "<section  id=\"method\">m</section></body></html>")
PAGE_COLOUR = (17, 22, 29)
INK_COLOUR = (228, 233, 240)


def page_with_content_rows(height, content_rows):
    page_image = Image.new("RGB", (40, height), PAGE_COLOUR)
    for row_index in content_rows:
        page_image.putpixel((10, row_index), INK_COLOUR)
    return page_image


class CapturePageTest(unittest.TestCase):
    # --- positive ---

    def test_should_find_every_section_in_page_order(self):
        self.assertEqual(capture.find_section_ids(SAMPLE_REPORT), ["job", "dag", "method"])

    def test_should_hide_every_section_not_kept_and_each_extra_selector(self):
        style = capture.build_section_filter_style(["dag"], ["job", "dag", "method"], ["nav"])
        self.assertIn("section#job { display: none !important; }", style)
        self.assertIn("section#method { display: none !important; }", style)
        self.assertIn("nav { display: none !important; }", style)
        self.assertNotIn("section#dag", style)

    def test_should_put_the_style_last_in_head_so_it_wins(self):
        page = capture.inject_style(SAMPLE_REPORT, "<style>x</style>")
        self.assertIn("body{}</style><style>x</style></head>", page)

    def test_should_run_the_page_script_after_the_reports_own_scripts(self):
        page = capture.inject_before_body_end(SAMPLE_REPORT, capture.build_page_script_element("go()"))
        self.assertTrue(page.endswith("</script>\n</body></html>"))
        self.assertIn("setTimeout(function () {\ngo()\n}", page)

    def test_should_trim_background_below_the_last_content_row(self):
        trimmed = capture.trim_trailing_background(page_with_content_rows(100, [5, 40]), 4)
        self.assertEqual(trimmed.height, 45)

    # --- edge / boundary ---

    def test_should_keep_a_page_without_content_whole(self):
        blank_page = page_with_content_rows(30, [])
        self.assertEqual(capture.trim_trailing_background(blank_page, 4).height, 30)

    def test_should_not_trim_past_the_page_when_the_margin_overshoots(self):
        trimmed = capture.trim_trailing_background(page_with_content_rows(50, [47]), 10)
        self.assertEqual(trimmed.height, 50)

    def test_should_append_markup_to_a_page_without_head_or_body(self):
        self.assertEqual(capture.inject_style("<p>x</p>", "<style/>"), "<style/><p>x</p>")
        self.assertEqual(capture.inject_before_body_end("<p>x</p>", "<s/>"), "<p>x</p><s/>")

    def test_should_ignore_blank_entries_in_a_comma_list(self):
        self.assertEqual(capture.split_comma_list(" dag, ,verdict ,"), ["dag", "verdict"])
        self.assertEqual(capture.split_comma_list(""), [])

    # --- negative ---

    def test_should_refuse_a_picture_cut_off_by_a_short_window(self):
        self.assertIsNone(capture.trim_trailing_background(page_with_content_rows(60, [59]), 4))

    def test_should_name_sections_the_report_does_not_have(self):
        self.assertEqual(capture.find_unknown_sections(["dag", "verdict"], ["job", "dag"]), ["verdict"])


if __name__ == "__main__":
    unittest.main()
