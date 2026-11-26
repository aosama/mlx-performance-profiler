#!/usr/bin/env python3
"""Turn sections of a profiler report into the PNG that opens release notes.

The visual at the top of a release must be something the profiler itself
drew, so readers see the product, not a mock-up. A screenshot of the whole
report is too long to read at a glance, so this keeps only the sections
named (for example the verdict and the timelines) by hiding the others with
an injected style sheet, renders the page in headless Chrome, and trims the
empty background below the last kept section.

Usage (from the repository root, after a tool has written its report):
    python3 .claude/skills/release-notes/scripts/capture_report_visual.py \\
        build/qwen35_decode.html release-notes/media/v0.2.0-verdict.png \\
        --sections verdict --hide "#verdict .note"

    # A graph view, driven first (each mounted view has a small controller:
    # openStage, showOverview, setColourMode, showHotPath, fit), at an exact
    # 16:9 size for social posts:
    python3 .claude/skills/release-notes/scripts/capture_report_visual.py \\
        build/qwen35_decode_eager.html docs/images/graph.png --sections dag \\
        --page-script "const view = mlxPerfGraphView.mountedViews[0];
                       view.setColourMode('heat'); view.openStage(3);" \\
        --width 1600 --height 900 --no-trim

Needs Google Chrome (or Chromium) and Pillow. It reads the report and writes
only the PNG; the report is left untouched.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

CHROME_CANDIDATE_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)
DEFAULT_WIDTH_CSS_PIXELS = 1440
DEFAULT_WINDOW_HEIGHT_CSS_PIXELS = 4000
DEFAULT_DEVICE_SCALE_FACTOR = 2
BOTTOM_MARGIN_CSS_PIXELS = 24
PAGE_SCRIPT_DELAY_MILLISECONDS = 400
# Virtual time headless Chrome grants the page before the screenshot: enough
# for the graph view to lay out, then for a page script and its re-layout.
RENDER_VIRTUAL_TIME_BUDGET_MILLISECONDS = 4000


def find_section_ids(report_html):
    """The ids of the report's top-level <section> elements, in page order."""
    return re.findall(r'<section\s+id="([^"]+)"', report_html)


def build_section_filter_style(kept_section_ids, all_section_ids, hidden_selectors=()):
    """CSS that hides every section not kept, plus any extra elements named by
    selector (a long explanatory note inside a kept section, say). Hiding,
    rather than cutting the HTML apart, keeps the report's own layout and
    styles exactly as the profiler drew them."""
    hidden_section_selectors = [f"section#{section_id}" for section_id in all_section_ids
                                if section_id not in kept_section_ids]
    rules = [f"{selector} {{ display: none !important; }}"
             for selector in hidden_section_selectors + list(hidden_selectors)]
    return "<style>\n" + "\n".join(rules) + "\n</style>\n"


def build_page_script_element(page_script):
    """The caller's script, run after the graph views have mounted and laid
    out. It is deferred because a view lays out a frame or two after it
    mounts; headless Chrome's virtual time budget lets the delay elapse."""
    return ("<script>setTimeout(function () {\n" + page_script +
            "\n}, " + str(PAGE_SCRIPT_DELAY_MILLISECONDS) + ");</script>\n")


def inject_before_body_end(report_html, element_html):
    """Adds markup last in <body>, after the report's own scripts."""
    body_end_index = report_html.lower().rfind("</body>")
    if body_end_index < 0:
        return report_html + element_html
    return report_html[:body_end_index] + element_html + report_html[body_end_index:]


def inject_style(report_html, style_element):
    """Adds the style last in <head>, so it wins over the report's own rules."""
    head_end_index = report_html.lower().find("</head>")
    if head_end_index < 0:
        return style_element + report_html
    return report_html[:head_end_index] + style_element + report_html[head_end_index:]


def split_comma_list(comma_separated_text):
    return [entry.strip() for entry in comma_separated_text.split(",") if entry.strip()]


def find_unknown_sections(requested_section_ids, all_section_ids):
    return [section_id for section_id in requested_section_ids
            if section_id not in all_section_ids]


def find_chrome_binary():
    for candidate_path in CHROME_CANDIDATE_PATHS:
        if Path(candidate_path).exists():
            return candidate_path
    sys.exit("error: Google Chrome or Chromium is needed to render the report; "
             f"looked in {', '.join(CHROME_CANDIDATE_PATHS)}")


def render_page_to_png(chrome_path, page_path, png_path, width, height, scale):
    """One headless Chrome screenshot of the page at a fixed window size."""
    completed = subprocess.run(
        [chrome_path, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
         f"--virtual-time-budget={RENDER_VIRTUAL_TIME_BUDGET_MILLISECONDS}",
         f"--screenshot={png_path}", page_path.resolve().as_uri()],
        capture_output=True, text=True, timeout=120)
    if completed.returncode != 0 or not Path(png_path).exists():
        sys.exit(f"error: Chrome could not render {page_path}\n{completed.stderr.strip()}")


def trim_trailing_background(page_image, bottom_margin_pixels):
    """Crops the solid background below the content. Returns None when the
    content reaches the bottom edge: the window was too short and the image
    would cut a section off, which must not be published."""
    background_colour = page_image.getpixel((0, page_image.height - 1))
    solid_background = Image.new(page_image.mode, page_image.size, background_colour)
    content_box = ImageChops.difference(page_image, solid_background).getbbox()
    if content_box is None:
        return page_image
    content_bottom = content_box[3]
    if content_bottom >= page_image.height - 1:
        return None
    cropped_height = min(page_image.height, content_bottom + bottom_margin_pixels)
    return page_image.crop((0, 0, page_image.width, cropped_height))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report_html", type=Path, help="a report the profiler wrote")
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--sections", required=True,
                        help="comma-separated section ids to keep, e.g. verdict,timelines")
    parser.add_argument("--hide", default="",
                        help="comma-separated CSS selectors to hide inside kept sections, "
                             "e.g. '#verdict .note'")
    parser.add_argument("--page-script", default="",
                        help="JavaScript to run once the graph views have mounted, e.g. "
                             "\"mlxPerfGraphView.mountedViews[0].openStage(3)\"")
    parser.add_argument("--no-trim", action="store_true",
                        help="keep the exact window size (a 16:9 social image) instead of "
                             "trimming the background below the content")
    parser.add_argument("--colours", type=int, default=0,
                        help="reduce to this many colours (256 keeps UI screenshots sharp "
                             "at about a third of the size); 0 keeps full colour")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH_CSS_PIXELS)
    parser.add_argument("--height", type=int, default=DEFAULT_WINDOW_HEIGHT_CSS_PIXELS,
                        help="window height; raise it if the kept sections are taller")
    parser.add_argument("--scale", type=int, default=DEFAULT_DEVICE_SCALE_FACTOR,
                        help="device pixels per CSS pixel; 2 stays sharp on high-DPI screens")
    return parser.parse_args()


def build_capture_page(report_html, arguments):
    """The report with only the kept sections showing, plus any page script.
    Exits naming the report's sections when a requested one does not exist."""
    all_section_ids = find_section_ids(report_html)
    kept_section_ids = split_comma_list(arguments.sections)
    unknown_section_ids = find_unknown_sections(kept_section_ids, all_section_ids)
    if not kept_section_ids or unknown_section_ids:
        sys.exit(f"error: unknown or no sections {unknown_section_ids or kept_section_ids}; "
                 f"this report has: {', '.join(all_section_ids)}")
    capture_html = inject_style(report_html, build_section_filter_style(
        kept_section_ids, all_section_ids, split_comma_list(arguments.hide)))
    if arguments.page_script:
        capture_html = inject_before_body_end(capture_html,
                                              build_page_script_element(arguments.page_script))
    return capture_html


def render_capture_page(capture_html, arguments):
    """Renders the page and returns the image, trimmed unless --no-trim."""
    chrome_path = find_chrome_binary()
    with tempfile.TemporaryDirectory() as scratch_directory:
        capture_page_path = Path(scratch_directory) / "capture_page.html"
        capture_page_path.write_text(capture_html, encoding="utf-8")
        raw_png_path = Path(scratch_directory) / "page.png"
        render_page_to_png(chrome_path, capture_page_path, raw_png_path,
                           arguments.width, arguments.height, arguments.scale)
        page_image = Image.open(raw_png_path).convert("RGB")
    if arguments.no_trim:
        return page_image
    trimmed_image = trim_trailing_background(page_image, BOTTOM_MARGIN_CSS_PIXELS * arguments.scale)
    if trimmed_image is None:
        sys.exit(f"error: the kept sections are taller than {arguments.height} px; raise --height")
    return trimmed_image


def reduce_to_palette(page_image, colour_count):
    """A report screenshot is flat UI colour and text, so a palette without
    dithering keeps it sharp at about a third of the bytes, which matters for
    images embedded in a README or a release page."""
    return page_image.quantize(colors=colour_count, method=Image.Quantize.MEDIANCUT,
                               dither=Image.Dither.NONE)


def main():
    arguments = parse_arguments()
    capture_html = build_capture_page(arguments.report_html.read_text(encoding="utf-8"), arguments)
    page_image = render_capture_page(capture_html, arguments)
    if arguments.colours:
        page_image = reduce_to_palette(page_image, arguments.colours)
    arguments.output_png.parent.mkdir(parents=True, exist_ok=True)
    page_image.save(arguments.output_png, optimize=True)
    print(f"wrote {arguments.output_png}: {page_image.width}x{page_image.height} px, "
          f"{arguments.output_png.stat().st_size // 1024} KiB, sections {arguments.sections}")


if __name__ == "__main__":
    main()
