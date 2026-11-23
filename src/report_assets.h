#pragma once

// Report assets: one concern only — the static CSS and JavaScript the HTML
// report inlines. Inlined (not linked) so the report is one self-contained
// file that opens from disk, offline, and can be attached or archived as is.

#include <string_view>

namespace mlx_plan {

std::string_view report_stylesheet();
// Click-to-sort for any <table class="sortable">: numeric when every cell in
// the column carries data-sort-value, text otherwise. No library: ~30 lines
// do not earn a dependency.
std::string_view report_sort_script();
// Just the enableSortableTables(root) definition, for pages that insert
// tables after load (the live view) and must re-apply it per insertion.
std::string_view report_sort_function();

} // namespace mlx_plan
