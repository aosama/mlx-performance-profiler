#include "report_launch_comparison_section.h"

#include "gpu_backend_vocabulary.h"
#include "text_format.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace mlx_perf {
namespace {

const KernelLaunchSummary* find_kernel(const std::vector<KernelLaunchSummary>& kernel_summaries,
                                       const std::string& kernel_key) {
  for (const KernelLaunchSummary& kernel_summary : kernel_summaries) {
    if (kernel_summary.kernel_key == kernel_key) {
      return &kernel_summary;
    }
  }
  return nullptr;
}

// Kernel keys in the order they first ran, the baseline's first, so the
// table reads like the job.
std::vector<std::string> collect_kernel_keys(const std::vector<KernelLaunchSummary>& baseline_kernels,
                                             const std::vector<KernelLaunchSummary>& candidate_kernels) {
  std::vector<std::string> kernel_keys;
  for (const auto* kernel_summaries : {&baseline_kernels, &candidate_kernels}) {
    for (const KernelLaunchSummary& kernel_summary : *kernel_summaries) {
      if (std::find(kernel_keys.begin(), kernel_keys.end(), kernel_summary.kernel_key) == kernel_keys.end()) {
        kernel_keys.push_back(kernel_summary.kernel_key);
      }
    }
  }
  return kernel_keys;
}

std::string render_variant_cells(const KernelLaunchSummary* kernel_summary, int gpu_core_count) {
  if (kernel_summary == nullptr) {
    return "<td class=\"number\">—</td><td class=\"number\">—</td><td class=\"number\">—</td>";
  }
  std::ostringstream cells_html;
  cells_html << "<td class=\"number\">" << kernel_summary->launch_count << "</td><td class=\"number\">"
             << kernel_summary->fewest_threadgroups;
  if (kernel_summary->most_threadgroups != kernel_summary->fewest_threadgroups) {
    cells_html << "–" << kernel_summary->most_threadgroups;
  }
  cells_html << "</td>";
  if (kernel_summary->lowest_core_coverage < 0.0) {
    cells_html << "<td class=\"number\">—</td>";
  } else {
    const bool is_unable_to_fill_gpu = kernel_summary->has_launch_unable_to_fill_gpu;
    cells_html << "<td class=\"number" << (is_unable_to_fill_gpu ? " underfills-gpu" : "") << "\">"
               << (is_unable_to_fill_gpu ? "⚠ " : "")
               << static_cast<int>(kernel_summary->lowest_core_coverage * gpu_core_count + 0.5)
               << " of " << gpu_core_count << "</td>";
  }
  return cells_html.str();
}

std::string render_header(const std::string& baseline_name, const std::string& candidate_name,
                          const std::string& key_heading) {
  std::ostringstream header_html;
  header_html << "<thead><tr><th rowspan=\"2\">" << escape_html_text(key_heading) << "</th>";
  for (const std::string* variant_name : {&baseline_name, &candidate_name}) {
    header_html << "<th colspan=\"3\">" << escape_html_text(*variant_name) << "</th>";
  }
  header_html << "</tr><tr>";
  for (int variant_index = 0; variant_index < 2; ++variant_index) {
    header_html << "<th>Launches</th><th>" << gpu_vocabulary::WORK_GROUPS_HEADING << "</th><th title=\""
                << gpu_vocabulary::COMPUTE_UNITS_TITLE << "\">" << gpu_vocabulary::COMPUTE_UNITS_HEADING << "</th>";
  }
  header_html << "</tr></thead>";
  return header_html.str();
}

} // namespace

std::string render_launch_comparison_section(const std::string& baseline_name,
                                             const std::vector<KernelLaunchSummary>& baseline_kernels,
                                             const std::string& candidate_name,
                                             const std::vector<KernelLaunchSummary>& candidate_kernels,
                                             int gpu_core_count, const std::string& key_heading) {
  if (baseline_kernels.empty() && candidate_kernels.empty()) {
    return std::string();
  }
  std::ostringstream section_html;
  section_html << "<section id=\"launches\"><h2>Kernel launches side by side</h2>"
               << "<div class=\"table-wrap\"><table>"
               << render_header(baseline_name, candidate_name, key_heading) << "<tbody>";
  for (const std::string& kernel_key : collect_kernel_keys(baseline_kernels, candidate_kernels)) {
    section_html << "<tr><td>" << escape_html_text(kernel_key) << "</td>"
                 << render_variant_cells(find_kernel(baseline_kernels, kernel_key), gpu_core_count)
                 << render_variant_cells(find_kernel(candidate_kernels, kernel_key), gpu_core_count)
                 << "</tr>";
  }
  section_html << "</tbody></table></div><p class=\"note\">Each side's kernel launches, matched by "
               << escape_html_text(key_heading == "Kernel" ? "kernel name (a custom kernel's name, otherwise"
                                                             " the MLX operation)" : gpu_vocabulary::KERNEL_FUNCTION_HEADING)
               << ". " << gpu_vocabulary::WORK_GROUP_PLACEMENT
               // An unknown core count (0) is never printed as a number.
               << (gpu_core_count > 0 ? std::to_string(gpu_core_count) + " " + gpu_vocabulary::COMPUTE_UNITS
                                      : std::string(gpu_vocabulary::COMPUTE_UNITS))
               << " leaves " << gpu_vocabulary::COMPUTE_UNITS << " idle (⚠): a common reason a kernel is slower"
               << " although it does the same work. Covering every core does not prove a launch is"
               << " efficient; it only rules this reason out.</p></section>";
  return section_html.str();
}

std::string render_launch_comparison_section(const std::string& baseline_name,
                                             const RunTelemetry& baseline_serial_telemetry,
                                             const std::string& candidate_name,
                                             const RunTelemetry& candidate_serial_telemetry) {
  return render_launch_comparison_section(
      baseline_name, summarize_kernel_launches(baseline_serial_telemetry), candidate_name,
      summarize_kernel_launches(candidate_serial_telemetry),
      baseline_serial_telemetry.gpu_device_facts.gpu_core_count, "Kernel");
}

} // namespace mlx_perf
