#include "report_verdict_section.h"

#include "text_format.h"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace mlx_perf {
namespace {

std::string format_percent(double relative_difference) {
  char percent_text[32];
  std::snprintf(percent_text, sizeof(percent_text), "%.1f%%",
                100.0 * std::abs(relative_difference));
  return percent_text;
}

std::string format_median_and_range(const SampleSummary& sample_summary) {
  return format_microseconds(sample_summary.median) + " (" +
         format_microseconds(sample_summary.minimum) + " – " +
         format_microseconds(sample_summary.maximum) + ")";
}

const char* verdict_css_class(ComparisonVerdict verdict) {
  switch (verdict) {
    case ComparisonVerdict::candidate_faster: return "faster";
    case ComparisonVerdict::candidate_slower: return "slower";
    case ComparisonVerdict::within_spread: break;
  }
  return "within-spread";
}

const char* verdict_cell_text(ComparisonVerdict verdict) {
  switch (verdict) {
    case ComparisonVerdict::candidate_faster: return "faster, beyond the spread";
    case ComparisonVerdict::candidate_slower: return "slower, beyond the spread";
    case ComparisonVerdict::within_spread: break;
  }
  return "inside the spread: not a result";
}

std::string render_metric_row(const char* metric_name, const PairedComparison& paired_comparison) {
  const double median_difference = paired_comparison.median_difference;
  std::ostringstream row_html;
  row_html << "<tr><td>" << metric_name << "</td><td class=\"number\">"
           << escape_html_text(format_median_and_range(paired_comparison.baseline))
           << "</td><td class=\"number\">"
           << escape_html_text(format_median_and_range(paired_comparison.candidate))
           << "</td><td class=\"number\">" << (median_difference < 0 ? "−" : "+")
           << escape_html_text(format_microseconds(std::abs(median_difference))) << " ("
           << (median_difference < 0 ? "−" : "+")
           << format_percent(paired_comparison.relative_difference) << ")</td><td>"
           << verdict_cell_text(paired_comparison.verdict) << "</td><td class=\"number\">"
           << paired_comparison.rounds_candidate_faster << " of "
           << paired_comparison.round_count << "</td></tr>";
  return row_html.str();
}

std::string describe_output_agreement(const OutputAgreement& output_agreement) {
  char difference_text[64];
  std::snprintf(difference_text, sizeof(difference_text), "%.3g",
                output_agreement.max_absolute_difference);
  return "Both variants produce " + std::to_string(output_agreement.output_count) +
         " output(s) of the same shapes and dtypes; the largest element-wise difference"
         " is " + difference_text + ".";
}

} // namespace

std::string describe_verdict(const std::string& baseline_name, const std::string& candidate_name,
                             const PairedComparison& paired_comparison) {
  const std::string spread_clause =
      "Spread: " + baseline_name + " " +
      format_microseconds(paired_comparison.baseline.spread()) + ", " + candidate_name + " " +
      format_microseconds(paired_comparison.candidate.spread()) + ".";
  const std::string difference =
      format_microseconds(std::abs(paired_comparison.median_difference));
  if (paired_comparison.verdict == ComparisonVerdict::within_spread) {
    return "The " + difference + " difference between " + candidate_name + " and " +
           baseline_name + " is inside the spread: not a result. " + spread_clause;
  }
  const bool is_faster = paired_comparison.verdict == ComparisonVerdict::candidate_faster;
  return candidate_name + " is " + format_percent(paired_comparison.relative_difference) +
         (is_faster ? " faster" : " slower") + " than " + baseline_name + " (" + difference +
         " of " + format_microseconds(paired_comparison.baseline.median) + "). " +
         spread_clause + " The difference exceeds the spread.";
}

std::string render_verdict_section(const std::string& baseline_name,
                                   const std::string& candidate_name,
                                   const PairComparisons& pair_comparisons,
                                   const OutputAgreement& output_agreement) {
  std::ostringstream section_html;
  section_html << "<p class=\"verdict " << verdict_css_class(pair_comparisons.job_wall.verdict)
               << "\">" << escape_html_text(describe_verdict(baseline_name, candidate_name,
                                                             pair_comparisons.job_wall))
               << "</p><div class=\"table-wrap\"><table><thead><tr><th>Metric</th><th>"
               << escape_html_text(baseline_name) << " median (range)</th><th>"
               << escape_html_text(candidate_name) << " median (range)</th><th>"
               << escape_html_text(candidate_name) << " − " << escape_html_text(baseline_name)
               << "</th><th>Verdict</th><th>Rounds " << escape_html_text(candidate_name)
               << " was faster</th></tr></thead><tbody>"
               << render_metric_row("Job duration", pair_comparisons.job_wall)
               << render_metric_row("GPU execution", pair_comparisons.gpu_execution)
               << render_metric_row("Host submission", pair_comparisons.host_submission)
               << "</tbody></table></div><p class=\"note\">A difference counts only when every"
               << " sample of one variant beats every sample of the other: overlapping ranges"
               << " are run-to-run noise. Passes alternated between the variants, flipping the"
               << " order each round. Host submission is host time inside async_eval (encoding"
               << " and committing the work). "
               << escape_html_text(describe_output_agreement(output_agreement)) << "</p>";
  return section_html.str();
}

} // namespace mlx_perf
