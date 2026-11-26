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

// "−11.0 µs" / "+5.0 µs": differences always carry their sign.
std::string format_signed_microseconds(double signed_microseconds) {
  return (signed_microseconds < 0 ? "−" : "+") +
         format_microseconds(std::abs(signed_microseconds));
}

// The middle half of the per-round differences, the range the verdict is
// judged on: "−11.0 µs to −10.5 µs".
std::string format_difference_middle_half(const SampleSummary& round_difference) {
  return format_signed_microseconds(round_difference.lower_quartile) + " to " +
         format_signed_microseconds(round_difference.upper_quartile);
}

// A variant's median beside the middle half of its own rounds.
std::string format_median_and_middle_half(const SampleSummary& sample_summary) {
  return format_microseconds(sample_summary.median) + " (" +
         format_microseconds(sample_summary.lower_quartile) + " – " +
         format_microseconds(sample_summary.upper_quartile) + ")";
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
  std::ostringstream row_html;
  row_html << "<tr><td>" << metric_name << "</td><td class=\"number\">"
           << escape_html_text(format_median_and_middle_half(paired_comparison.baseline))
           << "</td><td class=\"number\">"
           << escape_html_text(format_median_and_middle_half(paired_comparison.candidate))
           << "</td><td class=\"number\">"
           << escape_html_text(format_signed_microseconds(paired_comparison.median_difference))
           << " (" << (paired_comparison.median_difference < 0 ? "−" : "+")
           << format_percent(paired_comparison.relative_difference) << ")</td><td class=\"number\">"
           << escape_html_text(format_difference_middle_half(paired_comparison.round_difference))
           << "</td><td>" << verdict_cell_text(paired_comparison.verdict)
           << "</td><td class=\"number\">" << paired_comparison.rounds_candidate_faster << " of "
           << paired_comparison.round_count << "</td></tr>";
  return row_html.str();
}

std::string describe_output_agreement(const OutputAgreement& output_agreement) {
  const std::string same_kind = "Both variants produce " +
                                std::to_string(output_agreement.output_count) +
                                " output(s) of the same shapes and dtypes";
  if (std::isnan(output_agreement.max_absolute_difference)) {
    return same_kind + ", but where they differ the values are not finite (NaN or infinity),"
                       " so their agreement cannot be measured.";
  }
  char difference_text[64];
  std::snprintf(difference_text, sizeof(difference_text), "%.3g",
                output_agreement.max_absolute_difference);
  return same_kind + "; the largest element-wise difference is " + difference_text + ".";
}

} // namespace

std::string describe_verdict(const std::string& baseline_name, const std::string& candidate_name,
                             const PairedComparison& paired_comparison) {
  const std::string difference_middle_half =
      format_difference_middle_half(paired_comparison.round_difference);
  const std::string difference =
      format_microseconds(std::abs(paired_comparison.median_difference));
  if (paired_comparison.verdict == ComparisonVerdict::within_spread) {
    return "The " + difference + " difference between " + candidate_name + " and " +
           baseline_name + " is inside the spread: the middle half of the per-round" +
           " differences (" + difference_middle_half + ") includes zero. Not a result.";
  }
  const bool is_faster = paired_comparison.verdict == ComparisonVerdict::candidate_faster;
  return candidate_name + " is " + format_percent(paired_comparison.relative_difference) +
         (is_faster ? " faster" : " slower") + " than " + baseline_name + " (" + difference +
         " of " + format_microseconds(paired_comparison.baseline.median) +
         "). Middle half of the per-round differences: " + difference_middle_half +
         ". The difference exceeds the spread.";
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
               << escape_html_text(baseline_name) << " median (middle half)</th><th>"
               << escape_html_text(candidate_name) << " median (middle half)</th><th>"
               << escape_html_text(candidate_name) << " − " << escape_html_text(baseline_name)
               << " (median of rounds)</th><th>Middle half of per-round differences</th>"
               << "<th>Verdict</th><th>Rounds " << escape_html_text(candidate_name)
               << " was faster</th></tr></thead><tbody>"
               << render_metric_row("Job duration", pair_comparisons.job_wall)
               << render_metric_row("GPU execution", pair_comparisons.gpu_execution)
               << render_metric_row("Host submission", pair_comparisons.host_submission)
               << "</tbody></table></div><p class=\"note\">Each round runs both variants;"
               << " the verdict is judged on the per-round differences (candidate − baseline),"
               << " so drift both variants share within a round cancels. A difference counts"
               << " only when the middle half of those differences (25th to 75th percentile)"
               << " lies entirely on one side of zero, so one disturbed round cannot decide it."
               << " Every sample is in the comparison's JSON file. The order flips each round."
               << " Host submission is host time inside async_eval (encoding and committing the"
               << " work). "
               << escape_html_text(describe_output_agreement(output_agreement)) << "</p>";
  return section_html.str();
}

} // namespace mlx_perf
