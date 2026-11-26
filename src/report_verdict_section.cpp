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

// The interval the verdict is judged on: "−12.0 µs to −9.0 µs".
std::string format_difference_interval(const PairedComparison& paired_comparison) {
  return format_signed_microseconds(paired_comparison.difference_interval_low) + " to " +
         format_signed_microseconds(paired_comparison.difference_interval_high);
}

// "96%": the interval's coverage, as the page states it.
std::string format_coverage(const PairedComparison& paired_comparison) {
  char coverage_text[16];
  std::snprintf(coverage_text, sizeof(coverage_text), "%.0f%%",
                100.0 * paired_comparison.difference_interval_coverage);
  return coverage_text;
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
           << escape_html_text(format_difference_interval(paired_comparison)) << " ("
           << format_coverage(paired_comparison) << ")</td><td>" << verdict_cell_text(paired_comparison.verdict)
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
  const std::string difference =
      format_microseconds(std::abs(paired_comparison.median_difference));
  const std::string interval_clause = format_coverage(paired_comparison) +
                                      " interval for the per-round difference";
  if (paired_comparison.verdict == ComparisonVerdict::within_spread) {
    const std::string reason =
        paired_comparison.difference_interval_coverage < MINIMUM_INTERVAL_COVERAGE
        ? std::to_string(paired_comparison.round_count) + " rounds are too few to judge."
        : "the " + interval_clause + " (" + format_difference_interval(paired_comparison) +
              ") includes zero.";
    return "The " + difference + " difference between " + candidate_name + " and " +
           baseline_name + " is not a result: " + reason;
  }
  const bool is_faster = paired_comparison.verdict == ComparisonVerdict::candidate_faster;
  const int rounds_won = is_faster
      ? paired_comparison.rounds_candidate_faster
      : paired_comparison.round_count - paired_comparison.rounds_candidate_faster;
  return candidate_name + " is " + format_percent(paired_comparison.relative_difference) +
         (is_faster ? " faster" : " slower") + " than " + baseline_name + " (" + difference +
         " of " + format_microseconds(paired_comparison.baseline.median) + ", " +
         (is_faster ? "faster" : "slower") + " in " + std::to_string(rounds_won) + " of " +
         std::to_string(paired_comparison.round_count) + " rounds). " + interval_clause + ": " +
         format_difference_interval(paired_comparison) + ". The difference exceeds the spread.";
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
               << " (median of rounds)</th><th>Interval for the per-round difference</th>"
               << "<th>Verdict</th><th>Rounds " << escape_html_text(candidate_name)
               << " was faster</th></tr></thead><tbody>"
               << render_metric_row("Job duration", pair_comparisons.job_wall)
               << render_metric_row("GPU span", pair_comparisons.gpu_span)
               << render_metric_row("Host submission", pair_comparisons.host_submission)
               << "</tbody></table></div><p class=\"note\">Each round runs both variants,"
               << " uninstrumented, flipping which goes first; the verdict is judged on the"
               << " per-round differences (candidate − baseline), so drift both share within a"
               << " round cancels. A difference counts only when the whole interval for the"
               << " median per-round difference lies on one side of zero: at least 10 of 12"
               << " rounds, so one or two disturbed rounds cannot decide it, and two identical"
               << " variants are called different under 4% of the time per metric. Job"
               << " duration is the headline. Host submission is host time inside the job's"
               << " single async_eval. GPU span runs from the job's first command buffer"
               << " starting to its last one ending: it includes GPU gaps while the host is"
               << " still encoding, and its end is an upper bound by the same small amount for"
               << " both variants. Every sample is in the comparison's JSON file. "
               << escape_html_text(describe_output_agreement(output_agreement)) << "</p>";
  return section_html.str();
}

} // namespace mlx_perf
