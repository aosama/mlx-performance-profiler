// Component tests for report_procedure_page: the procedure comparison page,
// from a synthetic pair (no GPU).

#include "procedure_fixture.h"
#include "gpu_backend_vocabulary.h"
#include "wired_memory_fixture.h"
#include "report_procedure_page.h"
#include "test_harness.h"

#include <string>

namespace {

using namespace mlx_perf;

std::string page_for(const PairedProcedureMeasurement& paired_measurement) {
  return build_procedure_comparison_document("Hand-offs", paired_measurement,
                                             compare_procedure_pair(paired_measurement));
}

bool contains(const std::string& page_html, const std::string& needle) {
  return page_html.find(needle) != std::string::npos;
}

// --- positive ---

void should_lead_with_the_duration_verdict_and_judge_three_metrics() {
  const std::string page_html = page_for(procedure_fixture::make_chatty_vs_batched());
  EXPECT_TRUE(contains(page_html, "<title>Hand-offs</title>"));
  EXPECT_TRUE(contains(page_html, "class=\"verdict faster\""));
  EXPECT_TRUE(contains(page_html, "batched is"));
  EXPECT_TRUE(contains(page_html, "Procedure duration"));
  EXPECT_TRUE(contains(page_html, "GPU busy"));
  EXPECT_TRUE(contains(page_html, "GPU idle (waiting on the host)"));
}

void should_show_each_procedures_hand_offs_per_run() {
  const std::string page_html = page_for(procedure_fixture::make_chatty_vs_batched());
  EXPECT_TRUE(contains(page_html, "id=\"handoffs\""));
  EXPECT_TRUE(contains(page_html, "<tr><td>chatty</td><td class=\"number\">8</td><td class=\"number\">8</td>"));
  EXPECT_TRUE(contains(page_html, "<tr><td>batched</td><td class=\"number\">1</td><td class=\"number\">1</td>"));
}

void should_compare_launches_by_kernel_function() {
  const std::string page_html = page_for(procedure_fixture::make_chatty_vs_batched());
  EXPECT_TRUE(contains(page_html, "id=\"launches\""));
  EXPECT_TRUE(contains(page_html, mlx_perf::gpu_vocabulary::KERNEL_FUNCTION_HEADING));
  EXPECT_TRUE(contains(page_html, "v_Add_float32"));
  EXPECT_TRUE(contains(page_html, "gemv_float32"));
}

void should_judge_memory_per_run_in_its_own_section() {
  const std::string page_html = page_for(procedure_fixture::make_chatty_vs_batched());
  EXPECT_TRUE(contains(page_html, "<section id=\"memory\"><h2>Memory per run</h2>"));
  EXPECT_TRUE(contains(page_html, "Rounds batched used less"));
  EXPECT_TRUE(!contains(page_html, "memory-pressure"));
}

void should_note_rounds_that_crossed_the_wire_limit_beside_the_timing_verdict() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  paired_measurement.candidate.rounds[4].memory_figures.wire_requests_over_limit_count = 2;
  const std::string page_html = page_for(paired_measurement);
  const std::size_t note_at = page_html.find("class=\"note memory-pressure\"");
  EXPECT_TRUE(note_at != std::string::npos && note_at < page_html.find("id=\"handoffs\""));
  EXPECT_TRUE(contains(page_html, "batched crossed the wire limit in 1 of 12 rounds"));
}

// --- edge / boundary ---

void should_show_zero_percent_busy_for_a_zero_length_run() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  for (ProcedureRun& procedure_run : paired_measurement.candidate.rounds) {
    procedure_run = ProcedureRun{};
  }
  const std::string page_html = page_for(paired_measurement);
  EXPECT_TRUE(contains(page_html, "(0%)"));
}

void should_describe_a_tie_as_within_spread() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  paired_measurement.candidate.rounds = paired_measurement.baseline.rounds;
  const std::string page_html = page_for(paired_measurement);
  EXPECT_TRUE(contains(page_html, "class=\"verdict within-spread\""));
}

void should_show_launches_without_coverage_when_the_core_count_is_unknown() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  paired_measurement.gpu_device_facts = {};
  const std::string page_html = page_for(paired_measurement);
  EXPECT_TRUE(contains(page_html, "v_Add_float32"));
  EXPECT_TRUE(!contains(page_html, "class=\"number underfills-gpu\""));
  // An unknown count is never printed as a number, on either backend.
  const std::string compute_units = mlx_perf::gpu_vocabulary::COMPUTE_UNITS;
  EXPECT_TRUE(!contains(page_html, "GPU's 0 " + compute_units));
  EXPECT_TRUE(contains(page_html, "than the GPU's " + compute_units + " leaves"));
}

void should_draw_a_wired_memory_lane_only_for_a_procedure_with_samples() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  paired_measurement.candidate.observed_wired_memory = wired_memory_fixture::release_then_unwire();
  const std::string page_html = page_for(paired_measurement);
  EXPECT_TRUE(contains(page_html, "id=\"wired-memory-candidate\""));
  EXPECT_TRUE(contains(page_html, "Wired memory: batched"));
  EXPECT_TRUE(!contains(page_html, "id=\"wired-memory-baseline\""));
}

// --- negative ---

void should_escape_procedure_names_and_the_title() {
  PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  paired_measurement.baseline.procedure_name = "<script>alert(1)</script>";
  const std::string page_html = build_procedure_comparison_document(
      "a & b", paired_measurement, compare_procedure_pair(paired_measurement));
  EXPECT_TRUE(!contains(page_html, "<script>alert(1)"));
  EXPECT_TRUE(contains(page_html, "&lt;script&gt;"));
  EXPECT_TRUE(contains(page_html, "<title>a &amp; b</title>"));
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_lead_with_the_duration_verdict_and_judge_three_metrics),
      TEST_CASE(should_show_each_procedures_hand_offs_per_run),
      TEST_CASE(should_compare_launches_by_kernel_function),
      TEST_CASE(should_judge_memory_per_run_in_its_own_section),
      TEST_CASE(should_note_rounds_that_crossed_the_wire_limit_beside_the_timing_verdict),
      TEST_CASE(should_show_zero_percent_busy_for_a_zero_length_run),
      TEST_CASE(should_describe_a_tie_as_within_spread),
      TEST_CASE(should_draw_a_wired_memory_lane_only_for_a_procedure_with_samples),
      TEST_CASE(should_show_launches_without_coverage_when_the_core_count_is_unknown),
      TEST_CASE(should_escape_procedure_names_and_the_title),
  });
}
