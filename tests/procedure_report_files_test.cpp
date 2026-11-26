// Component tests for procedure_report_files: the page and record written to
// a directory and read back, from a synthetic pair (no GPU).

#include "captured_console.h"
#include "json_value.h"
#include "procedure_fixture.h"
#include "procedure_report_files.h"
#include "test_harness.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using namespace mlx_perf;

std::string read_whole_file(const std::filesystem::path& file_path) {
  std::ifstream file_stream(file_path);
  std::ostringstream file_text;
  file_text << file_stream.rdbuf();
  return file_text.str();
}

std::filesystem::path fresh_output_directory(const std::string& directory_name) {
  const std::filesystem::path output_directory =
      std::filesystem::current_path() / "procedure_report_files_test" / directory_name;
  std::filesystem::remove_all(output_directory);
  std::filesystem::create_directories(output_directory);
  return output_directory;
}

bool write_fixture(const std::string& output_directory) {
  const PairedProcedureMeasurement paired_measurement = procedure_fixture::make_chatty_vs_batched();
  return write_procedure_report_files({"Hand-offs", output_directory, "handoffs"}, paired_measurement,
                                      compare_procedure_pair(paired_measurement));
}

// --- positive ---

void should_write_the_page_and_the_record_side_by_side() {
  const std::filesystem::path output_directory = fresh_output_directory("both");
  EXPECT_TRUE(write_fixture(output_directory.string()));
  EXPECT_TRUE(read_whole_file(output_directory / "handoffs.html").find("id=\"handoffs\"") != std::string::npos);
  const JsonValue record = parse_json(read_whole_file(output_directory / "handoffs.json"));
  EXPECT_TRUE(record.at("kind").as_string() == "procedure_comparison");
}

// --- edge / boundary ---

void should_replace_files_left_by_an_earlier_comparison() {
  const std::filesystem::path output_directory = fresh_output_directory("replace");
  std::ofstream(output_directory / "handoffs.json") << "stale";
  EXPECT_TRUE(write_fixture(output_directory.string()));
  EXPECT_TRUE(read_whole_file(output_directory / "handoffs.json").find("stale") == std::string::npos);
}

// --- negative ---

void should_report_failure_for_a_missing_directory() {
  const std::filesystem::path missing_directory = fresh_output_directory("gone") / "does_not_exist";
  captured_console::CapturedConsole captured_output("procedure_report_files_missing");
  EXPECT_TRUE(!write_fixture(missing_directory.string()));
  EXPECT_TRUE(captured_output.stderr_text().find("could not write") != std::string::npos);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_write_the_page_and_the_record_side_by_side),
      TEST_CASE(should_replace_files_left_by_an_earlier_comparison),
      TEST_CASE(should_report_failure_for_a_missing_directory),
  });
}
