// qwen35_profile: profile one real decode step of a dense Qwen3.5 checkpoint.
// Orchestration only: proves the forward math by greedy generation, then
// measures the step eagerly and compiled and writes the Spark-style reports.
//
//   qwen35_profile <model-directory> [--context N] [--granularity auto|node|stage]
//   (defaults: N = 1024; auto times one stage per decoder layer at this size)

#include "graph_model.h"
#include "measurement_session.h"
#include "paired_measurement.h"
#include "qwen35/qwen35_config.h"
#include "qwen35/qwen35_decode_job.h"
#include "qwen35/qwen35_generation.h"
#include "qwen35/qwen35_weights.h"
#include "qwen35/token_text.h"
#include "report_bundle.h"
#include "timeline_console.h"
#include "dispatch_recorder.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace mlx_perf::qwen35;

const std::vector<std::string> CHECK_PROMPT_ENTRIES = {"The", "Ġcapital", "Ġof", "ĠFrance",
                                                       "Ġis"};
const std::string EXPECTED_NEXT_ENTRY = "ĠParis";
constexpr int CHECK_NEW_TOKEN_COUNT = 12;
constexpr int DEFAULT_CONTEXT_LENGTH = 1024;

struct ProfileOptions {
  std::string model_directory;
  int context_length = DEFAULT_CONTEXT_LENGTH;
  mlx_perf::TimelineOptions timeline_options;
};

const char* const USAGE =
    "usage: qwen35_profile <model-directory> [--context N] [--granularity auto|node|stage]";

int parse_context_length(const std::string& context_text) {
  char* parse_end = nullptr;
  const long context_length = std::strtol(context_text.c_str(), &parse_end, 10);
  if (context_text.empty() || *parse_end != '\0' || context_length < 0 ||
      context_length > 1 << 20) {
    throw std::invalid_argument("--context needs a whole number from 0 to 1048576");
  }
  return static_cast<int>(context_length);
}

mlx_perf::GranularityChoice parse_granularity(const std::string& granularity_text) {
  if (granularity_text == "auto") return mlx_perf::GranularityChoice::automatic;
  if (granularity_text == "node") return mlx_perf::GranularityChoice::per_node;
  if (granularity_text == "stage") return mlx_perf::GranularityChoice::per_stage;
  throw std::invalid_argument("--granularity needs auto, node or stage");
}

// Flags come in name/value pairs after the model directory, in any order.
ProfileOptions parse_profile_options(int argument_count, char** argument_values) {
  if (argument_count < 2 || argument_count % 2 != 0) {
    throw std::invalid_argument(USAGE);
  }
  ProfileOptions profile_options{argument_values[1], DEFAULT_CONTEXT_LENGTH, {}};
  for (int flag_index = 2; flag_index < argument_count; flag_index += 2) {
    const std::string flag_name = argument_values[flag_index];
    const std::string flag_value = argument_values[flag_index + 1];
    if (flag_name == "--context") {
      profile_options.context_length = parse_context_length(flag_value);
    } else if (flag_name == "--granularity") {
      profile_options.timeline_options.granularity_choice = parse_granularity(flag_value);
    } else {
      throw std::invalid_argument(USAGE);
    }
  }
  return profile_options;
}

std::vector<int> lookup_prompt(const TokenText& token_text) {
  std::vector<int> prompt_token_ids;
  for (const std::string& prompt_entry : CHECK_PROMPT_ENTRIES) {
    const int token_id = token_text.find_token_id(prompt_entry);
    if (token_id < 0) {
      throw std::runtime_error("vocabulary has no entry " + prompt_entry);
    }
    prompt_token_ids.push_back(token_id);
  }
  return prompt_token_ids;
}

// The gate every timing depends on: a wrong layer still runs (and times)
// fine, so only a correct continuation proves the graph is the real model.
bool prove_forward_is_correct(const Qwen35Weights& weights, const Qwen35TextConfig& config,
                              const TokenText& token_text) {
  const std::vector<int> prompt_token_ids = lookup_prompt(token_text);
  const auto started_at = std::chrono::steady_clock::now();
  const std::vector<int> generated_ids =
      generate_greedy(weights, config, prompt_token_ids, CHECK_NEW_TOKEN_COUNT);
  const double elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
  std::printf("correctness check: \"%s\" → \"%s\"  (%.2fs)\n",
              token_text.decode(prompt_token_ids).c_str(),
              token_text.decode(generated_ids).c_str(), elapsed_seconds);
  const bool is_correct = generated_ids.front() == token_text.find_token_id(EXPECTED_NEXT_ENTRY);
  std::printf("  first generated token %s the expected \"%s\"\n",
              is_correct ? "IS" : "is NOT", token_text.decode({token_text.find_token_id(EXPECTED_NEXT_ENTRY)}).c_str());
  return is_correct;
}

void print_variant_summary(const mlx_perf::VariantMeasurement& variant_measurement) {
  std::printf("\n######## %s ########", variant_measurement.variant_name.c_str());
  mlx_perf::print_job_summary(variant_measurement.job_totals,
                              variant_measurement.gpu_timeline);
  mlx_perf::print_stream_table(variant_measurement.gpu_timeline);
  mlx_perf::print_stage_table(variant_measurement.gpu_timeline);
}

// Measures one decode step at `context_length`, eager vs compiled, with the
// passes interleaved so the verdict is judged against the spread.
bool profile_decode_step(const Qwen35Weights& weights, const Qwen35TextConfig& config,
                         const ProfileOptions& profile_options, int token_id,
                         const std::string& model_name) {
  std::printf("\nprofiling one decode step at context %d...\n", profile_options.context_length);
  const auto flat_inputs =
      make_profiling_inputs(config, profile_options.context_length, token_id);
  const EagerDecodeGraphFactory eager_factory(weights, config, flat_inputs);
  const CompiledDecodeGraphFactory compiled_factory(weights, config, flat_inputs);
  const auto stream_index_to_stream =
      mlx_perf::collect_streams(mlx_perf::capture_graph(eager_factory.build_fresh_graph()));
  const auto paired_measurement = mlx_perf::measure_pair(
      {"eager", eager_factory}, {"compiled", compiled_factory}, stream_index_to_stream,
      profile_options.timeline_options);
  print_variant_summary(paired_measurement.baseline);
  print_variant_summary(paired_measurement.candidate);
  std::printf("\n");
  return mlx_perf::write_report_bundle(
      {model_name + " — one decode step at context " +
           std::to_string(profile_options.context_length),
       "build", "qwen35_decode"},
      paired_measurement);
}

} // namespace

int main(int argument_count, char** argument_values) {
  // Before the correctness check compiles the model's kernels, so the
  // profile names every kernel launch it records.
  mlx_perf::install_dispatch_recorder();
  try {
    const ProfileOptions profile_options = parse_profile_options(argument_count, argument_values);
    const Qwen35TextConfig config = load_qwen35_text_config(profile_options.model_directory);
    const Qwen35Weights weights = load_qwen35_weights(profile_options.model_directory, config);
    materialize_qwen35_weights(weights);
    const TokenText token_text(profile_options.model_directory);
    std::printf("loaded %zu layers from %s\n", weights.layers.size(),
                profile_options.model_directory.c_str());
    if (!prove_forward_is_correct(weights, config, token_text)) {
      std::fprintf(stderr, "error: the forward pass is wrong; refusing to profile it\n");
      return 1;
    }
    const std::string model_name =
        std::filesystem::path(profile_options.model_directory).filename().string();
    return profile_decode_step(weights, config, profile_options,
                               token_text.find_token_id(EXPECTED_NEXT_ENTRY), model_name)
        ? 0
        : 1;
  } catch (const std::exception& failure) {
    std::fprintf(stderr, "error: %s\n", failure.what());
    return 1;
  }
}
