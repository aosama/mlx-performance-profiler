// Unit tests for qwen35_config: reading config.json into the decode config.

#include "qwen35/qwen35_config.h"
#include "temporary_model_directory.h"
#include "test_harness.h"

#include <stdexcept>
#include <string>

namespace {

using namespace mlx_plan::qwen35;

const std::string TEXT_CONFIG_FIELDS = R"(
  "hidden_size": 1024, "intermediate_size": 3584, "vocab_size": 248320,
  "rms_norm_eps": 1e-06, "num_attention_heads": 8, "num_key_value_heads": 2,
  "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 16,
  "linear_key_head_dim": 128, "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"])";

std::string vision_language_config(const std::string& extra_text_fields) {
  return R"({"quantization": {"group_size": 64, "bits": 4},
             "text_config": {)" + TEXT_CONFIG_FIELDS + extra_text_fields + "}}";
}

bool loading_throws(const std::string& config_text) {
  const test_support::TemporaryModelDirectory model_directory;
  model_directory.write_file("config.json", config_text);
  try {
    load_qwen35_text_config(model_directory.path());
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

// --- positive ---

void should_read_a_vision_language_checkpoint_with_rope_parameters() {
  const test_support::TemporaryModelDirectory model_directory;
  model_directory.write_file(
      "config.json",
      vision_language_config(R"(, "rope_parameters": {"rope_theta": 10000000,
                                                    "partial_rotary_factor": 0.25})"));
  const Qwen35TextConfig config = load_qwen35_text_config(model_directory.path());
  EXPECT_TRUE(config.hidden_size == 1024 && config.attention_head_dim == 256);
  EXPECT_TRUE(config.rotary_dims == 64);
  EXPECT_TRUE(config.rope_theta == 1e7f);
  EXPECT_TRUE(config.quantization_group_size == 64);
  EXPECT_TRUE(config.layer_kinds.size() == 4);
  EXPECT_TRUE(config.layer_kinds[3] == LayerKind::full_attention);
}

// --- edge / boundary ---

void should_read_a_text_only_checkpoint_with_top_level_rope_fields() {
  const test_support::TemporaryModelDirectory model_directory;
  model_directory.write_file(
      "config.json", R"({"quantization": {"group_size": 32}, )" + TEXT_CONFIG_FIELDS +
                         R"(, "rope_theta": 5000, "partial_rotary_factor": 0.5})");
  const Qwen35TextConfig config = load_qwen35_text_config(model_directory.path());
  EXPECT_TRUE(config.rotary_dims == 128 && config.rope_theta == 5000.0f);
  EXPECT_TRUE(config.quantization_group_size == 32);
}

void should_accept_an_explicit_zero_expert_count() {
  EXPECT_TRUE(!loading_throws(vision_language_config(
      R"(, "num_experts": 0, "rope_theta": 1, "partial_rotary_factor": 1)")));
}

// --- negative ---

void should_reject_mixture_of_experts_checkpoints() {
  EXPECT_TRUE(loading_throws(vision_language_config(
      R"(, "num_experts": 256, "rope_theta": 1, "partial_rotary_factor": 1)")));
}

void should_reject_unknown_layer_types_and_missing_fields() {
  std::string unknown_layer_config = vision_language_config(
      R"(, "rope_theta": 1, "partial_rotary_factor": 1)");
  unknown_layer_config.replace(unknown_layer_config.find("full_attention"),
                               std::string("full_attention").size(), "sliding_window");
  EXPECT_TRUE(loading_throws(unknown_layer_config));
  EXPECT_TRUE(loading_throws(vision_language_config("")));  // no rope fields
  EXPECT_TRUE(loading_throws("{}"));
}

void should_report_a_missing_config_file() {
  bool has_thrown = false;
  try {
    load_qwen35_text_config("/nonexistent/model/directory");
  } catch (const std::runtime_error&) {
    has_thrown = true;
  }
  EXPECT_TRUE(has_thrown);
}

} // namespace

int main() {
  return test_harness::run_all({
      TEST_CASE(should_read_a_vision_language_checkpoint_with_rope_parameters),
      TEST_CASE(should_read_a_text_only_checkpoint_with_top_level_rope_fields),
      TEST_CASE(should_accept_an_explicit_zero_expert_count),
      TEST_CASE(should_reject_mixture_of_experts_checkpoints),
      TEST_CASE(should_reject_unknown_layer_types_and_missing_fields),
      TEST_CASE(should_report_a_missing_config_file),
  });
}
