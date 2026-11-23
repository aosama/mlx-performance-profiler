#pragma once

// A throwaway model directory for tests: created under the system temp
// directory, removed on destruction (it only ever holds files the test wrote).

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace test_support {

class TemporaryModelDirectory {
 public:
  TemporaryModelDirectory() {
    std::random_device entropy;
    directory_path_ = std::filesystem::temp_directory_path() /
        ("mlx_plan_test_model_" + std::to_string(entropy()));
    std::filesystem::create_directories(directory_path_);
  }
  ~TemporaryModelDirectory() {
    std::error_code ignored_error;
    std::filesystem::remove_all(directory_path_, ignored_error);
  }
  TemporaryModelDirectory(const TemporaryModelDirectory&) = delete;
  TemporaryModelDirectory& operator=(const TemporaryModelDirectory&) = delete;

  void write_file(const std::string& file_name, const std::string& file_text) const {
    std::ofstream(directory_path_ / file_name, std::ios::binary) << file_text;
  }
  std::string path() const { return directory_path_.string(); }

 private:
  std::filesystem::path directory_path_;
};

} // namespace test_support
