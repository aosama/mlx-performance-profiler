#pragma once

// Captures everything the process writes to stdout and stderr while alive,
// at the file-descriptor level (so printf, iostreams and unbuffered writes
// alike are caught), into files in the working directory.

#include <unistd.h>

#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <string>

namespace captured_console {

class CapturedConsole {
 public:
  explicit CapturedConsole(const std::string& capture_name)
      : stdout_path_(capture_name + "_stdout.txt"), stderr_path_(capture_name + "_stderr.txt") {
    std::fflush(stdout);
    std::fflush(stderr);
    saved_stdout_ = dup(STDOUT_FILENO);
    saved_stderr_ = dup(STDERR_FILENO);
    redirect(STDOUT_FILENO, stdout_path_);
    redirect(STDERR_FILENO, stderr_path_);
  }
  ~CapturedConsole() { restore(); }
  CapturedConsole(const CapturedConsole&) = delete;
  CapturedConsole& operator=(const CapturedConsole&) = delete;

  // Restores the real streams (idempotent) and returns what was captured.
  std::string stdout_text() { restore(); return read_file(stdout_path_); }
  std::string stderr_text() { restore(); return read_file(stderr_path_); }

 private:
  static void redirect(int stream_descriptor, const std::string& file_path) {
    const int file_descriptor = open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    dup2(file_descriptor, stream_descriptor);
    close(file_descriptor);
  }
  static std::string read_file(const std::string& file_path) {
    std::ifstream input_stream(file_path);
    return std::string(std::istreambuf_iterator<char>(input_stream), {});
  }
  void restore() {
    if (saved_stdout_ < 0) {
      return;
    }
    std::fflush(stdout);
    std::fflush(stderr);
    dup2(saved_stdout_, STDOUT_FILENO);
    dup2(saved_stderr_, STDERR_FILENO);
    close(saved_stdout_);
    close(saved_stderr_);
    saved_stdout_ = -1;
    saved_stderr_ = -1;
  }

  std::string stdout_path_;
  std::string stderr_path_;
  int saved_stdout_ = -1;
  int saved_stderr_ = -1;
};

} // namespace captured_console
