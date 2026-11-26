#include "svg_renderer.h"

#include "console_output.h"

#include <fstream>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>
#include <sys/wait.h>

// The process environment, passed through to `dot` unchanged.
extern char** environ;

namespace mlx_perf {
namespace {

bool write_text_file(
    const std::string& file_path,
    const std::string& file_text) {
  std::ofstream output_stream(file_path, std::ios::binary);
  return output_stream.good() &&
         output_stream.write(
             file_text.data(), static_cast<std::streamsize>(file_text.size()))
             .good();
}

std::string read_text_file(const std::string& file_path) {
  std::ifstream input_stream(file_path, std::ios::binary);
  std::ostringstream file_text;
  file_text << input_stream.rdbuf();
  return file_text.str();
}

// Runs `dot -Tsvg <dot_file_path> -o <svg_file_path>` with an argument
// vector, never through a shell: the paths carry caller-chosen names
// (report bases, variant names), and inside a shell even double quotes leave
// $(...), backticks and quotes live. True when dot ran and exited 0; false
// when it is missing or rejects the graph.
bool run_dot_layout(const std::string& dot_file_path, const std::string& svg_file_path) {
  std::string program_name = "dot";
  std::string format_flag = "-Tsvg";
  std::string output_flag = "-o";
  std::string dot_path_argument = dot_file_path;
  std::string svg_path_argument = svg_file_path;
  char* dot_arguments[] = {program_name.data(), format_flag.data(), dot_path_argument.data(),
                           output_flag.data(), svg_path_argument.data(), nullptr};
  // With console output off the library prints nothing, and that includes
  // what dot would say about a graph it rejects.
  posix_spawn_file_actions_t dot_file_actions;
  posix_spawn_file_actions_init(&dot_file_actions);
  if (!is_console_output_enabled()) {
    posix_spawn_file_actions_addopen(&dot_file_actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&dot_file_actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
  }
  pid_t dot_process_id = 0;
  const int spawn_status = posix_spawnp(&dot_process_id, "dot", &dot_file_actions, nullptr,
                                        dot_arguments, environ);
  posix_spawn_file_actions_destroy(&dot_file_actions);
  if (spawn_status != 0) {
    return false;
  }
  int dot_exit_status = 0;
  if (waitpid(dot_process_id, &dot_exit_status, 0) != dot_process_id) {
    return false;
  }
  return WIFEXITED(dot_exit_status) && WEXITSTATUS(dot_exit_status) == 0;
}

} // namespace

std::string render_plan_svg(
    const std::string& dot_source,
    const std::string& output_directory,
    const std::string& plan_file_base) {
  const std::string dot_file_path =
      output_directory + "/" + plan_file_base + ".dot";
  const std::string svg_file_path =
      output_directory + "/" + plan_file_base + ".svg";
  if (!write_text_file(dot_file_path, dot_source)) {
    return "";
  }

  if (!run_dot_layout(dot_file_path, svg_file_path)) {
    return "";
  }
  return read_text_file(svg_file_path);
}

} // namespace mlx_perf
