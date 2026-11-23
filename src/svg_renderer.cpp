#include "svg_renderer.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace mlx_plan {
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

  // Quoting keeps arbitrary artifact bases safe; dot exits nonzero when the
  // graph is malformed or the tool itself is missing.
  std::ostringstream layout_command;
  layout_command << "dot -Tsvg \"" << dot_file_path << "\" -o \"" << svg_file_path
                 << "\"";
  if (std::system(layout_command.str().c_str()) != 0) {
    return "";
  }
  return read_text_file(svg_file_path);
}

} // namespace mlx_plan
