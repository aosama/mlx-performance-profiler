#include "measurement_file.h"

#include "json_writer.h"
#include "measurement_document.h"
#include "report_writer.h"

namespace mlx_perf {

std::string write_measurement_file(const VariantMeasurement& variant_measurement,
                                   const std::string& output_directory,
                                   const std::string& file_base) {
  const std::string measurement_path = output_directory + "/" + file_base + ".json";
  const bool is_written = write_text_file(
      measurement_path, write_json(build_measurement_document(variant_measurement)));
  return is_written ? measurement_path : std::string();
}

} // namespace mlx_perf
