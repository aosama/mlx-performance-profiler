#include "measurement_file.h"

#include "json_writer.h"
#include "measurement_document.h"
#include "report_writer.h"

namespace mlx_perf {

std::string write_json_file(const JsonValue& document, const std::string& output_directory,
                            const std::string& file_base) {
  const std::string json_path = output_directory + "/" + file_base + ".json";
  return write_text_file(json_path, write_json(document)) ? json_path : std::string();
}

std::string write_measurement_file(const VariantMeasurement& variant_measurement,
                                   const std::string& output_directory,
                                   const std::string& file_base) {
  return write_json_file(build_measurement_document(variant_measurement), output_directory,
                         file_base);
}

} // namespace mlx_perf
