#pragma once

// Metal object classes: one concern only — find the concrete Objective-C
// classes behind the Metal objects MLX uses on this Mac's GPU (they differ by
// GPU family, e.g. AGXG17SDevice), so recorders can hook them.

#include <objc/runtime.h>

namespace mlx_perf {

struct MetalObjectClasses {
  Class device_class = nullptr;
  Class command_buffer_class = nullptr;
  Class compute_encoder_class = nullptr;
  Class shared_event_class = nullptr;
};

// Found once, by making one of each object on MLX's own Metal device the way
// MLX makes them (a concurrent compute encoder on a command buffer with
// unretained references from a new queue; a shared event) and never
// committing anything. Members are null when there is no Metal GPU.
const MetalObjectClasses& find_metal_object_classes();

} // namespace mlx_perf
