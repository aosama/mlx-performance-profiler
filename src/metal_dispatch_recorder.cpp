#include "metal_dispatch_recorder.h"

#include "metal_object_classes.h"
#include "objc_method_hooks.h"
#include "recording_slots.h"

#include "mlx/backend/metal/device.h"

#include <objc/runtime.h>

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mlx_perf {

struct DispatchRecordingScope::RecordingState {
  struct EncoderPipeline {
    std::string metal_function_name;
    std::uint64_t max_threads_per_threadgroup = 0;
    std::uint64_t thread_execution_width = 0;
    std::uint64_t static_threadgroup_memory_bytes = 0;
    // Dynamic threadgroup memory by argument index, set after the pipeline.
    std::map<std::uint64_t, std::uint64_t> index_to_dynamic_memory_bytes;
  };
  std::vector<KernelDispatch> recorded_dispatches;
  std::unordered_map<const void*, EncoderPipeline> encoder_to_pipeline;
};

namespace {

using RecordingState = DispatchRecordingScope::RecordingState;

using DispatchSlots = RecordingSlots<RecordingState>;
using ActiveRecording = DispatchSlots::ActiveRecording;

using DispatchImp = void (*)(id, SEL, MTL::Size, MTL::Size);
using SetPipelineImp = void (*)(id, SEL, id);
using SetThreadgroupMemoryImp = void (*)(id, SEL, NS::UInteger, NS::UInteger);
using NewPipelineWithFunctionImp = id (*)(id, SEL, id, id*);
using NewPipelineWithDescriptorImp = id (*)(id, SEL, id, NS::UInteger, id*, id*);

DispatchImp original_dispatch_threadgroups = nullptr;
DispatchImp original_dispatch_threads = nullptr;
SetPipelineImp original_set_compute_pipeline_state = nullptr;
SetThreadgroupMemoryImp original_set_threadgroup_memory_length = nullptr;
NewPipelineWithFunctionImp original_new_pipeline_with_function = nullptr;
NewPipelineWithDescriptorImp original_new_pipeline_with_descriptor = nullptr;

// Pipelines can be compiled on any thread and are cached by MLX for the
// process's life, so their names are kept process-wide, not per scope, and
// never erased. The address is the key: MLX never frees a cached pipeline,
// so an address names one pipeline for the whole process.
std::mutex pipeline_names_mutex;
std::unordered_map<const void*, std::string>& pipeline_to_function_name() {
  static std::unordered_map<const void*, std::string> pipeline_names;
  return pipeline_names;
}

void remember_pipeline_name(id pipeline_object, const MTL::Function* compiled_function) {
  if (pipeline_object == nullptr || compiled_function == nullptr ||
      compiled_function->name() == nullptr) {
    return;
  }
  const std::lock_guard<std::mutex> names_lock(pipeline_names_mutex);
  pipeline_to_function_name()[pipeline_object] = compiled_function->name()->utf8String();
}

std::string find_pipeline_name(const void* pipeline_object) {
  const std::lock_guard<std::mutex> names_lock(pipeline_names_mutex);
  const auto name_entry = pipeline_to_function_name().find(pipeline_object);
  return name_entry == pipeline_to_function_name().end() ? std::string() : name_entry->second;
}

GridSize to_grid_size(const MTL::Size& metal_size) {
  return GridSize{metal_size.width, metal_size.height, metal_size.depth};
}

void record_dispatch(RecordingState& recording_state, id encoder, DispatchGridUnit grid_unit,
                     const MTL::Size& grid, const MTL::Size& threads_per_threadgroup) {
  KernelDispatch kernel_dispatch;
  kernel_dispatch.grid_unit = grid_unit;
  kernel_dispatch.grid = to_grid_size(grid);
  kernel_dispatch.threads_per_threadgroup = to_grid_size(threads_per_threadgroup);
  const auto pipeline_entry = recording_state.encoder_to_pipeline.find(encoder);
  if (pipeline_entry != recording_state.encoder_to_pipeline.end()) {
    const RecordingState::EncoderPipeline& encoder_pipeline = pipeline_entry->second;
    kernel_dispatch.metal_function_name = encoder_pipeline.metal_function_name;
    kernel_dispatch.max_threads_per_threadgroup = encoder_pipeline.max_threads_per_threadgroup;
    kernel_dispatch.thread_execution_width = encoder_pipeline.thread_execution_width;
    kernel_dispatch.threadgroup_memory_bytes = encoder_pipeline.static_threadgroup_memory_bytes;
    for (const auto& [argument_index, dynamic_bytes] : encoder_pipeline.index_to_dynamic_memory_bytes) {
      kernel_dispatch.threadgroup_memory_bytes += dynamic_bytes;
    }
  }
  recording_state.recorded_dispatches.push_back(std::move(kernel_dispatch));
}

void dispatch_threadgroups_hook(id encoder, SEL selector, MTL::Size grid, MTL::Size group) {
  original_dispatch_threadgroups(encoder, selector, grid, group);
  const ActiveRecording active_recording;
  if (active_recording.recording_state() != nullptr) {
    record_dispatch(*active_recording.recording_state(), encoder, DispatchGridUnit::threadgroups, grid, group);
  }
}

void dispatch_threads_hook(id encoder, SEL selector, MTL::Size grid, MTL::Size group) {
  original_dispatch_threads(encoder, selector, grid, group);
  const ActiveRecording active_recording;
  if (active_recording.recording_state() != nullptr) {
    record_dispatch(*active_recording.recording_state(), encoder, DispatchGridUnit::threads, grid, group);
  }
}

// A new pipeline also clears dynamic threadgroup memory: Metal keeps lengths
// set for an earlier kernel, but MLX sets them after the pipeline for the
// kernel that needs them, so only those count toward this kernel.
void set_compute_pipeline_state_hook(id encoder, SEL selector, id pipeline_object) {
  original_set_compute_pipeline_state(encoder, selector, pipeline_object);
  const ActiveRecording active_recording;
  if (active_recording.recording_state() == nullptr || pipeline_object == nullptr) {
    return;
  }
  const auto* pipeline = reinterpret_cast<MTL::ComputePipelineState*>(pipeline_object);
  RecordingState::EncoderPipeline encoder_pipeline;
  encoder_pipeline.metal_function_name = find_pipeline_name(pipeline_object);
  encoder_pipeline.max_threads_per_threadgroup = pipeline->maxTotalThreadsPerThreadgroup();
  encoder_pipeline.thread_execution_width = pipeline->threadExecutionWidth();
  encoder_pipeline.static_threadgroup_memory_bytes = pipeline->staticThreadgroupMemoryLength();
  active_recording.recording_state()->encoder_to_pipeline[encoder] = std::move(encoder_pipeline);
}

void set_threadgroup_memory_length_hook(id encoder, SEL selector, NS::UInteger length_bytes,
                                        NS::UInteger argument_index) {
  original_set_threadgroup_memory_length(encoder, selector, length_bytes, argument_index);
  const ActiveRecording active_recording;
  if (active_recording.recording_state() != nullptr) {
    active_recording.recording_state()->encoder_to_pipeline[encoder]
        .index_to_dynamic_memory_bytes[argument_index] = length_bytes;
  }
}

// Pipeline creation is rare (MLX caches every pipeline), so naming is
// always on, whether or not a scope is open.
id new_pipeline_with_function_hook(id device, SEL selector, id function_object, id* error_out) {
  const id pipeline_object = original_new_pipeline_with_function(device, selector, function_object, error_out);
  remember_pipeline_name(pipeline_object, reinterpret_cast<const MTL::Function*>(function_object));
  return pipeline_object;
}

id new_pipeline_with_descriptor_hook(id device, SEL selector, id descriptor_object, NS::UInteger options,
                                     id* reflection_out, id* error_out) {
  const id pipeline_object = original_new_pipeline_with_descriptor(
      device, selector, descriptor_object, options, reflection_out, error_out);
  const auto* descriptor = reinterpret_cast<MTL::ComputePipelineDescriptor*>(descriptor_object);
  remember_pipeline_name(pipeline_object, descriptor == nullptr ? nullptr : descriptor->computeFunction());
  return pipeline_object;
}

bool install_hooks() {
  const MetalObjectClasses& metal_object_classes = find_metal_object_classes();
  const Class encoder_class = metal_object_classes.compute_encoder_class;
  const Class device_class = metal_object_classes.device_class;
  if (encoder_class == nullptr || device_class == nullptr) {
    return false;
  }
  bool is_installed = replace_method(encoder_class, "dispatchThreadgroups:threadsPerThreadgroup:",
                                     reinterpret_cast<IMP>(dispatch_threadgroups_hook),
                                     original_dispatch_threadgroups);
  is_installed &= replace_method(encoder_class, "dispatchThreads:threadsPerThreadgroup:",
                                 reinterpret_cast<IMP>(dispatch_threads_hook), original_dispatch_threads);
  is_installed &= replace_method(encoder_class, "setComputePipelineState:",
                                 reinterpret_cast<IMP>(set_compute_pipeline_state_hook),
                                 original_set_compute_pipeline_state);
  is_installed &= replace_method(encoder_class, "setThreadgroupMemoryLength:atIndex:",
                                 reinterpret_cast<IMP>(set_threadgroup_memory_length_hook),
                                 original_set_threadgroup_memory_length);
  // Naming is a bonus: without these hooks launches still record, unnamed.
  replace_method(device_class, "newComputePipelineStateWithFunction:error:",
                 reinterpret_cast<IMP>(new_pipeline_with_function_hook), original_new_pipeline_with_function);
  replace_method(device_class, "newComputePipelineStateWithDescriptor:options:reflection:error:",
                 reinterpret_cast<IMP>(new_pipeline_with_descriptor_hook),
                 original_new_pipeline_with_descriptor);
  return is_installed;
}

} // namespace

bool install_dispatch_recorder() {
  static const bool is_installed = install_hooks();
  return is_installed;
}

DispatchRecordingScope::DispatchRecordingScope(RecordingReach recording_reach)
    : recording_state_(std::make_unique<RecordingState>()), recording_reach_(recording_reach) {
  enclosing_recording_state_ = DispatchSlots::swap_open_recording(recording_reach_, recording_state_.get());
}

DispatchRecordingScope::~DispatchRecordingScope() {
  DispatchSlots::swap_open_recording(recording_reach_, enclosing_recording_state_);
}

// Paused while the list is taken, so a process-wide hook on another thread
// cannot write into it at the same moment.
std::vector<KernelDispatch> DispatchRecordingScope::take_recorded_dispatches() {
  DispatchSlots::swap_open_recording(recording_reach_, nullptr);
  std::vector<KernelDispatch> recorded_dispatches = std::move(recording_state_->recorded_dispatches);
  recording_state_->recorded_dispatches.clear();
  DispatchSlots::swap_open_recording(recording_reach_, recording_state_.get());
  return recorded_dispatches;
}

} // namespace mlx_perf
