#include "dispatch_recorder.h"

#include "cupti_session.h"
#include "recording_slots.h"

#include <cuda.h>

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mlx_perf {

struct DispatchRecordingScope::RecordingState {
  std::vector<KernelDispatch> recorded_dispatches;
};

namespace {

using RecordingState = DispatchRecordingScope::RecordingState;
using DispatchSlots = RecordingSlots<RecordingState>;
using ActiveRecording = DispatchSlots::ActiveRecording;

// Every thread of an NVIDIA warp runs in lock step, as an Apple SIMD group
// does; the width has been 32 on every NVIDIA GPU MLX's CUDA backend
// supports (compute capability 7.5 and later).
constexpr std::uint64_t NVIDIA_WARP_WIDTH = 32;

// What a launch's shape does not say: the kernel's name and limits. The
// driver answers them per function, and MLX launches the same few hundred
// functions over and over, so they are read once and kept for the process's
// life. A function handle stays valid while its module is loaded, and MLX
// never unloads its kernels.
struct KernelFacts {
  std::string kernel_function_name;
  std::uint64_t max_threads_per_block = 0;
  std::uint64_t static_shared_memory_bytes = 0;
};

std::mutex kernel_facts_mutex;
std::unordered_map<CUfunction, KernelFacts>& function_to_kernel_facts() {
  static std::unordered_map<CUfunction, KernelFacts> kernel_facts;
  return kernel_facts;
}

std::uint64_t to_unsigned_attribute(int attribute_value) {
  return attribute_value < 0 ? 0 : static_cast<std::uint64_t>(attribute_value);
}

// A launch may name its kernel by a function handle, or, since CUDA 12, by a
// library kernel handle (CUkernel) cast to one: cuBLAS and cuDNN launch that
// way, and the function queries reject such a handle. So each fact is asked
// of the function first and of the library kernel when that fails.
std::string read_kernel_function_name(CUfunction launched_handle) {
  const char* raw_function_name = nullptr;
  if (cuFuncGetName(&raw_function_name, launched_handle) == CUDA_SUCCESS && raw_function_name != nullptr) {
    return raw_function_name;
  }
  if (cuKernelGetName(&raw_function_name, reinterpret_cast<CUkernel>(launched_handle)) == CUDA_SUCCESS &&
      raw_function_name != nullptr) {
    return raw_function_name;
  }
  return std::string();
}

std::uint64_t read_kernel_attribute(CUfunction launched_handle, CUfunction_attribute attribute) {
  int attribute_value = 0;
  if (cuFuncGetAttribute(&attribute_value, attribute, launched_handle) == CUDA_SUCCESS) {
    return to_unsigned_attribute(attribute_value);
  }
  CUdevice current_device = 0;
  if (cuCtxGetDevice(&current_device) == CUDA_SUCCESS &&
      cuKernelGetAttribute(&attribute_value, attribute, reinterpret_cast<CUkernel>(launched_handle),
                           current_device) == CUDA_SUCCESS) {
    return to_unsigned_attribute(attribute_value);
  }
  return 0;
}

KernelFacts query_kernel_facts(CUfunction launched_handle) {
  KernelFacts kernel_facts;
  kernel_facts.kernel_function_name = read_kernel_function_name(launched_handle);
  kernel_facts.max_threads_per_block =
      read_kernel_attribute(launched_handle, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
  kernel_facts.static_shared_memory_bytes =
      read_kernel_attribute(launched_handle, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
  return kernel_facts;
}

KernelFacts find_kernel_facts(CUfunction cuda_function) {
  const std::lock_guard<std::mutex> facts_lock(kernel_facts_mutex);
  auto facts_entry = function_to_kernel_facts().find(cuda_function);
  if (facts_entry == function_to_kernel_facts().end()) {
    facts_entry = function_to_kernel_facts().emplace(cuda_function, query_kernel_facts(cuda_function)).first;
  }
  return facts_entry->second;
}

// The shape every launch path shares, in the driver's own terms.
struct LaunchShape {
  CUfunction cuda_function = nullptr;
  GridSize grid_in_blocks;
  GridSize threads_per_block;
  std::uint64_t dynamic_shared_memory_bytes = 0;
};

void record_launch(RecordingState& recording_state, const LaunchShape& launch_shape) {
  KernelDispatch kernel_dispatch;
  // CUDA grids are always counted in blocks, the unit Metal calls a
  // threadgroup.
  kernel_dispatch.grid_unit = DispatchGridUnit::threadgroups;
  kernel_dispatch.grid = launch_shape.grid_in_blocks;
  kernel_dispatch.threads_per_threadgroup = launch_shape.threads_per_block;
  kernel_dispatch.thread_execution_width = NVIDIA_WARP_WIDTH;
  kernel_dispatch.threadgroup_memory_bytes = launch_shape.dynamic_shared_memory_bytes;
  if (launch_shape.cuda_function != nullptr) {
    const KernelFacts kernel_facts = find_kernel_facts(launch_shape.cuda_function);
    kernel_dispatch.kernel_function_name = kernel_facts.kernel_function_name;
    kernel_dispatch.max_threads_per_threadgroup = kernel_facts.max_threads_per_block;
    kernel_dispatch.threadgroup_memory_bytes += kernel_facts.static_shared_memory_bytes;
  }
  recording_state.recorded_dispatches.push_back(std::move(kernel_dispatch));
}

// A graph kernel node names its kernel either as a function or, for a
// library kernel, as a CUkernel the driver resolves in the node's context.
CUfunction resolve_node_function(const CUDA_KERNEL_NODE_PARAMS& node_params) {
  if (node_params.func != nullptr || node_params.kern == nullptr) {
    return node_params.func;
  }
  CUfunction resolved_function = nullptr;
  return cuKernelGetFunction(&resolved_function, node_params.kern) == CUDA_SUCCESS ? resolved_function : nullptr;
}

LaunchShape shape_of_graph_node(const CUDA_KERNEL_NODE_PARAMS& node_params) {
  return LaunchShape{resolve_node_function(node_params),
                     {node_params.gridDimX, node_params.gridDimY, node_params.gridDimZ},
                     {node_params.blockDimX, node_params.blockDimY, node_params.blockDimZ},
                     node_params.sharedMemBytes};
}

LaunchShape shape_of_launch_kernel(const cuLaunchKernel_params& launch_params) {
  return LaunchShape{launch_params.f,
                     {launch_params.gridDimX, launch_params.gridDimY, launch_params.gridDimZ},
                     {launch_params.blockDimX, launch_params.blockDimY, launch_params.blockDimZ},
                     launch_params.sharedMemBytes};
}

LaunchShape shape_of_launch_kernel_ex(const cuLaunchKernelEx_params& launch_params) {
  const CUlaunchConfig& launch_config = *launch_params.config;
  return LaunchShape{launch_params.f,
                     {launch_config.gridDimX, launch_config.gridDimY, launch_config.gridDimZ},
                     {launch_config.blockDimX, launch_config.blockDimY, launch_config.blockDimZ},
                     launch_config.sharedMemBytes};
}

// The four ways MLX, cuBLAS and cuDNN put a kernel on the GPU: a node in a
// CUDA graph (MLX's default, both the v1 and v2 parameter layouts), or a
// direct launch (MLX with graphs off, and library kernels during stream
// capture), in both the classic and the extended form.
bool read_launch_shape(const DriverCall& driver_call, LaunchShape& launch_shape) {
  switch (driver_call.callback_id) {
    case CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2: {
      const auto* add_params = static_cast<const cuGraphAddKernelNode_v2_params*>(driver_call.function_params);
      if (add_params->nodeParams == nullptr) return false;
      launch_shape = shape_of_graph_node(*add_params->nodeParams);
      return true;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode: {
      const auto* add_params = static_cast<const cuGraphAddKernelNode_params*>(driver_call.function_params);
      if (add_params->nodeParams == nullptr) return false;
      // v1 is v2 without the trailing kern and ctx fields.
      CUDA_KERNEL_NODE_PARAMS widened_params{};
      std::memcpy(&widened_params, add_params->nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS_v1));
      launch_shape = shape_of_graph_node(widened_params);
      return true;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
      launch_shape = shape_of_launch_kernel(*static_cast<const cuLaunchKernel_params*>(driver_call.function_params));
      return true;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
      const auto* launch_params = static_cast<const cuLaunchKernelEx_params*>(driver_call.function_params);
      if (launch_params->config == nullptr) return false;
      launch_shape = shape_of_launch_kernel_ex(*launch_params);
      return true;
    }
    default:
      return false;
  }
}

class KernelLaunchObserver final : public DriverCallObserver {
 public:
  // On entry, like the Metal hooks after the original returns: the shape is
  // the caller's, whatever the driver then does with it.
  void on_driver_call(const DriverCall& driver_call) override {
    if (!driver_call.is_entering) {
      return;
    }
    const ActiveRecording active_recording;
    LaunchShape launch_shape;
    if (active_recording.recording_state() != nullptr && read_launch_shape(driver_call, launch_shape)) {
      record_launch(*active_recording.recording_state(), launch_shape);
    }
  }
};

constexpr CUpti_CallbackId OBSERVED_LAUNCH_CALLS[] = {
    CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2, CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,          CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
};

bool install_launch_observer() {
  static KernelLaunchObserver kernel_launch_observer;
  bool is_installed = true;
  for (const CUpti_CallbackId launch_call : OBSERVED_LAUNCH_CALLS) {
    is_installed &= observe_driver_call(launch_call, &kernel_launch_observer);
  }
  return is_installed;
}

} // namespace

bool install_dispatch_recorder() {
  static const bool is_installed = install_launch_observer();
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
