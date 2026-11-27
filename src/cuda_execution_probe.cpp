#include "cuda_execution_probe.h"

#include "cuda_mlx_streams.h"
#include "cupti_session.h"
#include "recording_slots.h"

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <utility>

namespace mlx_perf {

struct CudaExecutionProbe::ProbeState {
  std::uint64_t pass_anchor_nanoseconds = 0;
  // The graph launches attributed to each unit, by CUPTI correlation id.
  std::vector<std::vector<std::uint32_t>> position_to_correlation_ids;
  // Which unit each held CUDA stream's launches belong to right now. A few
  // streams at most (one per MLX stream in a stage), so a list beats a map.
  std::vector<std::pair<CUstream, int>> held_stream_positions;
  // Read once, on the first read after the pass: taking a span ends its
  // watch in the session, so it cannot be taken twice.
  bool are_spans_read = false;
  std::vector<GpuActivitySpan> position_to_span;
  int completed_launch_count = 0;
};

namespace {

using ProbeState = CudaExecutionProbe::ProbeState;
using ProbeSlots = RecordingSlots<ProbeState>;
using ActiveProbe = ProbeSlots::ActiveRecording;

std::uint64_t to_steady_nanoseconds(std::chrono::steady_clock::time_point steady_instant) {
  // libstdc++'s and libc++'s steady_clock read CLOCK_MONOTONIC on Linux, the
  // clock cupti_session hands CUPTI, so the anchor needs no conversion.
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(steady_instant.time_since_epoch()).count());
}

void attribute_launch(ProbeState& probe_state, const DriverCall& driver_call) {
  const CUstream launch_stream = static_cast<const cuGraphLaunch_params*>(driver_call.function_params)->hStream;
  for (const auto& [held_stream, position] : probe_state.held_stream_positions) {
    if (held_stream == launch_stream) {
      probe_state.position_to_correlation_ids[position].push_back(driver_call.correlation_id);
      watch_gpu_activity(driver_call.correlation_id);
      return;
    }
  }
}

class GraphLaunchAttribution final : public DriverCallObserver {
 public:
  void on_driver_call(const DriverCall& driver_call) override {
    if (!driver_call.is_entering) {
      return;
    }
    const ActiveProbe active_probe;
    if (active_probe.recording_state() != nullptr) {
      attribute_launch(*active_probe.recording_state(), driver_call);
    }
  }
};

bool install_graph_launch_attribution() {
  static GraphLaunchAttribution graph_launch_attribution;
  return observe_driver_call(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, &graph_launch_attribution) &&
      observe_driver_call(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz, &graph_launch_attribution);
}

GpuActivitySpan merge_spans(const GpuActivitySpan& accumulated_span, const GpuActivitySpan& launch_span) {
  if (!launch_span.is_measured) {
    return accumulated_span;
  }
  if (!accumulated_span.is_measured) {
    return launch_span;
  }
  GpuActivitySpan merged_span = accumulated_span;
  merged_span.started_nanoseconds = std::min(merged_span.started_nanoseconds, launch_span.started_nanoseconds);
  merged_span.finished_nanoseconds = std::max(merged_span.finished_nanoseconds, launch_span.finished_nanoseconds);
  merged_span.activity_record_count += launch_span.activity_record_count;
  return merged_span;
}

void read_spans_once(ProbeState& probe_state) {
  if (probe_state.are_spans_read) {
    return;
  }
  probe_state.are_spans_read = true;
  flush_gpu_activity();
  probe_state.position_to_span.assign(probe_state.position_to_correlation_ids.size(), GpuActivitySpan{});
  for (std::size_t position = 0; position < probe_state.position_to_correlation_ids.size(); ++position) {
    for (const std::uint32_t correlation_id : probe_state.position_to_correlation_ids[position]) {
      const GpuActivitySpan launch_span = take_gpu_activity_span(correlation_id);
      probe_state.completed_launch_count += launch_span.is_measured ? 1 : 0;
      probe_state.position_to_span[position] = merge_spans(probe_state.position_to_span[position], launch_span);
    }
  }
}

double microseconds_after_anchor(std::uint64_t nanoseconds, std::uint64_t anchor_nanoseconds) {
  return (static_cast<double>(nanoseconds) - static_cast<double>(anchor_nanoseconds)) / 1e3;
}

} // namespace

CudaExecutionProbe::CudaExecutionProbe(int slot_count, std::chrono::steady_clock::time_point pass_started_at)
    : probe_state_(std::make_unique<ProbeState>()) {
  static const bool is_attribution_installed = install_graph_launch_attribution();
  (void)is_attribution_installed;
  probe_state_->pass_anchor_nanoseconds = to_steady_nanoseconds(pass_started_at);
  probe_state_->position_to_correlation_ids.resize(static_cast<std::size_t>(std::max(0, slot_count)));
  enclosing_probe_state_ = ProbeSlots::swap_open_recording(RecordingReach::calling_thread, probe_state_.get());
}

// Closed first; spans never read are given up so the session drops them.
CudaExecutionProbe::~CudaExecutionProbe() {
  ProbeSlots::swap_open_recording(RecordingReach::calling_thread, enclosing_probe_state_);
  if (!probe_state_->are_spans_read) {
    for (const auto& correlation_ids : probe_state_->position_to_correlation_ids) {
      for (const std::uint32_t correlation_id : correlation_ids) {
        take_gpu_activity_span(correlation_id);
      }
    }
  }
}

void CudaExecutionProbe::hold_buffer_for(int position, const mlx::core::Stream& stream) {
  const CUstream held_stream = cuda_stream_of(stream);
  if (held_stream == nullptr) {
    return;
  }
  auto& held_stream_positions = probe_state_->held_stream_positions;
  for (auto& [already_held_stream, held_position] : held_stream_positions) {
    if (already_held_stream == held_stream) {
      held_position = position;
      return;
    }
  }
  held_stream_positions.emplace_back(held_stream, position);
}

void CudaExecutionProbe::commit_end_marker_for(int position) {
  auto& held_stream_positions = probe_state_->held_stream_positions;
  for (auto held_entry = held_stream_positions.begin(); held_entry != held_stream_positions.end();) {
    held_entry = held_entry->second == position ? held_stream_positions.erase(held_entry) : held_entry + 1;
  }
}

std::vector<GpuExecutionWindow> CudaExecutionProbe::read_windows() const {
  read_spans_once(*probe_state_);
  std::vector<GpuExecutionWindow> position_to_window(probe_state_->position_to_span.size());
  for (std::size_t position = 0; position < position_to_window.size(); ++position) {
    const GpuActivitySpan& unit_span = probe_state_->position_to_span[position];
    if (!unit_span.is_measured) {
      continue;
    }
    GpuExecutionWindow& window = position_to_window[position];
    window.is_measured = true;
    window.started_microseconds =
        microseconds_after_anchor(unit_span.started_nanoseconds, probe_state_->pass_anchor_nanoseconds);
    window.finished_microseconds =
        microseconds_after_anchor(unit_span.finished_nanoseconds, probe_state_->pass_anchor_nanoseconds);
  }
  return position_to_window;
}

int CudaExecutionProbe::count_completed_command_buffers() const {
  read_spans_once(*probe_state_);
  return probe_state_->completed_launch_count;
}

} // namespace mlx_perf
