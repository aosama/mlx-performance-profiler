#include "stage_timeline.h"

#include "gpu_execution_probe.h"
#include "progress_reporter.h"
#include "timed_units.h"
#include "timeline_dependencies.h"

#include "mlx/transforms.h"

#include <algorithm>
#include <map>
#include <thread>

namespace mlx_perf {
namespace {

// Stage tasks in dispatch order: by stage, then by stream index.
struct StageTaskLayout {
  std::vector<StageTimeline> stage_tasks;
  std::vector<int> position_to_task;
  std::vector<std::vector<int>> stage_to_tasks;
};

struct StageTaskDependencies {
  int previous_on_stream_task = NO_POSITION;
  std::vector<int> cross_stream_producer_tasks;  // earlier stages only
};

// Host instants per stage task (µs since pass start), filled during dispatch.
struct StageTaskInstants {
  std::vector<double> task_to_issue_started_microseconds;
  std::vector<double> task_to_issue_finished_microseconds;
  std::vector<double> task_to_completed_microseconds;
};

StageTaskLayout lay_out_stage_tasks(
    const CapturedGraph& captured_graph, const StagePlan& stage_plan) {
  const int stage_count = static_cast<int>(stage_plan.stage_names.size());
  std::vector<std::map<int, std::vector<int>>> stage_to_stream_members(stage_count);
  for (std::size_t position = 0; position < stage_plan.position_to_stage.size();
       ++position) {
    const int stream_index = node_at(captured_graph, static_cast<int>(position)).stream_index;
    stage_to_stream_members[stage_plan.position_to_stage[position]][stream_index]
        .push_back(static_cast<int>(position));
  }
  StageTaskLayout layout;
  layout.position_to_task.assign(stage_plan.position_to_stage.size(), NO_POSITION);
  layout.stage_to_tasks.resize(stage_count);
  for (int stage_index = 0; stage_index < stage_count; ++stage_index) {
    for (const auto& [stream_index, member_positions] :
         stage_to_stream_members[stage_index]) {
      const int task_position = static_cast<int>(layout.stage_tasks.size());
      StageTimeline stage_task;
      stage_task.stage_index = stage_index;
      stage_task.stage_name = stage_plan.stage_names[stage_index];
      stage_task.stream_index = stream_index;
      stage_task.node_positions = member_positions;
      for (const int member_position : member_positions) {
        layout.position_to_task[member_position] = task_position;
      }
      layout.stage_to_tasks[stage_index].push_back(task_position);
      layout.stage_tasks.push_back(std::move(stage_task));
    }
  }
  return layout;
}

void add_cross_stream_producers(
    const StageTaskLayout& layout,
    const std::vector<NodeDependencies>& position_to_dependencies,
    int task_position,
    StageTaskDependencies& task_dependencies) {
  const StageTimeline& stage_task = layout.stage_tasks[task_position];
  for (const int member_position : stage_task.node_positions) {
    for (const int producer_position :
         position_to_dependencies[member_position].cross_stream_input_positions) {
      const int producer_task = layout.position_to_task[producer_position];
      const bool is_earlier_stage =
          layout.stage_tasks[producer_task].stage_index < stage_task.stage_index;
      auto& producer_tasks = task_dependencies.cross_stream_producer_tasks;
      if (is_earlier_stage && std::find(producer_tasks.begin(), producer_tasks.end(),
                                        producer_task) == producer_tasks.end()) {
        producer_tasks.push_back(producer_task);
      }
    }
  }
}

// Streams are in-order queues, so the previous task on a stream already
// covers every same-stream producer from an earlier stage.
std::vector<StageTaskDependencies> derive_task_dependencies(
    const StageTaskLayout& layout,
    const std::vector<NodeDependencies>& position_to_dependencies) {
  const int task_count = static_cast<int>(layout.stage_tasks.size());
  std::vector<StageTaskDependencies> task_to_dependencies(task_count);
  std::unordered_map<int, int> stream_index_to_last_task;
  for (int task_position = 0; task_position < task_count; ++task_position) {
    const int stream_index = layout.stage_tasks[task_position].stream_index;
    const auto last_on_stream = stream_index_to_last_task.find(stream_index);
    if (last_on_stream != stream_index_to_last_task.end()) {
      task_to_dependencies[task_position].previous_on_stream_task =
          last_on_stream->second;
    }
    stream_index_to_last_task[stream_index] = task_position;
    add_cross_stream_producers(layout, position_to_dependencies, task_position,
                               task_to_dependencies[task_position]);
  }
  return task_to_dependencies;
}

// Every member of a task on one stream shares that async_eval's per-stream
// Event, so checking one member (the last) observes the whole task without
// paying an is_available call per node per sweep.
void sweep_task_completions(
    const CapturedGraph& captured_graph,
    const StageTaskLayout& layout,
    TimelineClock::time_point pass_started_at,
    std::vector<int>& pending_tasks,
    StageTaskInstants& task_instants) {
  std::vector<int> still_pending_tasks;
  for (const int task_position : pending_tasks) {
    const int representative_position =
        layout.stage_tasks[task_position].node_positions.back();
    if (node_at(captured_graph, representative_position).handle.is_available()) {
      task_instants.task_to_completed_microseconds[task_position] =
          microseconds_since(pass_started_at);
    } else {
      still_pending_tasks.push_back(task_position);
    }
  }
  pending_tasks = std::move(still_pending_tasks);
}

std::vector<mlx::core::array> collect_sink_handles(
    const CapturedGraph& captured_graph, const std::vector<int>& sink_positions) {
  std::vector<mlx::core::array> sink_handles;
  for (const int sink_position : sink_positions) {
    sink_handles.push_back(node_at(captured_graph, sink_position).handle);
  }
  return sink_handles;
}

// One async_eval per stage. Buffers are held BEFORE issue_started and markers
// committed AFTER issue_finished, so the probe's cost is never charged to
// the stage as host issue time.
void dispatch_one_stage(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const StagePlan& stage_plan,
    const StageTaskLayout& layout,
    int stage_index,
    TimelineClock::time_point pass_started_at,
    StageTaskInstants& task_instants,
    GpuExecutionProbe& command_buffer_probe) {
  const std::vector<int>& stage_tasks = layout.stage_to_tasks[stage_index];
  for (const int task_position : stage_tasks) {
    command_buffer_probe.hold_buffer_for(
        task_position,
        stream_index_to_stream.at(layout.stage_tasks[task_position].stream_index));
  }
  const std::vector<mlx::core::array> sink_handles = collect_sink_handles(
      captured_graph, stage_plan.stage_to_sink_positions[stage_index]);
  const double issue_started_microseconds = microseconds_since(pass_started_at);
  mlx::core::async_eval(sink_handles);
  const double issue_finished_microseconds = microseconds_since(pass_started_at);
  for (const int task_position : stage_tasks) {
    command_buffer_probe.commit_end_marker_for(task_position);
    task_instants.task_to_issue_started_microseconds[task_position] =
        issue_started_microseconds;
    task_instants.task_to_issue_finished_microseconds[task_position] =
        issue_finished_microseconds;
  }
}

StageTaskInstants dispatch_and_observe_stages(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const StagePlan& stage_plan,
    const StageTaskLayout& layout,
    TimelineClock::time_point pass_started_at,
    GpuExecutionProbe& command_buffer_probe) {
  const int task_count = static_cast<int>(layout.stage_tasks.size());
  StageTaskInstants task_instants;
  task_instants.task_to_issue_started_microseconds.assign(task_count, 0.0);
  task_instants.task_to_issue_finished_microseconds.assign(task_count, 0.0);
  task_instants.task_to_completed_microseconds.assign(task_count, 0.0);
  std::vector<int> pending_tasks;
  ProgressReporter progress("timeline stage tasks complete", task_count);
  const int stage_count = static_cast<int>(stage_plan.stage_names.size());
  for (int stage_index = 0; stage_index < stage_count; ++stage_index) {
    dispatch_one_stage(captured_graph, stream_index_to_stream, stage_plan, layout,
                       stage_index, pass_started_at, task_instants, command_buffer_probe);
    const std::vector<int>& stage_tasks = layout.stage_to_tasks[stage_index];
    pending_tasks.insert(pending_tasks.end(), stage_tasks.begin(), stage_tasks.end());
    sweep_task_completions(captured_graph, layout, pass_started_at, pending_tasks,
                           task_instants);
    progress.update(task_count - static_cast<int>(pending_tasks.size()));
  }
  // Yield rather than sleep: a sleep's wake-up granularity would dominate
  // the ~100 µs stages this pass resolves.
  while (!pending_tasks.empty()) {
    std::this_thread::yield();
    sweep_task_completions(captured_graph, layout, pass_started_at, pending_tasks,
                           task_instants);
    progress.update(task_count - static_cast<int>(pending_tasks.size()));
  }
  progress.finish(task_count);
  return task_instants;
}

// Both the marker's GPU start and host-observed completion trail the stage's
// true GPU end; the earlier of the two is the tighter honest bound.
std::vector<GpuExecutionWindow> tighten_marker_ends(
    std::vector<GpuExecutionWindow> task_to_gpu_window,
    const std::vector<double>& task_to_completed_microseconds) {
  for (std::size_t task_position = 0; task_position < task_to_gpu_window.size();
       ++task_position) {
    GpuExecutionWindow& gpu_window = task_to_gpu_window[task_position];
    if (gpu_window.is_measured && gpu_window.is_end_bounded_by_marker) {
      gpu_window.finished_microseconds =
          std::max(gpu_window.started_microseconds,
                   std::min(gpu_window.finished_microseconds,
                            task_to_completed_microseconds[task_position]));
    }
  }
  return task_to_gpu_window;
}

NodeTimestamps assemble_task_timestamps(
    int task_position,
    const StageTaskDependencies& task_dependencies,
    const StageTaskInstants& task_instants,
    const std::vector<double>& task_to_ready_microseconds,
    const GpuExecutionWindow& gpu_window) {
  NodeTimestamps timestamps;
  timestamps.has_gpu_window = gpu_window.is_measured;
  timestamps.gpu_started_microseconds = gpu_window.started_microseconds;
  timestamps.gpu_finished_microseconds = gpu_window.finished_microseconds;
  timestamps.issue_started_microseconds =
      task_instants.task_to_issue_started_microseconds[task_position];
  timestamps.issue_finished_microseconds =
      task_instants.task_to_issue_finished_microseconds[task_position];
  timestamps.completed_microseconds =
      task_instants.task_to_completed_microseconds[task_position];
  if (task_dependencies.previous_on_stream_task != NO_POSITION) {
    timestamps.stream_ready_microseconds =
        task_to_ready_microseconds[task_dependencies.previous_on_stream_task];
  }
  timestamps.cross_stream_inputs_ready_microseconds = latest_ready(
      task_dependencies.cross_stream_producer_tasks, task_to_ready_microseconds);
  return timestamps;
}

// Per-node entries keep identity and structure (streams, joins) for the
// DAG and primitive tallies; timing lives on the stage task.
std::vector<NodeTimeline> build_member_nodes(
    const CapturedGraph& captured_graph,
    const std::vector<NodeDependencies>& position_to_dependencies,
    const StageTaskLayout& layout) {
  std::vector<NodeTimeline> node_timelines;
  for (std::size_t position = 0; position < layout.position_to_task.size(); ++position) {
    NodeTimeline node_timeline;
    node_timeline.node = &node_at(captured_graph, static_cast<int>(position));
    node_timeline.stage_position = layout.position_to_task[position];
    record_structure(position_to_dependencies[position], node_timeline);
    node_timelines.push_back(std::move(node_timeline));
  }
  return node_timelines;
}

void finish_stage_task(
    StageTimeline& stage_task,
    const StageTaskDependencies& task_dependencies,
    GpuTimeline& gpu_timeline) {
  stage_task.breakdown = decompose_node_latency(stage_task.timestamps);
  stage_task.cross_stream_producer_stage_positions =
      task_dependencies.cross_stream_producer_tasks;
  stage_task.has_cross_stream_input =
      !stage_task.cross_stream_producer_stage_positions.empty();
  gpu_timeline.cross_stream_stall_microseconds +=
      stage_task.breakdown.wait_on_dependency_microseconds;
  if (stage_task.breakdown.is_compute_measured) {
    gpu_timeline.gpu_measured_node_count +=
        static_cast<int>(stage_task.node_positions.size());
    gpu_timeline.measured_gpu_compute_microseconds +=
        stage_task.breakdown.compute_microseconds;
  }
  gpu_timeline.timeline_wall_microseconds = std::max(
      gpu_timeline.timeline_wall_microseconds,
      stage_task.timestamps.completed_microseconds);
}

} // namespace

std::vector<StageCut> resolve_stage_cuts(
    const CapturedGraph& captured_graph,
    const std::vector<StageBoundary>& stage_boundaries) {
  const auto array_id_to_position = build_array_id_to_position(captured_graph);
  std::vector<StageCut> stage_cuts;
  for (const StageBoundary& stage_boundary : stage_boundaries) {
    StageCut stage_cut{stage_boundary.stage_name, {}};
    for (const auto& boundary_array : stage_boundary.boundary_arrays) {
      const auto boundary_entry = array_id_to_position.find(boundary_array.id());
      if (boundary_entry != array_id_to_position.end()) {
        stage_cut.boundary_positions.push_back(boundary_entry->second);
      }
    }
    stage_cuts.push_back(std::move(stage_cut));
  }
  return stage_cuts;
}

GpuTimeline capture_stage_timeline(
    const CapturedGraph& captured_graph,
    const std::unordered_map<int, mlx::core::Stream>& stream_index_to_stream,
    const std::vector<StageCut>& stage_cuts,
    const std::string& final_stage_name) {
  require_every_stream_known(captured_graph, stream_index_to_stream);
  const std::vector<NodeDependencies> position_to_dependencies =
      snapshot_dependencies(captured_graph);
  const StagePlan stage_plan = plan_stages(
      merge_input_positions(position_to_dependencies), stage_cuts, final_stage_name);
  StageTaskLayout layout = lay_out_stage_tasks(captured_graph, stage_plan);
  const std::vector<StageTaskDependencies> task_to_dependencies =
      derive_task_dependencies(layout, position_to_dependencies);
  const TimelineClock::time_point pass_started_at = TimelineClock::now();
  GpuExecutionProbe command_buffer_probe(
      static_cast<int>(layout.stage_tasks.size()), pass_started_at);
  const StageTaskInstants task_instants = dispatch_and_observe_stages(
      captured_graph, stream_index_to_stream, stage_plan, layout, pass_started_at,
      command_buffer_probe);
  const std::vector<GpuExecutionWindow> task_to_gpu_window = tighten_marker_ends(
      command_buffer_probe.read_windows(), task_instants.task_to_completed_microseconds);
  const std::vector<double> task_to_ready_microseconds = build_ready_microseconds(
      task_instants.task_to_completed_microseconds, task_to_gpu_window);

  GpuTimeline gpu_timeline;
  gpu_timeline.granularity = TimelineGranularity::per_stage;
  gpu_timeline.nodes =
      build_member_nodes(captured_graph, position_to_dependencies, layout);
  for (std::size_t task_position = 0; task_position < layout.stage_tasks.size();
       ++task_position) {
    StageTimeline& stage_task = layout.stage_tasks[task_position];
    stage_task.timestamps = assemble_task_timestamps(
        static_cast<int>(task_position), task_to_dependencies[task_position],
        task_instants, task_to_ready_microseconds, task_to_gpu_window[task_position]);
    finish_stage_task(stage_task, task_to_dependencies[task_position], gpu_timeline);
  }
  gpu_timeline.stages = std::move(layout.stage_tasks);
  gpu_timeline.observed_command_buffer_count =
      command_buffer_probe.count_completed_command_buffers();
  gpu_timeline.is_command_buffer_count_lower_bound = true;
  gpu_timeline.streams = summarize_streams(
      collect_timed_units(gpu_timeline), stream_index_to_stream);
  return gpu_timeline;
}

} // namespace mlx_perf
