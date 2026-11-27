#include "cupti_session.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <unordered_map>

namespace mlx_perf {
namespace {

// Room for every driver call the three observers watch (about a dozen), and
// a few observers per call: the probe and the submission recorder both
// watch graph launches.
constexpr int MAX_CALL_ROUTES = 32;
constexpr int MAX_OBSERVERS_PER_CALL = 4;
// CUPTI fills one buffer before asking for the next; 8 MiB holds tens of
// thousands of kernel records, so a pass rarely needs a second buffer, and
// CUPTI requires 8-byte alignment.
constexpr std::size_t ACTIVITY_BUFFER_BYTES = 8u << 20;
constexpr std::size_t ACTIVITY_BUFFER_ALIGNMENT_BYTES = 8;

struct CallRoute {
  std::atomic<CUpti_CallbackId> callback_id{0};
  std::array<std::atomic<DriverCallObserver*>, MAX_OBSERVERS_PER_CALL> observers{};
  std::atomic<int> observer_count{0};
};

// The probe and the submission recorder can both watch one graph launch, so
// a span is kept until every watcher has taken it.
struct WatchedSpan {
  GpuActivitySpan activity_span;
  int watcher_count = 0;
};

struct SessionState {
  CUpti_SubscriberHandle subscriber = nullptr;
  // Routes are only ever appended, under the registration mutex, and each
  // is complete before route_count publishes it, so the callback reads them
  // without a lock.
  std::array<CallRoute, MAX_CALL_ROUTES> call_routes;
  std::atomic<int> route_count{0};
  std::mutex registration_mutex;
  // Spans of the watched correlation ids; written by CUPTI's buffer thread
  // and the launching threads, read once the work is complete.
  std::mutex activity_mutex;
  std::unordered_map<std::uint32_t, WatchedSpan> correlation_id_to_span;
};

// Never destroyed: CUPTI can deliver activity buffers while the process
// exits, after function-local statics have been destroyed, and the buffer
// callback must still find a live mutex and map then.
SessionState& session_state() {
  static SessionState* const process_lifetime_state = new SessionState();
  return *process_lifetime_state;
}

std::uint64_t read_monotonic_nanoseconds() {
  timespec monotonic_time{};
  clock_gettime(CLOCK_MONOTONIC, &monotonic_time);
  return static_cast<std::uint64_t>(monotonic_time.tv_sec) * 1'000'000'000ull +
      static_cast<std::uint64_t>(monotonic_time.tv_nsec);
}

const CallRoute* find_route(const SessionState& state, CUpti_CallbackId callback_id) {
  const int route_count = state.route_count.load(std::memory_order_acquire);
  for (int route_index = 0; route_index < route_count; ++route_index) {
    if (state.call_routes[route_index].callback_id.load(std::memory_order_relaxed) == callback_id) {
      return &state.call_routes[route_index];
    }
  }
  return nullptr;
}

void CUPTIAPI on_cupti_callback(void*, CUpti_CallbackDomain callback_domain, CUpti_CallbackId callback_id,
                                const void* callback_data_pointer) {
  if (callback_domain != CUPTI_CB_DOMAIN_DRIVER_API) {
    return;
  }
  const CallRoute* call_route = find_route(session_state(), callback_id);
  if (call_route == nullptr) {
    return;
  }
  const auto* callback_data = static_cast<const CUpti_CallbackData*>(callback_data_pointer);
  DriverCall driver_call;
  driver_call.callback_id = callback_id;
  driver_call.is_entering = callback_data->callbackSite == CUPTI_API_ENTER;
  driver_call.correlation_id = callback_data->correlationId;
  driver_call.function_params = callback_data->functionParams;
  if (!driver_call.is_entering && callback_data->functionReturnValue != nullptr) {
    driver_call.returned_result = *static_cast<const CUresult*>(callback_data->functionReturnValue);
  }
  const int observer_count = call_route->observer_count.load(std::memory_order_acquire);
  for (int observer_index = 0; observer_index < observer_count; ++observer_index) {
    call_route->observers[observer_index].load(std::memory_order_relaxed)->on_driver_call(driver_call);
  }
}

void CUPTIAPI on_buffer_requested(std::uint8_t** activity_buffer, std::size_t* buffer_bytes,
                                  std::size_t* max_record_count) {
  *activity_buffer = static_cast<std::uint8_t*>(
      std::aligned_alloc(ACTIVITY_BUFFER_ALIGNMENT_BYTES, ACTIVITY_BUFFER_BYTES));
  *buffer_bytes = *activity_buffer == nullptr ? 0 : ACTIVITY_BUFFER_BYTES;
  *max_record_count = 0;  // as many as fit
}

struct RecordedWork {
  std::uint32_t correlation_id = 0;
  std::uint64_t started_nanoseconds = 0;
  std::uint64_t finished_nanoseconds = 0;
};

// The three record kinds that occupy the GPU: kernels, copies and fills.
// Every other kind CUPTI might deliver is not GPU work and is skipped.
bool read_recorded_work(const CUpti_Activity* activity_record, RecordedWork& recorded_work) {
  switch (activity_record->kind) {
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    case CUPTI_ACTIVITY_KIND_KERNEL: {
      const auto* kernel_record = reinterpret_cast<const CUpti_ActivityKernel13*>(activity_record);
      recorded_work = {kernel_record->correlationId, kernel_record->start, kernel_record->end};
      return true;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      const auto* copy_record = reinterpret_cast<const CUpti_ActivityMemcpy7*>(activity_record);
      recorded_work = {copy_record->correlationId, copy_record->start, copy_record->end};
      return true;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
      const auto* fill_record = reinterpret_cast<const CUpti_ActivityMemset5*>(activity_record);
      recorded_work = {fill_record->correlationId, fill_record->start, fill_record->end};
      return true;
    }
    default:
      return false;
  }
}

void merge_into_watched_span(SessionState& state, const RecordedWork& recorded_work) {
  const auto span_entry = state.correlation_id_to_span.find(recorded_work.correlation_id);
  if (span_entry == state.correlation_id_to_span.end() ||
      recorded_work.finished_nanoseconds < recorded_work.started_nanoseconds) {
    return;
  }
  GpuActivitySpan& activity_span = span_entry->second.activity_span;
  if (!activity_span.is_measured) {
    activity_span = {true, recorded_work.started_nanoseconds, recorded_work.finished_nanoseconds, 0};
  }
  activity_span.started_nanoseconds = std::min(activity_span.started_nanoseconds, recorded_work.started_nanoseconds);
  activity_span.finished_nanoseconds = std::max(activity_span.finished_nanoseconds, recorded_work.finished_nanoseconds);
  ++activity_span.activity_record_count;
}

void CUPTIAPI on_buffer_completed(CUcontext, std::uint32_t, std::uint8_t* activity_buffer, std::size_t,
                                  std::size_t valid_bytes) {
  SessionState& state = session_state();
  {
    const std::lock_guard<std::mutex> activity_lock(state.activity_mutex);
    CUpti_Activity* activity_record = nullptr;
    while (cuptiActivityGetNextRecord(activity_buffer, valid_bytes, &activity_record) == CUPTI_SUCCESS) {
      RecordedWork recorded_work;
      if (read_recorded_work(activity_record, recorded_work)) {
        merge_into_watched_span(state, recorded_work);
      }
    }
  }
  std::free(activity_buffer);
}

bool subscribe_and_enable_activity() {
  SessionState& state = session_state();
  // Before anything else: records CUPTI timestamps from now on use this clock.
  if (cuptiActivityRegisterTimestampCallback(read_monotonic_nanoseconds) != CUPTI_SUCCESS) {
    return false;
  }
  if (cuptiSubscribe(&state.subscriber, on_cupti_callback, nullptr) != CUPTI_SUCCESS) {
    return false;
  }
  if (cuptiActivityRegisterCallbacks(on_buffer_requested, on_buffer_completed) != CUPTI_SUCCESS) {
    return false;
  }
  return cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) == CUPTI_SUCCESS &&
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY) == CUPTI_SUCCESS &&
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET) == CUPTI_SUCCESS;
}

} // namespace

bool start_cupti_session() {
  static const bool is_started = subscribe_and_enable_activity();
  return is_started;
}

bool observe_driver_call(CUpti_CallbackId callback_id, DriverCallObserver* observer) {
  if (!start_cupti_session() || observer == nullptr) {
    return false;
  }
  SessionState& state = session_state();
  const std::lock_guard<std::mutex> registration_lock(state.registration_mutex);
  CallRoute* call_route = const_cast<CallRoute*>(find_route(state, callback_id));
  if (call_route == nullptr) {
    const int route_count = state.route_count.load(std::memory_order_relaxed);
    if (route_count == MAX_CALL_ROUTES) {
      return false;
    }
    call_route = &state.call_routes[route_count];
    call_route->callback_id.store(callback_id, std::memory_order_relaxed);
    state.route_count.store(route_count + 1, std::memory_order_release);
  }
  const int observer_count = call_route->observer_count.load(std::memory_order_relaxed);
  if (observer_count == MAX_OBSERVERS_PER_CALL) {
    return false;
  }
  call_route->observers[observer_count].store(observer, std::memory_order_relaxed);
  call_route->observer_count.store(observer_count + 1, std::memory_order_release);
  return cuptiEnableCallback(1, state.subscriber, CUPTI_CB_DOMAIN_DRIVER_API, callback_id) == CUPTI_SUCCESS;
}

void watch_gpu_activity(std::uint32_t correlation_id) {
  SessionState& state = session_state();
  const std::lock_guard<std::mutex> activity_lock(state.activity_mutex);
  ++state.correlation_id_to_span[correlation_id].watcher_count;
}

void flush_gpu_activity() {
  if (start_cupti_session()) {
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  }
}

GpuActivitySpan take_gpu_activity_span(std::uint32_t correlation_id) {
  SessionState& state = session_state();
  const std::lock_guard<std::mutex> activity_lock(state.activity_mutex);
  const auto span_entry = state.correlation_id_to_span.find(correlation_id);
  if (span_entry == state.correlation_id_to_span.end()) {
    return GpuActivitySpan{};
  }
  const GpuActivitySpan activity_span = span_entry->second.activity_span;
  if (--span_entry->second.watcher_count <= 0) {
    state.correlation_id_to_span.erase(span_entry);
  }
  return activity_span;
}

std::uint64_t current_session_nanoseconds() {
  return read_monotonic_nanoseconds();
}

} // namespace mlx_perf
