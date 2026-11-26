#pragma once

// Recording slots: one concern only — where an observation hook finds the
// recording it should write into. A recording is open either for the calling
// thread or for the whole process; the recorders (dispatch, submission) each
// keep their own slots and decide what to record.
//
// Why two reaches: the serial pass times one node at a time on the thread
// that evaluates it, and must not pick up other threads' Metal work, so it
// records on the calling thread only. A procedure may hand its MLX work to a
// worker thread (an engine with an owner thread for MLX), and procedures run
// one at a time by design, so everything the process submits while one runs
// is its work: it records process-wide (#44).
//
// Cost: with no recording open, a hook reads one thread-local pointer and one
// relaxed atomic. Only while a process-wide recording is open does a hook
// take a mutex, which also keeps the recording alive until the hook is done.

#include <atomic>
#include <mutex>
#include <utility>

namespace mlx_perf {

enum class RecordingReach { calling_thread, whole_process };

template <typename RecordingState>
class RecordingSlots {
 public:
  // The recording a hook should write into right now, or none. The calling
  // thread's own recording wins over a process-wide one. Hold it only for the
  // hook's bookkeeping, never across the original Metal call.
  class ActiveRecording {
   public:
    ActiveRecording() {
      if (thread_recording_state_ != nullptr) {
        recording_state_ = thread_recording_state_;
        return;
      }
      if (process_recording_state_.load(std::memory_order_acquire) == nullptr) {
        return;
      }
      process_lock_ = std::unique_lock<std::mutex>(process_recording_mutex_);
      recording_state_ = process_recording_state_.load(std::memory_order_relaxed);
    }
    RecordingState* recording_state() const { return recording_state_; }

   private:
    std::unique_lock<std::mutex> process_lock_;
    RecordingState* recording_state_ = nullptr;
  };

  // Opens `recording_state` (nullptr pauses) at `reach` and returns the
  // recording it replaces, which the caller restores the same way. Closing a
  // process-wide recording waits for any hook still writing into it.
  // Recordings of one reach must close in the reverse order they opened; for
  // whole-process ones that means one measurement at a time, which is how
  // procedures are measured.
  static RecordingState* swap_open_recording(RecordingReach reach, RecordingState* recording_state) {
    if (reach == RecordingReach::calling_thread) {
      std::swap(thread_recording_state_, recording_state);
      return recording_state;
    }
    const std::lock_guard<std::mutex> process_lock(process_recording_mutex_);
    return process_recording_state_.exchange(recording_state, std::memory_order_acq_rel);
  }

 private:
  static inline thread_local RecordingState* thread_recording_state_ = nullptr;
  static inline std::atomic<RecordingState*> process_recording_state_{nullptr};
  static inline std::mutex process_recording_mutex_;
};

} // namespace mlx_perf
