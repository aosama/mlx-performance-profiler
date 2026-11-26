//! Human-readable job summary for the terminal.

use crate::mlx_perf_ffi::JobSummary;

const MICROSECONDS_PER_MILLISECOND: f64 = 1000.0;

fn milliseconds(microseconds: f64) -> f64 {
    microseconds / MICROSECONDS_PER_MILLISECOND
}

/// Renders the headline numbers, each with what it measures.
pub fn describe_summary(job_summary: &JobSummary) -> String {
    format!(
        "job duration        {:.3} ms  median of warm runs (range {:.3}–{:.3} ms)\n\
         instrumented        {:.3} ms  timeline pass, one command buffer per node\n\
         sum of stages       {:.3} ms  each node synchronized alone\n\
         GPU execution       {:.3} ms  measured over {}/{} nodes\n\
         cross-stream stall  {:.3} ms  Spark fetchWaitTime analog\n\
         streams             {}",
        milliseconds(job_summary.job_wall_median_us),
        milliseconds(job_summary.job_wall_min_us),
        milliseconds(job_summary.job_wall_max_us),
        milliseconds(job_summary.timeline_wall_us),
        milliseconds(job_summary.sum_of_stages_us),
        milliseconds(job_summary.measured_gpu_execution_us),
        job_summary.gpu_measured_node_count,
        job_summary.executed_node_count,
        milliseconds(job_summary.cross_stream_stall_us),
        job_summary.stream_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_summary() -> JobSummary {
        JobSummary {
            job_wall_median_us: 1500.0,
            job_wall_min_us: 1200.0,
            job_wall_max_us: 2100.0,
            timeline_wall_us: 2400.0,
            sum_of_stages_us: 6000.0,
            measured_gpu_execution_us: 900.0,
            cross_stream_stall_us: 300.0,
            executed_node_count: 12,
            gpu_measured_node_count: 12,
            stream_count: 3,
        }
    }

    #[test]
    fn should_show_every_headline_number_in_milliseconds() {
        let described = describe_summary(&sample_summary());
        assert!(described.contains("1.500 ms"));
        assert!(described.contains("(range 1.200–2.100 ms)"));
        assert!(described.contains("measured over 12/12 nodes"));
        assert!(described.contains("streams             3"));
    }

    #[test]
    fn should_render_an_all_zero_summary_without_panicking() {
        let described = describe_summary(&JobSummary::default());
        assert!(described.contains("0.000 ms"));
        assert!(described.contains("0/0 nodes"));
    }

    #[test]
    fn should_keep_negative_inputs_visible_rather_than_clamped() {
        let mut corrupt_summary = sample_summary();
        corrupt_summary.cross_stream_stall_us = -50.0;
        assert!(describe_summary(&corrupt_summary).contains("-0.050 ms"));
    }
}
