//! The job this example profiles, built with mlx-bridge's safe API on three
//! GPU streams: two independent branches on their own streams, joined on the
//! main stream (the cross-stream "shuffle" edge the report shows as a stall).

use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDeviceHandle, MlxDtype, MlxStream, raw};

const BATCH_ROWS: i32 = 256;
const FEATURE_WIDTH: i32 = 1024;
const REDUCTION_ROWS: i32 = 2048;

/// One context per stream: mlx-bridge submits every operation to its
/// context's stream, which is how the branches land on different streams.
pub struct ThreeStreamJob {
    main_context: MlxBindingsContext,
    projection_context: MlxBindingsContext,
    reduction_context: MlxBindingsContext,
}

impl ThreeStreamJob {
    /// The main stream is the default GPU stream; each branch gets a new one.
    pub fn new() -> Result<Self, MlxCError> {
        let gpu_device = MlxDeviceHandle::default_device()?;
        Ok(Self {
            main_context: MlxBindingsContext::on_stream(MlxStream::default_gpu()?)?,
            projection_context: MlxBindingsContext::on_stream(MlxStream::on_device(&gpu_device)?)?,
            reduction_context: MlxBindingsContext::on_stream(MlxStream::on_device(&gpu_device)?)?,
        })
    }

    /// A brand-new, unevaluated graph for the job's single output.
    pub fn build_fresh_output(&self) -> Result<MlxArray, MlxCError> {
        let projection = &self.projection_context;
        let batch = projection.ones(&[BATCH_ROWS, FEATURE_WIDTH], MlxDtype::Float32)?;
        let weights = projection.ones(&[FEATURE_WIDTH, FEATURE_WIDTH], MlxDtype::Float32)?;
        let projected = projection.multiply_scalar(&projection.matmul(&batch, &weights)?, 1e-3)?;
        let gated = projection.multiply(&projected, &projection.sigmoid(&projected)?)?;

        let reduction = &self.reduction_context;
        let samples = reduction.ones(&[REDUCTION_ROWS, FEATURE_WIDTH], MlxDtype::Float32)?;
        let mixed = reduction.add(&reduction.exp(&samples)?, &reduction.sin(&samples)?)?;
        let column_totals = reduction.sum_axis(&mixed, 0, false)?;

        let joined = self.main_context.add(&gated, &column_totals)?;
        self.main_context.tanh(&joined)
    }
}

/// The C callback handed to mlx-performance-profiler. A panic must never unwind into
/// C++ frames (undefined behavior), so it is caught and reported as failure;
/// catch_unwind's closure is the one closure this crate cannot avoid.
///
/// # Safety
/// `payload` must point at a live `ThreeStreamJob`.
pub unsafe extern "C" fn build_three_stream_graph(
    fresh_outputs: raw::mlx_vector_array,
    payload: *mut c_void,
) -> c_int {
    // SAFETY: The caller passes the ThreeStreamJob it registered as payload.
    let three_stream_job = unsafe { &*(payload as *const ThreeStreamJob) };
    let build_outcome = catch_unwind(AssertUnwindSafe(|| append_fresh_output(three_stream_job, fresh_outputs)));
    match build_outcome {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => 1,
        Err(_) => 2,
    }
}

fn append_fresh_output(
    three_stream_job: &ThreeStreamJob,
    fresh_outputs: raw::mlx_vector_array,
) -> Result<(), MlxCError> {
    let job_output = three_stream_job.build_fresh_output()?;
    // SAFETY: Both handles are live; MLX-C copies the array handle on append,
    // so dropping `job_output` afterwards is fine.
    let append_status = unsafe { raw::mlx_vector_array_append_value(fresh_outputs, job_output.raw()) };
    if append_status == 0 {
        Ok(())
    } else {
        Err(MlxCError {
            operation: "append the job output",
            description: format!("mlx_vector_array_append_value returned {append_status}"),
        })
    }
}
