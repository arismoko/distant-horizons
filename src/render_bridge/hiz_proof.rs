//! One-shot Case 1 diagnostic. No buffer or ticket is retained between captures.

use super::{CameraMatrices, HIZ_WORKGROUP, HizTarget, Std140, id, next_serial, storage};
use crate::gpu_arena::unpack_section_pos;
use mod_api::{
    BindGroupLayout, BindGroupLayoutBuilder, ClientHost, ComputeDispatchContext,
    ComputeDispatchError, ComputePipeline, ComputePipelineRef, DrawBindings, ReadbackTicket,
    RegisterRenderPipelinesEvent, Sampler, SamplerAddress, SamplerFilter, StorageBufferDescriptor,
    StorageBufferRef,
};
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::sync::Mutex;

const DIR: &str = "distantsunrises-hiz-proof";
const STATUS: &str = "distantsunrises-hiz-proof/capture.txt";
const STATUS_TMP: &str = "distantsunrises-hiz-proof/capture.tmp";
const BUFFER: &str = "HizProofBuffer";
const PARAMS: &str = "HizProofParams";
// Match hiz_proof.glsl. Each stage gets one bounded list and a count independent of that bound.
const CAPACITY: usize = 4096;
const HEADER_WORDS: usize = 12;
const REJECT_WORDS: usize = 10;
const DEPTH_OFFSET: usize = (HEADER_WORDS + 2 * CAPACITY * REJECT_WORDS) * 4;
const TRAV_ITERS: u32 = crate::octree::MAX_OCTREE_DETAIL as u32 + 1;

enum State {
    Idle,
    Armed([u32; 2]),
    Recording,
    InFlight(Pending),
}

static STATE: Mutex<State> = Mutex::new(State::Idle);

fn write_status(text: &str) -> io::Result<()> {
    fs::write(STATUS_TMP, text)?;
    fs::rename(STATUS_TMP, STATUS)
}

pub(crate) fn arm() -> io::Result<Option<u64>> {
    let mut state = STATE.lock().unwrap();
    if !matches!(*state, State::Idle) {
        return Ok(None);
    }
    fs::create_dir_all(DIR)?;
    let serial = next_serial();
    // Invalidate the previous capture before arming, even if this frame cannot be recorded.
    write_status(&format!("status=armed\nserial={}\n", serial_number(serial)))?;
    *state = State::Armed(serial);
    Ok(Some(serial_number(serial)))
}

fn serial_number([lo, hi]: [u32; 2]) -> u64 {
    (u64::from(hi) << 32) | u64::from(lo)
}

fn failure(serial: [u32; 2], reason: &str) {
    let serial = serial_number(serial);
    eprintln!("distantsunrises: Hi-Z capture {serial} failed: {reason}");
    if let Err(err) = write_status(&format!(
        "status=failed\nserial={serial}\nreason={reason}\n"
    )) {
        eprintln!("distantsunrises: Hi-Z capture {serial} could not publish failure: {err}");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kernel {
    Mip0,
    Reduce,
    Init,
    Clear,
    Traversal,
    Cmdgen,
    Capture,
}

impl Kernel {
    fn id(self) -> mod_api::Identifier {
        id(match self {
            Self::Mip0 => "distantsunrises:voxy_hiz_copy_proof",
            Self::Reduce => "distantsunrises:voxy_hiz_reduce_proof",
            Self::Init => "distantsunrises:voxy_trav_init_proof",
            Self::Clear => "distantsunrises:voxy_trav_clear_proof",
            Self::Traversal => "distantsunrises:voxy_traversal_proof",
            Self::Cmdgen => "distantsunrises:voxy_cmdgen_proof",
            Self::Capture => "distantsunrises:voxy_hiz_capture",
        })
    }

    fn indexed(self) -> bool {
        matches!(self, Self::Mip0 | Self::Reduce | Self::Traversal)
    }

    fn proof_layout(self, layout: BindGroupLayoutBuilder) -> BindGroupLayoutBuilder {
        let layout = layout.entry(storage(BUFFER, false, (HEADER_WORDS as u32 + 1) * 4));
        if self.indexed() || self == Self::Capture {
            layout.uniform_buffer(PARAMS, 16)
        } else {
            layout
        }
    }
}

pub(super) fn register_pair(
    event: &mut RegisterRenderPipelinesEvent<'_>,
    name: &str,
    shader: &str,
    layout: BindGroupLayoutBuilder,
    kernel: Kernel,
) {
    let make = |pipeline_id, layout: BindGroupLayoutBuilder| {
        let builder = ComputePipeline::builder(pipeline_id, id(shader))
            .bind_group_layout(layout.build().expect("static compute layout"));
        if kernel == Kernel::Mip0 {
            builder.define("HIZ_FROM_WORLD_DEPTH")
        } else {
            builder
        }
    };
    event.compute_pipeline(make(id(name), layout.clone()).build());
    event.compute_pipeline(
        make(kernel.id(), kernel.proof_layout(layout))
            .define("HIZ_PROOF")
            .define_value("HIZ_PROOF_CAPACITY", CAPACITY.to_string())
            .build(),
    );
}

pub(super) fn register_capture(event: &mut RegisterRenderPipelinesEvent<'_>) {
    let kernel = Kernel::Capture;
    let layout = BindGroupLayout::builder().sampler(
        "HizMip0",
        Sampler::new(SamplerFilter::Nearest, SamplerAddress::ClampToEdge),
    );
    event.compute_pipeline(
        ComputePipeline::builder(kernel.id(), kernel.id())
            .bind_group_layout(
                kernel
                    .proof_layout(layout)
                    .build()
                    .expect("static capture layout"),
            )
            .define_value("HIZ_PROOF_CAPACITY", CAPACITY.to_string())
            .build(),
    );
}

struct Frame {
    serial: [u32; 2],
    size: [u32; 2],
    mips: u32,
    camera: CameraMatrices,
    byte_len: usize,
}

pub(super) struct Recording {
    frame: Frame,
    pipelines: [ComputePipelineRef; 7],
    buffer: StorageBufferRef,
    failure: Option<String>,
    ticket: Option<ReadbackTicket>,
}

struct Pending {
    frame: Frame,
    ticket: ReadbackTicket,
    buffer: StorageBufferRef,
}

pub(super) fn prepare(
    host: &dyn ClientHost,
    camera: &CameraMatrices,
    hiz: &HizTarget,
) -> Option<Recording> {
    let serial = {
        let mut state = STATE.lock().unwrap();
        let State::Armed(serial) = *state else {
            return None;
        };
        *state = State::Recording;
        serial
    };
    let prepared = (|| {
        use Kernel::*;
        let assets = host.render_assets();
        let pipeline = |k: Kernel| {
            assets
                .compute_pipeline(&k.id())
                .map_err(|err| format!("{k:?} pipeline: {err:?}"))
        };
        let pipelines = [
            pipeline(Mip0)?,
            pipeline(Reduce)?,
            pipeline(Init)?,
            pipeline(Clear)?,
            pipeline(Traversal)?,
            pipeline(Cmdgen)?,
            pipeline(Capture)?,
        ];
        let [w, h] = hiz.size;
        let byte_len = (w as usize)
            .checked_mul(h as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|bytes| bytes.checked_add(DEPTH_OFFSET))
            .ok_or("capture length overflow")?;
        // A fresh WebGPU buffer is zero-initialized, including the GPU witnesses and counters.
        let buffer = host
            .create_storage_buffer(&StorageBufferDescriptor::new(byte_len as u64).copy_src())
            .map_err(|err| format!("allocate full-resolution buffer: {err:?}"))?;
        Ok::<_, String>(Recording {
            frame: Frame {
                serial,
                size: hiz.size,
                mips: hiz.mips,
                camera: *camera,
                byte_len,
            },
            pipelines,
            buffer,
            failure: None,
            ticket: None,
        })
    })();
    match prepared {
        Ok(recording) => Some(recording),
        Err(err) => {
            failure(serial, &err);
            *STATE.lock().unwrap() = State::Idle;
            None
        }
    }
}

/// Extend the base bindings only on the one armed frame. A refusal during recording does not
/// enqueue a dispatch: run the base pipeline at this same spot and discard this capture.
pub(super) fn dispatch(
    recording: &mut Option<Recording>,
    kernel: Kernel,
    index: u32,
    base: ComputePipelineRef,
    bindings: DrawBindings<'_>,
    mut record: impl FnMut(ComputePipelineRef, &DrawBindings<'_>) -> Result<(), ComputeDispatchError>,
) -> Result<(), ComputeDispatchError> {
    if let Some(capture) = recording.as_mut().filter(|r| r.failure.is_none()) {
        let params = index.to_le_bytes();
        let mut uniform = [0; 16];
        uniform[..4].copy_from_slice(&params);
        let mut instrumented = bindings
            .clone()
            .storage_buffer(BUFFER, capture.buffer.slice(0, DEPTH_OFFSET as u64));
        if kernel.indexed() {
            instrumented = instrumented.inline_uniform(PARAMS, &uniform);
        }
        match record(capture.pipelines[kernel as usize], &instrumented) {
            Ok(()) => return Ok(()),
            Err(err) => capture.failure = Some(format!("record {kernel:?}: {err:?}")),
        }
    }
    record(base, &bindings)
}

impl Recording {
    pub(super) fn copy_mip0(&mut self, ctx: &mut dyn ComputeDispatchContext, hiz: &HizTarget) {
        if self.failure.is_some() {
            return;
        }
        let [w, h] = self.frame.size;
        let [lo, hi] = self.frame.serial;
        let params = Std140(Vec::with_capacity(16)).u32s(&[w, h, lo, hi]).0;
        let bindings = DrawBindings::new()
            .texture("HizMip0", hiz.source(0))
            .storage_buffer(BUFFER, self.buffer.slice(0, self.frame.byte_len as u64))
            .inline_uniform(PARAMS, &params);
        if let Err(err) = ctx.dispatch(
            self.pipelines[Kernel::Capture as usize],
            &bindings,
            [w.div_ceil(HIZ_WORKGROUP), h.div_ceil(HIZ_WORKGROUP), 1],
        ) {
            self.failure = Some(format!("record mip0 copy: {err:?}"));
        }
    }

    pub(super) fn request_readback(&mut self, host: &dyn ClientHost) {
        if self.failure.is_none() {
            match self
                .buffer
                .request_readback(host, 0, self.frame.byte_len as u64)
            {
                Ok(ticket) => self.ticket = Some(ticket),
                Err(err) => self.failure = Some(format!("request readback: {err:?}")),
            }
        }
    }

    pub(super) fn finish(self, host: &dyn ClientHost) {
        let mut state = STATE.lock().unwrap();
        if let Some(ticket) = self.ticket {
            *state = State::InFlight(Pending {
                frame: self.frame,
                ticket,
                buffer: self.buffer,
            });
        } else {
            failure(
                self.frame.serial,
                self.failure
                    .as_deref()
                    .unwrap_or("compute node not recorded"),
            );
            super::report("release Hi-Z proof buffer", self.buffer.release(host));
            *state = State::Idle;
        }
    }
}

pub(super) fn poll(host: &dyn ClientHost) {
    let mut state = STATE.lock().unwrap();
    let State::InFlight(pending) = &*state else {
        return;
    };
    let result = match pending.ticket.poll_readback(host) {
        Ok(None) => return,
        Ok(Some(bytes)) => pending.frame.publish(&bytes),
        Err(err) => Err(format!("poll readback: {err:?}")),
    };
    if let Err(err) = result {
        failure(pending.frame.serial, &err);
    }
    super::report("release Hi-Z proof buffer", pending.buffer.release(host));
    *state = State::Idle;
}

impl Frame {
    fn publish(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() != self.byte_len {
            return Err(format!(
                "partial readback: got {} of {} bytes",
                bytes.len(),
                self.byte_len
            ));
        }
        let word = |i: usize| u32::from_le_bytes(bytes[i * 4..][..4].try_into().unwrap());
        let [w, h] = self.size;
        let [
            lo,
            hi,
            mips,
            groups,
            init,
            clears,
            expected,
            ran,
            traversal,
            cmdgen_ran,
            cmdgen,
        ] = std::array::from_fn(word);
        let mip_mask = u32::MAX >> (32 - self.mips);
        let groups_expected = w.div_ceil(HIZ_WORKGROUP) * h.div_ceil(HIZ_WORKGROUP);
        let trav_mask = (1 << TRAV_ITERS) - 1;
        if [lo, hi] != self.serial
            || mips != mip_mask
            || groups != groups_expected
            || init != 1
            || clears != TRAV_ITERS
            || ran != expected & trav_mask
            || cmdgen_ran != 1
        {
            return Err(format!(
                "missing GPU witnesses: serial={}, mips={mips:#x}/{mip_mask:#x}, copy_groups={groups}/{groups_expected}, init={init}, clears={clears}/{TRAV_ITERS}, traversal={ran:#x}/{expected:#x}, cmdgen={cmdgen_ran}",
                serial_number([lo, hi])
            ));
        }
        let depth = &bytes[DEPTH_OFFSET..];
        let all_zero = depth
            .chunks_exact(4)
            .all(|p| f32::from_le_bytes(p.try_into().unwrap()) == 0.0);
        if all_zero && (traversal != 0 || cmdgen != 0) {
            return Err("all-zero mip0 disagrees with positive Hi-Z rejects".to_owned());
        }
        let mut rows = String::from(
            "stage\tid\tdetail\tx\ty\tz\tmip\tnearest\tfar_occluder\tnearest_bits\tfar_bits\tlo_x\tlo_y\thi_x\thi_y\n",
        );
        let mut totals = String::new();
        for (stage_index, (stage, total)) in [("traversal", traversal), ("cmdgen", cmdgen)]
            .into_iter()
            .enumerate()
        {
            let kept = (total as usize).min(CAPACITY);
            writeln!(
                totals,
                "{stage}_rejects={total}\n{stage}_recorded={kept}\n{stage}_truncated={}",
                total as usize > kept
            )
            .unwrap();
            for slot in 0..kept {
                let at = HEADER_WORDS + (stage_index * CAPACITY + slot) * REJECT_WORDS;
                let [p0, p1, id, mip, near_bits, far_bits, x0, y0, x1, y1] =
                    std::array::from_fn(|i| word(at + i));
                let (detail, [x, y, z]) = unpack_section_pos([p0, p1]);
                let nearest = f32::from_bits(near_bits);
                let far = f32::from_bits(far_bits);
                if !(far > 0.0 && far >= nearest)
                    || mip >= self.mips
                    || x0 > x1
                    || y0 > y1
                    || x1 >= (w >> mip).max(1)
                    || y1 >= (h >> mip).max(1)
                {
                    return Err(format!("invalid {stage} reject at {slot}"));
                }
                writeln!(rows, "{stage}\t{id}\t{detail}\t{x}\t{y}\t{z}\t{mip}\t{nearest}\t{far}\t{near_bits:08x}\t{far_bits:08x}\t{x0}\t{y0}\t{x1}\t{y1}").unwrap();
            }
        }
        let serial = serial_number(self.serial);
        let camera = self.camera;
        fs::write(format!("{DIR}/mip0.f32le"), depth).map_err(|err| err.to_string())?;
        fs::write(format!("{DIR}/rejects.tsv"), rows).map_err(|err| err.to_string())?;
        // Publish the matching manifest last; existing raw files without a complete status are
        // not a capture. A witnessed all-zero depth map is a result, not a proof of valid depth.
        write_status(&format!(
            "status=complete\nserial={serial}\nwidth={w}\nheight={h}\nmip_count={}\ndepth_all_zero={all_zero}\n\
             mip_written_mask={mips:#x}\ncopy_workgroups={groups}\ntraversal_init={init}\ntraversal_clears={clears}\n\
             traversal_expected_mask={:#x}\ntraversal_ran_mask={ran:#x}\ncmdgen_ran={cmdgen_ran}\n\
             camera_position={:?}\ncamera_yaw={}\ncamera_pitch={}\ncamera_fov_degrees={}\ncamera_near={}\ncamera_far={}\n\
             camera_view={:?}\ncamera_projection={:?}\n{totals}",
            self.mips, expected & trav_mask, camera.position, camera.yaw, camera.pitch,
            camera.fov_degrees, camera.near, camera.far, camera.view, camera.projection,
        )).map_err(|err| err.to_string())?;
        eprintln!("distantsunrises: Hi-Z capture {serial} complete: {STATUS}");
        Ok(())
    }
}
