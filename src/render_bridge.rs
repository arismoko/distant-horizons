//! Thread-safe bridge between Distant Sunrises mod hooks, background workers, and the GPU-driven
//! renderer (`gpu_arena`, GLSL shaders under `assets/distantsunrises/shaders/`).
//!
//! No `wgpu` types and no `[native.*]`: everything goes through Sunrise's general render API.

use crate::api::{LodWorldGenerator, RemoteLodProvider};
use crate::config::DhConfig;
use crate::gen_worker::GenWorkerPool;
use crate::lod::{DimensionLodStore, SectionPos};
use crate::stage3_camera::CameraMatrices;
use crate::storage::{IoResponse, LodIoWorker};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::Instant;

// ---------------------------------------------------------------------------------------------
// GPU-driven Voxy-style renderer on Sunrise's general mod API (zero `[native.*]`).
//
// Frame (all at `RenderLevelStage::AfterOpaqueBlocks`, ruling 16: world-stage callbacks):
//   1. `voxy_hiz.csh` builds an R32Float depth pyramid from vanilla `WorldDepth` (sunrise#5590/#5591),
//   2. `voxy_cmdgen.csh` frustum + Hi-Z culls every section and rewrites its indirect slots
//      (`instance_count = 0` when culled, ruling 18),
//   3. one `draw_indirect` for the opaque slots and one for the translucent slots (sunrise#5592)
//      pull 8-byte quads from the arena (`voxy_quads.vsh/fsh`),
//   4. a shader pack that compiled `dh_terrain` / `dh_water` / `dh_shadow` draws through the
//      pack-linked pipelines (`voxy_pack_*.vsh`) instead of `voxy_quads`.
// ---------------------------------------------------------------------------------------------

use crate::gpu_arena::{
    ArenaConfig, ArenaError, BuiltSection, GpuArena, OPAQUE_COUNT_OFFSET, TRANSLUCENT_COUNT_OFFSET,
};

/// Shader ids (`assets/distantsunrises/shaders/{core,compute}/<path>`).
pub const SHADER_QUADS: &str = "distantsunrises:core/voxy_quads";
pub const SHADER_HIZ: &str = "distantsunrises:voxy_hiz";
pub const SHADER_TRAV_INIT: &str = "distantsunrises:voxy_trav_init";
pub const SHADER_TRAV_CLEAR: &str = "distantsunrises:voxy_trav_clear";
pub const SHADER_TRAVERSAL: &str = "distantsunrises:voxy_traversal";
pub const SHADER_CMDGEN: &str = "distantsunrises:voxy_cmdgen";
pub const SHADER_TRANS_CLEAR: &str = "distantsunrises:voxy_trans_clear";
pub const SHADER_TRANS_PREFIX: &str = "distantsunrises:voxy_trans_prefix";
pub const SHADER_TRANS_SCATTER: &str = "distantsunrises:voxy_trans_scatter";
pub const SHADER_SSAO: &str = "distantsunrises:voxy_ssao";
pub const SHADER_SSAO_APPLY: &str = "distantsunrises:core/voxy_ssao_apply";

pub const PIPELINE_OPAQUE: &str = "distantsunrises:voxy_opaque";
pub const PIPELINE_TRANSLUCENT: &str = "distantsunrises:voxy_translucent";
pub const PIPELINE_SSAO_APPLY: &str = "distantsunrises:voxy_ssao_apply";
pub const PIPELINE_PACK_TERRAIN: &str = "distantsunrises:voxy_pack_terrain";
pub const PIPELINE_PACK_WATER: &str = "distantsunrises:voxy_pack_water";
pub const PIPELINE_PACK_SHADOW: &str = "distantsunrises:voxy_pack_shadow";

/// Compute workgroup sizes; must match `local_size_*` in the `.csh` files.
pub const CMDGEN_WORKGROUP: u32 = 128;
pub const TRAV_WORKGROUP: u32 = 64;
pub const HIZ_WORKGROUP: u32 = 8;

fn id(text: &str) -> mod_api::Identifier {
    mod_api::Identifier::parse(text).expect("static identifier")
}

/// A storage-buffer layout entry (`std430`, set 1). `min_size` is the size of the GLSL block's
/// fixed part (one array element): the bound slice must hold at least that.
fn storage(name: &str, read_only: bool, min_size: u32) -> mod_api::BindEntry {
    mod_api::BindEntry::new(
        name,
        mod_api::BindingKind::StorageBuffer {
            read_only,
            min_size,
        },
    )
}

/// Sampler of the baked face atlas: trilinear over all mip levels (`max_lod > 0.25` enables
/// linear blending between levels `0..=ATLAS_MIP_LEVELS - 1`, like vanilla's backends).
fn atlas_sampler() -> mod_api::Sampler {
    let mut sampler = mod_api::Sampler::new(
        mod_api::SamplerFilter::Linear,
        mod_api::SamplerAddress::ClampToEdge,
    );
    sampler.max_lod = Some((crate::pipeline::ATLAS_MIP_LEVELS - 1) as f32);
    sampler
}

/// The quad pipelines' set-1 layout (`voxy_quads.vsh/fsh`).
pub fn quad_layout() -> mod_api::BindGroupLayout {
    mod_api::BindGroupLayout::builder()
        .uniform_buffer("QuadParams", 160)
        .entry(storage("QuadBuffer", true, 8))
        .entry(storage("SectionBuffer", true, 32))
        .entry(storage("ModelBuffer", true, 64))
        .entry(storage("BiomeTintBuffer", true, 4))
        .sampler("BlockAtlas", atlas_sampler())
        .build()
        .expect("static quad layout")
}

/// The pack-linked pipelines' set-1 layout (`voxy_pack_*.vsh`): `SceneUniform` is 176 bytes
/// ([`crate::shader_pack::PACK_SCENE_UNIFORM_SIZE`]) and the quad arena block is `QuadArena`.
pub fn pack_layout() -> mod_api::BindGroupLayout {
    mod_api::BindGroupLayout::builder()
        .uniform_buffer(
            "SceneUniform",
            crate::shader_pack::PACK_SCENE_UNIFORM_SIZE as u32,
        )
        .entry(storage("QuadArena", true, 8))
        .entry(storage("SectionBuffer", true, 32))
        .entry(storage("ModelBuffer", true, 64))
        .entry(storage("BiomeTintBuffer", true, 4))
        .sampler("BlockAtlas", atlas_sampler())
        .build()
        .expect("static pack layout")
}

/// Registers the render pipelines. Called from `Hooks::register_render_pipelines`. The compute
/// pipelines (Hi-Z, command generation) are added with the compute primitives.
pub fn register_pipelines(event: &mut mod_api::RegisterRenderPipelinesEvent<'_>) {
    use mod_api::{PipelineBlend, PrimitiveTopology, RenderPipeline, WorldDepth};
    // Opaque: depth-tested, culled by the index order of the quad mesh; translucent: depth-tested
    // and depth-writing (matching Voxy `MDICSectionRenderer.renderTranslucent` and vanilla
    // `RenderType.translucent`) so closer water surfaces reject farther/back water faces instead
    // of double-blending.
    event.register(
        RenderPipeline::new(
            id(PIPELINE_OPAQUE),
            id(SHADER_QUADS),
            PipelineBlend::Opaque,
            WorldDepth::Tested,
        )
        .depth_write(true)
        .topology(PrimitiveTopology::Triangles)
        .cull(false)
        .bind_group_layout(quad_layout())
        .pack_binding(crate::shader_pack::DhPackSlot::Terrain.pack_binding()),
    );
    event.register(
        RenderPipeline::new(
            id(PIPELINE_TRANSLUCENT),
            id(SHADER_QUADS),
            PipelineBlend::Translucent,
            WorldDepth::Tested,
        )
        .define("TRANSLUCENT")
        .depth_write(true)
        .topology(PrimitiveTopology::Triangles)
        .cull(false)
        .bind_group_layout(quad_layout()),
    );
    // Pack-linked pipelines: the mod's vertex stage (`voxy_pack_*.vsh`) feeds the pack's
    // `dh_terrain` / `dh_water` / `dh_shadow` fragment. Optional: without a pack that provides
    // the program they are unbound and the plain pipelines above draw.
    for (slot, name, blend) in [
        (
            crate::shader_pack::DhPackSlot::Terrain,
            PIPELINE_PACK_TERRAIN,
            PipelineBlend::Opaque,
        ),
        (
            crate::shader_pack::DhPackSlot::Water,
            PIPELINE_PACK_WATER,
            PipelineBlend::Translucent,
        ),
        (
            crate::shader_pack::DhPackSlot::Shadow,
            PIPELINE_PACK_SHADOW,
            PipelineBlend::Opaque,
        ),
    ] {
        event.register(
            RenderPipeline::new(
                id(name),
                id(slot.vertex_shader()),
                blend,
                WorldDepth::Tested,
            )
            .depth_write(true)
            .topology(PrimitiveTopology::Triangles)
            .cull(false)
            .bind_group_layout(pack_layout())
            .pack_binding(slot.pack_binding())
            .optional(),
        );
    }
    register_compute(event);
}

/// Hi-Z copy / reduce and command generation (`assets/distantsunrises/shaders/compute`).
fn register_compute(event: &mut mod_api::RegisterRenderPipelinesEvent<'_>) {
    use mod_api::{
        BindGroupLayout, ComputePipeline, Sampler, SamplerAddress, SamplerFilter,
        StorageImageAccess,
    };
    let nearest = || Sampler::new(SamplerFilter::Nearest, SamplerAddress::ClampToEdge);
    let hiz_layout = || {
        BindGroupLayout::builder()
            .uniform_buffer("HizParams", 16)
            .sampler("HizSource", nearest())
            .entry(mod_api::BindEntry::new(
                "HizDest",
                mod_api::BindingKind::StorageImage {
                    access: StorageImageAccess::WriteOnly,
                },
            ))
            .build()
            .expect("static hiz layout")
    };
    event.compute_pipeline(
        ComputePipeline::builder(id("distantsunrises:voxy_hiz_copy"), id(SHADER_HIZ))
            .bind_group_layout(hiz_layout())
            .define("HIZ_FROM_WORLD_DEPTH")
            .build(),
    );
    event.compute_pipeline(
        ComputePipeline::builder(id("distantsunrises:voxy_hiz_reduce"), id(SHADER_HIZ))
            .bind_group_layout(hiz_layout())
            .build(),
    );
    event.compute_pipeline(
        ComputePipeline::builder(id("distantsunrises:voxy_trav_init"), id(SHADER_TRAV_INIT))
            .bind_group_layout(
                BindGroupLayout::builder()
                    .uniform_buffer("TravInitParams", 16)
                    .entry(storage("RootBuffer", true, 4))
                    .entry(storage("VisibilityBuffer", false, 4))
                    .entry(storage("QueueA", false, 20))
                    .entry(storage("RequestBuffer", false, 24))
                    .entry(storage("DispatchA", false, 16))
                    .build()
                    .expect("static trav_init layout"),
            )
            .build(),
    );
    event.compute_pipeline(
        ComputePipeline::builder(id("distantsunrises:voxy_trav_clear"), id(SHADER_TRAV_CLEAR))
            .bind_group_layout(
                BindGroupLayout::builder()
                    .entry(storage("QueueOut", false, 20))
                    .entry(storage("DispatchOut", false, 16))
                    .build()
                    .expect("static trav_clear layout"),
            )
            .build(),
    );
    event.compute_pipeline(
        ComputePipeline::builder(id("distantsunrises:voxy_traversal"), id(SHADER_TRAVERSAL))
            .bind_group_layout(
                BindGroupLayout::builder()
                    .uniform_buffer("TraversalParams", 128)
                    .entry(storage("NodeBuffer", true, 32))
                    .entry(storage("QueueIn", true, 20))
                    .entry(storage("QueueOut", false, 20))
                    .entry(storage("DispatchOut", false, 16))
                    .entry(storage("VisibilityBuffer", false, 4))
                    .entry(storage("RequestBuffer", false, 24))
                    .sampler("HizPyramid", nearest())
                    .build()
                    .expect("static traversal layout"),
            )
            .build(),
    );
    event.compute_pipeline(
        ComputePipeline::builder(
            id("distantsunrises:voxy_trans_clear"),
            id(SHADER_TRANS_CLEAR),
        )
        .bind_group_layout(
            BindGroupLayout::builder()
                .entry(storage("TransSortMeta", false, 2056))
                .build()
                .expect("static trans_clear layout"),
        )
        .build(),
    );
    event.compute_pipeline(
        ComputePipeline::builder(id("distantsunrises:voxy_cmdgen"), id(SHADER_CMDGEN))
            .bind_group_layout(
                BindGroupLayout::builder()
                    .uniform_buffer("CmdgenParams", 112)
                    .entry(storage("SectionBuffer", true, 32))
                    .entry(storage("VisibilityBuffer", true, 4))
                    .entry(storage("OpaqueCommands", false, 20))
                    .entry(storage("TranslucentCommands", false, 20))
                    .entry(storage("TransSortMeta", false, 2056))
                    .entry(storage("TransSortEntries", false, 8))
                    .sampler("HizPyramid", nearest())
                    .build()
                    .expect("static cmdgen layout"),
            )
            .build(),
    );
    event.compute_pipeline(
        ComputePipeline::builder(
            id("distantsunrises:voxy_trans_prefix"),
            id(SHADER_TRANS_PREFIX),
        )
        .bind_group_layout(
            BindGroupLayout::builder()
                .entry(storage("TransSortMeta", false, 2056))
                .build()
                .expect("static trans_prefix layout"),
        )
        .build(),
    );
    event.compute_pipeline(
        ComputePipeline::builder(
            id("distantsunrises:voxy_trans_scatter"),
            id(SHADER_TRANS_SCATTER),
        )
        .bind_group_layout(
            BindGroupLayout::builder()
                .entry(storage("SectionBuffer", true, 32))
                .entry(storage("TranslucentCommands", false, 20))
                .entry(storage("TransSortMeta", false, 2056))
                .entry(storage("TransSortEntries", true, 8))
                .build()
                .expect("static trans_scatter layout"),
        )
        .build(),
    );
    event.compute_pipeline(
        ComputePipeline::builder(id("distantsunrises:voxy_ssao"), id(SHADER_SSAO))
            .bind_group_layout(
                BindGroupLayout::builder()
                    .uniform_buffer("SsaoParams", 48)
                    .sampler("WorldDepthTex", nearest())
                    .entry(storage("SsaoStamp", false, STAMP_BYTES as u32))
                    .entry(mod_api::BindEntry::new(
                        "SsaoOut",
                        mod_api::BindingKind::StorageImage {
                            access: StorageImageAccess::WriteOnly,
                        },
                    ))
                    .build()
                    .expect("static ssao layout"),
            )
            .build(),
    );
    event.register(
        mod_api::RenderPipeline::new(
            id(PIPELINE_SSAO_APPLY),
            id(SHADER_SSAO_APPLY),
            mod_api::PipelineBlend::Translucent,
            mod_api::WorldDepth::ThroughWalls,
        )
        .topology(mod_api::PrimitiveTopology::Triangles)
        .cull(false)
        .bind_group_layout(
            BindGroupLayout::builder()
                .uniform_buffer("SsaoApplyParams", 16)
                .entry(storage("SsaoStamp", true, STAMP_BYTES as u32))
                .sampler("SsaoTex", nearest())
                .build()
                .expect("static ssao_apply layout"),
        ),
    );
}

/// Writes `v` as little-endian f32s / u32s / i32s into a std140 byte buffer.
struct Std140(Vec<u8>);

impl Std140 {
    fn f32s(mut self, v: &[f32]) -> Self {
        v.iter()
            .for_each(|x| self.0.extend_from_slice(&x.to_le_bytes()));
        self
    }
    fn u32s(mut self, v: &[u32]) -> Self {
        v.iter()
            .for_each(|x| self.0.extend_from_slice(&x.to_le_bytes()));
        self
    }
    fn i32s(mut self, v: &[i32]) -> Self {
        v.iter()
            .for_each(|x| self.0.extend_from_slice(&x.to_le_bytes()));
        self
    }
}

/// `QuadParams` (160 bytes), shared by `voxy_quads.vsh` / `.fsh`.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the 160-byte QuadParams block field by field"
)]
pub fn quad_params_bytes(
    view_proj: [f32; 16],
    camera: [f64; 3],
    near_dh: f32,
    far_dh: f32,
    clip_dist: f32,
    sun: f32,
    fog: [f32; 4],
    fog_color: [f32; 4],
    is_hdr: bool,
    shading: bool,
) -> Vec<u8> {
    let (base, sub) = camera_split(camera);
    Std140(Vec::with_capacity(160))
        .f32s(&view_proj)
        .i32s(&[base[0], base[1], base[2], 0])
        .f32s(&[sub[0], sub[1], sub[2], 0.0])
        .f32s(&[near_dh, far_dh, clip_dist, sun])
        .f32s(&fog)
        .f32s(&fog_color)
        .u32s(&[u32::from(is_hdr), u32::from(shading), 0, 0])
        .0
}

/// `CmdgenParams` (112 bytes). `compact_opaque` (`camera_sub_pos.w`) makes cmdgen compact the
/// opaque commands behind the GPU count in `TransSortMeta` instead of writing fixed slots.
pub fn cmdgen_params_bytes(
    view_proj: [f32; 16],
    camera: [f64; 3],
    section_count: u32,
    hiz_size: [u32; 2],
    hiz_mips: u32,
    hiz_enabled: bool,
    compact_opaque: bool,
) -> Vec<u8> {
    let (base, sub) = camera_split(camera);
    Std140(Vec::with_capacity(112))
        .f32s(&view_proj)
        .i32s(&[base[0], base[1], base[2], section_count as i32])
        .f32s(&[sub[0], sub[1], sub[2], f32::from(u8::from(compact_opaque))])
        .i32s(&[
            hiz_size[0] as i32,
            hiz_size[1] as i32,
            hiz_mips as i32,
            i32::from(hiz_enabled),
        ])
        .0
}

/// Camera block position split into its 32-block section (`>> 5`) and the sub-section remainder,
/// so shader math stays in f32 near the camera (Voxy `baseSectionPos` / `cameraSubPos`).
pub fn camera_split(camera: [f64; 3]) -> ([i32; 3], [f32; 3]) {
    let base = camera.map(|c| (c.floor() as i64 >> 5) as i32);
    let sub = std::array::from_fn(|i| (camera[i] - f64::from(base[i] << 5)) as f32);
    (base, sub)
}

/// Mip count of the Hi-Z pyramid: `32 - leading_zeros(max(w, h))` (sunrise#5590 rule).
pub fn hiz_mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).leading_zeros()
}

/// The mod-owned Hi-Z pyramid (sunrise#5590): an `R32Float` target with a full mip chain,
/// created for `WriteOnly` storage images and sampled by cmdgen. Resized with the viewport.
struct HizTarget {
    target: mod_api::RenderTargetRef,
    size: [u32; 2],
    mips: u32,
}

static HIZ: Mutex<Option<HizTarget>> = Mutex::new(None);

impl HizTarget {
    /// Mip views: level `l` as a storage-image destination and as a sampled source.
    fn dest(&self, level: u32) -> mod_api::StorageImageBinding {
        mod_api::StorageImageBinding::new(self.target, 0, level)
    }
    fn source(&self, level: u32) -> mod_api::TextureSource<'static> {
        mod_api::TextureSource::RenderTargetColorMip(mod_api::RenderTargetMipView::single(
            self.target,
            0,
            level,
        ))
    }
    /// All mips, for cmdgen's `HizPyramid` sampler.
    fn all_mips(&self) -> mod_api::TextureSource<'static> {
        mod_api::TextureSource::RenderTargetColorMip(mod_api::RenderTargetMipView::new(
            self.target,
            0,
            0,
            self.mips,
        ))
    }
}

/// The Hi-Z pyramid for a `size` viewport, created or resized on demand. `None` when the
/// adapter refuses the target; the caller then skips Hi-Z culling (`hiz_enabled = false`).
fn ensure_hiz(host: &dyn mod_api::ClientHost, size: [u32; 2]) -> Option<HizTarget> {
    use mod_api::{RenderTargetDescriptor, RenderTargetFormat, StorageImageAccess};
    let mut slot = HIZ.lock().unwrap();
    if let Some(h) = slot.as_ref().filter(|h| h.size == size) {
        return Some(HizTarget { ..*h });
    }
    if let Some(old) = slot.take() {
        let _ = old.target.release(host);
    }
    let mips = hiz_mip_levels(size[0], size[1]);
    let desc = RenderTargetDescriptor::new(size)
        .color_target(RenderTargetFormat::R32Float)
        .mip_levels(mips)
        .storage(StorageImageAccess::WriteOnly);
    let target = host.create_render_target(&desc).ok()?;
    let made = HizTarget { target, size, mips };
    *slot = Some(HizTarget { ..made });
    Some(made)
}

/// Bytes of a publication stamp: one serial as its low and high `u32` words.
const STAMP_BYTES: u64 = 8;

/// The next GPU publication serial. It starts at 1 (0 is a fresh or cleared stamp) and counts
/// every submission that records a producer, as a `u64` that is never reused.
static PUBLICATION_SERIAL: AtomicU64 = AtomicU64::new(1);

fn next_serial() -> [u32; 2] {
    let serial = PUBLICATION_SERIAL.fetch_add(1, Ordering::Relaxed);
    [serial as u32, (serial >> 32) as u32]
}

/// A zeroed retained buffer holding one publication stamp. A compute dispatch binds it
/// read-write (`read_only: false`) and writes its serial; a draw's fragment stage binds it
/// `read_only: true` (the engine allows one buffer to be read-write in a compute node and
/// read-only in a draw pass, and refuses only both within one draw pass). The host never reads
/// it back: the consuming shader compares it with the serial it was drawn with.
fn create_stamp_buffer(host: &dyn mod_api::ClientHost) -> Option<mod_api::StorageBufferRef> {
    let buffer = report(
        "create stamp",
        host.create_storage_buffer(&mod_api::StorageBufferDescriptor::new(STAMP_BYTES)),
    )?;
    // Queued writes land in call order before the frame's first pass.
    if report(
        "clear stamp",
        buffer.write_range(host, 0, &[0; STAMP_BYTES as usize]),
    )
    .is_none()
    {
        report("release stamp", buffer.release(host));
        return None;
    }
    Some(buffer)
}

/// Mod-owned `R32Float` SSAO output target and the stamp its producer publishes, both
/// recreated with the viewport.
#[derive(Clone, Copy)]
struct SsaoTarget {
    target: mod_api::RenderTargetRef,
    size: [u32; 2],
    stamp: mod_api::StorageBufferRef,
}

static SSAO: Mutex<Option<SsaoTarget>> = Mutex::new(None);

/// What the SSAO compute node recorded by this submission's `setup_compute` publishes: its
/// target (the one the apply draw must sample) and the serial one `voxy_ssao` invocation
/// writes into that target's stamp. Admission to the frame graph is not execution: the graph
/// may still drop every node (a cycle after `prepare_draws`), and a dispatch may be refused.
/// So this only says which serial to expect; whether it was published is decided on the GPU by
/// `voxy_ssao_apply.fsh`. `prepare_draws` takes it, so a serial is drawn at most once.
static SSAO_PENDING: Mutex<Option<(SsaoTarget, [u32; 2])>> = Mutex::new(None);

impl SsaoTarget {
    fn dest(&self) -> mod_api::StorageImageBinding {
        mod_api::StorageImageBinding::new(self.target, 0, 0)
    }
    fn source(&self) -> mod_api::TextureSource<'static> {
        mod_api::TextureSource::RenderTargetColorMip(mod_api::RenderTargetMipView::single(
            self.target,
            0,
            0,
        ))
    }
}

/// The SSAO target for a `size` viewport, created or recreated on demand together with its
/// stamp (a resize drops the old stamp, so nothing published for the old target can be read
/// for the new one). Called only from `setup_compute`, where a resize replaces the handles
/// before this submission records anything that binds them.
fn ensure_ssao(host: &dyn mod_api::ClientHost, size: [u32; 2]) -> Option<SsaoTarget> {
    use mod_api::{RenderTargetDescriptor, RenderTargetFormat, StorageImageAccess};
    let mut slot = SSAO.lock().unwrap();
    if let Some(s) = slot.filter(|s| s.size == size) {
        return Some(s);
    }
    if let Some(old) = slot.take() {
        let _ = old.target.release(host);
        let _ = old.stamp.release(host);
    }
    let desc = RenderTargetDescriptor::new(size)
        .color_target(RenderTargetFormat::R32Float)
        .mip_levels(1)
        .storage(StorageImageAccess::WriteOnly);
    let target = host.create_render_target(&desc).ok()?;
    let Some(stamp) = create_stamp_buffer(host) else {
        let _ = target.release(host);
        return None;
    };
    let made = SsaoTarget {
        target,
        size,
        stamp,
    };
    *slot = Some(made);
    Some(made)
}

/// Consistent circular vanilla clip distance shared between opaque LODs and compute SSAO.
fn vanilla_clip_dist(frame: &RenderFrameParams) -> f32 {
    if frame.config.overdraw_prevention > 0.0 {
        (frame.vanilla_far_blocks * frame.config.overdraw_prevention - 2.0).max(0.0)
    } else {
        0.0
    }
}

/// Sections built off-thread and waiting for a `&mut ClientHost` to reach the arena.
static PENDING_SECTIONS: Mutex<Vec<(SectionKey, BuiltSection)>> = Mutex::new(Vec::new());
static ARENA: Mutex<Option<ArenaState<GpuArena, mod_api::MeshRef>>> = Mutex::new(None);

/// Identity of an uploaded section: `(dimension, lod level, section position)`.
pub type SectionKey = (String, u8, [i32; 3]);

struct ArenaState<Arena, Mesh> {
    arena: Arena,
    /// Shared quad index pattern (`0,1,2,2,1,3` per quad) the indirect commands index into.
    quad_mesh: Option<Mesh>,
    ids: HashMap<SectionKey, u32>,
    node_cap: NodeCapGate,
}

/// Whether the cap was logged in this play session, independent of GPU arena lifetime.
static NODE_CAP_LOGGED: AtomicBool = AtomicBool::new(false);

/// Sections the arena refused for lack of octree nodes. A refused key is not tried again
/// until a section is removed and nodes may be free, or the arena is recreated.
#[derive(Default)]
struct NodeCapGate {
    refused: HashSet<SectionKey>,
}

impl NodeCapGate {
    fn is_suppressed(&self, key: &SectionKey) -> bool {
        self.refused.contains(key)
    }

    /// Suppresses `key`; returns the log line the first time in a session.
    fn refuse(
        &mut self,
        key: SectionKey,
        (needed_nodes, max_nodes, needed_roots, max_roots): (usize, usize, usize, usize),
        logged: &AtomicBool,
    ) -> Option<String> {
        self.refused.insert(key);
        (!logged.swap(true, Ordering::AcqRel)).then(|| {
            format!(
                "distantsunrises: GPU octree node cap reached ({needed_nodes}/{max_nodes} nodes, {needed_roots}/{max_roots} roots); skipping additional sections until sections are removed"
            )
        })
    }

    /// A section was removed, so refused keys may fit again.
    fn lift(&mut self) {
        self.refused.clear();
    }
}

/// GPU operations used by the terrain lifetime decisions. Draws and uploads use the concrete arena.
trait TerrainGpu {
    type Arena;
    type Mesh;
    type Ticket: Copy;

    fn create_arena(&mut self, config: ArenaConfig) -> Result<Self::Arena, ArenaError>;
    fn create_quad_mesh(&mut self) -> Result<Self::Mesh, ArenaError>;
    fn release_quad_mesh(&self, mesh: Self::Mesh) -> Result<(), mod_api::MeshError>;
    fn request_buffer_size(&self, arena: &Self::Arena) -> Result<u64, mod_api::BufferError>;
    fn poll_readback(&self, ticket: Self::Ticket) -> Result<Option<Vec<u8>>, mod_api::BufferError>;
    fn release_readback(&self, ticket: Self::Ticket) -> Result<(), mod_api::BufferError>;
}

/// Model words, biome tint words and the baked face atlas mip chain (RGBA8; level `l` is
/// `ATLAS_SIDE >> l` texels wide).
type Tables = (Vec<u32>, Vec<u32>, Vec<Vec<u8>>);
static PENDING_TABLES: Mutex<Option<Tables>> = Mutex::new(None);
static ATLAS_CREATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Id of the dynamic texture holding the baked per-face texels.
pub const ATLAS_TEXTURE: &str = "distantsunrises:voxy_atlas";

/// Queues the baked model words and biome tint words for the next `upload_pending` (they replace
/// the previous tables, which a resource reload or new block id makes stale). `atlas_mips` holds
/// all `ATLAS_MIP_LEVELS` levels of the face atlas, finest first.
pub fn queue_model_tables(
    model_words: Vec<u32>,
    biome_tint_words: Vec<u32>,
    atlas_mips: Vec<Vec<u8>>,
) {
    *PENDING_TABLES.lock().unwrap() = Some((model_words, biome_tint_words, atlas_mips));
}

/// Most GPU node requests kept between two [`drain_gpu_node_requests`] calls.
const MAX_GPU_NODE_REQUESTS: usize = 512;

struct PendingReadback<Ticket> {
    ticket: Ticket,
    retired: bool,
}

struct NodeRequests<Ticket> {
    // One ticket, even when the world has changed and its bytes are no longer wanted.
    pending: Option<PendingReadback<Ticket>>,
    ready: Vec<(u8, [i32; 3])>,
}

static NODE_REQUESTS: Mutex<NodeRequests<mod_api::ReadbackTicket>> =
    Mutex::new(NodeRequests::new());

impl<Ticket: Copy> NodeRequests<Ticket> {
    const fn new() -> Self {
        Self {
            pending: None,
            ready: Vec::new(),
        }
    }

    fn start(&mut self, request: impl FnOnce() -> Option<Ticket>) {
        if self.pending.is_none() {
            self.pending = request().map(|ticket| PendingReadback {
                ticket,
                retired: false,
            });
        }
    }

    fn retire(&mut self) {
        self.ready.clear();
        if let Some(pending) = &mut self.pending {
            pending.retired = true;
        }
    }

    fn service<G: TerrainGpu<Ticket = Ticket> + ?Sized>(&mut self, gpu: &G, arena: &G::Arena) {
        // In #6138 the host can exist without mod GPU resources. A ticket operation would
        // return the same Unavailable as a spent ticket; probe the retained request buffer.
        if self.pending.is_none() || gpu.request_buffer_size(arena).is_err() {
            return;
        }
        let pending = self.pending.take().unwrap();
        if pending.retired {
            report("release_readback", gpu.release_readback(pending.ticket));
        } else {
            match report("poll_readback", gpu.poll_readback(pending.ticket)) {
                Some(Some(bytes)) => {
                    self.ready
                        .extend(crate::gpu_arena::decode_gpu_requests(&bytes));
                    self.ready.truncate(MAX_GPU_NODE_REQUESTS);
                }
                Some(None) => self.pending = Some(pending),
                None => {} // With the renderer present, a lost/failed copy is already retired.
            }
        }
    }
}

/// Takes the octree sections the GPU traversal asked for (a visible node wanting finer detail
/// whose child sections are not in the arena), for the pipeline to load ahead of its blind sweep.
pub fn drain_gpu_node_requests() -> Vec<(u8, [i32; 3])> {
    std::mem::take(&mut NODE_REQUESTS.lock().unwrap().ready)
}

pub(crate) fn poll_node_requests(host: &dyn mod_api::ClientHost) {
    let guard = ARENA.lock().unwrap();
    let Some(state) = guard.as_ref() else { return };
    NODE_REQUESTS.lock().unwrap().service(host, &state.arena);
}

/// Called by the pipeline's mesher for each finished section (empty = remove).
pub fn submit_built_section(key: SectionKey, section: BuiltSection) {
    PENDING_SECTIONS.lock().unwrap().push((key, section));
}

/// Most quads one indirect command can span (a 32³ section's worth of faces in one bucket).
const MAX_QUADS_PER_COMMAND: u32 = 1 << 16;

impl TerrainGpu for dyn mod_api::ClientHost + '_ {
    type Arena = GpuArena;
    type Mesh = mod_api::MeshRef;
    type Ticket = mod_api::ReadbackTicket;

    fn create_arena(&mut self, config: ArenaConfig) -> Result<Self::Arena, ArenaError> {
        GpuArena::new(self, config)
    }

    /// A mesh holding only the shared quad index buffer; its one vertex binding is never read
    /// (the vertex stages pull quads from the arena by `gl_VertexID`). It has the stride of the
    /// default `position_tex_color` binding, which a pipeline with no declared binding uses.
    fn create_quad_mesh(&mut self) -> Result<Self::Mesh, ArenaError> {
        use mod_api::{IndexFormat, MeshBufferInit, MeshDescriptor, MeshIndexInit};
        let mut bytes = Vec::with_capacity(MAX_QUADS_PER_COMMAND as usize * 24);
        for q in 0..MAX_QUADS_PER_COMMAND {
            for i in [0u32, 1, 2, 2, 1, 3] {
                bytes.extend_from_slice(&(q * 4 + i).to_le_bytes());
            }
        }
        let dummy = [0u8; 24];
        let vertex = [MeshBufferInit::from_bytes(24, &dummy)];
        let desc = MeshDescriptor::new(&vertex)
            .with_indices(MeshIndexInit::from_bytes(IndexFormat::Uint32, &bytes));
        self.create_mesh(&desc)
            .map_err(|_| ArenaError::Buffer(mod_api::BufferError::Unavailable))
    }

    fn release_quad_mesh(&self, mesh: Self::Mesh) -> Result<(), mod_api::MeshError> {
        mesh.release(self)
    }

    fn request_buffer_size(&self, arena: &Self::Arena) -> Result<u64, mod_api::BufferError> {
        arena.requests.size(self)
    }

    fn poll_readback(&self, ticket: Self::Ticket) -> Result<Option<Vec<u8>>, mod_api::BufferError> {
        ticket.poll_readback(self)
    }

    fn release_readback(&self, ticket: Self::Ticket) -> Result<(), mod_api::BufferError> {
        ticket.release(self)
    }
}

/// Creates the arena on first use (ruling 2: it then survives reloads and world close),
/// uploads pending sections and flushes metadata. Call from `client_tick_pre`. Errors are logged
/// through [`report`]; the frame goes on without them. A retained arena whose buffers are
/// unavailable is skipped quietly before any state or upload is touched.
pub fn upload_pending(host: &mut dyn mod_api::ClientHost, config: ArenaConfig) {
    report("arena upload", try_upload_pending(host, config));
}

fn ensure_arena<'a, G: TerrainGpu + ?Sized>(
    slot: &'a mut Option<ArenaState<G::Arena, G::Mesh>>,
    gpu: &mut G,
    config: ArenaConfig,
) -> Result<&'a mut ArenaState<G::Arena, G::Mesh>, ArenaError> {
    if slot.is_none() {
        let quad_mesh = gpu.create_quad_mesh()?;
        let arena = match gpu.create_arena(config) {
            Ok(arena) => arena,
            Err(error) => {
                // The arena builder has rolled back its buffers, including all size fallbacks.
                report("release unused quad mesh", gpu.release_quad_mesh(quad_mesh));
                return Err(error);
            }
        };
        *slot = Some(ArenaState {
            arena,
            quad_mesh: Some(quad_mesh),
            ids: HashMap::new(),
            node_cap: NodeCapGate::default(),
        });
    }
    let state = slot.as_mut().unwrap();
    if state.quad_mesh.is_none() {
        state.quad_mesh = Some(gpu.create_quad_mesh()?);
    }
    Ok(state)
}

fn try_upload_pending(
    host: &mut dyn mod_api::ClientHost,
    config: ArenaConfig,
) -> Result<(), crate::gpu_arena::ArenaError> {
    let mut guard = ARENA.lock().unwrap();
    let retained = ArenaState::discard_if_retired(&mut guard, host)?;
    if retained == Retained::Unavailable {
        // The host exists without mod GPU resources (#6138); nothing can be uploaded or mutated.
        return Ok(());
    }
    if retained == Retained::Retired {
        // These storage, target, mesh and ticket handles were retired together by Sunrise.
        // Drop their mod-side state without reading or releasing the dead handles.
        PENDING_SECTIONS.lock().unwrap().clear();
        *PENDING_TABLES.lock().unwrap() = None;
        *NODE_REQUESTS.lock().unwrap() = NodeRequests::new();
        *FRAME.lock().unwrap() = None;
        PREPARED.lock().unwrap().clear();
        *HIZ.lock().unwrap() = None;
        *SSAO.lock().unwrap() = None;
        *SSAO_PENDING.lock().unwrap() = None;
        // No arena is left to have prepared these counts; the replacement records its own.
        clear_arena_stats();
        // The named texture is retired separately at the frame boundary. If it is still
        // registered, retire our atlas before rebaking so its name can be created again.
        if ATLAS_CREATED.load(Ordering::Relaxed) {
            match host.release_dynamic_texture(&id(ATLAS_TEXTURE)) {
                Ok(()) | Err(mod_api::DynamicTextureError::Unknown) => {
                    ATLAS_CREATED.store(false, Ordering::Relaxed);
                }
                Err(error) => {
                    report::<(), _>("release retired atlas", Err(error));
                }
            }
        }
        // Reset/requeue old mesh generations with PIPELINE unlocked from ARENA; server-side
        // cleanup acquires those locks in the opposite direction.
        drop(guard);
        crate::pipeline::recover_retired_arena();
        guard = ARENA.lock().unwrap();
    }
    let state = ensure_arena(&mut guard, host, config)?;
    let pending_tables = PENDING_TABLES.lock().unwrap().take();
    if let Some((models, tints, atlas_mips)) = pending_tables {
        let wrote_models = report("write models", state.arena.write_models(host, 0, &models));
        let wrote_tints = report(
            "write biome tints",
            state.arena.write_biome_tints(host, 0, &tints),
        );
        let id = mod_api::Identifier::parse(ATLAS_TEXTURE).expect("static texture id");
        let side = crate::pipeline::ATLAS_SIDE as u32;
        // Level 0 creates (with the whole mip chain allocated) or replaces the texture; levels
        // `1..ATLAS_MIP_LEVELS` are then written as regions.
        let level0 = if ATLAS_CREATED.load(std::sync::atomic::Ordering::Relaxed) {
            host.update_dynamic_texture(&id, side, side, &atlas_mips[0])
        } else {
            host.create_dynamic_texture(
                &id,
                side,
                side,
                crate::pipeline::ATLAS_MIP_LEVELS,
                &atlas_mips[0],
            )
        };
        if level0.is_ok() {
            ATLAS_CREATED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let wrote_atlas = report(
            "upload atlas",
            level0.and_then(|()| {
                (1..crate::pipeline::ATLAS_MIP_LEVELS).try_for_each(|level| {
                    let size = side >> level;
                    host.update_dynamic_texture_region(
                        &id,
                        level,
                        [0, 0],
                        [size, size],
                        &atlas_mips[level as usize],
                    )
                })
            }),
        );
        // A failed table or atlas write is retried next frame unless newer tables were queued
        // meanwhile.
        if wrote_models.is_none() || wrote_tints.is_none() || wrote_atlas.is_none() {
            let mut slot = PENDING_TABLES.lock().unwrap();
            if slot.is_none() {
                *slot = Some((models, tints, atlas_mips));
            }
        }
    }
    // One section failing (arena full, bad write) never drops the rest of the batch: each is
    // handled on its own, logged through `report`, and the metadata flush still runs.
    let mut removed = false;
    for (key, section) in std::mem::take(&mut *PENDING_SECTIONS.lock().unwrap()) {
        let old = state.ids.get(&key).copied();
        if section.is_empty() {
            state.node_cap.refused.remove(&key);
            if let Some(old) = old
                && report("remove section", state.arena.remove(old)).is_some()
            {
                state.ids.remove(&key);
                state.node_cap.lift();
                removed = true;
            }
            continue;
        }
        if state.node_cap.is_suppressed(&key) {
            continue;
        }
        // `upload_replace` keeps the old section live when the new one cannot be stored.
        match state.arena.upload_replace(host, old, &section) {
            Err(ArenaError::NodesFull {
                needed_nodes,
                max_nodes,
                needed_roots,
                max_roots,
            }) => {
                let counts = (needed_nodes, max_nodes, needed_roots, max_roots);
                if let Some(line) = state.node_cap.refuse(key, counts, &NODE_CAP_LOGGED) {
                    eprintln!("{line}");
                }
            }
            result => {
                if let Some(new) = report("upload section", result) {
                    state.ids.insert(key, new);
                }
            }
        }
    }
    // An emptied arena prepares no draw, so nothing would overwrite the last prepared counts.
    if removed && state.arena.section_high_water() == 0 {
        clear_arena_stats();
    }
    state.arena.flush(host)
}

/// Retire terrain content, not GPU allocations or the baked asset tables.
pub(crate) fn clear_sections(dimension: Option<&str>) {
    let mut guard = ARENA.lock().unwrap();
    let keep = |key: &SectionKey| dimension.is_some_and(|dim| key.0 != dim);
    PENDING_SECTIONS
        .lock()
        .unwrap()
        .retain(|(key, _)| keep(key));
    NODE_REQUESTS.lock().unwrap().retire();
    PREPARED.lock().unwrap().clear();
    if let Some(state) = guard.as_mut() {
        state.ids.retain(|key, id| {
            if keep(key) {
                return true;
            }
            if report("remove section", state.arena.remove(*id)).is_some() {
                state.node_cap.lift();
            }
            false
        });
        state.node_cap.refused.retain(|key| keep(key));
        if state.arena.section_high_water() == 0 {
            clear_arena_stats();
        }
    }
    if dimension.is_none() {
        NODE_CAP_LOGGED.store(false, Ordering::Release);
    }
}

/// Sunrise retires mesh handles on world close and asset reload. The terrain arena survives.
pub(crate) fn invalidate_quad_mesh() {
    ArenaState::invalidate_quad_mesh(&mut ARENA.lock().unwrap());
}

/// What the retained-buffer probe found out about the arena slot.
#[derive(Debug, PartialEq, Eq)]
enum Retained {
    /// No arena yet, or its request buffer answered.
    Live,
    /// Typed staleness: the arena was dropped from the slot.
    Retired,
    /// The host has no mod GPU resources right now; the arena is left untouched.
    Unavailable,
}

impl<Arena, Mesh> ArenaState<Arena, Mesh> {
    fn discard_if_retired(
        slot: &mut Option<Self>,
        gpu: &(impl TerrainGpu<Arena = Arena> + ?Sized),
    ) -> Result<Retained, ArenaError> {
        let Some(state) = slot else {
            return Ok(Retained::Live);
        };
        // Unlike a generic unavailable context, Stale proves this retained generation is gone.
        // Storage survives ordinary reload, resize and world close on the renderer contract.
        match gpu.request_buffer_size(&state.arena) {
            Ok(_) => Ok(Retained::Live),
            Err(mod_api::BufferError::Stale) => {
                *slot = None;
                Ok(Retained::Retired)
            }
            Err(mod_api::BufferError::Unavailable) => Ok(Retained::Unavailable),
            Err(error) => Err(error.into()),
        }
    }

    fn invalidate_quad_mesh(slot: &mut Option<Self>) {
        if let Some(state) = slot {
            state.quad_mesh = None;
        }
    }
}

static FRAME_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The frame's parameters, set by [`begin_frame`] in `prepare_render_buffers`.
static FRAME: Mutex<Option<RenderFrameParams>> = Mutex::new(None);

/// Draws prepared in `prepare_render_buffers`, issued in their stage by [`draw_stage`].
static PREPARED: Mutex<Vec<(mod_api::RenderLevelStage, mod_api::PreparedWorldDraw)>> =
    Mutex::new(Vec::new());

/// Stage the opaque LOD draw runs in; the Hi-Z/cmdgen compute node runs just before it, after
/// the vanilla opaque blocks are in the depth buffer.
pub const OPAQUE_STAGE: mod_api::RenderLevelStage = mod_api::RenderLevelStage::AfterOpaqueFeatures;
/// Stage of the translucent LOD draw.
pub const TRANSLUCENT_STAGE: mod_api::RenderLevelStage =
    mod_api::RenderLevelStage::AfterTranslucentBlocks;

/// Starts the frame: drains workers, stores the frame parameters and drops last frame's prepared
/// draws. Called first from `Hooks::prepare_render_buffers`.
pub fn begin_frame(camera: [f64; 3]) {
    FRAME_ID.fetch_add(1, Ordering::Relaxed);
    let frame = begin_render_frame(camera);
    *FRAME.lock().unwrap() = Some(frame);
    PREPARED.lock().unwrap().clear();
}

/// Registers the Hi-Z + cmdgen compute node of this frame in the frame graph. Called from
/// `Hooks::frame_graph_setup`.
pub fn setup_compute(
    host: &dyn mod_api::ClientHost,
    camera: &CameraMatrices,
    graph: &mut dyn mod_api::FrameGraphContext,
) {
    use mod_api::{
        ComputePassDescriptor, DrawBindings, FrameTarget, RenderLevelStage, TextureSource,
    };
    // Every early return below leaves no SSAO publication to expect: `prepare_draws` of this
    // submission then draws no `ssao_apply`.
    *SSAO_PENDING.lock().unwrap() = None;
    let Some(frame) = FRAME.lock().unwrap().clone() else {
        return;
    };
    let guard = ARENA.lock().unwrap();
    let Some(state) = guard.as_ref() else { return };
    let arena = &state.arena;
    let sections = arena.section_high_water();
    if !frame.enabled || sections == 0 {
        return;
    }
    let Some([w, h]) = host.window_size() else {
        return;
    };
    let assets = host.render_assets();
    let ids = [
        "voxy_hiz_copy",
        "voxy_hiz_reduce",
        "voxy_trav_init",
        "voxy_trav_clear",
        "voxy_traversal",
        "voxy_trans_clear",
        "voxy_cmdgen",
        "voxy_trans_prefix",
        "voxy_trans_scatter",
    ]
    .map(|n| assets.compute_pipeline(&id(&format!("distantsunrises:{n}"))));
    let [
        Ok(copy),
        Ok(reduce),
        Ok(trav_init),
        Ok(trav_clear),
        Ok(traversal),
        Ok(trans_clear),
        Ok(cmdgen),
        Ok(trans_prefix),
        Ok(trans_scatter),
    ] = ids
    else {
        return;
    };
    let root_count = arena.root_count();
    let dispatch_a_ref = arena.dispatch_a;
    let dispatch_b_ref = arena.dispatch_b;
    let requests_ref = arena.requests;
    let focal_px = (h as f32 * 0.5 * camera.projection[5].abs()).max(1.0);
    let lod_threshold = 2.0f32;
    let Some(hiz) = ensure_hiz(host, [w, h]) else {
        return;
    };
    let mips = hiz.mips;
    let view_proj = camera.view_projection();
    let use_count_buf = host.render_capabilities().multi_draw_indirect_count;
    let desc = ComputePassDescriptor::new("distantsunrises:hiz_cmdgen")
        .reads(FrameTarget::MainDepth)
        .writes(FrameTarget::Mod(hiz.target))
        .after_stage(RenderLevelStage::AfterOpaqueBlocks)
        .before_stage(OPAQUE_STAGE);
    let slices = arena.slices;
    let added = graph.add_compute_pass(desc, &mut |ctx| {
        let groups = |n: u32| n.div_ceil(HIZ_WORKGROUP);
        let (mut sw, mut sh) = (w, h);
        for level in 0..mips {
            let (dw, dh) = if level == 0 {
                (w, h)
            } else {
                ((sw / 2).max(1), (sh / 2).max(1))
            };
            let params = Std140(Vec::new())
                .i32s(&[sw as i32, sh as i32, dw as i32, dh as i32])
                .0;
            let source = if level == 0 {
                TextureSource::WorldDepth
            } else {
                hiz.source(level - 1)
            };
            let bindings = DrawBindings::new()
                .inline_uniform("HizParams", &params)
                .texture("HizSource", source)
                .storage_image("HizDest", hiz.dest(level));
            report(
                "hiz dispatch",
                ctx.dispatch(
                    if level == 0 { copy } else { reduce },
                    &bindings,
                    [groups(dw), groups(dh), 1],
                ),
            );
            (sw, sh) = (dw, dh);
        }
        let init_threads = sections.max(root_count).max(1);
        let init_params = Std140(Vec::with_capacity(16))
            .u32s(&[
                sections,
                root_count,
                crate::gpu_arena::GPU_TRAV_QUEUE_CAP,
                0,
            ])
            .0;
        let init_bindings = DrawBindings::new()
            .inline_uniform("TravInitParams", &init_params)
            .storage_buffer("RootBuffer", slices.roots)
            .storage_buffer("VisibilityBuffer", slices.visibility)
            .storage_buffer("QueueA", slices.queue_a)
            .storage_buffer("RequestBuffer", slices.requests)
            .storage_buffer("DispatchA", slices.dispatch_a);
        report(
            "trav_init dispatch",
            ctx.dispatch(
                trav_init,
                &init_bindings,
                [init_threads.div_ceil(TRAV_WORKGROUP), 1, 1],
            ),
        );
        let (base, sub) = camera_split(camera.position);
        let trav_params = Std140(Vec::with_capacity(128))
            .f32s(&view_proj)
            .i32s(&[
                base[0],
                base[1],
                base[2],
                crate::gpu_arena::GPU_TRAV_QUEUE_CAP as i32,
            ])
            .f32s(&[sub[0], sub[1], sub[2], focal_px])
            .i32s(&[w as i32, h as i32, mips as i32, 1])
            .f32s(&[
                lod_threshold,
                crate::gpu_arena::GPU_MAX_REQUESTS as f32,
                0.0,
                0.0,
            ])
            .0;
        for iter in 0..=crate::octree::MAX_OCTREE_DETAIL {
            let (q_in, q_out, d_in_ref, d_out) = if iter % 2 == 0 {
                (
                    slices.queue_a,
                    slices.queue_b,
                    dispatch_a_ref,
                    slices.dispatch_b,
                )
            } else {
                (
                    slices.queue_b,
                    slices.queue_a,
                    dispatch_b_ref,
                    slices.dispatch_a,
                )
            };
            let tclear_bindings = DrawBindings::new()
                .storage_buffer("QueueOut", q_out)
                .storage_buffer("DispatchOut", d_out);
            report(
                "trav_clear dispatch",
                ctx.dispatch(trav_clear, &tclear_bindings, [1, 1, 1]),
            );
            let trav_bindings = DrawBindings::new()
                .inline_uniform("TraversalParams", &trav_params)
                .storage_buffer("NodeBuffer", slices.nodes)
                .storage_buffer("QueueIn", q_in)
                .storage_buffer("QueueOut", q_out)
                .storage_buffer("DispatchOut", d_out)
                .storage_buffer("VisibilityBuffer", slices.visibility)
                .storage_buffer("RequestBuffer", slices.requests)
                .texture("HizPyramid", hiz.all_mips());
            report(
                "traversal dispatch_indirect",
                ctx.dispatch_indirect(traversal, &trav_bindings, d_in_ref, 0),
            );
        }
        // The client tick services the one readback slot without blocking.
        NODE_REQUESTS.lock().unwrap().start(|| {
            report(
                "request_readback",
                requests_ref.request_readback(host, 0, crate::gpu_arena::GPU_REQUEST_BUF_BYTES),
            )
        });
        let clear_bindings =
            DrawBindings::new().storage_buffer("TransSortMeta", slices.trans_sort_meta);
        report(
            "trans_clear dispatch",
            ctx.dispatch(trans_clear, &clear_bindings, [1, 1, 1]),
        );
        let params = cmdgen_params_bytes(
            view_proj,
            camera.position,
            sections,
            [w, h],
            mips,
            true,
            use_count_buf,
        );
        let bindings = DrawBindings::new()
            .inline_uniform("CmdgenParams", &params)
            .storage_buffer("SectionBuffer", slices.sections)
            .storage_buffer("VisibilityBuffer", slices.visibility)
            .storage_buffer("OpaqueCommands", slices.opaque_commands)
            .storage_buffer("TranslucentCommands", slices.translucent_commands)
            .storage_buffer("TransSortMeta", slices.trans_sort_meta)
            .storage_buffer("TransSortEntries", slices.trans_sort_entries)
            .texture("HizPyramid", hiz.all_mips());
        report(
            "cmdgen dispatch",
            ctx.dispatch(
                cmdgen,
                &bindings,
                [sections.div_ceil(CMDGEN_WORKGROUP), 1, 1],
            ),
        );
        let prefix_bindings =
            DrawBindings::new().storage_buffer("TransSortMeta", slices.trans_sort_meta);
        report(
            "trans_prefix dispatch",
            ctx.dispatch(trans_prefix, &prefix_bindings, [1, 1, 1]),
        );
        let scatter_bindings = DrawBindings::new()
            .storage_buffer("SectionBuffer", slices.sections)
            .storage_buffer("TranslucentCommands", slices.translucent_commands)
            .storage_buffer("TransSortMeta", slices.trans_sort_meta)
            .storage_buffer("TransSortEntries", slices.trans_sort_entries);
        report(
            "trans_scatter dispatch",
            ctx.dispatch(
                trans_scatter,
                &scatter_bindings,
                [sections.div_ceil(CMDGEN_WORKGROUP), 1, 1],
            ),
        );
    });
    report("add_compute_pass", added);

    // Compute SSAO pass on opaque LOD depth (runs after opaque LODs, before translucents).
    // `frame_graph_setup` runs before `prepare_render_buffers` (sunrise `gfx/draw_frame.rs`), so
    // the target is (re)created here and `prepare_draws` binds this exact target through
    // `SSAO_PENDING`; the handles of a resized-away target are never bound by this submission.
    if frame.config.ssao_enabled
        && let (Ok(ssao_pipe), Some(ssao_target)) = (
            assets.compute_pipeline(&id("distantsunrises:voxy_ssao")),
            ensure_ssao(host, [w, h]),
        )
    {
        let clip_dist = vanilla_clip_dist(&frame);
        // The serial this submission's one publishing invocation writes into the target's stamp
        // (`voxy_ssao.csh`).
        let serial = next_serial();
        let ssao_params = Std140(Vec::with_capacity(48))
            .f32s(&[
                camera.projection[0].abs().max(1e-4),
                camera.projection[5].abs().max(1e-4),
                camera.near,
                clip_dist,
            ])
            .i32s(&[w as i32, h as i32, 0, 0])
            .u32s(&[serial[0], serial[1], 0, 0])
            .0;
        let ssao_desc = ComputePassDescriptor::new("distantsunrises:ssao")
            .reads(FrameTarget::MainDepth)
            .writes(FrameTarget::Mod(ssao_target.target))
            .after_stage(OPAQUE_STAGE)
            .before_stage(TRANSLUCENT_STAGE);
        let added_ssao = graph.add_compute_pass(ssao_desc, &mut |ctx| {
            let bindings = DrawBindings::new()
                .inline_uniform("SsaoParams", &ssao_params)
                .texture("WorldDepthTex", TextureSource::WorldDepth)
                .storage_buffer("SsaoStamp", ssao_target.stamp.slice(0, STAMP_BYTES))
                .storage_image("SsaoOut", ssao_target.dest());
            report(
                "ssao dispatch",
                ctx.dispatch(
                    ssao_pipe,
                    &bindings,
                    [w.div_ceil(HIZ_WORKGROUP), h.div_ceil(HIZ_WORKGROUP), 1],
                ),
            );
        });
        if report("add_ssao_compute_pass", added_ssao).is_some() {
            *SSAO_PENDING.lock().unwrap() = Some((ssao_target, serial));
        }
    }
}

/// Prepares this frame's two indirect draws (opaque in [`OPAQUE_STAGE`], translucent in
/// [`TRANSLUCENT_STAGE`]); a shader pack that compiled `dh_terrain` / `dh_water` draws the
/// pack-linked pipelines instead. Called from `Hooks::prepare_render_buffers`.
pub fn prepare_draws(
    host: &dyn mod_api::ClientHost,
    camera: &CameraMatrices,
    context: &mut dyn mod_api::PrepareRenderBuffersContext,
) {
    use mod_api::{DrawBindings, IndirectDraw, TextureSource};
    // Taken before any early return: a submission's expectation is never carried to the next.
    let ssao_expected = SSAO_PENDING.lock().unwrap().take();
    let Some(frame) = FRAME.lock().unwrap().clone() else {
        return;
    };
    let guard = ARENA.lock().unwrap();
    let Some(state) = guard.as_ref() else { return };
    let Some(quad_mesh) = state.quad_mesh else {
        return;
    };
    let arena = &state.arena;
    let section_high_water = arena.section_high_water();
    if !frame.enabled || section_high_water == 0 {
        return;
    }
    let assets = host.render_assets();
    let view_proj = camera.view_projection();
    let near_dh = 8.0;
    let rd_blocks = frame.config.render_distance_blocks().max(256.0);
    let far_dh = (rd_blocks * 1.4).max(1024.0);
    let atlas_id = id(ATLAS_TEXTURE);
    let clip_dist = vanilla_clip_dist(&frame);
    // Keep mid-ground hills crisp like Voxy (`NormalRenderPipeline.java`), respecting
    // `fog_enabled`, `fog_falloff`, and `fog_spherical`.
    let (fog_start, fog_end) = if frame.config.fog_enabled {
        let start = (rd_blocks * 0.55).max(frame.vanilla_far_blocks + 64.0);
        (start, rd_blocks.max(start + 64.0))
    } else {
        (far_dh * 4.0, far_dh * 8.0)
    };
    let quad_params = quad_params_bytes(
        view_proj,
        camera.position,
        camera.near,
        far_dh,
        clip_dist,
        frame.sun_brightness,
        [
            fog_start,
            fog_end,
            frame.config.fog_falloff.min(2) as f32,
            f32::from(frame.config.fog_spherical),
        ],
        frame.fog_color,
        false, // frames store vanilla gamma; no sRGB decode in the shader
        frame.config.shading_enabled,
    );
    let (base, sub) = camera_split(camera.position);
    let pack_params = |depth_remap: [f32; 2]| {
        let view: [[f32; 4]; 4] =
            std::array::from_fn(|c| std::array::from_fn(|r| camera.view[c * 4 + r]));
        let proj: [[f32; 4]; 4] =
            std::array::from_fn(|c| std::array::from_fn(|r| camera.projection[c * 4 + r]));
        crate::shader_pack::PackSceneUniform {
            model_view: view,
            projection: proj,
            base_block_pos: [base[0] << 5, base[1] << 5, base[2] << 5],
            frame_id: FRAME_ID.load(Ordering::Relaxed),
            camera_sub_pos: sub,
            depth_remap,
        }
        .to_bytes()
    };
    let s = arena.slices;
    // Apply compute SSAO multiplicatively onto opaque LODs before sorted translucent water draws,
    // for the target and serial this submission's `setup_compute` recorded. The draw is
    // prepared whether or not the compute node will run (the graph may be dropped after this
    // hook); `voxy_ssao_apply.fsh` draws only where the target's stamp holds the serial, so a
    // dropped graph, a refused dispatch or a route that never runs the node before this stage
    // darkens nothing. Under OIT the stage's draws are not issued at all.
    if frame.config.ssao_enabled
        && let Some((ssao_target, serial)) = ssao_expected
        && let Ok(ssao_apply) = assets.render_pipeline(&id(PIPELINE_SSAO_APPLY))
    {
        let expected = Std140(Vec::with_capacity(16))
            .u32s(&[serial[0], serial[1], 0, 0])
            .0;
        let bindings = DrawBindings::new()
            .inline_uniform("SsaoApplyParams", &expected)
            .storage_buffer("SsaoStamp", ssao_target.stamp.slice(0, STAMP_BYTES))
            .texture("SsaoTex", ssao_target.source());
        if let Some(batch) = report(
            "ssao_apply draw",
            context.draw_mesh(
                TRANSLUCENT_STAGE,
                ssao_apply,
                quad_mesh,
                mod_api::MeshDraw::indexed(0..3, 0),
                &bindings,
            ),
        ) {
            PREPARED.lock().unwrap().push((TRANSLUCENT_STAGE, batch));
        }
    }
    let use_count_buf = host.render_capabilities().multi_draw_indirect_count;
    for (stage, plain, pack, cmds, count, count_offset) in [
        (
            OPAQUE_STAGE,
            PIPELINE_OPAQUE,
            PIPELINE_PACK_TERRAIN,
            arena.opaque_commands,
            arena.opaque_command_count(),
            OPAQUE_COUNT_OFFSET,
        ),
        (
            TRANSLUCENT_STAGE,
            PIPELINE_TRANSLUCENT,
            PIPELINE_PACK_WATER,
            arena.translucent_commands,
            arena.translucent_command_count(),
            TRANSLUCENT_COUNT_OFFSET,
        ),
    ] {
        let packed = assets
            .render_pipeline(&id(pack))
            .ok()
            .filter(|p| assets.pipeline_pack_bound(*p));
        let pack_bytes = pack_params(crate::shader_pack::DepthRemap::standard(near_dh));
        let pipeline;
        let bindings = if let Some(p) = packed {
            pipeline = p;
            DrawBindings::new()
                .inline_uniform("SceneUniform", &pack_bytes)
                .storage_buffer("QuadArena", s.quads)
                .storage_buffer("SectionBuffer", s.sections)
                .storage_buffer("ModelBuffer", s.models)
                .storage_buffer("BiomeTintBuffer", s.biome_tints)
                .texture("BlockAtlas", TextureSource::DynamicTexture(&atlas_id))
        } else {
            let Ok(p) = assets.render_pipeline(&id(plain)) else {
                continue;
            };
            pipeline = p;
            DrawBindings::new()
                .inline_uniform("QuadParams", &quad_params)
                .storage_buffer("QuadBuffer", s.quads)
                .storage_buffer("SectionBuffer", s.sections)
                .storage_buffer("ModelBuffer", s.models)
                .storage_buffer("BiomeTintBuffer", s.biome_tints)
                .texture("BlockAtlas", TextureSource::DynamicTexture(&atlas_id))
        };
        let mut draw = IndirectDraw::new(pipeline, cmds, count, &bindings).with_mesh(quad_mesh);
        if use_count_buf {
            draw = draw.with_count_buffer(arena.trans_sort_meta, count_offset);
        }
        if let Some(batch) = report("draw_indirect", context.draw_indirect(stage, &draw)) {
            PREPARED.lock().unwrap().push((stage, batch));
        }
    }
    // The shadow pass takes the pack-linked shadow pipeline through the same world stage when the
    // pack compiled `dh_shadow` (`pipeline_pack_bound`), drawn with the pack's shadow matrices.
    if let (Some(shadow), Ok(p)) = (
        host.shader_pack_state().and_then(|s| s.shadow),
        assets.render_pipeline(&id(PIPELINE_PACK_SHADOW)),
    ) && assets.pipeline_pack_bound(p)
    {
        let bytes = crate::shader_pack::PackSceneUniform {
            model_view: std::array::from_fn(|c| {
                std::array::from_fn(|r| shadow.model_view[c * 4 + r])
            }),
            projection: std::array::from_fn(|c| {
                std::array::from_fn(|r| shadow.projection[c * 4 + r])
            }),
            base_block_pos: [base[0] << 5, base[1] << 5, base[2] << 5],
            frame_id: FRAME_ID.load(Ordering::Relaxed),
            camera_sub_pos: sub,
            depth_remap: [0.0; 2],
        }
        .to_bytes();
        let bindings = DrawBindings::new()
            .inline_uniform("SceneUniform", &bytes)
            .storage_buffer("QuadArena", s.quads)
            .storage_buffer("SectionBuffer", s.sections)
            .storage_buffer("ModelBuffer", s.models)
            .storage_buffer("BiomeTintBuffer", s.biome_tints)
            .texture("BlockAtlas", TextureSource::DynamicTexture(&atlas_id));
        let mut draw = IndirectDraw::new(
            p,
            arena.opaque_commands,
            arena.opaque_command_count(),
            &bindings,
        )
        .with_mesh(quad_mesh);
        if use_count_buf {
            draw = draw.with_count_buffer(arena.trans_sort_meta, OPAQUE_COUNT_OFFSET);
        }
        if let Some(batch) = report(
            "shadow draw_indirect",
            context.draw_indirect(OPAQUE_STAGE, &draw),
        ) {
            PREPARED.lock().unwrap().push((OPAQUE_STAGE, batch));
        }
    }
    record_frame_stats(u64::from(section_high_water), arena.live_sections() as u64);
}

/// Issues the draws prepared for `stage`. Called from `Hooks::render_level_stage`.
pub fn draw_stage(
    stage: mod_api::RenderLevelStage,
    context: &mut dyn mod_api::RenderLevelStageContext,
) {
    let Some(pass) = context.world_pass() else {
        return;
    };
    for (s, batch) in PREPARED.lock().unwrap().iter() {
        if *s == stage {
            report("pass.draw", pass.draw(*batch));
        }
    }
}

/// Snapshot of client frame parameters passed from the mod crate to the stage renderer.
#[derive(Clone, Debug)]
pub struct RenderFrameParams {
    pub enabled: bool,
    pub active_dimension: String,
    pub config: DhConfig,
    pub fog_color: [f32; 4],
    pub vanilla_far_blocks: f32,
    pub sun_brightness: f32,
}

/// Telemetry counters surfaced in F3 and `/ds status`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DhStats {
    pub active_sections: usize,
    /// Last prepared frame: one past the highest occupied GPU section slot; includes holes.
    /// Zero once the arena is retired or emptied, or rendering is disabled.
    pub section_high_water: u64,
    /// Last prepared frame: sections resident in the GPU arena, independent of draw visibility.
    /// Cleared with `section_high_water`.
    pub resident_sections: u64,
    pub generated_columns: u64,
    pub saved_sections: u64,
}

/// Shared runtime state accessible from `Hooks` and the stage renderer.
pub(crate) struct DhRuntime {
    pub enabled_flag: AtomicBool,
    pub section_high_water: AtomicU64,
    pub resident_sections: AtomicU64,
    pub saved_sections: AtomicU64,
    pub io: LodIoWorker,
    pub gen_pool: GenWorkerPool,
    pub inner: Mutex<DhState>,
}

pub struct DhState {
    pub config: DhConfig,
    pub world_dir: Option<PathBuf>,
    pub server_open: bool,
    pub live_levels: HashSet<String>,
    pub active_dimension: String,
    pub camera: Option<CameraMatrices>,
    pub camera_pos: [f64; 3],
    pub has_camera: bool,
    pub fog_color: [f32; 4],
    pub vanilla_far_blocks: f32,
    pub sun_brightness: f32,
    pub rain_level: f32,
    pub thunder_level: f32,
    /// Client's current vanilla render distance in chunks (from `ClientHost::graphics_options`).
    pub vanilla_render_distance_chunks: i32,
    /// Loaded vanilla client chunk coordinates for accurate seam handling.
    pub client_loaded_chunks: HashSet<[i32; 2]>,
    pub dimensions: HashMap<String, DimensionLodStore>,
    pub pending_chunks: HashMap<String, VecDeque<[i32; 2]>>,
    pub pending_chunk_set: HashMap<String, HashSet<[i32; 2]>>,
    /// Public API: per-dimension custom world generators overriding built-in distant sampling.
    pub world_generators: HashMap<String, Arc<dyn LodWorldGenerator>>,
    /// Public API: per-dimension client-side remote LOD providers.
    pub remote_providers: HashMap<String, Arc<dyn RemoteLodProvider>>,
    /// Public API: dimensions where built-in distant generation is explicitly disabled.
    pub disabled_builtin_gen: HashSet<String>,
    /// Public API: dimensions where built-in multiplayer LOD networking is explicitly disabled.
    pub disabled_builtin_net: HashSet<String>,
    /// Public API: section positions already requested from a registered `RemoteLodProvider`.
    pub remote_requested: HashMap<String, HashSet<SectionPos>>,
    /// Per-dimension round-robin cursor across connected players for dedicated/LAN server distant
    /// generation (per dimension so two dimensions with equal player counts do not alias).
    pub server_player_round_robin: HashMap<String, usize>,
    /// Cached desired-section order keyed by `(dimension, player_uuid)` (`0` for local camera),
    /// rebuilt when config changes or the camera moves at least `SECTION_SIZE` (64) blocks
    /// Manhattan distance (matching DH's `QuadTree.setCenterBlockPos` threshold
    /// `FullDataSourceV2.WIDTH`).
    pub desired_cache: HashMap<(String, u128), DesiredCache>,
    /// Earliest instant the next synchronous distant-sampling batch may run for each dimension.
    pub next_distant_sample: HashMap<String, Instant>,
}

/// Cached result of `DimensionLodStore::desired_sections_coarse_to_fine`.
pub struct DesiredCache {
    pub center: [i32; 2],
    pub config: DhConfig,
    pub list: Vec<SectionPos>,
    pub disk_cursor: usize,
    pub sample_cursor: usize,
}

impl Default for DhState {
    fn default() -> Self {
        Self {
            config: DhConfig::default(),
            world_dir: None,
            server_open: false,
            live_levels: HashSet::new(),
            active_dimension: "minecraft:overworld".to_owned(),
            camera: None,
            camera_pos: [0.0, 64.0, 0.0],
            has_camera: false,
            fog_color: [0.75, 0.85, 1.0, 1.0],
            vanilla_far_blocks: 80.0,
            sun_brightness: 1.0,
            rain_level: 0.0,
            thunder_level: 0.0,
            vanilla_render_distance_chunks: 12,
            client_loaded_chunks: HashSet::new(),
            dimensions: HashMap::new(),
            pending_chunks: HashMap::new(),
            pending_chunk_set: HashMap::new(),
            world_generators: HashMap::new(),
            remote_providers: HashMap::new(),
            disabled_builtin_gen: HashSet::new(),
            disabled_builtin_net: HashSet::new(),
            remote_requested: HashMap::new(),
            server_player_round_robin: HashMap::new(),
            desired_cache: HashMap::new(),
            next_distant_sample: HashMap::new(),
        }
    }
}

static RUNTIME: OnceLock<Arc<DhRuntime>> = OnceLock::new();

pub(crate) fn runtime() -> &'static Arc<DhRuntime> {
    RUNTIME.get_or_init(|| {
        Arc::new(DhRuntime {
            enabled_flag: AtomicBool::new(true),
            section_high_water: AtomicU64::new(0),
            resident_sections: AtomicU64::new(0),
            saved_sections: AtomicU64::new(0),
            io: LodIoWorker::spawn(),
            gen_pool: GenWorkerPool::new(),
            inner: Mutex::new(DhState::default()),
        })
    })
}

/// Drains background IO results into the section store and returns this frame's parameters.
pub fn begin_render_frame(camera: [f64; 3]) -> RenderFrameParams {
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let io_responses = rt.io.drain_responses();
    st.camera_pos = camera;
    st.has_camera = true;

    for res in io_responses {
        match res {
            IoResponse::Saved => {
                rt.saved_sections.fetch_add(1, Ordering::Relaxed);
            }
            IoResponse::Loaded { dimension, section } => {
                if !st.live_levels.contains(&dimension) {
                    continue;
                }
                let dim = st.dimensions.entry(dimension).or_default();
                let newer = dim
                    .sections
                    .get(&section.pos)
                    .is_none_or(|existing| existing.populated < section.populated);
                if newer {
                    dim.sections.insert(section.pos, section);
                }
            }
        }
    }

    RenderFrameParams {
        enabled: st.config.enabled,
        active_dimension: st.active_dimension.clone(),
        config: st.config,
        fog_color: st.fog_color,
        vanilla_far_blocks: st.vanilla_far_blocks,
        sun_brightness: st.sun_brightness,
    }
}

/// Record arena occupancy during draw preparation, before any draw is issued.
pub fn record_frame_stats(section_high_water: u64, resident_sections: u64) {
    let rt = runtime();
    rt.section_high_water
        .store(section_high_water, Ordering::Relaxed);
    rt.resident_sections
        .store(resident_sections, Ordering::Relaxed);
}

/// Forgets the last prepared arena occupancy at a lifecycle change (arena retired or emptied,
/// rendering disabled), where no later preparation would overwrite it. Atomics only, so it
/// nests under any lock and never touches a replacement arena's own later record.
fn clear_arena_stats() {
    let rt = runtime();
    rt.section_high_water.store(0, Ordering::Relaxed);
    rt.resident_sections.store(0, Ordering::Relaxed);
}

/// Publishes the `enabled` setting. Turning rendering off clears the arena counters once, on the
/// transition, instead of on every disabled frame.
pub(crate) fn set_enabled(enabled: bool) {
    if runtime().enabled_flag.swap(enabled, Ordering::Relaxed) && !enabled {
        clear_arena_stats();
    }
}

/// Snapshot stats for F3 and `/ds status`.
pub(crate) fn stats() -> DhStats {
    let rt = runtime();
    let st = rt.inner.lock().unwrap();
    let active_dim = &st.active_dimension;
    let active_sections = st
        .dimensions
        .get(active_dim)
        .map(|d| d.sections.len())
        .unwrap_or(0);
    let generated_columns = st
        .dimensions
        .get(active_dim)
        .map(|d| d.generated_columns)
        .unwrap_or(0);
    DhStats {
        active_sections,
        section_high_water: rt.section_high_water.load(Ordering::Relaxed),
        resident_sections: rt.resident_sections.load(Ordering::Relaxed),
        generated_columns,
        saved_sections: rt.saved_sections.load(Ordering::Relaxed),
    }
}

/// Compute sun brightness `0.18..=1.0` from Minecraft's `day_time` (`0..24000`), dimension, and
/// live rain/thunder weather levels (`0.0..=1.0`).
pub fn sun_brightness_from_day_time_and_weather(
    day_time: i64,
    dimension: &str,
    rain_level: f32,
    thunder_level: f32,
) -> f32 {
    if dimension == "minecraft:the_nether" {
        return 0.72;
    }
    if dimension == "minecraft:the_end" {
        return 0.80;
    }
    let t = (day_time.rem_euclid(24_000) as f32) / 24_000.0;
    // Vanilla celestial angle: noon at t = 0.25 (6000), midnight at t = 0.75 (18000).
    let angle = (t - 0.25) * std::f32::consts::TAU;
    let sun_h = angle.cos();
    let day_factor = ((sun_h * 2.0) + 0.5).clamp(0.0, 1.0);
    // Match vanilla `ClientLevel.getSkyDarken`: rain dims sky up to 31.25%, thunder another 25%.
    let weather_scale =
        (1.0 - rain_level.clamp(0.0, 1.0) * 0.3125) * (1.0 - thunder_level.clamp(0.0, 1.0) * 0.25);
    (0.24 + 0.76 * day_factor * weather_scale).clamp(0.18, 1.0)
}

/// Compute sun brightness `0.2..=1.0` from Minecraft's `day_time` (`0..24000`) in clear weather.
pub fn sun_brightness_from_day_time(day_time: i64, dimension: &str) -> f32 {
    sun_brightness_from_day_time_and_weather(day_time, dimension, 0.0, 0.0)
}

/// Messages logged per `report` call site before it goes quiet.
const REPORT_LIMIT_PER_SITE: u32 = 20;

/// Counts one more message for `what` and says whether it is still under the per-site limit.
fn take_report_slot(counts: &mut HashMap<&'static str, u32>, what: &'static str) -> bool {
    let logged = counts.entry(what).or_insert(0);
    if *logged < REPORT_LIMIT_PER_SITE {
        *logged += 1;
        true
    } else {
        false
    }
}

/// Logs the first few errors of each render call site the frame cannot recover from, then keeps
/// that site quiet; one noisy site never silences unrelated errors.
fn report<T, E: std::fmt::Debug>(what: &'static str, result: Result<T, E>) -> Option<T> {
    static LOGGED: LazyLock<Mutex<HashMap<&'static str, u32>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    match result {
        Ok(v) => Some(v),
        Err(e) => {
            let log = take_report_slot(&mut LOGGED.lock().unwrap_or_else(|p| p.into_inner()), what);
            if log {
                eprintln!("distantsunrises: {what} failed: {e:?}");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mod_api::BufferError;
    use std::cell::RefCell;

    #[derive(Default)]
    struct RecordingGpu {
        available: bool,
        retired_arena: Option<u32>,
        arenas: u32,
        arena_error: Option<BufferError>,
        meshes: u32,
        live_meshes: RefCell<HashSet<u32>>,
        ready: RefCell<Option<Vec<u8>>>,
        polled: RefCell<Vec<u32>>,
        released: RefCell<Vec<u32>>,
    }

    impl TerrainGpu for RecordingGpu {
        type Arena = u32;
        type Mesh = u32;
        type Ticket = u32;

        fn create_arena(&mut self, _: ArenaConfig) -> Result<Self::Arena, ArenaError> {
            if let Some(error) = self.arena_error {
                assert!(
                    self.live_meshes.get_mut().contains(&self.meshes),
                    "mesh allocated first"
                );
                return Err(error.into());
            }
            self.arenas += 1;
            Ok(self.arenas)
        }

        fn create_quad_mesh(&mut self) -> Result<Self::Mesh, ArenaError> {
            self.meshes += 1;
            self.live_meshes.get_mut().insert(self.meshes);
            Ok(self.meshes)
        }

        fn release_quad_mesh(&self, mesh: Self::Mesh) -> Result<(), mod_api::MeshError> {
            assert!(self.live_meshes.borrow_mut().remove(&mesh));
            Ok(())
        }

        fn request_buffer_size(&self, arena: &Self::Arena) -> Result<u64, BufferError> {
            if !self.available {
                Err(BufferError::Unavailable)
            } else if self.retired_arena == Some(*arena) {
                Err(BufferError::Stale)
            } else {
                Ok(4)
            }
        }

        fn poll_readback(&self, ticket: Self::Ticket) -> Result<Option<Vec<u8>>, BufferError> {
            assert!(self.available, "polled without a renderer");
            self.polled.borrow_mut().push(ticket);
            Ok(self.ready.borrow_mut().take())
        }

        fn release_readback(&self, ticket: Self::Ticket) -> Result<(), BufferError> {
            assert!(self.available, "released without a renderer");
            self.released.borrow_mut().push(ticket);
            // A retire discards the completed bytes without polling or decoding them.
            self.ready.borrow_mut().take();
            Ok(())
        }
    }

    fn single_node_request(x: i32) -> Vec<u8> {
        [1, 0, 0, 0]
            .into_iter()
            .chain(crate::gpu_arena::pack_section_pos(1, x, 0, 0))
            .flat_map(u32::to_le_bytes)
            .collect()
    }

    #[test]
    fn retired_traversal_ticket_survives_missing_renderer_and_cannot_publish_into_next_world() {
        let mut host = RecordingGpu {
            available: true,
            ..Default::default()
        };
        let arena = 1;
        let mut requests = NodeRequests::new();

        requests.start(|| Some(1));
        *host.ready.borrow_mut() = Some(single_node_request(17));
        requests.service(&host, &arena);
        assert_eq!(requests.ready, [(1, [17, 0, 0])]);
        requests.start(|| Some(2));
        requests.service(&host, &arena); // still in flight
        *host.ready.borrow_mut() = Some(single_node_request(23));

        requests.retire();
        assert!(requests.ready.is_empty());
        host.available = false;
        requests.service(&host, &arena);
        requests.retire(); // another world retired before another usable client tick
        requests.start(|| panic!("retired ticket must still occupy the single slot"));
        assert_eq!(requests.pending.as_ref().unwrap().ticket, 2);
        assert!(host.released.borrow().is_empty());

        host.available = true;
        requests.service(&host, &arena);
        assert_eq!(*host.released.borrow(), [2]);
        assert_eq!(*host.polled.borrow(), [1, 2]);
        assert!(requests.ready.is_empty());
        assert!(requests.pending.is_none());

        requests.start(|| Some(3));
        *host.ready.borrow_mut() = Some(single_node_request(31));
        requests.service(&host, &arena);
        assert_eq!(requests.ready, [(1, [31, 0, 0])]);
        assert_eq!(*host.polled.borrow(), [1, 2, 3]);
    }

    #[test]
    fn only_typed_staleness_retires_an_arena_and_its_installed_ids() {
        let mut host = RecordingGpu {
            available: true,
            ..Default::default()
        };
        let config = ArenaConfig {
            max_sections: 4,
            geometry_bytes: 0,
        };
        let mut slot = None;
        let live = ensure_arena(&mut slot, &mut host, config).unwrap();
        let old_arena = live.arena;
        let key = ("minecraft:overworld".into(), 0, [2, 0, -2]);
        live.ids.insert(key, 3);
        assert_eq!(
            ArenaState::discard_if_retired(&mut slot, &host).unwrap(),
            Retained::Live
        );
        host.available = false;
        assert_eq!(
            ArenaState::discard_if_retired(&mut slot, &host).unwrap(),
            Retained::Unavailable
        );
        assert_eq!(slot.as_ref().unwrap().ids.len(), 1);

        host.available = true;
        host.retired_arena = Some(old_arena);
        let mut requests = NodeRequests::new();
        requests.start(|| Some(71));
        *host.ready.borrow_mut() = Some(single_node_request(17));
        requests.service(&host, &old_arena);
        assert!(host.polled.borrow().is_empty());
        assert!(host.released.borrow().is_empty());
        assert_eq!(
            ArenaState::discard_if_retired(&mut slot, &host).unwrap(),
            Retained::Retired
        );
        assert!(slot.is_none());
        let new = ensure_arena(&mut slot, &mut host, config).unwrap();
        assert_ne!(new.arena, old_arena);
        assert!(new.ids.is_empty());
        assert_eq!(host.arenas, 2);
        assert_eq!(host.meshes, 2);
    }

    #[test]
    fn final_arena_refusal_releases_new_mesh_and_next_ensure_can_allocate() {
        let mut gpu = RecordingGpu {
            arena_error: Some(BufferError::ExceedsDeviceLimit),
            ..Default::default()
        };
        let config = ArenaConfig {
            max_sections: 4,
            geometry_bytes: 0,
        };
        let mut slot = None;
        assert!(matches!(
            ensure_arena(&mut slot, &mut gpu, config),
            Err(ArenaError::Buffer(BufferError::ExceedsDeviceLimit))
        ));
        assert!(slot.is_none());
        assert_eq!(gpu.meshes, 1);
        assert!(gpu.live_meshes.get_mut().is_empty());
        assert_eq!(gpu.arenas, 0);

        gpu.arena_error = None;
        let state = ensure_arena(&mut slot, &mut gpu, config).unwrap();
        assert_eq!(state.quad_mesh, Some(2));
        assert_eq!(*gpu.live_meshes.get_mut(), HashSet::from([2]));
        assert_eq!(gpu.arenas, 1);
        assert_eq!(
            ensure_arena(&mut slot, &mut gpu, config).unwrap().quad_mesh,
            Some(2)
        );
        assert_eq!(gpu.meshes, 2);
        assert_eq!(gpu.arenas, 1);
    }

    #[test]
    fn quad_mesh_recovery_only_recreates_mesh_after_each_invalidation() {
        let mut host = RecordingGpu::default();
        let mut slot = None;
        let config = ArenaConfig {
            max_sections: 4,
            geometry_bytes: 0,
        };
        let first = ensure_arena(&mut slot, &mut host, config).unwrap();
        let arena = first.arena;
        let mut old_mesh = first.quad_mesh.unwrap();
        assert_eq!(host.arenas, 1);
        // Exercise the shared invalidation twice with a retained arena, not an actual world
        // rejoin or pack apply. Pack reload retention is blocked by Sunrise #6264.
        for generation in [2, 3] {
            ArenaState::invalidate_quad_mesh(&mut slot);
            assert!(slot.as_ref().unwrap().quad_mesh.is_none());
            let recovered = ensure_arena(&mut slot, &mut host, config).unwrap();
            let mesh = recovered.quad_mesh.unwrap();
            assert_ne!(mesh, old_mesh);
            assert_eq!(recovered.arena, arena);
            assert_eq!(host.arenas, 1);
            assert_eq!(host.meshes, generation);
            assert_eq!(
                ensure_arena(&mut slot, &mut host, config)
                    .unwrap()
                    .quad_mesh,
                Some(mesh)
            );
            assert_eq!(host.arenas, 1);
            assert_eq!(host.meshes, generation);
            old_mesh = mesh;
        }
    }

    #[test]
    fn cmdgen_params_carry_compact_opaque_in_camera_sub_pos_w() {
        let params =
            |compact| cmdgen_params_bytes([0.0; 16], [0.0; 3], 4, [8, 8], 4, true, compact);
        let w = |bytes: &[u8]| f32::from_le_bytes(bytes[92..96].try_into().unwrap());
        assert_eq!(params(true).len(), 112);
        assert_eq!(w(&params(true)), 1.0);
        assert_eq!(w(&params(false)), 0.0);
    }

    #[test]
    fn node_cap_gate_suppresses_logs_once_and_resets_with_removals_and_sessions() {
        let key = |x: i32| -> SectionKey { ("minecraft:overworld".into(), 0, [x, 0, 0]) };
        let logged = AtomicBool::new(false);
        let counts = (66, 33, 2, 8);
        {
            let mut gate = NodeCapGate::default();
            assert!(!gate.is_suppressed(&key(1)));
            assert_eq!(
                gate.refuse(key(1), counts, &logged).as_deref(),
                Some(
                    "distantsunrises: GPU octree node cap reached (66/33 nodes, 2/8 roots); skipping additional sections until sections are removed"
                )
            );
            assert!(gate.is_suppressed(&key(1)));
            assert!(!gate.is_suppressed(&key(2)));

            // A second refusal in the same session is suppressed but not logged again.
            assert_eq!(gate.refuse(key(2), counts, &logged), None);
            assert!(gate.is_suppressed(&key(2)));

            // A removal lifts suppression, but does not rearm the log.
            gate.lift();
            assert!(!gate.is_suppressed(&key(1)) && !gate.is_suppressed(&key(2)));
            assert_eq!(gate.refuse(key(1), counts, &logged), None);
        }

        // GPU retirement drops the gate, not the session diagnostic.
        {
            let mut gate = NodeCapGate::default();
            assert!(!gate.is_suppressed(&key(1)));
            assert_eq!(gate.refuse(key(1), counts, &logged), None);
            assert!(gate.is_suppressed(&key(1)));
        }

        // A session reset rearms the log even if no arena exists at the time.
        logged.store(false, Ordering::Release);
        let mut gate = NodeCapGate::default();
        assert!(gate.refuse(key(1), counts, &logged).is_some());
        assert_eq!(gate.refuse(key(2), counts, &logged), None);
    }

    #[test]
    fn report_limit_is_per_call_site() {
        let mut counts = HashMap::new();
        for _ in 0..REPORT_LIMIT_PER_SITE {
            assert!(take_report_slot(&mut counts, "noisy"));
        }
        assert!(!take_report_slot(&mut counts, "noisy"));
        assert!(!take_report_slot(&mut counts, "noisy"));
        // Another site still logs after "noisy" is silenced.
        assert!(take_report_slot(&mut counts, "quiet"));
    }

    #[test]
    fn weather_dims_overworld_sun_brightness_but_not_nether_or_end() {
        let clear_noon =
            sun_brightness_from_day_time_and_weather(6_000, "minecraft:overworld", 0.0, 0.0);
        let storm_noon =
            sun_brightness_from_day_time_and_weather(6_000, "minecraft:overworld", 1.0, 1.0);
        let clear_midnight =
            sun_brightness_from_day_time_and_weather(18_000, "minecraft:overworld", 0.0, 0.0);
        assert!(clear_noon > 0.99);
        assert!(storm_noon < clear_noon - 0.25);
        assert!(storm_noon > clear_midnight);
        assert_eq!(
            sun_brightness_from_day_time_and_weather(18_000, "minecraft:the_nether", 1.0, 1.0),
            0.72
        );
        assert_eq!(
            sun_brightness_from_day_time_and_weather(18_000, "minecraft:the_end", 1.0, 1.0),
            0.80
        );
    }
}
