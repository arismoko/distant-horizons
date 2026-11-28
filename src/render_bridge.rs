//! Thread-safe bridge between Distant Sunrises mod hooks, background workers, and the GPU-driven
//! renderer (`gpu_arena`, GLSL shaders under `assets/distantsunrises/shaders/`).
//!
//! No `wgpu` types and no `[native.*]`: everything goes through Sunrise's general render API.

use crate::api::{LodWorldGenerator, RemoteLodProvider};
use crate::config::DhConfig;
use crate::gen_worker::GenWorkerPool;
use crate::lod::{DimensionLodStore, SectionPos};
use crate::octree::OctreeSectionPos;
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

use crate::gpu_arena::{ArenaConfig, ArenaError, BuiltSection, GpuArena};

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

/// The quad pipelines' set-1 layout (`voxy_quads.vsh/fsh`).
pub fn quad_layout() -> mod_api::BindGroupLayout {
    use mod_api::{SamplerAddress, SamplerFilter};
    mod_api::BindGroupLayout::builder()
        .uniform_buffer("QuadParams", 160)
        .entry(storage("QuadBuffer", true, 8))
        .entry(storage("SectionBuffer", true, 32))
        .entry(storage("ModelBuffer", true, 64))
        .entry(storage("BiomeTintBuffer", true, 4))
        .sampler(
            "BlockAtlas",
            SamplerFilter::Linear,
            SamplerAddress::ClampToEdge,
        )
        .build()
        .expect("static quad layout")
}

/// The pack-linked pipelines' set-1 layout (`voxy_pack_*.vsh`): `SceneUniform` is 176 bytes
/// ([`crate::shader_pack::PACK_SCENE_UNIFORM_SIZE`]) and the quad arena block is `QuadArena`.
pub fn pack_layout() -> mod_api::BindGroupLayout {
    use mod_api::{SamplerAddress, SamplerFilter};
    mod_api::BindGroupLayout::builder()
        .uniform_buffer(
            "SceneUniform",
            crate::shader_pack::PACK_SCENE_UNIFORM_SIZE as u32,
        )
        .entry(storage("QuadArena", true, 8))
        .entry(storage("SectionBuffer", true, 32))
        .entry(storage("ModelBuffer", true, 64))
        .entry(storage("BiomeTintBuffer", true, 4))
        .sampler(
            "BlockAtlas",
            SamplerFilter::Linear,
            SamplerAddress::ClampToEdge,
        )
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
        BindGroupLayout, ComputePipeline, SamplerAddress, SamplerFilter, StorageImageAccess,
    };
    let hiz_layout = || {
        BindGroupLayout::builder()
            .uniform_buffer("HizParams", 16)
            .sampler(
                "HizSource",
                SamplerFilter::Nearest,
                SamplerAddress::ClampToEdge,
            )
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
                    .sampler(
                        "HizPyramid",
                        SamplerFilter::Nearest,
                        SamplerAddress::ClampToEdge,
                    )
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
                    .sampler(
                        "HizPyramid",
                        SamplerFilter::Nearest,
                        SamplerAddress::ClampToEdge,
                    )
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
                    .uniform_buffer("SsaoParams", 32)
                    .sampler(
                        "WorldDepthTex",
                        SamplerFilter::Nearest,
                        SamplerAddress::ClampToEdge,
                    )
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
                .sampler(
                    "SsaoTex",
                    SamplerFilter::Nearest,
                    SamplerAddress::ClampToEdge,
                )
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

/// `CmdgenParams` (112 bytes).
pub fn cmdgen_params_bytes(
    view_proj: [f32; 16],
    camera: [f64; 3],
    section_count: u32,
    hiz_size: [u32; 2],
    hiz_mips: u32,
    hiz_enabled: bool,
) -> Vec<u8> {
    let (base, sub) = camera_split(camera);
    Std140(Vec::with_capacity(112))
        .f32s(&view_proj)
        .i32s(&[base[0], base[1], base[2], section_count as i32])
        .f32s(&[sub[0], sub[1], sub[2], 0.0])
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

/// Half-resolution SSAO target size, rounded up.
fn ssao_size(viewport: [u32; 2]) -> [u32; 2] {
    [viewport[0].div_ceil(2), viewport[1].div_ceil(2)]
}

/// Mod-owned half-resolution `R32Float` SSAO output target, resized with the viewport.
struct SsaoTarget {
    target: mod_api::RenderTargetRef,
    size: [u32; 2],
}

static SSAO: Mutex<Option<SsaoTarget>> = Mutex::new(None);

/// Whether the SSAO compute pass has been added to the frame graph for the current target. False
/// after (re)allocating the target and at the start of every `setup_compute`, so `ssao_apply`
/// never multiplies the screen by an uninitialised or stale AO image.
static SSAO_WRITTEN: AtomicBool = AtomicBool::new(false);

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

fn ensure_ssao(host: &dyn mod_api::ClientHost, size: [u32; 2]) -> Option<SsaoTarget> {
    use mod_api::{RenderTargetDescriptor, RenderTargetFormat, StorageImageAccess};
    let mut slot = SSAO.lock().unwrap();
    if let Some(s) = slot.as_ref().filter(|s| s.size == size) {
        return Some(SsaoTarget { ..*s });
    }
    if let Some(old) = slot.take() {
        let _ = old.target.release(host);
    }
    SSAO_WRITTEN.store(false, Ordering::Release);
    let desc = RenderTargetDescriptor::new(size)
        .color_target(RenderTargetFormat::R32Float)
        .mip_levels(1)
        .storage(StorageImageAccess::WriteOnly);
    let target = host.create_render_target(&desc).ok()?;
    let made = SsaoTarget { target, size };
    *slot = Some(SsaoTarget { ..made });
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
static PENDING_Y_BOUNDS: Mutex<std::collections::BTreeMap<String, (i32, i32)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// One shared GPU geometry arena ([`crate::gpu_arena::VOXY_GEOMETRY_BYTES`], 128 MiB by default)
/// survives dimension and world changes. Like Voxy's `MixinLevelRenderer.voxy$setWorld`, only
/// the active world owns arena section slots (flushed to the GPU on the next enabled upload tick).
/// `RenderResourceReuse` recycles the geometry buffer via `getOrCreateGeometryBuffer` /
/// `giveBackGeometryBuffer` rather than reallocating VRAM.
/// Integrated unloads clear the departing dimension's GPU slots and upload residency but keep
/// its CPU octree for fast re-upload. [`crate::pipeline::client_tick`] admits completions only
/// for the active live dimension.
static ARENA: Mutex<Option<ArenaState>> = Mutex::new(None);

/// Identity of an uploaded section: `(dimension, lod level, section position)`.
pub type SectionKey = (String, u8, [i32; 3]);

struct ArenaState {
    arena: GpuArena,
    /// Shared quad index pattern (`0,1,2,2,1,3` per quad) the indirect commands index into.
    /// `None` after a Sunrise mesh retirement until recreated on demand in `prepare_draws`.
    quad_mesh: Option<mod_api::MeshRef>,
    ids: HashMap<SectionKey, u32>,
}

/// Model words, biome tint words and the baked face atlas (RGBA8, `ATLAS_SIDE²`).
type Tables = (Vec<u32>, Vec<u32>, Vec<u8>);
static PENDING_TABLES: Mutex<Option<Tables>> = Mutex::new(None);
static ATLAS_CREATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Id of the dynamic texture holding the baked per-face texels.
pub const ATLAS_TEXTURE: &str = "distantsunrises:voxy_atlas";

/// Queues the baked model words and biome tint words for the next `upload_pending` (they replace
/// the previous tables, which a resource reload or new block id makes stale).
pub fn queue_model_tables(model_words: Vec<u32>, biome_tint_words: Vec<u32>, atlas: Vec<u8>) {
    *PENDING_TABLES.lock().unwrap() = Some((model_words, biome_tint_words, atlas));
}

/// Queues a mesh. An empty mesh retires old geometry but only resolves its octant with coverage.
pub fn submit_built_section(key: SectionKey, section: BuiltSection) {
    PENDING_SECTIONS.lock().unwrap().push((key, section));
}

/// Publish the build height alongside meshes, including on the tick that creates the arena.
pub(crate) fn set_dimension_y_bounds(dimension: &str, bounds: (i32, i32)) {
    PENDING_Y_BOUNDS
        .lock()
        .unwrap()
        .insert(dimension.to_owned(), bounds);
}

/// Most quads one indirect command can span (a 32³ section's worth of faces in one bucket).
const MAX_QUADS_PER_COMMAND: u32 = 1 << 16;

/// A mesh holding only the shared quad index buffer; its one vertex binding is never read (the
/// vertex stages pull quads from the arena by `gl_VertexID`). It has the stride of the default
/// `position_tex_color` binding, which a pipeline with no declared binding must be drawn with.
fn create_quad_mesh(
    host: &dyn mod_api::ClientHost,
) -> Result<mod_api::MeshRef, crate::gpu_arena::ArenaError> {
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
    host.create_mesh(&desc)
        .map_err(|_| crate::gpu_arena::ArenaError::Buffer(mod_api::BufferError::Unavailable))
}

/// Creates the arena on first use (ruling 2: it then survives reloads and world close),
/// uploads pending sections and flushes metadata. Call from `client_tick_pre`. Errors are logged
/// through [`report`]; the frame goes on without them.
pub fn upload_pending(host: &mut dyn mod_api::ClientHost, config: ArenaConfig) {
    report_arena("arena upload", try_upload_pending(host, config));
}

/// Allocates the arena buffers first, then the shared quad mesh; a mesh failure releases every
/// buffer so nothing leaks.
fn create_state(
    host: &mut dyn mod_api::ClientHost,
    config: ArenaConfig,
) -> Result<ArenaState, ArenaError> {
    let arena = GpuArena::new(host, config)?;
    match create_quad_mesh(&*host) {
        Ok(quad_mesh) => Ok(ArenaState {
            arena,
            quad_mesh: Some(quad_mesh),
            ids: HashMap::new(),
        }),
        Err(e) => {
            arena.release(host);
            Err(e)
        }
    }
}

/// Releases the quad mesh and every arena buffer, ignoring failures.
fn release_state(host: &mut dyn mod_api::ClientHost, state: ArenaState) {
    if let Some(mesh) = state.quad_mesh {
        let _ = mesh.release(&*host);
    }
    state.arena.release(host);
}

/// The live quad mesh, recreated into `slot` when a Sunrise mesh retirement cleared it.
fn ensure_quad_mesh(
    host: &dyn mod_api::ClientHost,
    slot: &mut Option<mod_api::MeshRef>,
) -> Result<mod_api::MeshRef, ArenaError> {
    if let Some(mesh) = *slot {
        return Ok(mesh);
    }
    let mesh = create_quad_mesh(host)?;
    *slot = Some(mesh);
    Ok(mesh)
}

/// Answers world close / client disconnect (`recipe_properties.rs:37 -> 46` ->
/// `native_lifecycle.rs:232`, `clear_mod_meshes()`), which runs after `client_player_logging_out`
/// on both integrated and remote clients (never on dimension change). Releases best-effort and
/// clears the handle, leaving `GpuArena` intact; `prepare_draws` recreates it on demand.
pub fn drop_quad_mesh(host: &dyn mod_api::ClientHost) {
    if let Some(state) = ARENA.lock().unwrap().as_mut()
        && let Some(old) = state.quad_mesh.take()
    {
        let _ = old.release(host);
    }
}

/// Answers resource reload (`packs/mod.rs:650 -> 742`), where `clear_mod_meshes()` has already
/// retired the handle before `client_resources_reloaded` fires (and `ClientReloadHost` has no
/// `MeshContext`). Clears the handle, leaving `GpuArena` intact.
pub fn clear_quad_mesh() {
    if let Some(state) = ARENA.lock().unwrap().as_mut() {
        state.quad_mesh = None;
    }
}

/// A draw named a mesh Sunrise already retired.
fn is_mesh_stale(err: &mod_api::PipelineDrawError) -> bool {
    matches!(
        err,
        mod_api::PipelineDrawError::Mesh(mod_api::MeshError::Stale)
    )
}

/// `(drawn sections, drawn quads)` for the status line when a terrain draw was prepared;
/// `None` clears the status counters without counting a drawn frame.
fn prepared_frame_counts(
    prepared_terrain_draws: u32,
    section_high_water: u32,
    live_sections: usize,
) -> Option<(u64, u64)> {
    (prepared_terrain_draws > 0).then_some((section_high_water as u64, live_sections as u64))
}

/// Puts `tables` back in the queue unless newer ones were queued meanwhile.
fn restore_tables_if_empty(slot: &mut Option<Tables>, tables: Tables) {
    if slot.is_none() {
        *slot = Some(tables);
    }
}

/// Drops every queued mesh of `dimension`.
fn purge_pending_dimension(pending: &mut Vec<(SectionKey, BuiltSection)>, dimension: &str) {
    pending.retain(|(key, _)| key.0 != dimension);
}

/// The arena is dead: keeps the `queued` tables for the replacement, hands the old state to
/// `teardown_state`, and drops every queued mesh (they may belong to dimensions that unload
/// before the replacement exists, and the replacement is re-filled from the stores).
fn apply_arena_loss<T>(
    arena_slot: &mut Option<T>,
    tables_slot: &mut Option<Tables>,
    pending: &mut Vec<(SectionKey, BuiltSection)>,
    queued: Tables,
    teardown_state: impl FnOnce(T),
) {
    restore_tables_if_empty(tables_slot, queued);
    if let Some(state) = arena_slot.take() {
        teardown_state(state);
    }
    pending.clear();
}

/// The renderer or a handle is gone, so a new arena is needed. Payload errors such as
/// `OutOfRange` are not: an identical arena would refuse the same payload again.
fn is_arena_lost(err: &ArenaError) -> bool {
    matches!(
        err,
        ArenaError::Buffer(
            mod_api::BufferError::Unavailable
                | mod_api::BufferError::Stale
                | mod_api::BufferError::NotOwned
        )
    )
}

/// What became of a table publication.
#[derive(Debug, PartialEq, Eq)]
enum Published {
    Done,
    /// The atlas was refused; keep the arena and try again next tick.
    Retry(Tables),
    /// A write found the arena dead; the tables are kept for the replacement.
    ArenaLost(Tables),
}

/// Writes the tables through the given writers. Write failures go to `on_error` with their
/// site; only a lost arena stops the publication.
fn publish_tables_with(
    tables: Tables,
    mut on_error: impl FnMut(&'static str, ArenaError),
    mut write_models: impl FnMut(&[u32]) -> Result<(), ArenaError>,
    mut write_tints: impl FnMut(&[u32]) -> Result<(), ArenaError>,
    mut write_atlas: impl FnMut(&[u8]) -> bool,
) -> Published {
    if let Err(e) = write_models(&tables.0) {
        let lost = is_arena_lost(&e);
        on_error("write models", e);
        if lost {
            return Published::ArenaLost(tables);
        }
    }
    if let Err(e) = write_tints(&tables.1) {
        let lost = is_arena_lost(&e);
        on_error("write biome tints", e);
        if lost {
            return Published::ArenaLost(tables);
        }
    }
    if write_atlas(&tables.2) {
        Published::Done
    } else {
        Published::Retry(tables)
    }
}

/// Creates the baked face atlas on first use and updates it afterwards. A refused atlas leaves
/// every model on its flat colour: tile size stays in the table, so the caller keeps trying
/// rather than drawing from a missing texture.
fn write_atlas(host: &mut dyn mod_api::ClientHost, atlas: &[u8]) -> bool {
    let id = mod_api::Identifier::parse(ATLAS_TEXTURE).expect("static texture id");
    let side = crate::pipeline::ATLAS_SIDE as u32;
    let done = if ATLAS_CREATED.load(std::sync::atomic::Ordering::Relaxed) {
        host.update_dynamic_texture(&id, side, side, atlas)
    } else {
        host.create_dynamic_texture(&id, side, side, atlas)
    };
    let wrote = done.is_ok();
    if wrote {
        ATLAS_CREATED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    wrote
}

// Lock order: `ARENA` -> `PENDING_TABLES` -> `PENDING_SECTIONS` when nested.
// `PENDING_Y_BOUNDS` is taken alone or immediately under `ARENA`.
// `ARENA` is never held across a call into `crate::pipeline`.
fn try_upload_pending(
    host: &mut dyn mod_api::ClientHost,
    config: ArenaConfig,
) -> Result<(), crate::gpu_arena::ArenaError> {
    let mut guard = ARENA.lock().unwrap();
    let created = guard.is_none();
    if created {
        *guard = Some(create_state(host, config)?);
    }
    let tables = PENDING_TABLES.lock().unwrap().take();
    if let Some(queued) = tables {
        let state = guard.as_ref().unwrap();
        // Each writer needs the host mutably, but only one runs at a time.
        let host_cell = std::cell::RefCell::new(&mut *host);
        let published = publish_tables_with(
            queued,
            |what, e| {
                report_arena::<()>(what, Err(e));
            },
            |words| {
                state
                    .arena
                    .write_models(&mut **host_cell.borrow_mut(), 0, words)
            },
            |words| {
                state
                    .arena
                    .write_biome_tints(&mut **host_cell.borrow_mut(), 0, words)
            },
            |atlas| write_atlas(&mut **host_cell.borrow_mut(), atlas),
        );
        match published {
            Published::Done => {}
            Published::Retry(queued) => {
                restore_tables_if_empty(&mut PENDING_TABLES.lock().unwrap(), queued);
            }
            Published::ArenaLost(queued) => {
                apply_arena_loss(
                    &mut *guard,
                    &mut PENDING_TABLES.lock().unwrap(),
                    &mut PENDING_SECTIONS.lock().unwrap(),
                    queued,
                    |state| release_state(host, state),
                );
                drop(guard);
                crate::pipeline::reset_arena_state();
                return Ok(());
            }
        }
    }
    let state = guard.as_mut().unwrap();
    for (dimension, bounds) in std::mem::take(&mut *PENDING_Y_BOUNDS.lock().unwrap()) {
        state.arena.set_dimension_y_bounds(dimension, bounds);
    }
    // One section failing (arena full, bad write) never drops the rest of the batch: each is
    // handled on its own, logged through `report`, and the metadata flush still runs.
    for (key, section) in std::mem::take(&mut *PENDING_SECTIONS.lock().unwrap()) {
        let old = state.ids.get(&key).copied();
        let [x, y, z] = section.pos;
        let pos = OctreeSectionPos::new(section.level, x, y, z);
        if section.is_empty() {
            if let Some(old) = old {
                if report_arena("remove section", state.arena.remove(old)).is_none() {
                    continue;
                }
                state.ids.remove(&key);
            }
            if section.complete {
                state.arena.mark_empty(key.0, pos);
            } else {
                state.arena.forget_empty(&key.0, pos);
            }
            continue;
        }
        state.arena.forget_empty(&key.0, pos);
        // `upload_replace` keeps the old section live when the new one cannot be stored.
        if let Some(new) = report_arena(
            "upload section",
            state.arena.upload_replace(host, old, &section),
        ) {
            state.ids.insert(key, new);
        }
    }
    // Only the active dimension owns arena section slots.
    let dimension = state.ids.keys().next().map(|key| key.0.as_str());
    let flushed = state.arena.flush(host, dimension);
    drop(guard);
    if created {
        crate::pipeline::mark_models_dirty();
    }
    flushed
}

/// Drops every section of `dimension` (world / dimension change); the buffers stay. Queued
/// meshes of the dimension are dropped even when no arena exists.
pub fn clear_dimension(dimension: &str) -> Result<(), crate::gpu_arena::ArenaError> {
    flush_report_summaries();
    let mut guard = ARENA.lock().unwrap();
    purge_pending_dimension(&mut PENDING_SECTIONS.lock().unwrap(), dimension);
    PENDING_Y_BOUNDS.lock().unwrap().remove(dimension);
    let Some(state) = guard.as_mut() else {
        drop(guard);
        clear_frame_stats();
        return Ok(());
    };
    state.arena.clear_dimension_coverage(Some(dimension));
    let doomed: Vec<_> = state
        .ids
        .keys()
        .filter(|k| k.0 == dimension)
        .cloned()
        .collect();
    let mut result = Ok(());
    for key in doomed {
        let id = state.ids.remove(&key).unwrap();
        if let Err(e) = state.arena.remove(id) {
            result = Err(e);
        }
    }
    drop(guard);
    clear_frame_stats();
    result
}

fn clear_all_pending_and_ids(
    pending: &mut Vec<(SectionKey, BuiltSection)>,
    ids: &mut HashMap<SectionKey, u32>,
    mut remove: impl FnMut(u32) -> Result<(), crate::gpu_arena::ArenaError>,
) -> Result<(), crate::gpu_arena::ArenaError> {
    pending.clear();
    let mut result = Ok(());
    for (_, id) in ids.drain() {
        if let Err(e) = remove(id) {
            result = Err(e);
        }
    }
    result
}

/// Drops every uploaded and pending section across all dimensions (session logout or remote
/// level unload); the reusable arena buffers and tables stay.
pub fn clear_all_sections() -> Result<(), crate::gpu_arena::ArenaError> {
    flush_report_summaries();
    let mut guard = ARENA.lock().unwrap();
    PENDING_Y_BOUNDS.lock().unwrap().clear();
    let mut pending = PENDING_SECTIONS.lock().unwrap();
    let result = match guard.as_mut() {
        Some(state) => {
            state.arena.clear_dimension_coverage(None);
            clear_all_pending_and_ids(&mut pending, &mut state.ids, |id| state.arena.remove(id))
        }
        None => {
            pending.clear();
            Ok(())
        }
    };
    drop(pending);
    drop(guard);
    clear_frame_stats();
    result
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
    // Every early return below leaves SSAO unwritten for the next frame's `ssao_apply`.
    SSAO_WRITTEN.store(false, Ordering::Release);
    let Some(frame) = FRAME.lock().unwrap().clone() else {
        return;
    };
    let guard = ARENA.lock().unwrap();
    let Some(state) = guard.as_ref() else { return };
    let arena = &state.arena;
    let sections = arena.section_high_water();
    if !frame.enabled || sections == 0 || !arena.is_published() {
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
    let focal_px = (h as f32 * 0.5 * camera.projection[5].abs()).max(1.0);
    let lod_threshold = 2.0f32;
    let Some(hiz) = ensure_hiz(host, [w, h]) else {
        return;
    };
    let mips = hiz.mips;
    let view_proj = camera.view_projection();
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
            .f32s(&[lod_threshold, 0.0, 0.0, 0.0])
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
                .texture("HizPyramid", hiz.all_mips());
            report(
                "traversal dispatch_indirect",
                ctx.dispatch_indirect(traversal, &trav_bindings, d_in_ref, 0),
            );
        }
        let clear_bindings =
            DrawBindings::new().storage_buffer("TransSortMeta", slices.trans_sort_meta);
        report(
            "trans_clear dispatch",
            ctx.dispatch(trans_clear, &clear_bindings, [1, 1, 1]),
        );
        let params = cmdgen_params_bytes(view_proj, camera.position, sections, [w, h], mips, true);
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
    // `frame_graph_setup` runs before `prepare_render_buffers` in Dev 2, so compute ensures the
    // `ssao_size` target first and `prepare_draws` ensures the same size before binding it.
    let [sw, sh] = ssao_size([w, h]);
    if frame.config.ssao_enabled
        && let (Ok(ssao_pipe), Some(ssao_target)) = (
            assets.compute_pipeline(&id("distantsunrises:voxy_ssao")),
            ensure_ssao(host, [sw, sh]),
        )
    {
        let clip_dist = vanilla_clip_dist(&frame);
        let ssao_params = Std140(Vec::with_capacity(32))
            .f32s(&[
                camera.projection[0].abs().max(1e-4),
                camera.projection[5].abs().max(1e-4),
                camera.near,
                clip_dist,
            ])
            .i32s(&[sw as i32, sh as i32, w as i32, h as i32])
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
                .storage_image("SsaoOut", ssao_target.dest());
            report(
                "ssao dispatch",
                ctx.dispatch(
                    ssao_pipe,
                    &bindings,
                    [sw.div_ceil(HIZ_WORKGROUP), sh.div_ceil(HIZ_WORKGROUP), 1],
                ),
            );
        });
        if report("add_ssao_compute_pass", added_ssao).is_some() {
            SSAO_WRITTEN.store(true, Ordering::Release);
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
    let Some(frame) = FRAME.lock().unwrap().clone() else {
        clear_frame_stats();
        return;
    };
    let mut guard = ARENA.lock().unwrap();
    let Some(state) = guard.as_mut() else {
        drop(guard);
        clear_frame_stats();
        return;
    };
    let arena = &state.arena;
    let sections = arena.section_high_water();
    if !frame.enabled || sections == 0 || !arena.is_published() {
        drop(guard);
        clear_frame_stats();
        return;
    }
    let Some(quad_mesh) = report_arena(
        "quad mesh recreate",
        ensure_quad_mesh(host, &mut state.quad_mesh),
    ) else {
        drop(guard);
        clear_frame_stats();
        return;
    };
    let mut prepared_terrain_draws = 0u32;
    let assets = host.render_assets();
    let view_proj = camera.view_projection();
    let near_dh = 8.0;
    let rd_blocks = frame.config.render_distance_blocks().max(256.0);
    let far_dh = (rd_blocks * 1.4).max(1024.0);
    let atlas_id = id(ATLAS_TEXTURE);
    let clip_dist = vanilla_clip_dist(&frame);
    // Keep mid-ground hills crisp like Voxy (`NormalRenderPipeline.java`), respecting
    // `fog_enabled` and `fog_spherical`.
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
            0.0, // unused fog.z lane
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
    // Apply compute SSAO multiplicatively onto opaque LODs before sorted translucent water draws.
    // Ensure the same `ssao_size` as `setup_compute` so the draw uses its target.
    if frame.config.ssao_enabled
        && let (Some(win_size), Ok(ssao_apply), Ok(_)) = (
            host.window_size(),
            assets.render_pipeline(&id(PIPELINE_SSAO_APPLY)),
            assets.compute_pipeline(&id("distantsunrises:voxy_ssao")),
        )
        && let Some(ssao_target) = ensure_ssao(host, ssao_size(win_size))
        // Checked after `ensure_ssao`, which clears the flag when it reallocates the target.
        && SSAO_WRITTEN.load(Ordering::Acquire)
    {
        let bindings = DrawBindings::new().texture("SsaoTex", ssao_target.source());
        let drawn = context.draw_mesh(
            TRANSLUCENT_STAGE,
            ssao_apply,
            quad_mesh,
            mod_api::MeshDraw::indexed(0..3, 0),
            &bindings,
        );
        // Proxy reconfiguration retires every mod mesh without `client_player_logging_out` or
        // `client_resources_reloaded` (`raw_ingress.rs:518`, `Event::Reconfiguring` ->
        // `native_lifecycle_world_closed()` -> `clear_mod_meshes()`, sunrise-org/sunrise#6622).
        if let Err(e) = &drawn
            && is_mesh_stale(e)
        {
            state.quad_mesh = None;
        }
        if let Some(batch) = report("ssao_apply draw", drawn) {
            PREPARED.lock().unwrap().push((TRANSLUCENT_STAGE, batch));
        }
    }
    for (stage, plain, pack, cmds, count) in [
        (
            OPAQUE_STAGE,
            PIPELINE_OPAQUE,
            PIPELINE_PACK_TERRAIN,
            arena.opaque_commands,
            arena.opaque_command_count(),
        ),
        (
            TRANSLUCENT_STAGE,
            PIPELINE_TRANSLUCENT,
            PIPELINE_PACK_WATER,
            arena.translucent_commands,
            arena.translucent_command_count(),
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
        let draw = IndirectDraw::new(pipeline, cmds, count, &bindings).with_mesh(quad_mesh);
        let drawn = context.draw_indirect(stage, &draw);
        // Stale after proxy reconfiguration (`raw_ingress.rs:518`, sunrise-org/sunrise#6622).
        if let Err(e) = &drawn
            && is_mesh_stale(e)
        {
            state.quad_mesh = None;
        }
        if let Some(batch) = report("draw_indirect", drawn) {
            PREPARED.lock().unwrap().push((stage, batch));
            prepared_terrain_draws += 1;
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
        let draw = IndirectDraw::new(
            p,
            arena.opaque_commands,
            arena.opaque_command_count(),
            &bindings,
        )
        .with_mesh(quad_mesh);
        let drawn = context.draw_indirect(OPAQUE_STAGE, &draw);
        // Stale after proxy reconfiguration (`raw_ingress.rs:518`, sunrise-org/sunrise#6622).
        if let Err(e) = &drawn
            && is_mesh_stale(e)
        {
            state.quad_mesh = None;
        }
        if let Some(batch) = report("shadow draw_indirect", drawn) {
            PREPARED.lock().unwrap().push((OPAQUE_STAGE, batch));
        }
    }
    let counts = prepared_frame_counts(prepared_terrain_draws, sections, arena.live_sections());
    drop(guard);
    match counts {
        Some((drawn_sections, drawn_quads)) => record_frame_stats(drawn_sections, drawn_quads),
        None => clear_frame_stats(),
    }
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

/// Telemetry counters surfaced on the F3 debug screen (`RegisterDebugEntriesEvent`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DhStats {
    pub active_sections: usize,
    pub rendered_sections: u64,
    pub rendered_quads: u64,
    pub generated_columns: u64,
    pub saved_sections: u64,
    pub frames_drawn: u64,
}

/// Shared runtime state accessible from `Hooks` and the stage renderer.
pub struct DhRuntime {
    pub enabled_flag: AtomicBool,
    pub rendered_sections: AtomicU64,
    pub rendered_quads: AtomicU64,
    pub saved_sections: AtomicU64,
    pub frames_drawn: AtomicU64,
    pub io: LodIoWorker,
    pub gen_pool: GenWorkerPool,
    pub inner: Mutex<DhState>,
}

pub struct DhState {
    pub config: DhConfig,
    pub world_dir: Option<PathBuf>,
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
    /// Rotating index over sorted active `(dimension, player)` targets for full-RD `.dhoct` reload.
    pub disk_reload_cursor: usize,
    /// Earliest instant the next synchronous distant-sampling batch may run for each dimension.
    pub next_distant_sample: HashMap<String, Instant>,
}

/// Cached result of `DimensionLodStore::desired_sections_coarse_to_fine`.
pub struct DesiredCache {
    pub center: [i32; 2],
    pub config: DhConfig,
    pub list: Arc<[SectionPos]>,
    pub disk_cursor: usize,
    pub octree_disk_cursor: usize,
    pub sample_cursor: usize,
}

impl Default for DhState {
    fn default() -> Self {
        Self {
            config: DhConfig::default(),
            world_dir: None,
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
            disk_reload_cursor: 0,
            next_distant_sample: HashMap::new(),
        }
    }
}

impl DhState {
    pub fn clear_dimension_server_state(&mut self, dimension: &str) {
        self.dimensions.remove(dimension);
        self.pending_chunks.remove(dimension);
        self.pending_chunk_set.remove(dimension);
        self.server_player_round_robin.remove(dimension);
        self.next_distant_sample.remove(dimension);
        self.desired_cache.retain(|(dim, _), _| dim != dimension);
    }

    pub fn clear_server_world_state(&mut self) {
        self.world_dir = None;
        self.dimensions.clear();
        self.desired_cache.clear();
        self.disk_reload_cursor = 0;
        self.pending_chunks.clear();
        self.pending_chunk_set.clear();
        self.server_player_round_robin.clear();
        self.next_distant_sample.clear();
    }

    pub fn clear_client_session_state(&mut self) {
        self.camera = None;
        self.has_camera = false;
        self.client_loaded_chunks.clear();
        self.remote_requested.clear();
    }

    fn adopt_io_responses(&mut self, responses: Vec<IoResponse>) -> u64 {
        let mut saved = 0;
        for res in responses {
            match res {
                IoResponse::Saved => saved += 1,
                IoResponse::Loaded { dimension, section } => {
                    let Some(dim) = self.dimensions.get_mut(&dimension) else {
                        continue;
                    };
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
        saved
    }
}

static RUNTIME: OnceLock<Arc<DhRuntime>> = OnceLock::new();

pub fn runtime() -> &'static Arc<DhRuntime> {
    RUNTIME.get_or_init(|| {
        Arc::new(DhRuntime {
            enabled_flag: AtomicBool::new(true),
            rendered_sections: AtomicU64::new(0),
            rendered_quads: AtomicU64::new(0),
            saved_sections: AtomicU64::new(0),
            frames_drawn: AtomicU64::new(0),
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
    st.camera_pos = camera;
    st.has_camera = true;

    // Drain under rt.inner, matching close's lock order, so a pre-bump reply cannot
    // repopulate retired state. Adoption drops replies for unloaded dimensions.
    let saved = st.adopt_io_responses(rt.io.drain_responses());
    if saved > 0 {
        rt.saved_sections.fetch_add(saved, Ordering::Relaxed);
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

/// Called by `record_stage` at the end of a frame.
pub fn record_frame_stats(rendered_sections: u64, rendered_quads: u64) {
    let rt = runtime();
    rt.rendered_sections
        .store(rendered_sections, Ordering::Relaxed);
    rt.rendered_quads.store(rendered_quads, Ordering::Relaxed);
    rt.frames_drawn.fetch_add(1, Ordering::Relaxed);
}

/// Clears the status line's drawn counts without counting a drawn frame (unload or no-draw).
pub fn clear_frame_stats() {
    let rt = runtime();
    rt.rendered_sections.store(0, Ordering::Relaxed);
    rt.rendered_quads.store(0, Ordering::Relaxed);
}

/// Current arena occupancy, independent of whether the last frame prepared a draw.
pub(crate) fn arena_usage() -> (usize, u64) {
    ARENA.lock().unwrap().as_ref().map_or((0, 0), |state| {
        (
            state.arena.live_sections(),
            state.arena.used_geometry_bytes(),
        )
    })
}

/// Snapshot stats for the F3 debug screen.
pub fn stats() -> DhStats {
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
        rendered_sections: rt.rendered_sections.load(Ordering::Relaxed),
        rendered_quads: rt.rendered_quads.load(Ordering::Relaxed),
        generated_columns,
        saved_sections: rt.saved_sections.load(Ordering::Relaxed),
        frames_drawn: rt.frames_drawn.load(Ordering::Relaxed),
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

const REPORT_INITIAL_BURST: u64 = 5;
type ReportCause = (&'static str, &'static str);

#[derive(Default)]
struct ReportCount {
    total: u64,
    reported: u64,
}

#[derive(Default)]
struct Reports(HashMap<ReportCause, ReportCount>);

impl Reports {
    fn record(&mut self, cause: ReportCause) -> Option<(u64, u64)> {
        let count = self.0.entry(cause).or_default();
        count.total += 1;
        let total = count.total;
        if total <= REPORT_INITIAL_BURST || is_summary_milestone(total) {
            let suppressed = total - count.reported - 1;
            count.reported = total;
            Some((total, suppressed))
        } else {
            None
        }
    }

    fn flush(&mut self, mut emit: impl FnMut(ReportCause, u64, u64)) {
        for (&cause, count) in &mut self.0 {
            if count.total > count.reported {
                emit(cause, count.total, count.total - count.reported);
                count.reported = count.total;
            }
        }
    }
}

fn is_summary_milestone(mut total: u64) -> bool {
    if total < 10 {
        return false;
    }
    while total.is_multiple_of(10) {
        total /= 10;
    }
    matches!(total, 1 | 2 | 5)
}

static REPORTS: LazyLock<Mutex<Reports>> = LazyLock::new(|| Mutex::new(Reports::default()));

pub(crate) fn report_warning(
    what: &'static str,
    cause: &'static str,
    detail: std::fmt::Arguments<'_>,
) {
    if let Some((total, suppressed)) = REPORTS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .record((what, cause))
    {
        log::warn!(
            "distantsunrises: {what} ({cause}): {detail}; total={total}, suppressed={suppressed}"
        );
    }
}

/// Reports errors suppressed since the last warning, retaining cumulative totals.
pub fn flush_report_summaries() {
    REPORTS.lock().unwrap_or_else(|p| p.into_inner()).flush(|(what, cause), total, suppressed| {
        log::warn!("distantsunrises: {what} ({cause}): final summary; total={total}, suppressed={suppressed}");
    });
}

fn report<T, E: std::fmt::Debug>(what: &'static str, result: Result<T, E>) -> Option<T> {
    result
        .inspect_err(|e| report_warning(what, what, format_args!("{e:?}")))
        .ok()
}

fn report_arena<T>(what: &'static str, result: Result<T, ArenaError>) -> Option<T> {
    result
        .inspect_err(|e| {
            let cause = match e {
                ArenaError::GeometryFull => "GeometryFull",
                ArenaError::SectionsFull => "SectionsFull",
                ArenaError::NodesFull => "NodesFull",
                ArenaError::EmptySection => "EmptySection",
                ArenaError::UnknownSection(_) => "UnknownSection",
                ArenaError::Buffer(_) => "Buffer",
            };
            report_warning(what, cause, format_args!("{e:?}"));
        })
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mod_api::BufferError;
    use std::cell::Cell;

    #[test]
    fn ssao_size_rounds_up_to_half_resolution() {
        assert_eq!(ssao_size([1920, 1080]), [960, 540]);
        assert_eq!(ssao_size([1921, 1081]), [961, 541]);
        assert_eq!(ssao_size([1, 1]), [1, 1]);
    }

    #[test]
    fn dh_state_clears_dimension_world_and_client_session_state_preserving_api_hooks() {
        use crate::api::{LodGenOutput, LodGenRequest, LodSectionPos};

        struct Hooks;
        impl LodWorldGenerator for Hooks {
            fn generate(&self, _: &LodGenRequest, _: &mut dyn FnMut(LodGenOutput)) {}
        }
        impl RemoteLodProvider for Hooks {
            fn request_section(&self, _: &str, _: LodSectionPos, _: i64) {}
        }

        let config = DhConfig {
            enabled: false,
            ..DhConfig::default()
        };
        let (unloaded, active) = ("test:unloaded", "test:active");
        let camera = CameraMatrices {
            position: [1.0, 2.0, 3.0],
            yaw: 30.0,
            pitch: 20.0,
            fov_degrees: 70.0,
            near: 0.1,
            far: 100.0,
            view: [0.0; 16],
            projection: [0.0; 16],
        };
        let path = PathBuf::from("test-world");
        let pos = SectionPos::new(0, 0, 0);
        let chunks = HashSet::from([[2, 3]]);
        let mut st = DhState {
            config,
            world_dir: Some(path.clone()),
            active_dimension: active.into(),
            camera: Some(camera),
            camera_pos: camera.position,
            has_camera: true,
            fog_color: [0.1, 0.2, 0.3, 1.0],
            vanilla_far_blocks: 512.0,
            sun_brightness: 0.5,
            rain_level: 0.25,
            thunder_level: 0.75,
            vanilla_render_distance_chunks: 32,
            client_loaded_chunks: chunks.clone(),
            disk_reload_cursor: 3,
            ..DhState::default()
        };
        let generator: Arc<dyn LodWorldGenerator> = Arc::new(Hooks);
        let provider: Arc<dyn RemoteLodProvider> = Arc::new(Hooks);
        for dim in [unloaded, active].map(str::to_owned) {
            st.dimensions.entry(dim.clone()).or_default();
            st.pending_chunks
                .insert(dim.clone(), chunks.iter().copied().collect());
            st.pending_chunk_set.insert(dim.clone(), chunks.clone());
            st.server_player_round_robin.insert(dim.clone(), 3);
            st.next_distant_sample.insert(dim.clone(), Instant::now());
            st.desired_cache.insert(
                (dim.clone(), 0),
                DesiredCache {
                    center: [2, 3],
                    config,
                    list: Arc::from([pos]),
                    disk_cursor: 1,
                    octree_disk_cursor: 1,
                    sample_cursor: 0,
                },
            );
            st.world_generators.insert(dim.clone(), generator.clone());
            st.remote_providers.insert(dim.clone(), provider.clone());
            st.disabled_builtin_gen.insert(dim.clone());
            st.disabled_builtin_net.insert(dim.clone());
            st.remote_requested.insert(dim, HashSet::from([pos]));
        }

        fn keys<V>(map: &HashMap<String, V>) -> Vec<&str> {
            map.keys().map(String::as_str).collect()
        }

        st.clear_dimension_server_state(unloaded);
        assert_eq!(st.disk_reload_cursor, 3);
        for expected in [Some(active), None] {
            assert_eq!(st.world_dir.as_deref(), expected.map(|_| path.as_path()));
            for actual in [
                keys(&st.dimensions),
                keys(&st.pending_chunks),
                keys(&st.pending_chunk_set),
                keys(&st.server_player_round_robin),
                keys(&st.next_distant_sample),
                st.desired_cache
                    .keys()
                    .map(|(dim, _)| dim.as_str())
                    .collect(),
            ] {
                assert!(actual.into_iter().eq(expected));
            }
            assert_eq!(st.camera, Some(camera));
            assert!(st.has_camera);
            assert_eq!(st.client_loaded_chunks, chunks);
            for dim in [active, unloaded] {
                assert_eq!(st.remote_requested[dim], HashSet::from([pos]));
            }
            if expected.is_some() {
                st.clear_server_world_state();
                assert_eq!(st.disk_reload_cursor, 0);
            }
        }
        st.clear_client_session_state();
        assert!(!st.has_camera);
        assert!(st.camera.is_none());
        assert!(st.client_loaded_chunks.is_empty());
        assert!(st.remote_requested.is_empty());
        assert_eq!(st.config, config);
        assert_eq!(st.active_dimension, active);
        assert_eq!(st.camera_pos, camera.position);
        assert_eq!(st.fog_color, [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(st.vanilla_far_blocks, 512.0);
        assert_eq!(st.sun_brightness, 0.5);
        assert_eq!((st.rain_level, st.thunder_level), (0.25, 0.75));
        assert_eq!(st.vanilla_render_distance_chunks, 32);
        for dim in [unloaded, active] {
            assert!(Arc::ptr_eq(&st.world_generators[dim], &generator));
            assert!(Arc::ptr_eq(&st.remote_providers[dim], &provider));
            assert!(st.disabled_builtin_gen.contains(dim));
            assert!(st.disabled_builtin_net.contains(dim));
        }
    }

    #[test]
    fn adopt_io_responses_drops_loaded_section_for_unloaded_dimension() {
        use crate::lod::LodSection;

        let (reopened, active) = ("custom:reopened", "minecraft:overworld");
        let pos = SectionPos::new(0, 0, 0);
        let dir = crate::storage::fresh_tmp_dir("reopen-column-store");
        let io = LodIoWorker::spawn();
        let mut st = DhState::default();
        for dim in [reopened, active] {
            st.dimensions.entry(dim.into()).or_default();
            io.request_save(dir.clone(), dim.into(), LodSection::new(pos));
            io.request_load(dir.clone(), dim.into(), pos);
        }
        io.flush();
        st.clear_dimension_server_state(reopened);
        st.adopt_io_responses(vec![IoResponse::Loaded {
            dimension: reopened.into(),
            section: LodSection::new(pos),
        }]);
        assert!(!st.dimensions.contains_key(reopened));

        io.discard_dimension_load_replies(reopened);
        st.dimensions.entry(reopened.into()).or_default();
        let saved = st.adopt_io_responses(io.drain_responses());
        drop(io);
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(saved, 2);
        assert!(st.dimensions[reopened].sections.is_empty());
        assert!(st.dimensions[active].sections.contains_key(&pos));
    }

    fn tables() -> Tables {
        (vec![1, 2], vec![3], vec![4, 5, 6])
    }

    fn buffer_err(e: BufferError) -> Result<(), ArenaError> {
        Err(ArenaError::Buffer(e))
    }

    /// `(site, Debug of the error)` pairs, since `ArenaError` is not comparable.
    fn sites(errors: &[(&'static str, ArenaError)]) -> Vec<(&'static str, String)> {
        errors.iter().map(|(w, e)| (*w, format!("{e:?}"))).collect()
    }

    fn mesh() -> BuiltSection {
        BuiltSection {
            level: 0,
            pos: [0; 3],
            aabb: crate::gpu_arena::SectionAabb::FULL,
            quads: vec![crate::gpu_arena::PackedQuad::default()],
            counts: [1, 0, 0, 0, 0, 0, 0, 0],
            complete: false,
        }
    }

    fn key(dim: &str) -> SectionKey {
        (dim.to_owned(), 0u8, [0; 3])
    }

    fn pending_dims(pending: &[(SectionKey, BuiltSection)]) -> Vec<&str> {
        pending.iter().map(|(k, _)| k.0.as_str()).collect()
    }

    #[test]
    fn publish_tables_fatal_buffer_error_marks_arena_lost_and_skips_later_writes() {
        let fatal = |i: u8| match i {
            0 => BufferError::Stale,
            1 => BufferError::Unavailable,
            _ => BufferError::NotOwned,
        };
        for kind in 0..3 {
            for on_models in [true, false] {
                let (tints, atlas) = (Cell::new(0), Cell::new(0));
                let mut errors = Vec::new();
                let published = publish_tables_with(
                    tables(),
                    |what, e| errors.push((what, e)),
                    |_| {
                        if on_models {
                            buffer_err(fatal(kind))
                        } else {
                            Ok(())
                        }
                    },
                    |_| {
                        tints.set(tints.get() + 1);
                        if on_models {
                            Ok(())
                        } else {
                            buffer_err(fatal(kind))
                        }
                    },
                    |_| {
                        atlas.set(atlas.get() + 1);
                        true
                    },
                );
                assert_eq!(published, Published::ArenaLost(tables()));
                assert_eq!(tints.get(), u32::from(!on_models));
                assert_eq!(atlas.get(), 0);
                let site = if on_models {
                    "write models"
                } else {
                    "write biome tints"
                };
                assert_eq!(
                    sites(&errors),
                    [(site, format!("{:?}", ArenaError::Buffer(fatal(kind))))]
                );
            }
        }
    }

    #[test]
    fn publish_tables_out_of_range_keeps_arena_without_retry_loop() {
        for on_models in [true, false] {
            let atlas = Cell::new(0);
            let mut errors = Vec::new();
            let published = publish_tables_with(
                tables(),
                |what, e| errors.push((what, e)),
                |_| {
                    if on_models {
                        buffer_err(BufferError::OutOfRange)
                    } else {
                        Ok(())
                    }
                },
                |_| {
                    if on_models {
                        Ok(())
                    } else {
                        buffer_err(BufferError::OutOfRange)
                    }
                },
                |_| {
                    atlas.set(atlas.get() + 1);
                    true
                },
            );
            assert_eq!(published, Published::Done);
            assert_eq!(atlas.get(), 1);
            let site = if on_models {
                "write models"
            } else {
                "write biome tints"
            };
            assert_eq!(
                sites(&errors),
                [(
                    site,
                    format!("{:?}", ArenaError::Buffer(BufferError::OutOfRange))
                )]
            );
        }
    }

    #[test]
    fn publish_tables_refused_atlas_retries_without_losing_arena() {
        let mut errors = Vec::new();
        let published = publish_tables_with(
            tables(),
            |what, e| errors.push((what, e)),
            |_| Ok(()),
            |_| Ok(()),
            |_| false,
        );
        assert_eq!(published, Published::Retry(tables()));
        assert!(errors.is_empty());
    }

    #[test]
    fn restore_tables_if_empty_keeps_newer_queued_tables() {
        let mut slot = None;
        restore_tables_if_empty(&mut slot, tables());
        assert_eq!(slot, Some(tables()));
        let newer = (vec![9], vec![], vec![]);
        let mut slot = Some(newer.clone());
        restore_tables_if_empty(&mut slot, tables());
        assert_eq!(slot, Some(newer));
    }

    #[test]
    fn clear_all_pending_and_ids_retires_every_dimension() {
        let overworld = key("minecraft:overworld");
        let nether = key("minecraft:the_nether");
        let mut pending = vec![(overworld.clone(), mesh()), (nether.clone(), mesh())];
        let mut ids = HashMap::from([(overworld, 1), (nether, 2)]);
        let mut attempted = Vec::new();

        let result = clear_all_pending_and_ids(&mut pending, &mut ids, |id| {
            attempted.push(id);
            // Fail the first removal regardless of the map's iteration order.
            if attempted.len() == 1 {
                Err(ArenaError::UnknownSection(id))
            } else {
                Ok(())
            }
        });

        assert!(pending.is_empty());
        assert!(ids.is_empty());
        assert!(
            matches!(result, Err(ArenaError::UnknownSection(id)) if Some(&id) == attempted.first())
        );
        attempted.sort_unstable();
        assert_eq!(attempted, [1, 2]);
    }

    #[test]
    fn arena_loss_and_dimension_unload_leave_no_stale_meshes_for_recovery() {
        let mut arena_slot = Some(1u32);
        let mut tables_slot: Option<Tables> = None;
        let mut pending = vec![(key("dim_a"), mesh()), (key("dim_b"), mesh())];
        let mut released = Vec::new();
        let mut errors = Vec::new();

        // The table write finds the arena dead.
        let lost = publish_tables_with(
            tables(),
            |what, e| errors.push((what, e)),
            |_| buffer_err(BufferError::Stale),
            |_| Ok(()),
            |_| true,
        );
        let Published::ArenaLost(queued) = lost else {
            panic!("expected ArenaLost, got {lost:?}");
        };
        assert_eq!(
            sites(&errors),
            [(
                "write models",
                format!("{:?}", ArenaError::Buffer(BufferError::Stale))
            )]
        );

        apply_arena_loss(
            &mut arena_slot,
            &mut tables_slot,
            &mut pending,
            queued,
            |state| released.push(state),
        );
        assert!(arena_slot.is_none());
        assert_eq!(released, [1]);
        assert_eq!(tables_slot, Some(tables()));
        assert!(pending.is_empty());

        // Before recovery, meshes arrive and one dimension unloads.
        pending.push((key("dim_a"), mesh()));
        pending.push((key("dim_b"), mesh()));
        purge_pending_dimension(&mut pending, "dim_a");

        // The replacement arena takes the kept tables.
        arena_slot = Some(2u32);
        let recovered = publish_tables_with(
            tables_slot.take().unwrap(),
            |what, e| errors.push((what, e)),
            |_| Ok(()),
            |_| Ok(()),
            |_| true,
        );
        assert_eq!(recovered, Published::Done);
        assert_eq!(errors.len(), 1);
        assert_eq!(arena_slot, Some(2));
        assert_eq!(released, [1]);
        assert_eq!(pending_dims(&pending), ["dim_b"]);
    }

    #[test]
    fn prepared_frame_counts_skips_frame_when_no_terrain_draw_was_prepared() {
        assert_eq!(prepared_frame_counts(0, 7, 5), None);
        assert_eq!(prepared_frame_counts(1, 7, 5), Some((7, 5)));
        assert_eq!(prepared_frame_counts(2, 7, 5), Some((7, 5)));
    }

    #[test]
    fn is_mesh_stale_matches_only_a_retired_quad_mesh() {
        use mod_api::{MeshError, PipelineDrawError};
        assert!(is_mesh_stale(&PipelineDrawError::Mesh(MeshError::Stale)));
        assert!(!is_mesh_stale(&PipelineDrawError::Mesh(
            MeshError::Unavailable
        )));
        assert!(!is_mesh_stale(&PipelineDrawError::InvalidRange));
        assert!(!is_mesh_stale(&PipelineDrawError::Buffer(
            BufferError::Stale
        )));
    }

    #[test]
    fn reporting_counts_each_site_and_cause_independently() {
        let mut reports = Reports::default();
        for total in 1..=REPORT_INITIAL_BURST {
            assert_eq!(reports.record(("upload", "GeometryFull")), Some((total, 0)));
        }
        assert_eq!(reports.record(("upload", "GeometryFull")), None);
        assert_eq!(reports.record(("upload", "SectionsFull")), Some((1, 0)));
        assert_eq!(reports.record(("replace", "GeometryFull")), Some((1, 0)));
    }

    #[test]
    fn milestones_follow_the_exact_one_two_five_cadence() {
        let milestones: Vec<_> = (0..=1000).filter(|&n| is_summary_milestone(n)).collect();
        assert_eq!(milestones, [10, 20, 50, 100, 200, 500, 1000]);
    }

    #[test]
    fn flush_reports_the_final_count_once_between_milestones() {
        let mut reports = Reports::default();
        let cause = ("upload", "GeometryFull");
        for total in 1..=63 {
            let expected = match total {
                1..=5 => Some((total, 0)),
                10 => Some((10, 4)),
                20 => Some((20, 9)),
                50 => Some((50, 29)),
                _ => None,
            };
            assert_eq!(reports.record(cause), expected);
        }
        let mut flushed = Vec::new();
        for _ in 0..2 {
            reports.flush(|key, total, suppressed| flushed.push((key, total, suppressed)));
        }
        assert_eq!(flushed, [(cause, 63, 13)]);
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
