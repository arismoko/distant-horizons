//! Thread-safe bridge between Distant Sunrises mod hooks, background workers, and the GPU-driven
//! renderer (`gpu_arena`, GLSL shaders under `assets/distantsunrises/shaders/`).
//!
//! No `wgpu` types and no `[native.*]`: everything goes through Sunrise's general render API.

use crate::api::{LodWorldGenerator, RemoteLodProvider};
use crate::config::DhConfig;
use crate::gen_worker::GenWorkerPool;
use crate::generic_render::{BeaconTracker, CloudField, GenericGroupMesh};
use crate::lod::{DimensionLodStore, SectionPos};
use crate::mesh::{MeshWorker, SectionMeshData};
use crate::network::SessionConfig;
use crate::stage3_camera::CameraMatrices;
use crate::storage::{IoResponse, LodIoWorker};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
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
//   4. `voxy_composite.vsh/fsh` composites the LOD target over vanilla sky pixels.
// ---------------------------------------------------------------------------------------------

use crate::gpu_arena::{ArenaConfig, BuiltSection, GpuArena};

/// Shader ids (`assets/distantsunrises/shaders/{core,compute}/<path>`).
pub const SHADER_QUADS: &str = "distantsunrises:core/voxy_quads";
pub const SHADER_COMPOSITE: &str = "distantsunrises:core/voxy_composite";
pub const SHADER_HIZ: &str = "distantsunrises:voxy_hiz";
pub const SHADER_CMDGEN: &str = "distantsunrises:voxy_cmdgen";

pub const PIPELINE_OPAQUE: &str = "distantsunrises:voxy_opaque";
pub const PIPELINE_TRANSLUCENT: &str = "distantsunrises:voxy_translucent";
pub const PIPELINE_COMPOSITE: &str = "distantsunrises:voxy_composite";
pub const PIPELINE_PACK_TERRAIN: &str = "distantsunrises:voxy_pack_terrain";
pub const PIPELINE_PACK_WATER: &str = "distantsunrises:voxy_pack_water";
pub const PIPELINE_PACK_SHADOW: &str = "distantsunrises:voxy_pack_shadow";

/// Compute workgroup sizes; must match `local_size_*` in the `.csh` files.
pub const CMDGEN_WORKGROUP: u32 = 128;
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
            SamplerFilter::Nearest,
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
            SamplerFilter::Nearest,
            SamplerAddress::ClampToEdge,
        )
        .build()
        .expect("static pack layout")
}

/// Registers the render pipelines. Called from `Hooks::register_render_pipelines`. The compute
/// pipelines (Hi-Z, command generation) are added with the compute primitives.
pub fn register_pipelines(event: &mut mod_api::RegisterRenderPipelinesEvent<'_>) {
    use mod_api::{
        PipelineBlend, PrimitiveTopology, RenderPipeline, SamplerAddress, SamplerFilter, WorldDepth,
    };
    // Opaque: depth-tested, culled by the index order of the quad mesh; translucent: sorted by the
    // arena's bucket order, blended, no depth write.
    event.register(
        RenderPipeline::new(
            id(PIPELINE_OPAQUE),
            id(SHADER_QUADS),
            PipelineBlend::Opaque,
            WorldDepth::Tested,
        )
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
        .depth_write(false)
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
            .topology(PrimitiveTopology::Triangles)
            .cull(false)
            .bind_group_layout(pack_layout())
            .pack_binding(slot.pack_binding())
            .optional(),
        );
    }
    register_compute(event);
    event.register(
        RenderPipeline::new(
            id(PIPELINE_COMPOSITE),
            id(SHADER_COMPOSITE),
            PipelineBlend::Opaque,
            WorldDepth::ThroughWalls,
        )
        .topology(PrimitiveTopology::Triangles)
        .bind_group_layout(
            mod_api::BindGroupLayout::builder()
                .uniform_buffer("CompositeParams", 176)
                .sampler(
                    "LodColor",
                    SamplerFilter::Nearest,
                    SamplerAddress::ClampToEdge,
                )
                .sampler(
                    "LodDepth",
                    SamplerFilter::Nearest,
                    SamplerAddress::ClampToEdge,
                )
                .sampler(
                    "WorldDepth",
                    SamplerFilter::Nearest,
                    SamplerAddress::ClampToEdge,
                )
                .build()
                .expect("static composite layout"),
        ),
    );
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
        ComputePipeline::builder(id("distantsunrises:voxy_cmdgen"), id(SHADER_CMDGEN))
            .bind_group_layout(
                BindGroupLayout::builder()
                    .uniform_buffer("CmdgenParams", 112)
                    .entry(storage("SectionBuffer", true, 32))
                    .entry(storage("OpaqueCommands", false, 20))
                    .entry(storage("TranslucentCommands", false, 20))
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
fn ensure_hiz(host: &dyn mod_api::ClientHost, size: [u32; 2]) -> Option<(HizTarget, bool)> {
    use mod_api::{RenderTargetDescriptor, RenderTargetFormat, StorageImageAccess};
    let mut slot = HIZ.lock().unwrap();
    if let Some(h) = slot.as_ref().filter(|h| h.size == size) {
        return Some((HizTarget { ..*h }, false));
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
    Some((made, true))
}

/// Whether the Hi-Z sampler can filter linearly. `R32Float` without float32 filtering must be
/// bound `Nearest` (the cmdgen layout always does), so this only reports the adapter.
pub fn hiz_linear_filter(host: &dyn mod_api::ClientHost) -> bool {
    host.render_capabilities().float32_filterable
}

/// Sections built off-thread and waiting for a `&mut ClientHost` to reach the arena.
static PENDING_SECTIONS: Mutex<Vec<(SectionKey, BuiltSection)>> = Mutex::new(Vec::new());
static ARENA: Mutex<Option<ArenaState>> = Mutex::new(None);

/// Identity of an uploaded section: `(dimension, lod level, section position)`.
pub type SectionKey = (String, u8, [i32; 3]);

struct ArenaState {
    arena: GpuArena,
    /// Shared quad index pattern (`0,1,2,2,1,3` per quad) the indirect commands index into.
    quad_mesh: mod_api::MeshRef,
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

/// Called by the bakery / mesher worker for each finished section (empty = remove).
pub fn submit_built_section(key: SectionKey, section: BuiltSection) {
    PENDING_SECTIONS.lock().unwrap().push((key, section));
}

/// Most quads one indirect command can span (a 32³ section's worth of faces in one bucket).
const MAX_QUADS_PER_COMMAND: u32 = 1 << 16;

/// A mesh holding only the shared quad index buffer; its one vertex binding is never read (the
/// vertex stages pull quads from the arena by `gl_VertexID`). It has the stride of the default
/// `position_tex_color` binding, which a pipeline with no declared binding must be drawn with.
fn create_quad_mesh(
    host: &mut dyn mod_api::ClientHost,
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
/// uploads pending sections and flushes metadata. Call from `client_tick_pre`.
pub fn upload_pending(
    host: &mut dyn mod_api::ClientHost,
    config: ArenaConfig,
) -> Result<(), crate::gpu_arena::ArenaError> {
    let mut guard = ARENA.lock().unwrap();
    if guard.is_none() {
        let quad_mesh = create_quad_mesh(host)?;
        *guard = Some(ArenaState {
            arena: GpuArena::new(host, config)?,
            quad_mesh,
            ids: HashMap::new(),
        });
    }
    let state = guard.as_mut().unwrap();
    if let Some((models, tints, atlas)) = PENDING_TABLES.lock().unwrap().take() {
        state.arena.write_models(host, 0, &models)?;
        state.arena.write_biome_tints(host, 0, &tints)?;
        let id = mod_api::Identifier::parse(ATLAS_TEXTURE).expect("static texture id");
        let side = crate::pipeline::ATLAS_SIDE as u32;
        let done = if ATLAS_CREATED.load(std::sync::atomic::Ordering::Relaxed) {
            host.update_dynamic_texture(&id, side, side, &atlas)
        } else {
            host.create_dynamic_texture(&id, side, side, &atlas)
        };
        // A refused atlas leaves every model on its flat colour: tile size stays in the table, so
        // keep trying next time rather than drawing from a missing texture.
        if done.is_ok() {
            ATLAS_CREATED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    for (key, section) in std::mem::take(&mut *PENDING_SECTIONS.lock().unwrap()) {
        let old = state.ids.remove(&key);
        if section.is_empty() {
            if let Some(old) = old {
                state.arena.remove(old)?;
            }
            continue;
        }
        let id = state.arena.upload_replace(host, old, &section)?;
        state.ids.insert(key, id);
    }
    state.arena.flush(host)
}

/// Drops every section of `dimension` (world / dimension change); the buffers stay.
pub fn clear_dimension(dimension: &str) -> Result<(), crate::gpu_arena::ArenaError> {
    let mut guard = ARENA.lock().unwrap();
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    let doomed: Vec<_> = state
        .ids
        .keys()
        .filter(|k| k.0 == dimension)
        .cloned()
        .collect();
    for key in doomed {
        let id = state.ids.remove(&key).unwrap();
        state.arena.remove(id)?;
    }
    Ok(())
}

/// Frees the arena's GPU buffers (mod unload / re-init only).
pub fn release_arena(
    host: &mut dyn mod_api::ClientHost,
) -> Result<(), crate::gpu_arena::ArenaError> {
    match ARENA.lock().unwrap().take() {
        Some(state) => state.arena.release(host),
        None => Ok(()),
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
    let (frame, _, _, _) = begin_render_frame(camera);
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
    let ids = ["voxy_hiz_copy", "voxy_hiz_reduce", "voxy_cmdgen"]
        .map(|n| assets.compute_pipeline(&id(&format!("distantsunrises:{n}"))));
    let [Ok(copy), Ok(reduce), Ok(cmdgen)] = ids else {
        return;
    };
    let Some((hiz, _)) = ensure_hiz(host, [w, h]) else {
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
        let params = cmdgen_params_bytes(view_proj, camera.position, sections, [w, h], mips, true);
        let bindings = DrawBindings::new()
            .inline_uniform("CmdgenParams", &params)
            .storage_buffer("SectionBuffer", slices.sections)
            .storage_buffer("OpaqueCommands", slices.opaque_commands)
            .storage_buffer("TranslucentCommands", slices.translucent_commands)
            .texture("HizPyramid", hiz.all_mips());
        report(
            "cmdgen dispatch",
            ctx.dispatch(
                cmdgen,
                &bindings,
                [sections.div_ceil(CMDGEN_WORKGROUP), 1, 1],
            ),
        );
    });
    report("add_compute_pass", added);
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
        return;
    };
    let guard = ARENA.lock().unwrap();
    let Some(state) = guard.as_ref() else { return };
    let arena = &state.arena;
    let sections = arena.section_high_water();
    if !frame.enabled || sections == 0 {
        return;
    }
    let assets = host.render_assets();
    let view_proj = camera.view_projection();
    let near_dh = 8.0;
    let far_dh = (frame.config.render_distance_blocks() * 1.4).max(1024.0);
    let atlas_id = id(ATLAS_TEXTURE);
    let quad_params = quad_params_bytes(
        view_proj,
        camera.position,
        near_dh,
        far_dh,
        (frame.vanilla_far_blocks * frame.config.overdraw_prevention).max(0.0),
        frame.sun_brightness,
        // DH's own far fog: vanilla's environmental end is about the vanilla render distance, where
        // the LOD starts, so using it would fog every LOD quad into the sky colour.
        [far_dh * 0.75, far_dh, 0.0, 0.0],
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
        let draw = IndirectDraw::new(pipeline, cmds, count, &bindings).with_mesh(state.quad_mesh);
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
        let draw = IndirectDraw::new(
            p,
            arena.opaque_commands,
            arena.opaque_command_count(),
            &bindings,
        )
        .with_mesh(state.quad_mesh);
        if let Some(batch) = report(
            "shadow draw_indirect",
            context.draw_indirect(OPAQUE_STAGE, &draw),
        ) {
            PREPARED.lock().unwrap().push((OPAQUE_STAGE, batch));
        }
    }
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed).is_multiple_of(300) {
        eprintln!(
            "distantsunrises: arena sections high-water {sections}, live {}, opaque commands {}, quad bytes used {}, vanilla far {} blocks",
            arena.live_sections(),
            arena.opaque_command_count(),
            arena.used_geometry_bytes(),
            frame.vanilla_far_blocks
        );
    }
    record_frame_stats(sections as u64, arena.live_sections() as u64);
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
    pub fog_type: u32,
    pub env_fog_start: f32,
    pub env_fog_end: f32,
    pub vanilla_far_blocks: f32,
    pub sun_brightness: f32,
    pub visible_sections: Vec<SectionPos>,
    pub generic_active: Vec<(String, [f64; 3])>,
}

/// Telemetry counters surfaced on the F3 debug screen (`RegisterDebugEntriesEvent`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DhStats {
    pub active_sections: usize,
    pub ready_meshes: usize,
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
    pub mesher: MeshWorker,
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
    pub improved_transparency: bool,
    pub visible_section_count: usize,
    pub fog_color: [f32; 4],
    pub fog_type: u32,
    pub env_fog_start: f32,
    pub env_fog_end: f32,
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
    /// Ready CPU meshes awaiting upload to GPU (`world_drawn` drains this).
    pub pending_gpu_uploads: Vec<SectionMeshData>,
    /// Set of section positions known to have a valid non-empty mesh for each dimension.
    pub meshed_sections: HashMap<String, HashSet<SectionPos>>,
    /// Counter bumped when active dimension resets so the GPU cache drops stale buffers.
    pub gpu_reset_epoch: u64,
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
    /// Server-enforced multiplayer session policy received via `MSG_SESSION_CONFIG`.
    pub remote_session_config: Option<SessionConfig>,
    /// Whether `MSG_SESSION_CONFIG_REQUEST` has been sent for the current remote session.
    pub remote_config_requested: bool,
    /// Per-server local disk cache root when connected to a remote multiplayer server.
    pub remote_cache_dir: Option<PathBuf>,
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
    /// Generic box groups (beacons, clouds, and third-party mod groups) per dimension.
    pub generic_groups: HashMap<String, HashMap<String, GenericGroupMesh>>,
    pub generic_groups_revision: u64,
    pub last_uploaded_generic_revision: u64,
    /// Per-dimension distant beacon trackers.
    pub beacons: HashMap<String, BeaconTracker>,
    /// Procedural 3D LOD cloud generator.
    pub cloud_field: CloudField,
    /// Start time for smooth LOD cloud drift.
    pub start_instant: Instant,
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
            active_dimension: "minecraft:overworld".to_owned(),
            camera: None,
            camera_pos: [0.0, 64.0, 0.0],
            has_camera: false,
            improved_transparency: false,
            visible_section_count: 0,
            fog_color: [0.75, 0.85, 1.0, 1.0],
            fog_type: 0,
            env_fog_start: 0.0,
            env_fog_end: 64.0,
            vanilla_far_blocks: 80.0,
            sun_brightness: 1.0,
            rain_level: 0.0,
            thunder_level: 0.0,
            vanilla_render_distance_chunks: 12,
            client_loaded_chunks: HashSet::new(),
            dimensions: HashMap::new(),
            pending_chunks: HashMap::new(),
            pending_chunk_set: HashMap::new(),
            pending_gpu_uploads: Vec::new(),
            meshed_sections: HashMap::new(),
            gpu_reset_epoch: 0,
            world_generators: HashMap::new(),
            remote_providers: HashMap::new(),
            disabled_builtin_gen: HashSet::new(),
            disabled_builtin_net: HashSet::new(),
            remote_requested: HashMap::new(),
            remote_session_config: None,
            remote_config_requested: false,
            remote_cache_dir: None,
            server_player_round_robin: HashMap::new(),
            desired_cache: HashMap::new(),
            next_distant_sample: HashMap::new(),
            generic_groups: HashMap::new(),
            generic_groups_revision: 0,
            last_uploaded_generic_revision: 0,
            beacons: HashMap::new(),
            cloud_field: CloudField::default(),
            start_instant: Instant::now(),
        }
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
            mesher: MeshWorker::spawn(),
            gen_pool: GenWorkerPool::new(),
            inner: Mutex::new(DhState::default()),
        })
    })
}

/// Drain background IO/mesher results and return frame parameters, pending GPU section uploads,
/// pending generic box-group uploads, and reset epoch for the stage renderer.
pub fn begin_render_frame(
    camera: [f64; 3],
) -> (
    RenderFrameParams,
    Vec<SectionMeshData>,
    Vec<GenericGroupMesh>,
    u64,
) {
    let rt = runtime();

    // First pull any newly finished CPU meshes from the background meshing worker.
    let finished_meshes = rt.mesher.drain_finished();
    let io_responses = rt.io.drain_responses();

    let mut st = rt.inner.lock().unwrap();
    st.camera_pos = camera;
    st.has_camera = true;

    for res in io_responses {
        match res {
            IoResponse::Saved => {
                rt.saved_sections.fetch_add(1, Ordering::Relaxed);
            }
            IoResponse::Loaded { dimension, section } => {
                let pos = section.pos;
                let dim = st.dimensions.entry(dimension.clone()).or_default();
                let should_mesh = match dim.sections.get(&pos) {
                    Some(existing) => existing.populated < section.populated,
                    None => true,
                };
                if should_mesh {
                    dim.sections.insert(pos, section.clone());
                    rt.mesher.submit(dimension, section);
                }
            }
        }
    }

    for mesh in finished_meshes {
        if !mesh.is_empty() {
            st.meshed_sections
                .entry(mesh.dimension.clone())
                .or_default()
                .insert(mesh.pos);
            st.pending_gpu_uploads.push(mesh);
        }
    }

    let dim_name = st.active_dimension.clone();
    let uploads: Vec<SectionMeshData> = st
        .pending_gpu_uploads
        .drain(..)
        .filter(|m| m.dimension == dim_name)
        .collect();

    let mut effective_cfg = st.config;
    if let Some(scfg) = st.remote_session_config {
        effective_cfg.render_distance_chunks = effective_cfg
            .render_distance_chunks
            .min(scfg.max_render_distance_chunks);
    }

    let ready_set = st
        .meshed_sections
        .get(&dim_name)
        .cloned()
        .unwrap_or_default();
    let visible_sections = if effective_cfg.enabled {
        DimensionLodStore::select_visible_sections(camera[0], camera[2], effective_cfg, &ready_set)
    } else {
        Vec::new()
    };

    let generic_uploads = if st.last_uploaded_generic_revision != st.generic_groups_revision {
        st.last_uploaded_generic_revision = st.generic_groups_revision;
        st.generic_groups
            .get(&dim_name)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let generic_active: Vec<(String, [f64; 3])> = st
        .generic_groups
        .get(&dim_name)
        .map(|m| {
            m.values()
                .filter(|g| g.active && !g.is_empty())
                .map(|g| (g.id.clone(), g.origin))
                .collect()
        })
        .unwrap_or_default();

    let params = RenderFrameParams {
        enabled: effective_cfg.enabled,
        active_dimension: dim_name,
        config: effective_cfg,
        fog_color: st.fog_color,
        fog_type: st.fog_type,
        env_fog_start: st.env_fog_start,
        env_fog_end: st.env_fog_end,
        vanilla_far_blocks: st.vanilla_far_blocks,
        sun_brightness: st.sun_brightness,
        visible_sections,
        generic_active,
    };
    let epoch = st.gpu_reset_epoch;
    (params, uploads, generic_uploads, epoch)
}

/// Called by `record_stage` at the end of a frame.
pub fn record_frame_stats(rendered_sections: u64, rendered_quads: u64) {
    let rt = runtime();
    rt.rendered_sections
        .store(rendered_sections, Ordering::Relaxed);
    rt.rendered_quads.store(rendered_quads, Ordering::Relaxed);
    rt.frames_drawn.fetch_add(1, Ordering::Relaxed);
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
    let ready_meshes = st
        .meshed_sections
        .get(active_dim)
        .map(|m| m.len())
        .unwrap_or(0);
    DhStats {
        active_sections,
        ready_meshes,
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

/// Logs the first few errors of a render call the frame cannot recover from, then stays quiet.
fn report<T, E: std::fmt::Debug>(what: &str, result: Result<T, E>) -> Option<T> {
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    match result {
        Ok(v) => Some(v),
        Err(e) => {
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 20 {
                eprintln!("distantsunrises: {what} failed: {e:?}");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
