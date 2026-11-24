//! Thread-safe bridge between Distant Horizons mod hooks, background workers, and the
//! `sunrise-mod-render` mixin helpers (`render_needs` and `world_drawn`).
//!
//! Keeps all `wgpu` types inside `src/mixins/world_drawn.rs` while the mod crate owns LOD data,
//! settings, visibility selection, shaders, and telemetry.

use crate::api::{LodWorldGenerator, RemoteLodProvider};
use crate::config::DhConfig;
#[cfg(feature = "sunrise-gaps-g1-g4")]
use crate::data_pipeline::sunrise_adapters::{BlockVisualTable, DimensionGapHandles};
use crate::data_pipeline::{
    DataPipelineWorker, PipelineBatch, apply_coarse_section, apply_extracted_chunk,
};
use crate::lod::{DimensionLodStore, SectionPos};
use crate::mesh::{MeshWorker, SectionMeshData};
use crate::stage3_camera::CameraMatrices;
use crate::storage::{IoResponse, LodIoWorker};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// WGSL terrain shader for solid and translucent LOD geometry.
///
/// Exact reversed-Z depth replacement:
/// Sunrise's `world.jittered` matrix carries exact FOV, aspect, bob, spin, and TAA jitter,
/// with `clip.w = -z_view` (`Mat4::perspective_rh` reversed Z). Replacing only `clip.z` with
/// `(near_dh * far_dh - near_dh * w) / (far_dh - near_dh)` produces an exact reversed-Z depth
/// in `[0, 1]` over `[near_dh, far_dh]` without fighting vanilla's far plane.
pub const TERRAIN_WGSL: &str = r#"
struct FrameUniforms {
    jittered: mat4x4<f32>,
    // [near_dh, far_dh, clip_dist, sun_brightness]
    clip_and_light: vec4<f32>,
    // [noise_intensity, fade_band, brightness_multiplier, saturation_multiplier]
    noise_and_grade: vec4<f32>,
    // [shading_enabled, noise_enabled, fade_enabled, is_hdr]
    flags: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> u_frame: FrameUniforms;

struct VsIn {
    @location(0) local_pos: vec3<f32>,
    @location(1) color_packed: u32,
    @location(2) packed_meta: u32,
    // Per-section instance attribute: [origin_x - cam_x, -cam_y, origin_z - cam_z, cell_size]
    @location(3) section_offset: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) rel_pos: vec3<f32>,
    @location(2) block_pos: vec3<f32>,
    @location(3) @interpolate(flat) packed_meta: u32,
};

fn unpack_rgba(c: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(c & 0xFFu) / 255.0,
        f32((c >> 8u) & 0xFFu) / 255.0,
        f32((c >> 16u) & 0xFFu) / 255.0,
        f32((c >> 24u) & 0xFFu) / 255.0,
    );
}

fn dir_shade(normal_idx: u32) -> f32 {
    switch normal_idx {
        case 0u: { return 0.50; } // down
        case 1u: { return 1.00; } // up
        case 2u, 3u: { return 0.80; } // north / south
        default: { return 0.65; } // west / east
    }
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var out: VsOut;
    let rel = v.local_pos + v.section_offset.xyz;
    var clip = u_frame.jittered * vec4<f32>(rel, 1.0);

    let near_dh = u_frame.clip_and_light.x;
    let far_dh = u_frame.clip_and_light.y;
    // Reversed-Z projection on [near_dh, far_dh]: w = near_dh -> z/w = 1.0, w = far_dh -> z/w = 0.0
    clip.z = (near_dh * far_dh - near_dh * clip.w) / max(far_dh - near_dh, 1.0);

    let base = unpack_rgba(v.color_packed);
    let normal_idx = v.packed_meta & 0x7u;
    let sky_light = f32((v.packed_meta >> 4u) & 0xFu) / 15.0;
    let block_light = f32((v.packed_meta >> 8u) & 0xFu) / 15.0;
    let sun = u_frame.clip_and_light.w;
    let light = clamp(max(sky_light * sun, block_light) * 0.90 + 0.10, 0.08, 1.0);

    var shade = 1.0;
    if (u_frame.flags.x != 0u) {
        shade = dir_shade(normal_idx);
        // Subtle altitude lift so ridges and valleys read clearly at horizon scale.
        let height_hint = clamp((v.local_pos.y - 62.0) / 180.0, -0.06, 0.08);
        shade = clamp(shade + height_hint, 0.45, 1.06);
    }

    let brightness = u_frame.noise_and_grade.z;
    let saturation = u_frame.noise_and_grade.w;
    var graded = base.rgb * light * shade * brightness;
    let luma = dot(graded, vec3<f32>(0.2126, 0.7152, 0.0722));
    graded = clamp(mix(vec3<f32>(luma), graded, saturation), vec3<f32>(0.0), vec3<f32>(1.0));

    out.clip_pos = clip;
    out.color = vec4<f32>(graded, base.a);
    out.rel_pos = rel;
    out.block_pos = v.local_pos;
    out.packed_meta = v.packed_meta;
    return out;
}

fn hash3(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(12.9898, 78.233, 37.719));
    return fract(sin(h) * 43758.5453);
}

fn bayer4x4(p: vec2<f32>) -> f32 {
    let x = i32(p.x) & 3;
    let y = i32(p.y) & 3;
    let idx = y * 4 + x;
    var m = array<f32, 16>(
        0.0,  8.0,  2.0, 10.0,
        12.0, 4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
        15.0, 7.0, 13.0,  5.0
    );
    return (m[idx] + 0.5) / 16.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dist = length(in.rel_pos.xz);
    let clip_dist = u_frame.clip_and_light.z;
    let fade_band = u_frame.noise_and_grade.y;
    if (clip_dist > 0.0) {
        if (u_frame.flags.z != 0u && fade_band > 0.5) {
            let t = smoothstep(clip_dist, clip_dist + fade_band, dist);
            if (t <= bayer4x4(in.clip_pos.xy)) {
                discard;
            }
        } else if (dist < clip_dist) {
            discard;
        }
    }

    var rgb = in.color.rgb;
    let is_fluid = (in.packed_meta >> 12u) & 1u;
    if (u_frame.flags.y != 0u && is_fluid == 0u) {
        let intensity = u_frame.noise_and_grade.x;
        let lum = (rgb.r + rgb.g + rgb.b) / 3.0;
        let amp = (1.0 - pow(lum * 2.0 - 1.0, 2.0)) * intensity;
        let qpos = floor(in.block_pos);
        let rnd = hash3(qpos) * 2.0 * amp - amp;
        let dropoff = clamp(dist / 512.0, 0.0, 1.0);
        let noisy = clamp(rgb + (vec3<f32>(1.0) - rgb) * rnd, vec3<f32>(0.0), vec3<f32>(1.0));
        rgb = mix(noisy, rgb, dropoff);
    }

    return vec4<f32>(rgb, in.color.a);
}
"#;

/// Fullscreen sky-masked composite + atmospheric distance fog shader.
///
/// Reads Sunrise's reversed-Z `vanilla_depth` (`0.0` = sky) and Distant Horizons' offscreen
/// `dh_color` and `dh_depth` (`0.0` = empty). Only writes where vanilla drew sky AND DH drew
/// terrain, blending into vanilla's atmospheric fog color in gamma-2.2 space (matching
/// `docs/hdr-passes.md`).
pub const COMPOSITE_WGSL: &str = r#"
struct CompositeUniforms {
    fog_color: vec4<f32>,
    // [near_dh, far_dh, fog_start, fog_end]
    params: vec4<f32>,
    // [fog_enabled, is_hdr, 0, 0]
    flags: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> u_comp: CompositeUniforms;
@group(0) @binding(1)
var t_dh_color: texture_2d<f32>;
@group(0) @binding(2)
var t_dh_depth: texture_depth_2d;
@group(0) @binding(3)
var t_mc_depth: texture_depth_2d;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle covering [-1, 1]^2
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dh_dims = textureDimensions(t_dh_color);
    let mc_dims = textureDimensions(t_mc_depth);
    let px = clamp(vec2<i32>(in.pos.xy), vec2<i32>(0), vec2<i32>(dh_dims) - vec2<i32>(1));
    let mc_px = clamp(
        vec2<i32>((vec2<f32>(px) + vec2<f32>(0.5)) * vec2<f32>(mc_dims) / max(vec2<f32>(dh_dims), vec2<f32>(1.0))),
        vec2<i32>(0),
        vec2<i32>(mc_dims) - vec2<i32>(1)
    );

    let mc_depth = textureLoad(t_mc_depth, mc_px, 0);
    // Vanilla reversed-Z: anything > 0.0 is foreground vanilla geometry.
    if (mc_depth > 0.0) {
        discard;
    }

    let dh_depth = textureLoad(t_dh_depth, px, 0);
    if (dh_depth <= 0.0) {
        discard;
    }

    var col = textureLoad(t_dh_color, px, 0).rgb;

    if (u_comp.flags.x != 0u) {
        let near_dh = u_comp.params.x;
        let far_dh = u_comp.params.y;
        let fog_start = u_comp.params.z;
        let fog_end = max(u_comp.params.w, fog_start + 1.0);
        // Recover linear view distance from reversed-Z DH depth.
        let view_dist = (near_dh * far_dh) / max(dh_depth * (far_dh - near_dh) + near_dh, 1e-5);
        let t = clamp((view_dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
        // Smooth exponential-like atmospheric fog curve matching DH's horizon blend.
        let fog_factor = clamp(1.0 - exp(-2.8 * t * t), 0.0, 1.0);
        if (u_comp.flags.y != 0u) {
            // Convert sRGB LOD albedo and fog color to linear HDR (docs/hdr-passes.md).
            let lin_col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(2.2));
            let lin_fog = pow(max(u_comp.fog_color.rgb, vec3<f32>(0.0)), vec3<f32>(2.2));
            col = mix(lin_col, lin_fog, fog_factor);
        } else {
            col = mix(col, u_comp.fog_color.rgb, fog_factor);
        }
    } else if (u_comp.flags.y != 0u) {
        col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(2.2));
    }

    return vec4<f32>(max(col, vec3<f32>(0.0)), 1.0);
}
"#;

/// Snapshot of client frame parameters passed from the mod crate to `world_drawn`.
#[derive(Clone, Debug)]
pub struct RenderFrameParams {
    pub enabled: bool,
    pub active_dimension: String,
    pub config: DhConfig,
    pub fog_color: [f32; 4],
    pub vanilla_far_blocks: f32,
    pub sun_brightness: f32,
    pub visible_sections: Vec<SectionPos>,
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

/// Shared runtime state accessible from both `Hooks` and the `sunrise-mod-render` mixins.
pub struct DhRuntime {
    pub enabled_flag: AtomicBool,
    pub rendered_sections: AtomicU64,
    pub rendered_quads: AtomicU64,
    pub saved_sections: AtomicU64,
    pub frames_drawn: AtomicU64,
    pub io: LodIoWorker,
    pub mesher: MeshWorker,
    pub data_pipeline: DataPipelineWorker,
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
    /// Stage 2b (G1–G4): immutable block visual table built once from `Host` for worker threads.
    #[cfg(feature = "sunrise-gaps-g1-g4")]
    pub block_visuals: Option<BlockVisualTable>,
    /// Stage 2b (G2–G4): per-dimension saved-region readers, detached generators & chunk request drivers.
    #[cfg(feature = "sunrise-gaps-g1-g4")]
    pub gap_handles: HashMap<String, DimensionGapHandles>,
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
            #[cfg(feature = "sunrise-gaps-g1-g4")]
            block_visuals: None,
            #[cfg(feature = "sunrise-gaps-g1-g4")]
            gap_handles: HashMap::new(),
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
            data_pipeline: DataPipelineWorker::spawn(),
            inner: Mutex::new(DhState::default()),
        })
    })
}

/// Called from `src/mixins/render_needs.rs`.
pub fn needs_world_depth() -> bool {
    runtime().enabled_flag.load(Ordering::Relaxed)
}

/// Drain background IO/mesher results and return frame parameters + pending GPU mesh uploads for
/// `src/mixins/world_drawn.rs`.
pub fn begin_render_frame(camera: [f64; 3]) -> (RenderFrameParams, Vec<SectionMeshData>, u64) {
    let rt = runtime();

    // First pull any newly finished CPU meshes, IO responses, and off-thread data pipeline batches.
    let finished_meshes = rt.mesher.drain_finished();
    let io_responses = rt.io.drain_responses();
    let pipeline_batches = rt.data_pipeline.drain_finished();

    let mut st = rt.inner.lock().unwrap();
    st.camera_pos = camera;
    st.has_camera = true;

    if !pipeline_batches.is_empty() {
        let save_root = st.world_dir.clone();
        let mut dirty_by_dim: HashMap<String, HashSet<SectionPos>> = HashMap::new();
        for batch in pipeline_batches {
            match batch {
                PipelineBatch::Chunk(extracted) => {
                    let dim = extracted.dimension.clone();
                    let store = st.dimensions.entry(dim.clone()).or_default();
                    for pos in apply_extracted_chunk(store, extracted) {
                        dirty_by_dim.entry(dim.clone()).or_default().insert(pos);
                    }
                }
                PipelineBatch::CoarseSection(coarse) => {
                    let dim = coarse.dimension.clone();
                    let store = st.dimensions.entry(dim.clone()).or_default();
                    if let Some(pos) = apply_coarse_section(store, coarse) {
                        dirty_by_dim.entry(dim).or_default().insert(pos);
                    }
                }
            }
        }
        for (dim, dirty_sections) in dirty_by_dim {
            let store = st.dimensions.entry(dim.clone()).or_default();
            let mut to_mesh: HashSet<SectionPos> = dirty_sections.clone();
            for &pos in &dirty_sections {
                for parent_pos in store.propagate_up(pos) {
                    to_mesh.insert(parent_pos);
                }
            }
            for pos in to_mesh {
                if let Some(sec) = store.sections.get_mut(&pos) {
                    if sec.dirty_mesh && sec.populated > 0 {
                        sec.dirty_mesh = false;
                        rt.mesher.submit(dim.clone(), sec.clone());
                    }
                    if sec.dirty_save
                        && sec.populated > 0
                        && let Some(ref dir) = save_root
                    {
                        sec.dirty_save = false;
                        rt.io.request_save(dir.clone(), dim.clone(), sec.clone());
                    }
                }
            }
        }
    }

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

    let ready_set = st
        .meshed_sections
        .get(&dim_name)
        .cloned()
        .unwrap_or_default();
    let visible_sections = if st.config.enabled {
        DimensionLodStore::select_visible_sections(camera[0], camera[2], st.config, &ready_set)
    } else {
        Vec::new()
    };

    let params = RenderFrameParams {
        enabled: st.config.enabled,
        active_dimension: dim_name,
        config: st.config,
        fog_color: st.fog_color,
        vanilla_far_blocks: st.vanilla_far_blocks,
        sun_brightness: st.sun_brightness,
        visible_sections,
    };
    let epoch = st.gpu_reset_epoch;
    (params, uploads, epoch)
}

/// Called from `src/mixins/world_drawn.rs` at the end of a frame.
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
