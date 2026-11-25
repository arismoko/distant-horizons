//! Thread-safe bridge between Distant Sunrises mod hooks, background workers, and the
//! `sunrise-mod-render` mixin helpers (`render_needs` and `world_drawn`).
//!
//! Keeps all `wgpu` types inside `src/mixins/world_drawn.rs` while the mod crate owns LOD data,
//! settings, visibility selection, shaders, and telemetry.

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

/// WGSL terrain shader for solid, translucent, and generic box-group LOD geometry.
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
    // [shading_enabled, noise_enabled, fade_enabled, debug_mode]
    flags: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> u_frame: FrameUniforms;

struct VsIn {
    @location(0) local_pos: vec3<f32>,
    @location(1) color_packed: u32,
    @location(2) packed_meta: u32,
    // Per-section instance attribute: [origin_x - cam_x, origin_y - cam_y, origin_z - cam_z, cell_size]
    @location(3) section_offset: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) rel_pos: vec3<f32>,
    @location(2) block_pos: vec3<f32>,
    @location(3) @interpolate(flat) packed_meta: u32,
    @location(4) @interpolate(flat) cell_size: f32,
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

fn detail_debug_color(detail: u32) -> vec3<f32> {
    switch detail {
        case 0u: { return vec3<f32>(0.20, 0.90, 0.30); } // d0 green
        case 1u: { return vec3<f32>(0.15, 0.85, 0.85); } // d1 cyan
        case 2u: { return vec3<f32>(0.25, 0.55, 0.98); } // d2 blue
        case 3u: { return vec3<f32>(0.55, 0.30, 0.95); } // d3 indigo
        case 4u: { return vec3<f32>(0.85, 0.25, 0.85); } // d4 magenta
        case 5u: { return vec3<f32>(0.95, 0.25, 0.25); } // d5 red
        default: { return vec3<f32>(0.98, 0.60, 0.15); } // d6 orange
    }
}

fn source_debug_color(src: u32) -> vec3<f32> {
    switch src {
        case 1u: { return vec3<f32>(0.95, 0.75, 0.20); } // distant surface
        case 2u: { return vec3<f32>(0.20, 0.85, 0.35); } // loaded chunk
        case 3u: { return vec3<f32>(0.30, 0.70, 0.98); } // generic object
        default: { return vec3<f32>(0.60, 0.60, 0.60); }
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
    let detail = (v.packed_meta >> 17u) & 0x7u;
    let source = (v.packed_meta >> 20u) & 0x3u;
    let is_generic = (v.packed_meta >> 22u) & 1u;
    let unshaded = (v.packed_meta >> 23u) & 1u;
    let debug_mode = u_frame.flags.w;

    var graded: vec3<f32>;
    if (debug_mode == 1u || debug_mode == 4u) {
        graded = detail_debug_color(detail) * dir_shade(normal_idx);
    } else if (debug_mode == 2u) {
        graded = source_debug_color(source) * dir_shade(normal_idx);
    } else if (debug_mode == 3u) {
        graded = vec3<f32>(block_light, 0.5 * (sky_light + block_light), sky_light);
    } else if (unshaded != 0u) {
        graded = base.rgb;
    } else {
        let sun = u_frame.clip_and_light.w;
        let light = clamp(max(sky_light * sun, block_light) * 0.90 + 0.10, 0.08, 1.0);

        var shade = 1.0;
        if (u_frame.flags.x != 0u && is_generic == 0u) {
            shade = dir_shade(normal_idx);
            let height_hint = clamp((v.local_pos.y - 62.0) / 180.0, -0.06, 0.08);
            shade = clamp(shade + height_hint, 0.45, 1.06);
        }

        let brightness = u_frame.noise_and_grade.z;
        let saturation = u_frame.noise_and_grade.w;
        graded = base.rgb * light * shade * brightness;
        let luma = dot(graded, vec3<f32>(0.2126, 0.7152, 0.0722));
        graded = clamp(mix(vec3<f32>(luma), graded, saturation), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    out.clip_pos = clip;
    out.color = vec4<f32>(graded, base.a);
    out.rel_pos = rel;
    out.block_pos = v.local_pos;
    out.packed_meta = v.packed_meta;
    out.cell_size = max(v.section_offset.w, 1.0);
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
    let skip_clip = (in.packed_meta >> 24u) & 1u;
    let dist = length(in.rel_pos.xz);
    let clip_dist = u_frame.clip_and_light.z;
    let fade_band = u_frame.noise_and_grade.y;
    if (skip_clip == 0u && clip_dist > 0.0) {
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
    let debug_mode = u_frame.flags.w;
    let is_fluid = (in.packed_meta >> 12u) & 1u;
    let is_generic = (in.packed_meta >> 22u) & 1u;

    if (debug_mode == 4u) {
        let cell = in.cell_size;
        let f = fract(in.block_pos.xz / cell);
        let edge = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
        if (edge < 0.05) {
            rgb = vec3<f32>(0.05, 0.05, 0.05);
        }
    } else if (debug_mode == 0u && u_frame.flags.y != 0u && is_fluid == 0u && is_generic == 0u) {
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

/// Fullscreen sky-masked composite + spiral SSAO + full DH distance/height/environmental fog shader.
///
/// Reads Sunrise's reversed-Z `vanilla_depth` (`0.0` = sky) and Distant Sunrises' offscreen
/// `dh_color` and `dh_depth` (`0.0` = empty). Only writes where vanilla drew sky AND DH drew
/// terrain, applying depth-buffer SSAO and DH far/height/environmental fog before converting
/// cleanly to Sunrise's linear HDR light space (`docs/hdr-passes.md`).
pub const COMPOSITE_WGSL: &str = r#"
struct CompositeUniforms {
    inv_view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    // [near_dh, far_dh, fog_start, fog_end]
    fog_params: vec4<f32>,
    // [camera_y, height_fog_base_y, env_fog_start, env_fog_end]
    height_and_env: vec4<f32>,
    // [ssao_radius, ssao_strength, ssao_bias, ssao_fade_dist]
    ssao_params: vec4<f32>,
    // [fog_enabled, is_hdr, fog_falloff, fog_spherical]
    flags0: vec4<u32>,
    // [height_fog_enabled, height_fog_mix_mode, ssao_samples, fog_type]
    flags1: vec4<u32>,
    // [cas_amount, 0.0, 0.0, 0.0]
    sharpen_and_pad: vec4<f32>,
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
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    return out;
}

fn unproject_dh(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let wpos = u_comp.inv_view_proj * ndc;
    return wpos.xyz / max(abs(wpos.w), 1e-6) * sign(select(1.0, wpos.w, wpos.w != 0.0));
}

fn interleaved_gradient_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

fn compute_ssao(
    px: vec2<i32>,
    dims: vec2<i32>,
    rel_pos: vec3<f32>,
    view_dist: f32,
) -> f32 {
    let sample_count = u_comp.flags1.z;
    if (sample_count == 0u) {
        return 0.0;
    }
    let fade_dist = u_comp.ssao_params.w;
    if (view_dist >= fade_dist) {
        return 0.0;
    }

    let f_dims = max(vec2<f32>(dims), vec2<f32>(1.0));
    let px_r = clamp(px + vec2<i32>(1, 0), vec2<i32>(0), dims - vec2<i32>(1));
    let px_l = clamp(px - vec2<i32>(1, 0), vec2<i32>(0), dims - vec2<i32>(1));
    let px_d = clamp(px + vec2<i32>(0, 1), vec2<i32>(0), dims - vec2<i32>(1));
    let px_u = clamp(px - vec2<i32>(0, 1), vec2<i32>(0), dims - vec2<i32>(1));

    let dr = textureLoad(t_dh_depth, px_r, 0);
    let dl = textureLoad(t_dh_depth, px_l, 0);
    let dd = textureLoad(t_dh_depth, px_d, 0);
    let du = textureLoad(t_dh_depth, px_u, 0);

    let pos_r = unproject_dh((vec2<f32>(px_r) + 0.5) / f_dims, max(dr, 1e-6));
    let pos_l = unproject_dh((vec2<f32>(px_l) + 0.5) / f_dims, max(dl, 1e-6));
    let pos_d = unproject_dh((vec2<f32>(px_d) + 0.5) / f_dims, max(dd, 1e-6));
    let pos_u = unproject_dh((vec2<f32>(px_u) + 0.5) / f_dims, max(du, 1e-6));

    let dx_r = pos_r - rel_pos;
    let dx_l = rel_pos - pos_l;
    let dpdx = select(dx_l, dx_r, dr > 0.0 && (dl <= 0.0 || dot(dx_r, dx_r) < dot(dx_l, dx_l)));

    let dy_d = pos_d - rel_pos;
    let dy_u = rel_pos - pos_u;
    let dpdy = select(dy_u, dy_d, dd > 0.0 && (du <= 0.0 || dot(dy_d, dy_d) < dot(dy_u, dy_u)));

    var n = cross(dpdy, dpdx);
    let n_len = length(n);
    if (n_len < 1e-5) {
        return 0.0;
    }
    n = n / n_len;
    // Orient normal toward the camera (-rel_pos).
    if (dot(n, -rel_pos) < 0.0) {
        n = -n;
    }

    let radius_world = u_comp.ssao_params.x;
    let strength = u_comp.ssao_params.y;
    let bias = u_comp.ssao_params.z;
    let px_radius = clamp(radius_world * f_dims.y / max(view_dist * 1.4, 1.0), 2.5, 42.0);
    let r_step = px_radius / f32(sample_count);

    var angle = interleaved_gradient_noise(vec2<f32>(px)) * 6.2831853;
    var r = r_step;
    var ao = 0.0;
    var hits = 0.0;

    for (var i = 0u; i < sample_count; i = i + 1u) {
        let off = vec2<f32>(cos(angle), sin(angle)) * r;
        r = r + r_step;
        angle = angle + 2.39996323; // golden angle

        let sp = clamp(px + vec2<i32>(round(off)), vec2<i32>(0), dims - vec2<i32>(1));
        let sd = textureLoad(t_dh_depth, sp, 0);
        if (sd <= 0.0) {
            continue;
        }
        let s_rel = unproject_dh((vec2<f32>(sp) + 0.5) / f_dims, sd);
        let diff = s_rel - rel_pos;
        let s_dist = length(diff);
        if (s_dist > 1e-3 && s_dist < radius_world * 2.5) {
            let ndotl = max(dot(n, diff / s_dist) - bias, 0.0);
            let falloff = 1.0 - clamp(s_dist / (radius_world * 2.5), 0.0, 1.0);
            ao = ao + ndotl * falloff;
            hits = hits + 1.0;
        }
    }

    let raw_ao = ao / max(hits, 1.0);
    let dist_fade = 1.0 - clamp(view_dist / max(fade_dist, 1.0), 0.0, 1.0);
    return clamp(smoothstep(0.0, 0.55, raw_ao) * strength * dist_fade, 0.0, 0.65);
}

fn fog_curve(t_raw: f32, falloff: u32) -> f32 {
    let t = clamp(t_raw, 0.0, 1.0);
    if (falloff == 0u) {
        return t;
    } else if (falloff == 1u) {
        return clamp(1.0 - exp(-2.5 * t), 0.0, 1.0);
    } else {
        return clamp(1.0 - exp(-2.8 * t * t), 0.0, 1.0);
    }
}

fn mix_height_fog(far_f: f32, height_f: f32, mode: u32) -> f32 {
    switch mode {
        case 0u: { return far_f; }
        case 1u: { return max(far_f, height_f); }
        case 2u: { return clamp(far_f + height_f, 0.0, 1.0); }
        case 3u: { return far_f * height_f; }
        default: { return 1.0 - (1.0 - far_f) * (1.0 - height_f); }
    }
}

fn fetch_dh_masked(sp: vec2<i32>, dims: vec2<i32>, fallback: vec3<f32>) -> vec3<f32> {
    let c = clamp(sp, vec2<i32>(0), dims - vec2<i32>(1));
    if (textureLoad(t_dh_depth, c, 0) <= 0.0) {
        return fallback;
    }
    return textureLoad(t_dh_color, c, 0).rgb;
}

fn apply_cas(px: vec2<i32>, dims: vec2<i32>, e: vec3<f32>, amount: f32) -> vec3<f32> {
    if (amount <= 0.001) {
        return e;
    }
    let a = fetch_dh_masked(px + vec2<i32>(-1, -1), dims, e);
    let b = fetch_dh_masked(px + vec2<i32>( 0, -1), dims, e);
    let c = fetch_dh_masked(px + vec2<i32>( 1, -1), dims, e);
    let d = fetch_dh_masked(px + vec2<i32>(-1,  0), dims, e);
    let f = fetch_dh_masked(px + vec2<i32>( 1,  0), dims, e);
    let g = fetch_dh_masked(px + vec2<i32>(-1,  1), dims, e);
    let h = fetch_dh_masked(px + vec2<i32>( 0,  1), dims, e);
    let i = fetch_dh_masked(px + vec2<i32>( 1,  1), dims, e);

    let mn_rgb = min(min(min(d, e), min(f, b)), h) + min(min(min(a, g), c), i);
    let mx_rgb = max(max(max(d, e), max(f, b)), h) + max(max(max(a, g), c), i);

    let rcp_mx = vec3<f32>(1.0) / max(mx_rgb, vec3<f32>(1e-4));
    let amp = inverseSqrt(max(clamp(min(mn_rgb, vec3<f32>(2.0) - mx_rgb) * rcp_mx, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1e-4)));
    let peak = 8.0 - 3.0 * clamp(amount, 0.0, 1.0);
    let w_rgb = vec3<f32>(-1.0) / (amp * peak);
    let rcp_w = vec3<f32>(1.0) / max(vec3<f32>(1.0) + 4.0 * w_rgb, vec3<f32>(1e-4));
    return clamp(((b + d + f + h) * w_rgb + e) * rcp_w, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dh_dims = textureDimensions(t_dh_color);
    let mc_dims = textureDimensions(t_mc_depth);
    let i_dh_dims = vec2<i32>(dh_dims);
    let px = clamp(vec2<i32>(in.pos.xy), vec2<i32>(0), i_dh_dims - vec2<i32>(1));
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
    col = apply_cas(px, i_dh_dims, col, u_comp.sharpen_and_pad.x);

    let near_dh = u_comp.fog_params.x;
    let far_dh = u_comp.fog_params.y;
    let view_dist = (near_dh * far_dh) / max(dh_depth * (far_dh - near_dh) + near_dh, 1e-5);
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / max(vec2<f32>(dh_dims), vec2<f32>(1.0));
    let rel_pos = unproject_dh(uv, dh_depth);

    // 1. Distant SSAO (applied before atmospheric fog, matching upstream DH ao -> fog order).
    let ao = compute_ssao(px, i_dh_dims, rel_pos, view_dist);
    col = col * (1.0 - ao);

    // 2. Atmospheric / height / environmental fog.
    let fog_type = u_comp.flags1.w;
    var fog_factor = 0.0;
    if (fog_type != 0u) {
        // Underwater / Lava / PowderSnow: use vanilla environmental fog distances.
        let env_start = u_comp.height_and_env.z;
        let env_end = max(u_comp.height_and_env.w, env_start + 1.0);
        let sphere_d = length(rel_pos);
        fog_factor = clamp((sphere_d - env_start) / (env_end - env_start), 0.0, 1.0);
    } else if (u_comp.flags0.x != 0u) {
        let fog_start = u_comp.fog_params.z;
        let fog_end = max(u_comp.fog_params.w, fog_start + 1.0);
        let active_dist = select(length(rel_pos.xz), length(rel_pos), u_comp.flags0.w != 0u);
        let far_t = (active_dist - fog_start) / (fog_end - fog_start);
        let far_fog = fog_curve(far_t, u_comp.flags0.z);

        var height_fog = 0.0;
        if (u_comp.flags1.x != 0u) {
            let world_y = rel_pos.y + u_comp.height_and_env.x;
            let base_y = u_comp.height_and_env.y;
            let below = max(base_y - world_y, 0.0);
            let h_t = clamp(below / 96.0, 0.0, 1.0) * clamp((active_dist - fog_start * 0.45) / max(fog_end * 0.55, 1.0), 0.0, 1.0);
            height_fog = fog_curve(h_t, u_comp.flags0.z) * 0.65;
        }
        fog_factor = mix_height_fog(far_fog, height_fog, u_comp.flags1.y);
    }

    if (u_comp.flags0.y != 0u) {
        // Convert sRGB LOD albedo and fog color to linear HDR (docs/hdr-passes.md).
        let lin_col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(2.2));
        let lin_fog = pow(max(u_comp.fog_color.rgb, vec3<f32>(0.0)), vec3<f32>(2.2));
        col = mix(lin_col, lin_fog, fog_factor);
    } else {
        col = mix(col, u_comp.fog_color.rgb, fog_factor);
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

/// Shared runtime state accessible from both `Hooks` and the `sunrise-mod-render` mixins.
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

/// Called from `src/mixins/render_needs.rs`.
pub fn needs_world_depth() -> bool {
    runtime().enabled_flag.load(Ordering::Relaxed)
}

/// Drain background IO/mesher results and return frame parameters, pending GPU section uploads,
/// pending generic box-group uploads, and reset epoch for `src/mixins/world_drawn.rs`.
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
