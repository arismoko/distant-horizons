//! Camera snapshot & Stage 3 general mod-API renderer bridge aligned with `#4758`
//! (`mod_api::Camera` and `mod_api::WorldRenderView<'a>`) and `#2005` (GLSL mod render
//! pipelines, retained meshes, uniforms, and render targets).
//!
//! Zero `wgpu` types and zero `[[native.mixin]]` references: `CameraMatrices::from_camera` reads
//! `#4758`'s `mod_api::Camera` fields (`position`, `view`, `projection`, `fov_degrees`, `near`,
//! `far`, `yaw`, `pitch`) directly.

use crate::config::DhConfig;
use crate::lod::SectionPos;

/// Camera-relative view + reversed-Z projection matrices extracted from `#4758`'s `mod_api::Camera`
/// (`position`, `view`, `projection`, `fov_degrees`, `near`, `far`, `yaw`, `pitch`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraMatrices {
    pub position: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
    /// Column-major 4x4 camera-relative view matrix (`Camera.view`).
    pub view: [f32; 16],
    /// Column-major 4x4 reversed-Z projection matrix (`Camera.projection`).
    pub projection: [f32; 16],
}

impl CameraMatrices {
    /// Extract matrices and lens parameters directly from `#4758`'s `mod_api::Camera`.
    pub fn from_camera(camera: &mod_api::Camera) -> Self {
        Self {
            position: camera.position,
            yaw: camera.yaw,
            pitch: camera.pitch,
            fov_degrees: camera.fov_degrees,
            near: camera.near,
            far: camera.far,
            view: camera.view,
            projection: camera.projection,
        }
    }

    /// Multiply `projection * view` in column-major order to obtain the combined camera-relative
    /// clip matrix matching Sunrise's renderer and `assets/distantsunrises/shaders/core/lod_terrain.vsh`.
    pub fn view_projection(&self) -> [f32; 16] {
        mul_mat4_col_major(&self.projection, &self.view)
    }

    /// Extract the 4 side frustum planes (left, right, bottom, top) from `projection * view`.
    /// Each plane `[a, b, c, d]` accepts a camera-relative point `p` when
    /// `a*p.x + b*p.y + c*p.z + d >= 0`.
    pub fn side_frustum_planes(&self) -> [[f32; 4]; 4] {
        let m = self.view_projection();
        let row = |r: usize| [m[r], m[4 + r], m[8 + r], m[12 + r]];
        let r0 = row(0);
        let r1 = row(1);
        let r3 = row(3);
        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        [
            add(r3, r0), // left
            sub(r3, r0), // right
            add(r3, r1), // bottom
            sub(r3, r1), // top
        ]
    }
}

/// CPU-side `DhFrameUniforms` layout (`std140`, 112 bytes) matching
/// `assets/distantsunrises/shaders/core/lod_terrain.vsh` and `lod_terrain.fsh`.
#[derive(Clone, Copy, Debug)]
pub struct TerrainUniformBlock {
    pub view_proj: [f32; 16],
    pub clip_and_light: [f32; 4],
    pub noise_and_fade: [f32; 4],
    pub flags: [u32; 4],
}

impl TerrainUniformBlock {
    pub fn build(
        camera: &CameraMatrices,
        cfg: DhConfig,
        vanilla_far_blocks: f32,
        sun_brightness: f32,
        is_hdr: bool,
    ) -> Self {
        let near_dh = 8.0f32;
        let far_dh = (cfg.render_distance_blocks() * 1.4).max(1024.0);
        let clip_dist =
            (vanilla_far_blocks * cfg.overdraw_prevention).clamp(0.0, vanilla_far_blocks * 0.85);
        let fade_band = (vanilla_far_blocks * 0.25).clamp(8.0, 48.0);
        Self {
            view_proj: camera.view_projection(),
            clip_and_light: [near_dh, far_dh, clip_dist, sun_brightness],
            noise_and_fade: [
                cfg.noise_intensity,
                fade_band,
                cfg.brightness_multiplier,
                cfg.saturation_multiplier,
            ],
            flags: [
                u32::from(cfg.shading_enabled),
                u32::from(cfg.noise_enabled),
                u32::from(cfg.fade_enabled),
                u32::from(is_hdr),
            ],
        }
    }

    pub fn to_std140_bytes(&self) -> [u8; 112] {
        let mut out = [0u8; 112];
        for (i, f) in self.view_proj.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
        }
        for (i, f) in self.clip_and_light.iter().enumerate() {
            out[64 + i * 4..68 + i * 4].copy_from_slice(&f.to_le_bytes());
        }
        for (i, f) in self.noise_and_fade.iter().enumerate() {
            out[80 + i * 4..84 + i * 4].copy_from_slice(&f.to_le_bytes());
        }
        for (i, v) in self.flags.iter().enumerate() {
            out[96 + i * 4..100 + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// Returns `true` if a section's camera-relative bounding box intersects the 4 side frustum planes.
pub fn section_in_side_frustum(
    planes: &[[f32; 4]; 4],
    pos: SectionPos,
    min_y: f32,
    max_y: f32,
    cam: [f64; 3],
) -> bool {
    let (ox, oz) = pos.block_origin();
    let span = pos.block_span() as f32;
    let rel_x = (ox as f64 - cam[0]) as f32;
    let rel_y = (-cam[1]) as f32;
    let rel_z = (oz as f64 - cam[2]) as f32;
    let min_p = [rel_x, min_y + rel_y, rel_z];
    let max_p = [rel_x + span, max_y + rel_y, rel_z + span];
    for p in planes {
        let px = if p[0] >= 0.0 { max_p[0] } else { min_p[0] };
        let py = if p[1] >= 0.0 { max_p[1] } else { min_p[1] };
        let pz = if p[2] >= 0.0 { max_p[2] } else { min_p[2] };
        if p[0] * px + p[1] * py + p[2] * pz + p[3] < 0.0 {
            return false;
        }
    }
    true
}

fn mul_mat4_col_major(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.0, // col 2
        0.0, 0.0, 0.0, 1.0, // col 3
    ];

    #[test]
    fn camera_matrices_extract_from_mod_api_camera_and_pack_std140() {
        let api_cam = mod_api::Camera {
            position: [12.0, 64.0, -8.0],
            eye_height: 1.62,
            partial_tick: 0.5,
            yaw: 45.0,
            pitch: -15.0,
            fov_degrees: 80.0,
            near: 0.05,
            far: 256.0,
            view: IDENTITY,
            projection: IDENTITY,
            dimension: "minecraft:overworld".to_owned(),
        };
        let cam = CameraMatrices::from_camera(&api_cam);
        assert_eq!(cam.position, [12.0, 64.0, -8.0]);
        assert_eq!(cam.fov_degrees, 80.0);
        assert_eq!(cam.near, 0.05);
        assert_eq!(cam.far, 256.0);
        assert_eq!(cam.view_projection(), IDENTITY);
        let block = TerrainUniformBlock::build(&cam, DhConfig::default(), 160.0, 0.9, true);
        let bytes = block.to_std140_bytes();
        assert_eq!(bytes.len(), 112);
        // flags.w = is_hdr = 1
        assert_eq!(u32::from_le_bytes(bytes[108..112].try_into().unwrap()), 1);
    }
}
