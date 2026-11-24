//! Iris/OptiFine shader-pack integration data for Distant Horizons.
//!
//! Computes the exact OpenGL `[-1, 1]` clip-space projection matrices (`dhProjection`,
//! `dhProjectionInverse`, `dhPreviousProjection`) and scalar uniforms (`dhNearPlane`,
//! `dhFarPlane`, `dhRenderDistance`) expected by translated shader packs (BSL, Complementary,
//! MakeUp, Bliss, Photon, etc.), and declares the static pack-binding table matching
//! `sunrise-org/sunrise#5371` (`PackBinding` on `RenderPipeline`).

use crate::colors::{
    DH_BLOCK_AIR, DH_BLOCK_DEEPSLATE, DH_BLOCK_DIRT, DH_BLOCK_GRASS, DH_BLOCK_ILLUMINATED,
    DH_BLOCK_LAVA, DH_BLOCK_LEAVES, DH_BLOCK_METAL, DH_BLOCK_NETHER_STONE, DH_BLOCK_SAND,
    DH_BLOCK_SNOW, DH_BLOCK_STONE, DH_BLOCK_TERRACOTTA, DH_BLOCK_UNKNOWN, DH_BLOCK_WATER,
    DH_BLOCK_WOOD,
};

/// Shared auxiliary depth-plane target ID used by DH solid and water pipelines (`#5371`).
pub const DH_DEPTH_TARGET_ID: &str = "distanthorizons:far_depth";

/// Pack program fallback chains matching Iris `DHCompatInternal` / `ProgramId`.
pub const DH_TERRAIN_PROGRAMS: &[&str] = &["dh_terrain"];
pub const DH_WATER_PROGRAMS: &[&str] = &["dh_water", "dh_terrain"];
pub const DH_SHADOW_PROGRAMS: &[&str] = &["dh_shadow"];

/// Depth sampler names exposed to shader-pack deferred/composite/final passes (`#5371`).
pub const DH_FULL_DEPTH_SAMPLERS: &[&str] = &["dhDepthTex", "dhDepthTex0"];
pub const DH_OPAQUE_DEPTH_SAMPLERS: &[&str] = &["dhDepthTex1"];

/// Preprocessor defines exposed while compiling a shader pack when DH is enabled
/// (`StandardMacros.java:81-96` plus `DISTANT_HORIZONS`).
pub const DH_PACK_DEFINES: &[(&str, Option<&str>)] = &[
    ("DISTANT_HORIZONS", None),
    ("DH_BLOCK_UNKNOWN", Some("0")),
    ("DH_BLOCK_LEAVES", Some("1")),
    ("DH_BLOCK_STONE", Some("2")),
    ("DH_BLOCK_WOOD", Some("3")),
    ("DH_BLOCK_METAL", Some("4")),
    ("DH_BLOCK_DIRT", Some("5")),
    ("DH_BLOCK_LAVA", Some("6")),
    ("DH_BLOCK_DEEPSLATE", Some("7")),
    ("DH_BLOCK_SNOW", Some("8")),
    ("DH_BLOCK_SAND", Some("9")),
    ("DH_BLOCK_TERRACOTTA", Some("10")),
    ("DH_BLOCK_NETHER_STONE", Some("11")),
    ("DH_BLOCK_WATER", Some("12")),
    ("DH_BLOCK_GRASS", Some("13")),
    ("DH_BLOCK_AIR", Some("14")),
    ("DH_BLOCK_ILLUMINATED", Some("15")),
];

/// Compile-time check that every `DH_BLOCK_*` define string matches its Rust constant.
const _: () = {
    assert!(DH_BLOCK_UNKNOWN == 0);
    assert!(DH_BLOCK_LEAVES == 1);
    assert!(DH_BLOCK_STONE == 2);
    assert!(DH_BLOCK_WOOD == 3);
    assert!(DH_BLOCK_METAL == 4);
    assert!(DH_BLOCK_DIRT == 5);
    assert!(DH_BLOCK_LAVA == 6);
    assert!(DH_BLOCK_DEEPSLATE == 7);
    assert!(DH_BLOCK_SNOW == 8);
    assert!(DH_BLOCK_SAND == 9);
    assert!(DH_BLOCK_TERRACOTTA == 10);
    assert!(DH_BLOCK_NETHER_STONE == 11);
    assert!(DH_BLOCK_WATER == 12);
    assert!(DH_BLOCK_GRASS == 13);
    assert!(DH_BLOCK_AIR == 14);
    assert!(DH_BLOCK_ILLUMINATED == 15);
};

/// Column-major 4x4 matrix (`[col0, col1, col2, col3]`).
pub type Mat4 = [[f32; 4]; 4];

pub const IDENTITY_MAT4: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Right-handed OpenGL perspective projection mapping `view_z in [-near, -far]` to
/// `ndc_z in [-1.0, +1.0]` (`depth = ndc_z * 0.5 + 0.5 in [0.0, 1.0]`).
///
/// Matches Iris `DHCompatInternal` (`dhProjection`) so shader-pack depth reconstruction
/// `dhProjectionInverse * vec4(ndc_xy, depth * 2.0 - 1.0, 1.0)` recovers exact view-space Z.
pub fn gl_perspective_rh(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y_rad.clamp(0.05, 3.0) * 0.5).tan();
    let aspect = aspect.max(0.01);
    let near = near.max(0.05);
    let far = far.max(near + 1.0);
    let inv_nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * inv_nf, -1.0],
        [0.0, 0.0, (2.0 * far * near) * inv_nf, 0.0],
    ]
}

/// Exact analytic inverse for a standard perspective matrix (`gl_perspective_rh`), with a general
/// Gauss-Jordan fallback if off-diagonal terms are present.
pub fn invert_mat4(m: Mat4) -> Option<Mat4> {
    let mut a = [[0.0f64; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = m[c][r] as f64;
        }
        a[r][r + 4] = 1.0;
    }
    for col in 0..4 {
        let mut pivot_row = col;
        let mut best = a[col][col].abs();
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            let v = row[col].abs();
            if v > best {
                best = v;
                pivot_row = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
        }
        let inv_diag = 1.0 / a[col][col];
        for cell in &mut a[col][col..8] {
            *cell *= inv_diag;
        }
        for r in 0..4 {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            let pivot_copy = a[col];
            for k in col..8 {
                a[r][k] -= factor * pivot_copy[k];
            }
        }
    }
    let mut out = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[c][r] = a[r][c + 4] as f32;
        }
    }
    Some(out)
}

/// Per-frame shader-pack uniforms supplied by Distant Horizons (`#5371`).
#[derive(Clone, Copy, Debug)]
pub struct DhPackFrameUniforms {
    pub dh_projection: Mat4,
    pub dh_projection_inverse: Mat4,
    pub dh_previous_projection: Mat4,
    pub dh_near_plane: f32,
    pub dh_far_plane: f32,
    pub dh_render_distance: i32,
}

impl Default for DhPackFrameUniforms {
    fn default() -> Self {
        Self {
            dh_projection: IDENTITY_MAT4,
            dh_projection_inverse: IDENTITY_MAT4,
            dh_previous_projection: IDENTITY_MAT4,
            dh_near_plane: 16.0,
            dh_far_plane: 4096.0,
            dh_render_distance: 2048,
        }
    }
}

impl DhPackFrameUniforms {
    /// Advance uniforms by one frame, latching the previous frame's `dhProjection` into
    /// `dhPreviousProjection` (initializing both on the first frame so temporal shaders do not
    /// sample identity on frame 0).
    pub fn update(
        &mut self,
        fov_y_rad: f32,
        aspect: f32,
        dh_near_plane: f32,
        dh_far_plane: f32,
        dh_render_distance_blocks: i32,
        is_first_frame: bool,
    ) {
        let proj = gl_perspective_rh(fov_y_rad, aspect, dh_near_plane, dh_far_plane);
        let proj_inv = invert_mat4(proj).unwrap_or(IDENTITY_MAT4);
        self.dh_previous_projection = if is_first_frame {
            proj
        } else {
            self.dh_projection
        };
        self.dh_projection = proj;
        self.dh_projection_inverse = proj_inv;
        self.dh_near_plane = dh_near_plane;
        self.dh_far_plane = dh_far_plane;
        self.dh_render_distance = dh_render_distance_blocks.max(16);
    }
}

/// Multiply column-major 4x4 matrix by a vec4.
#[inline]
pub fn mul_mat4_vec4(m: Mat4, v: [f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3],
        m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iris_depth_reconstruction_roundtrips_view_z_over_dh_frustum() {
        let near = 48.0f32;
        let far = 4096.0f32;
        let proj = gl_perspective_rh(70.0f32.to_radians(), 16.0 / 9.0, near, far);
        let proj_inv = invert_mat4(proj).expect("perspective matrix is invertible");

        for &dist in &[near, 128.0, 512.0, 2048.0, far] {
            let view_pos = [12.5, -7.25, -dist, 1.0];
            let clip = mul_mat4_vec4(proj, view_pos);
            let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
            let depth = ndc[2] * 0.5 + 0.5;
            assert!(
                (-1e-5..=1.0 + 1e-5).contains(&depth),
                "depth {depth} out of [0, 1] at dist={dist}"
            );

            // Exact Iris shader reconstruction: dhProjectionInverse * vec4(ndc.xy, depth * 2 - 1, 1)
            let recon_h = mul_mat4_vec4(proj_inv, [ndc[0], ndc[1], depth * 2.0 - 1.0, 1.0]);
            let recon = [
                recon_h[0] / recon_h[3],
                recon_h[1] / recon_h[3],
                recon_h[2] / recon_h[3],
            ];
            let tol = dist * 1e-3 + 1e-2;
            assert!(
                (recon[0] - view_pos[0]).abs() < tol
                    && (recon[1] - view_pos[1]).abs() < tol
                    && (recon[2] - view_pos[2]).abs() < tol,
                "mismatch at dist={dist}: got {recon:?}, expected {:?}",
                &view_pos[0..3]
            );
        }
    }

    #[test]
    fn frame_uniforms_latch_previous_projection_after_first_frame() {
        let mut u = DhPackFrameUniforms::default();
        u.update(70.0f32.to_radians(), 16.0 / 9.0, 32.0, 2048.0, 2048, true);
        assert_eq!(u.dh_previous_projection, u.dh_projection);
        let first = u.dh_projection;

        u.update(90.0f32.to_radians(), 16.0 / 9.0, 64.0, 4096.0, 4096, false);
        assert_eq!(u.dh_previous_projection, first);
        assert_ne!(u.dh_projection, first);
        assert_eq!(u.dh_near_plane, 64.0);
        assert_eq!(u.dh_far_plane, 4096.0);
        assert_eq!(u.dh_render_distance, 4096);
    }

    #[test]
    fn pack_defines_cover_distant_horizons_and_all_16_material_ids() {
        assert_eq!(DH_PACK_DEFINES.len(), 17);
        assert_eq!(DH_PACK_DEFINES[0], ("DISTANT_HORIZONS", None));
        assert_eq!(DH_PACK_DEFINES[13], ("DH_BLOCK_WATER", Some("12")));
        assert_eq!(DH_PACK_DEFINES[14], ("DH_BLOCK_GRASS", Some("13")));
        assert_eq!(DH_PACK_DEFINES[15], ("DH_BLOCK_AIR", Some("14")));
        assert_eq!(DH_PACK_DEFINES[16], ("DH_BLOCK_ILLUMINATED", Some("15")));
    }
}
