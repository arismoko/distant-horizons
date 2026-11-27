//! Camera snapshot & Stage 3 general mod-API renderer bridge aligned with `#4758`
//! (`mod_api::Camera` and `mod_api::WorldRenderView<'a>`) and `#2005` (GLSL mod render
//! pipelines, retained meshes, uniforms, and render targets).
//!
//! Zero `wgpu` types and zero `[[native.mixin]]` references: `CameraMatrices::from_camera` reads
//! `#4758`'s `mod_api::Camera` fields (`position`, `view`, `projection`, `fov_degrees`, `near`,
//! `far`, `yaw`, `pitch`) directly.

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
    /// clip matrix matching Sunrise's renderer.
    pub fn view_projection(&self) -> [f32; 16] {
        mul_mat4_col_major(&self.projection, &self.view)
    }
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
    fn camera_matrices_extract_from_mod_api_camera() {
        let api_cam = mod_api::Camera {
            position: [12.0, 64.0, -8.0],
            eye_height: 1.62,
            partial_tick: 0.5,
            yaw: 45.0,
            pitch: -15.0,
            roll: 0.0,
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
    }
}
