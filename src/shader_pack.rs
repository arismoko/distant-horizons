//! Iris/OptiFine shader-pack integration data for Distant Sunrises.
//!
//! Computes the exact OpenGL `[-1, 1]` clip-space projection matrices (`dhProjection`,
//! `dhProjectionInverse`, `dhPreviousProjection`) and scalar uniforms (`dhNearPlane`,
//! `dhFarPlane`, `dhRenderDistance`) expected by translated shader packs (BSL, Complementary,
//! MakeUp, Bliss, Photon, etc.), and declares the static pack-binding table matching
//! `sunrise-org/sunrise#5371` (`PackBinding` on `RenderPipeline`).

use mod_api::{
    PackBinding, PackDepthTarget, PackPassSlot, PackProgramMode, PackUniformDecl, PackUniformType,
    ShaderPackState,
};

use crate::colors::{
    DH_BLOCK_AIR, DH_BLOCK_DEEPSLATE, DH_BLOCK_DIRT, DH_BLOCK_GRASS, DH_BLOCK_ILLUMINATED,
    DH_BLOCK_LAVA, DH_BLOCK_LEAVES, DH_BLOCK_METAL, DH_BLOCK_NETHER_STONE, DH_BLOCK_SAND,
    DH_BLOCK_SNOW, DH_BLOCK_STONE, DH_BLOCK_TERRACOTTA, DH_BLOCK_UNKNOWN, DH_BLOCK_WATER,
    DH_BLOCK_WOOD,
};

/// Shared auxiliary depth-plane target ID used by DH solid and water pipelines (`#5371`).
pub const DH_DEPTH_TARGET_ID: &str = "distantsunrises:far_depth";

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

/// Per-frame shader-pack uniforms supplied by Distant Sunrises (`#5371`).
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

/// GLSL vertex entrypoints for `PackProgramMode::ModVertexPackFragment` (`#5700` part C). Each
/// links directly against the active pack's `dh_terrain` / `dh_water` / `dh_shadow` fragment shader.
pub const VOXY_PACK_TERRAIN_VSH: &str = "distantsunrises:core/voxy_pack_terrain";
pub const VOXY_PACK_WATER_VSH: &str = "distantsunrises:core/voxy_pack_water";
pub const VOXY_PACK_SHADOW_VSH: &str = "distantsunrises:core/voxy_pack_shadow";

/// Voxy `PackedQuad` bit layout (`lod/quad_format.glsl`, 64-bit / `ivec2` path, little endian).
/// The `.vsh` files decode exactly these fields from an `uvec2` (`x` = bits 0..31).
pub mod packed_quad {
    pub const FACE_SHIFT: u32 = 0; // 3 bits
    pub const WIDTH_SHIFT: u32 = 3; // 4 bits, stored minus one
    pub const HEIGHT_SHIFT: u32 = 7; // 4 bits, stored minus one
    pub const POS_Z_SHIFT: u32 = 11; // 5 bits
    pub const POS_Y_SHIFT: u32 = 16; // 5 bits
    pub const POS_X_SHIFT: u32 = 21; // 5 bits
    pub const STATE_SHIFT: u32 = 26; // 20 bits (`extractStateId`, `ivec2` path)
    pub const BIOME_SHIFT: u32 = 46; // 9 bits
    pub const LIGHT_SHIFT: u32 = 55; // 8 bits (block light in the high nibble, sky in the low)

    /// Decoded fields of one quad.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Decoded {
        pub face: u8,
        pub width: u8,
        pub height: u8,
        pub pos: [u8; 3],
        pub state: u32,
        pub biome: u16,
        pub light: u8,
    }

    const fn bits(q: u64, shift: u32, n: u32) -> u64 {
        (q >> shift) & ((1u64 << n) - 1)
    }

    /// Rust mirror of the GLSL decode in `voxy_pack_*.vsh`.
    pub const fn decode(q: u64) -> Decoded {
        Decoded {
            face: bits(q, FACE_SHIFT, 3) as u8,
            width: bits(q, WIDTH_SHIFT, 4) as u8 + 1,
            height: bits(q, HEIGHT_SHIFT, 4) as u8 + 1,
            pos: [
                bits(q, POS_X_SHIFT, 5) as u8,
                bits(q, POS_Y_SHIFT, 5) as u8,
                bits(q, POS_Z_SHIFT, 5) as u8,
            ],
            state: bits(q, STATE_SHIFT, 20) as u32,
            biome: bits(q, BIOME_SHIFT, 9) as u16,
            light: bits(q, LIGHT_SHIFT, 8) as u8,
        }
    }

    /// Inverse of [`decode`]; `width` / `height` are `1..=16`.
    pub const fn encode(d: Decoded) -> u64 {
        (d.face as u64 & 7) << FACE_SHIFT
            | ((d.width as u64 - 1) & 15) << WIDTH_SHIFT
            | ((d.height as u64 - 1) & 15) << HEIGHT_SHIFT
            | (d.pos[2] as u64 & 31) << POS_Z_SHIFT
            | (d.pos[1] as u64 & 31) << POS_Y_SHIFT
            | (d.pos[0] as u64 & 31) << POS_X_SHIFT
            | (d.state as u64 & 0xF_FFFF) << STATE_SHIFT
            | (d.biome as u64 & 0x1FF) << BIOME_SHIFT
            | (d.light as u64) << LIGHT_SHIFT
    }
}

/// Model-buffer word offsets (`voxy block_model.glsl` `BlockModel`, 16 `uint`s per model).
pub const MODEL_STRIDE_WORDS: usize = 16;
pub const MODEL_FLAGS_WORD: usize = 6;
pub const MODEL_CUSTOM_ID_WORD: usize = 8;
/// `flagsA` bits 8..11 carry the `DH_BLOCK_*` material id read as `dhMaterialId` by the `.vsh`.
pub const MODEL_FLAGS_DH_MATERIAL_SHIFT: u32 = 8;

/// `flagsA` word for a baked model: Voxy's low flags (`2` biome LUT, `4` translucent, `8` shaded)
/// plus the `DH_BLOCK_*` material in bits 8..11.
pub const fn model_flags(biome_lut: bool, translucent: bool, shaded: bool, dh_material: u8) -> u32 {
    (biome_lut as u32) << 1
        | (translucent as u32) << 2
        | (shaded as u32) << 3
        | ((dh_material as u32) & 0xF) << MODEL_FLAGS_DH_MATERIAL_SHIFT
}

/// std140 `SceneUniform` shared by the three `.vsh` files (176 bytes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackSceneUniform {
    /// Camera-relative world -> view. In the shadow slot: the pack's `shadowModelView`.
    pub model_view: Mat4,
    /// View -> clip with `w = -view.z`. In the shadow slot: the pack's `shadowProjection`.
    pub projection: Mat4,
    /// Block origin the section offsets are relative to (multiple of 32).
    pub base_block_pos: [i32; 3],
    pub frame_id: u32,
    /// Camera position minus `base_block_pos`.
    pub camera_sub_pos: [f32; 3],
    /// `clip.z = x + y * clip.w`; see [`DepthRemap`]. Unused by the shadow slot.
    pub depth_remap: [f32; 2],
}

pub const PACK_SCENE_UNIFORM_SIZE: usize = 176;

impl PackSceneUniform {
    pub fn to_bytes(&self) -> [u8; PACK_SCENE_UNIFORM_SIZE] {
        let mut out = [0u8; PACK_SCENE_UNIFORM_SIZE];
        let mut put = |off: usize, bytes: [u8; 4]| out[off..off + 4].copy_from_slice(&bytes);
        for c in 0..4 {
            for r in 0..4 {
                put(c * 16 + r * 4, self.model_view[c][r].to_le_bytes());
                put(64 + c * 16 + r * 4, self.projection[c][r].to_le_bytes());
            }
        }
        for i in 0..3 {
            put(128 + i * 4, self.base_block_pos[i].to_le_bytes());
            put(144 + i * 4, self.camera_sub_pos[i].to_le_bytes());
        }
        put(140, self.frame_id.to_le_bytes());
        put(160, self.depth_remap[0].to_le_bytes());
        put(164, self.depth_remap[1].to_le_bytes());
        out
    }
}

/// How the vertex shader writes depth so LOD fragments land in the pack's main depth
/// (`PackDepthTarget::Main`, ruling point 40) with the vanilla terrain's depth convention.
/// `clip.z = x + y * clip.w` gives an infinite-far projection, matching the `far`-free LOD range.
pub struct DepthRemap;

impl DepthRemap {
    /// Reversed-Z main depth (`1.0` near, `0.0` far, vanilla 26.3 world passes): `z/w = near / -view.z`.
    pub const fn reversed(near: f32) -> [f32; 2] {
        [near, 0.0]
    }

    /// Standard-Z main depth (`0.0` near, `1.0` far; a pack's GL depth): `z/w = 1 - near / -view.z`.
    pub const fn standard(near: f32) -> [f32; 2] {
        [-near, 1.0]
    }
}

/// The three pack-bound LOD pipelines (`#5370` / `#5371` / `#5700`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DhPackSlot {
    Terrain,
    Water,
    Shadow,
}

impl DhPackSlot {
    pub const ALL: [DhPackSlot; 3] = [DhPackSlot::Terrain, DhPackSlot::Water, DhPackSlot::Shadow];

    /// Pack program fallback chain (Iris `DHCompatInternal`: water falls back to the terrain program).
    pub const fn programs(self) -> &'static [&'static str] {
        match self {
            DhPackSlot::Terrain => DH_TERRAIN_PROGRAMS,
            DhPackSlot::Water => DH_WATER_PROGRAMS,
            DhPackSlot::Shadow => DH_SHADOW_PROGRAMS,
        }
    }

    pub const fn pass_slot(self) -> PackPassSlot {
        match self {
            DhPackSlot::Terrain => PackPassSlot::AfterOpaqueGbuffers,
            DhPackSlot::Water => PackPassSlot::AfterTranslucentGbuffers,
            DhPackSlot::Shadow => PackPassSlot::Shadow,
        }
    }

    pub const fn vertex_shader(self) -> &'static str {
        match self {
            DhPackSlot::Terrain => VOXY_PACK_TERRAIN_VSH,
            DhPackSlot::Water => VOXY_PACK_WATER_VSH,
            DhPackSlot::Shadow => VOXY_PACK_SHADOW_VSH,
        }
    }

    /// Whether the slot is drawn against vanilla terrain with the stencil mask.
    pub const fn stencil_masked(self) -> bool {
        !matches!(self, DhPackSlot::Shadow)
    }

    /// The `PackBinding` registered on this slot's `RenderPipeline`.
    pub fn pack_binding(self) -> PackBinding {
        let mut binding = PackBinding::new()
            .program_mode(PackProgramMode::ModVertexPackFragment)
            .slot(self.pass_slot());
        for program in self.programs() {
            binding = binding.program(*program);
        }
        for (name, value) in DH_PACK_DEFINES {
            binding = match value {
                Some(v) => binding.define_value(*name, *v),
                None => binding.define(*name),
            };
        }
        for uniform in dh_frame_uniform_decls() {
            binding = binding.frame_uniform(uniform);
        }
        if self.stencil_masked() {
            binding = binding.depth_target(PackDepthTarget::Main);
        }
        binding
    }
}

fn dh_frame_uniform_decls() -> Vec<PackUniformDecl> {
    let decl = |name: &str, ty| PackUniformDecl::new(name, ty);
    vec![
        decl("dhNearPlane", PackUniformType::Float),
        decl("dhFarPlane", PackUniformType::Float),
        decl("dhRenderDistance", PackUniformType::Int),
        decl("dhProjection", PackUniformType::Mat4).with_previous("dhPreviousProjection"),
        decl("dhProjectionInverse", PackUniformType::Mat4),
    ]
}

/// Stencil state of the LOD opaque / translucent slots against vanilla terrain (`#5700` part A +
/// D2): vanilla terrain writes stencil `1`; LOD draws only where it reads `0` and never writes.
/// Voxy keeps its own buffer with the opposite polarity (`AbstractRenderPipeline.initDepthStencil`:
/// clear `1`, vanilla pixels `0`, test `GL_EQUAL 1`); the design inverts it for the shared buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilMask {
    pub reference: u8,
    pub read_mask: u8,
    pub write_mask: u8,
}

pub const LOD_STENCIL_MASK: StencilMask = StencilMask {
    reference: 0,
    read_mask: 0xFF,
    write_mask: 0,
};

/// Per-frame snapshot of the fields of [`ShaderPackState`] this mod reads. Built once at the
/// boundary so the gating logic is plain data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PackView {
    pub active: bool,
    pub hides_clouds: bool,
    /// `state.shadow.is_some()`.
    pub shadow: bool,
    pub generation: u32,
}

impl PackView {
    pub const INACTIVE: PackView = PackView {
        active: false,
        hides_clouds: false,
        shadow: false,
        generation: 0,
    };

    pub fn from_state(state: Option<&ShaderPackState>) -> PackView {
        match state {
            None => PackView::INACTIVE,
            Some(s) => PackView {
                active: true,
                hides_clouds: s.hides_clouds,
                shadow: s.shadow.is_some(),
                generation: s.generation,
            },
        }
    }
}

/// `RenderAssets::pipeline_pack_bound` result per slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PackBound {
    pub terrain: bool,
    pub water: bool,
    pub shadow: bool,
}

/// What to do this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackFrame {
    /// The pack was loaded, switched or turned off since the last frame: drop every value derived
    /// from the pack (`mc_Entity` ids, baked model words) and rebuild them.
    pub rebuild: bool,
    /// Issue the shadow-pass indirect draws. Iris cancels DH in the shadow pass unless
    /// `shouldOverrideShadow` (the pack has `dh_shadow`, `DHCompatInternal`); here that is
    /// `state.shadow.is_some()` and the slot being bound.
    pub draw_shadow: bool,
    /// Draw LOD clouds: false when the pack hides clouds (`avoidRenderingClouds`).
    pub draw_clouds: bool,
}

/// Tracks `ShaderPackState::generation` (`#5698`, ruling point 33).
///
/// Pipelines and arenas are mod GPU objects that survive reloads (ruling point 2), so a pack
/// switch never frees or recreates them: only the derived CPU/GPU *contents* are rebuilt.
#[derive(Debug, Default)]
pub struct PackSession {
    last: Option<(bool, u32)>,
}

impl PackSession {
    pub fn begin_frame(&mut self, view: PackView, bound: PackBound) -> PackFrame {
        let key = (view.active, if view.active { view.generation } else { 0 });
        let rebuild = self.last != Some(key);
        self.last = Some(key);
        PackFrame {
            rebuild,
            draw_shadow: view.active && view.shadow && bound.shadow,
            draw_clouds: !(view.active && view.hides_clouds),
        }
    }
}

/// Per-generation cache of the pack's `block.properties` ids (`RenderAssets::shader_pack_block_id`,
/// `#5699`). Unmapped states get `0`, as Voxy's `ModelFactory` writes `0` when
/// `customBlockStateIdMapping` has no entry.
#[derive(Debug)]
pub struct MaterialIdTable<K> {
    ids: std::collections::HashMap<K, i32>,
    epoch: u32,
}

impl<K: std::hash::Hash + Eq> Default for MaterialIdTable<K> {
    fn default() -> Self {
        Self {
            ids: Default::default(),
            epoch: 0,
        }
    }
}

impl<K: std::hash::Hash + Eq> MaterialIdTable<K> {
    /// Id for `key`, resolving it once per generation. `resolve` is `shader_pack_block_id`.
    pub fn get_or_resolve(&mut self, key: K, resolve: impl FnOnce(&K) -> Option<i32>) -> i32 {
        *self
            .ids
            .entry(key)
            .or_insert_with_key(|k| resolve(k).unwrap_or(0))
    }

    /// Drop every cached id and release the map's memory. `epoch` changes so the arena owner
    /// knows to rewrite the model buffer's `customId` words.
    pub fn clear(&mut self) {
        self.ids = Default::default();
        self.epoch = self.epoch.wrapping_add(1);
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
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
    fn pack_defines_cover_distant_sunrises_and_all_16_material_ids() {
        assert_eq!(DH_PACK_DEFINES.len(), 17);
        assert_eq!(DH_PACK_DEFINES[0], ("DISTANT_HORIZONS", None));
        assert_eq!(DH_PACK_DEFINES[13], ("DH_BLOCK_WATER", Some("12")));
        assert_eq!(DH_PACK_DEFINES[14], ("DH_BLOCK_GRASS", Some("13")));
        assert_eq!(DH_PACK_DEFINES[15], ("DH_BLOCK_AIR", Some("14")));
        assert_eq!(DH_PACK_DEFINES[16], ("DH_BLOCK_ILLUMINATED", Some("15")));
    }

    #[test]
    fn packed_quad_roundtrips_voxy_layout() {
        let d = packed_quad::Decoded {
            face: 5,
            width: 16,
            height: 3,
            pos: [31, 0, 17],
            state: 0xA_BCDE,
            biome: 0x1FF,
            light: 0xF1,
        };
        assert_eq!(packed_quad::decode(packed_quad::encode(d)), d);
        // x = bits 21..25 per voxy `extractPos`
        assert_eq!(
            packed_quad::encode(packed_quad::Decoded {
                pos: [1, 0, 0],
                width: 1,
                height: 1,
                ..d_zero()
            }),
            1 << 21
        );
    }

    fn d_zero() -> packed_quad::Decoded {
        packed_quad::Decoded {
            face: 0,
            width: 1,
            height: 1,
            pos: [0; 3],
            state: 0,
            biome: 0,
            light: 0,
        }
    }

    #[test]
    fn scene_uniform_is_std140_176_bytes() {
        let u = PackSceneUniform {
            model_view: IDENTITY_MAT4,
            projection: IDENTITY_MAT4,
            base_block_pos: [32, -64, 96],
            frame_id: 7,
            camera_sub_pos: [1.0, 2.0, 3.0],
            depth_remap: DepthRemap::reversed(0.5),
        };
        let b = u.to_bytes();
        assert_eq!(b.len(), 176);
        assert_eq!(i32::from_le_bytes(b[128..132].try_into().unwrap()), 32);
        assert_eq!(i32::from_le_bytes(b[136..140].try_into().unwrap()), 96);
        assert_eq!(u32::from_le_bytes(b[140..144].try_into().unwrap()), 7);
        assert_eq!(f32::from_le_bytes(b[152..156].try_into().unwrap()), 3.0);
        assert_eq!(f32::from_le_bytes(b[160..164].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(b[80..84].try_into().unwrap()), 1.0);
    }

    #[test]
    fn depth_remap_matches_projection_conventions() {
        let (near, view_z) = (0.5f32, -200.0f32);
        let w = -view_z;
        let r = DepthRemap::reversed(near);
        assert!(((r[0] + r[1] * w) / w - near / w).abs() < 1e-7);
        let s = DepthRemap::standard(near);
        assert!(((s[0] + s[1] * w) / w - (1.0 - near / w)).abs() < 1e-6);
    }

    #[test]
    fn model_flags_place_material_above_voxy_flags() {
        let f = model_flags(true, false, true, DH_BLOCK_WATER);
        assert_eq!(f & 0xF, 0b1010);
        assert_eq!(
            (f >> MODEL_FLAGS_DH_MATERIAL_SHIFT) & 0xF,
            DH_BLOCK_WATER as u32
        );
    }

    #[test]
    fn pack_binding_configured_for_all_three_slots() {
        for slot in DhPackSlot::ALL {
            let b = slot.pack_binding();
            assert_eq!(
                b.program_mode,
                PackProgramMode::ModVertexPackFragment,
                "{slot:?}"
            );
            assert_eq!(b.slot, Some(slot.pass_slot()), "{slot:?}");
            assert_eq!(b.programs, slot.programs().to_vec(), "{slot:?}");
            assert_eq!(b.defines.len(), DH_PACK_DEFINES.len());
            assert_eq!(b.frame_uniforms.len(), 5);
            assert!(slot.vertex_shader().ends_with(".vsh"));
        }
        let t = DhPackSlot::Terrain.pack_binding();
        assert_eq!(t.programs, vec!["dh_terrain"]);
        assert_eq!(t.slot, Some(PackPassSlot::AfterOpaqueGbuffers));
        assert_eq!(t.depth_target, Some(PackDepthTarget::Main));
        let w = DhPackSlot::Water.pack_binding();
        assert_eq!(w.programs, vec!["dh_water", "dh_terrain"]);
        assert_eq!(w.slot, Some(PackPassSlot::AfterTranslucentGbuffers));
        assert_eq!(w.depth_target, Some(PackDepthTarget::Main));
        let sh = DhPackSlot::Shadow.pack_binding();
        assert_eq!(sh.programs, vec!["dh_shadow"]);
        assert_eq!(sh.slot, Some(PackPassSlot::Shadow));
        assert_eq!(sh.depth_target, None);
    }

    #[test]
    fn stencil_mask_reads_zero_and_never_writes() {
        assert_eq!(
            LOD_STENCIL_MASK,
            StencilMask {
                reference: 0,
                read_mask: 0xFF,
                write_mask: 0
            }
        );
        assert!(DhPackSlot::Terrain.stencil_masked() && DhPackSlot::Water.stencil_masked());
        assert!(!DhPackSlot::Shadow.stencil_masked());
    }

    fn view(active: bool, clouds: bool, shadow: bool, generation: u32) -> PackView {
        PackView {
            active,
            hides_clouds: clouds,
            shadow,
            generation,
        }
    }
    const ALL_BOUND: PackBound = PackBound {
        terrain: true,
        water: true,
        shadow: true,
    };

    #[test]
    fn generation_transitions_trigger_exactly_one_rebuild() {
        let mut s = PackSession::default();
        assert!(
            s.begin_frame(PackView::INACTIVE, PackBound::default())
                .rebuild
        ); // first frame
        assert!(
            !s.begin_frame(PackView::INACTIVE, PackBound::default())
                .rebuild
        );
        assert!(s.begin_frame(view(true, false, true, 1), ALL_BOUND).rebuild); // pack loaded
        assert!(!s.begin_frame(view(true, false, true, 1), ALL_BOUND).rebuild);
        assert!(s.begin_frame(view(true, false, true, 2), ALL_BOUND).rebuild); // F3+T / switch
        assert!(
            s.begin_frame(PackView::INACTIVE, PackBound::default())
                .rebuild
        ); // turned off
        assert!(
            !s.begin_frame(PackView::INACTIVE, PackBound::default())
                .rebuild
        );
        assert!(s.begin_frame(view(true, false, true, 2), ALL_BOUND).rebuild); // back on, same gen
    }

    #[test]
    fn clouds_suppressed_only_when_active_pack_hides_them() {
        let mut s = PackSession::default();
        assert!(
            s.begin_frame(PackView::INACTIVE, PackBound::default())
                .draw_clouds
        );
        assert!(
            s.begin_frame(view(true, false, false, 1), ALL_BOUND)
                .draw_clouds
        );
        assert!(
            !s.begin_frame(view(true, true, false, 1), ALL_BOUND)
                .draw_clouds
        );
        assert!(
            s.begin_frame(view(false, true, false, 0), PackBound::default())
                .draw_clouds
        );
    }

    #[test]
    fn shadow_draws_need_pack_shadow_and_a_bound_slot() {
        let mut s = PackSession::default();
        assert!(
            s.begin_frame(view(true, false, true, 1), ALL_BOUND)
                .draw_shadow
        );
        assert!(
            !s.begin_frame(view(true, false, false, 1), ALL_BOUND)
                .draw_shadow
        );
        let unbound = PackBound {
            shadow: false,
            ..ALL_BOUND
        };
        assert!(
            !s.begin_frame(view(true, false, true, 1), unbound)
                .draw_shadow
        );
        assert!(!s.begin_frame(PackView::INACTIVE, ALL_BOUND).draw_shadow);
    }

    #[test]
    fn material_ids_resolve_once_per_generation_and_clear_releases() {
        let mut t = MaterialIdTable::<u32>::default();
        let mut calls = 0;
        assert_eq!(
            t.get_or_resolve(1, |_| {
                calls += 1;
                Some(10008)
            }),
            10008
        );
        assert_eq!(
            t.get_or_resolve(1, |_| {
                calls += 1;
                Some(0)
            }),
            10008
        );
        assert_eq!(
            t.get_or_resolve(2, |_| {
                calls += 1;
                None
            }),
            0
        ); // unmapped -> 0 (voxy)
        assert_eq!(calls, 2);
        let epoch = t.epoch();
        t.clear();
        assert!(t.is_empty());
        assert_ne!(t.epoch(), epoch);
        assert_eq!(t.get_or_resolve(1, |_| Some(10020)), 10020);
    }
}
