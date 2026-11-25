//! Baked per-face block models for the 32³ octree mesher (Voxy `ModelFactory` / `ModelBakery`).
//!
//! Voxy bakes every block state once by rasterising its model orthographically from each of the
//! six sides into a `MODEL_TEXTURE_SIZE = 16` colour+depth image
//! (`client.core.model.ModelFactory`, `bakery.*`), derives per-face metadata (exists, occludes,
//! can-be-occluded, self-lighting, cut-out, tint state, covered rectangle, depth) and a model-level
//! metadata long (`ModelQueries`: biome coloured, translucent, double sided, contains fluid, is
//! fluid, culls same, fully opaque, light emission), deduplicates identical bakes, and stores a
//! custom block-state id (`customBlockStateIdMapping`, fed from Iris'
//! `WorldRenderingSettings.getBlockStateIds()`) next to every model.
//!
//! This module does the same from Sunrise's mod API instead of a GL rasteriser: the caller turns
//! `RenderAssets::block_state_models` + `baked_model` quads into [`BakeQuad`]s (they carry
//! per-face geometry, UVs, sprite key and tint index), supplies sprite pixels through
//! [`SpriteSource`] (`#4601` `RenderAssets::atlas_sprite_pixels`: frame 0, untinted RGBA8, errors
//! `NoSuchSprite` / `NotReady` per ruling 6-8), and the Iris `block.properties` id
//! (`RenderAssets::shader_pack_block_id`, `#5699`). Each face is rasterised into a 4×4 texel grid
//! (the compact GPU palette; Voxy keeps 16×16 and notes 8×8 would also do, `ModelFactory` TODO at
//! `MODEL_TEXTURE_SIZE`) plus an alpha-weighted average used by coarse detail levels.
//!
//! Biome grass/foliage/water colours are not baked in: each face records a tint state and a
//! 16-bit texel mask, and the GPU multiplies by the biome row of [`BiomeTintTable`]
//! (`quads3.vert`: `colourData[tintColour + biomeId]`). Water is `water_still.png` multiplied by
//! the biome water colour with the alpha forced to the pack's `water_still` alpha (vanilla: 180).

use crate::colors::{TintSource, classify_dh_material, is_water_block, tint_source_for_block};
use crate::octree::Face;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

/// Texels along one edge of a baked face grid.
pub const FACE_GRID: usize = 4;
/// Texels per baked face.
pub const FACE_TEXELS: usize = FACE_GRID * FACE_GRID;

/// `ModelQueries` model flag bits (`metadata >> 8*6`).
pub const MODEL_BIOME_COLOURED: u8 = 1;
pub const MODEL_TRANSLUCENT: u8 = 2;
pub const MODEL_DOUBLE_SIDED: u8 = 4;
pub const MODEL_CONTAINS_FLUID: u8 = 8;
pub const MODEL_FLUID: u8 = 16;
pub const MODEL_CULLS_SAME: u8 = 32;
pub const MODEL_FULLY_OPAQUE: u8 = 64;

/// Per-face GPU flag bits (`face_flags`): bits `0..16` are the tint texel mask.
pub const FACE_EXISTS: u32 = 1 << 16;
pub const FACE_OCCLUDES: u32 = 1 << 17;
pub const FACE_CAN_BE_OCCLUDED: u32 = 1 << 18;
pub const FACE_CUTOUT: u32 = 1 << 19;
pub const FACE_SELF_LIGHTING: u32 = 1 << 20;
/// Tint state at bits 21..23: 0 none, 1 partial, 2 full (`ModelQueries` / `computeFaceTint`).
pub const FACE_TINT_SHIFT: u32 = 21;
/// Depth offset at bits 24..30: `min(round(offset * 64), 62)` (`ModelFactory`).
pub const FACE_DEPTH_SHIFT: u32 = 24;

/// `sprite_pixels` failures. Ruling 7 merges `NotASpriteKey` into `NoSuchSprite` and renames
/// `Unsupported` to `NotReady`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpriteError {
    NoSuchSprite,
    NotReady,
}

/// Frame 0 of an atlas sprite, untinted, tightly packed top-left RGBA8
/// (`AtlasSpritePixels { width, height, rgba }`).
#[derive(Clone, Debug)]
pub struct SpritePixels {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl SpritePixels {
    fn pixel(&self, u: f32, v: f32) -> [u8; 4] {
        if self.width == 0
            || self.height == 0
            || self.rgba.len() < (self.width * self.height * 4) as usize
        {
            return [0; 4];
        }
        let x = ((u * self.width as f32).floor().max(0.0) as u32).min(self.width - 1);
        let y = ((v * self.height as f32).floor().max(0.0) as u32).min(self.height - 1);
        let i = ((y * self.width + x) * 4) as usize;
        [
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ]
    }
}

/// Where sprite pixels come from; the mod wires this to `RenderAssets::atlas_sprite_pixels`.
pub trait SpriteSource {
    fn sprite_pixels(&self, sprite: &str) -> Result<SpritePixels, SpriteError>;
}

/// One baked-model quad (`BakedQuad`): corners counter-clockwise seen from the facing side, in
/// block units; UVs 0..1 across the sprite.
#[derive(Clone, Debug, PartialEq)]
pub struct BakeQuad {
    pub positions: [[f32; 3]; 4],
    pub uvs: [[f32; 2]; 4],
    pub sprite: String,
    pub facing: Face,
    pub tint_index: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluidKind {
    Water,
    Lava,
}

/// Everything needed to bake one block state.
#[derive(Clone, Debug)]
pub struct BakeRequest {
    /// Id the octree voxels carry (`Voxel::block_id`).
    pub block_id: u32,
    /// Namespaced block id (`minecraft:oak_leaves`) for tint and material classification.
    pub name: String,
    /// Quads of the state's model (empty for pure fluids and invisible blocks).
    pub quads: Vec<BakeQuad>,
    /// Set for fluid blocks, which have no baked model (`ModelFactory` bakes fluid states from the
    /// fluid renderer).
    pub fluid: Option<FluidKind>,
    /// Render layer is translucent (`ChunkSectionLayer.TRANSLUCENT`); not carried by
    /// `BakedModelView`, so the caller decides (see [`default_translucent`]).
    pub translucent: bool,
    /// `RenderAssets::shader_pack_block_id` (`#5699`); `None` when no pack or no entry.
    pub mc_entity: Option<i32>,
    /// `0..=15` block light emission (`ModelFactory.getBlockLightEmission`).
    pub light_emission: u8,
}

/// Heuristic translucent-layer test for blocks whose model carries no render type.
pub fn default_translucent(name: &str) -> bool {
    let n = name.strip_prefix("minecraft:").unwrap_or(name);
    is_water_block(n)
        || n == "glass"
        || n.ends_with("_stained_glass")
        || n.ends_with("glass_pane")
        || n == "tinted_glass"
        || n == "ice"
        || n == "frosted_ice"
        || n == "slime_block"
        || n == "honey_block"
}

/// Baked data of one face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BakedFace {
    pub exists: bool,
    pub occludes: bool,
    pub can_be_occluded: bool,
    pub cutout: bool,
    pub self_lighting: bool,
    /// 0 none, 1 partial, 2 full.
    pub tint_state: u8,
    /// Texels (row-major `[b][a]`) whose colour the biome tint multiplies.
    pub tint_mask: u16,
    /// Covered rectangle on the `0..16` scale: `[a0, a1, b0, b1]` (`faceSize`).
    pub rect: [u8; 4],
    /// `min(round(depth * 64), 62)`.
    pub depth: u8,
    /// Alpha-weighted average RGBA (untinted).
    pub avg: [u8; 4],
    /// Packed RGBA8 texels (`r | g << 8 | b << 16 | a << 24`).
    pub texels: [u32; FACE_TEXELS],
}

impl BakedFace {
    const EMPTY: BakedFace = BakedFace {
        exists: false,
        occludes: false,
        can_be_occluded: false,
        cutout: false,
        self_lighting: false,
        tint_state: 0,
        tint_mask: 0,
        rect: [0; 4],
        depth: 0,
        avg: [0; 4],
        texels: [0; FACE_TEXELS],
    };

    fn gpu_flags(&self) -> u32 {
        let mut f = self.tint_mask as u32;
        if self.exists {
            f |= FACE_EXISTS;
        }
        if self.occludes {
            f |= FACE_OCCLUDES;
        }
        if self.can_be_occluded {
            f |= FACE_CAN_BE_OCCLUDED;
        }
        if self.cutout {
            f |= FACE_CUTOUT;
        }
        if self.self_lighting {
            f |= FACE_SELF_LIGHTING;
        }
        f | ((self.tint_state as u32) << FACE_TINT_SHIFT)
            | ((self.depth as u32 & 0x3F) << FACE_DEPTH_SHIFT)
    }

    fn gpu_rect(&self) -> u32 {
        (self.rect[0] as u32 & 15)
            | ((self.rect[1] as u32 & 15) << 4)
            | ((self.rect[2] as u32 & 15) << 8)
            | ((self.rect[3] as u32 & 15) << 12)
    }
}

/// One baked model: six faces plus model metadata and the shader-pack ids.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BakedModel {
    pub faces: [BakedFace; 6],
    /// `MODEL_*` bits.
    pub flags: u8,
    pub tint: BakedTint,
    /// Active shader pack `block.properties` id (`mc_Entity`), `-1` when none (`BlockIds` sentinel).
    pub mc_entity: i32,
    /// Iris/DH 4-bit `dhMaterialId` (`colors::DH_BLOCK_*`).
    pub dh_material_id: u8,
    pub light_emission: u8,
}

/// Tint source of a model, kept for CPU-side colour lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BakedTint {
    None,
    Grass,
    Foliage,
    Water,
}

/// What the mesher reads per block id (`ModelQueries` metadata subset).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelMeta {
    /// Index into the GPU model buffer; `0` = nothing to draw (air-like).
    pub model_id: u32,
    pub flags: u8,
    /// Bit `Face::id()` set when the face exists.
    pub face_exists: u8,
    /// Bit `Face::id()` set when the face occludes the neighbouring face.
    pub face_occludes: u8,
    pub light_emission: u8,
}

impl ModelMeta {
    pub const EMPTY: ModelMeta = ModelMeta {
        model_id: 0,
        flags: 0,
        face_exists: 0,
        face_occludes: 0,
        light_emission: 0,
    };

    pub fn is_empty(self) -> bool {
        self.model_id == 0
    }
    pub fn translucent(self) -> bool {
        self.flags & MODEL_TRANSLUCENT != 0
    }
    pub fn fluid(self) -> bool {
        self.flags & MODEL_FLUID != 0
    }
    pub fn contains_fluid(self) -> bool {
        self.flags & MODEL_CONTAINS_FLUID != 0
    }
    pub fn biome_coloured(self) -> bool {
        self.flags & MODEL_BIOME_COLOURED != 0
    }
    pub fn culls_same(self) -> bool {
        self.flags & MODEL_CULLS_SAME != 0
    }
    pub fn fully_opaque(self) -> bool {
        self.flags & MODEL_FULLY_OPAQUE != 0
    }
    pub fn double_sided(self) -> bool {
        self.flags & MODEL_DOUBLE_SIDED != 0
    }
    pub fn has_face(self, face: Face) -> bool {
        self.face_exists & (1 << face.id()) != 0
    }
    /// `ModelQueries.faceOccludes`.
    pub fn face_occludes(self, face: Face) -> bool {
        self.face_occludes & (1 << face.id()) != 0
    }
}

/// Block id -> [`ModelMeta`]. Implemented by [`BakedModelTable`] and by any closure.
pub trait ModelLookup {
    fn meta(&self, block_id: u32) -> ModelMeta;
}

impl<F: Fn(u32) -> ModelMeta> ModelLookup for F {
    fn meta(&self, block_id: u32) -> ModelMeta {
        self(block_id)
    }
}

/// GPU model entry, 112 bytes (`std430`-friendly, 16-byte multiple): per face average colour,
/// flags, covered rectangle and texel offset implied as `(model_id * 6 + face) * 16` into the
/// texel buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuModelEntry {
    pub face_avg: [u32; 6],
    pub face_flags: [u32; 6],
    pub face_rect: [u32; 6],
    pub face_pad: [u32; 6],
    pub flags: u32,
    pub mc_entity: i32,
    pub dh_material_id: u32,
    pub light_emission: u32,
}

#[inline]
pub const fn pack_rgba(c: [u8; 4]) -> u32 {
    (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
}

/// Row of the biome tint buffer: packed RGBA8 grass, foliage, water and padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuBiomeTint {
    pub grass: u32,
    pub foliage: u32,
    pub water: u32,
    pub pad: u32,
}

/// Biome id -> tint colours, indexed like the octree voxel `biome_id`.
#[derive(Clone, Debug, Default)]
pub struct BiomeTintTable {
    pub rows: Vec<crate::colors::BiomeTints>,
}

impl BiomeTintTable {
    /// `biomes[i]` is the namespaced biome id with voxel biome id `i`.
    pub fn build(biomes: &[&str], water_color_hint: impl Fn(&str) -> u32) -> Self {
        Self {
            rows: biomes
                .iter()
                .map(|b| crate::colors::BiomeTints::for_biome(b, water_color_hint(b)))
                .collect(),
        }
    }

    pub fn gpu_rows(&self) -> Vec<GpuBiomeTint> {
        self.rows
            .iter()
            .map(|t| GpuBiomeTint {
                grass: pack_rgba([t.grass[0], t.grass[1], t.grass[2], 255]),
                foliage: pack_rgba([t.foliage[0], t.foliage[1], t.foliage[2], 255]),
                water: pack_rgba([t.water[0], t.water[1], t.water[2], 255]),
                pad: 0,
            })
            .collect()
    }
}

/// All baked models of one resource generation (+ shader-pack generation, `#5698`).
#[derive(Debug)]
pub struct BakedModelTable {
    models: Vec<BakedModel>,
    by_model: HashMap<BakedModel, u32>,
    block_to_model: HashMap<u32, u32>,
}

impl Default for BakedModelTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Tangent coordinates of a point for a face: `(a, b, depth_axis_coordinate)`. Tangent axes match
/// the mesher: Y faces `(x, z)`, Z faces `(x, y)`, X faces `(y, z)` (`RenderDataFactory.Mesher.emitQuad`).
fn face_coords(face: Face, p: [f32; 3]) -> (f32, f32, f32) {
    match face.axis() {
        0 => (p[0], p[2], p[1]),
        1 => (p[0], p[1], p[2]),
        _ => (p[1], p[2], p[0]),
    }
}

fn mix_rgb(c: [u8; 3], m: [u8; 3]) -> [u8; 3] {
    [
        ((c[0] as u16 * m[0] as u16) / 255) as u8,
        ((c[1] as u16 * m[1] as u16) / 255) as u8,
        ((c[2] as u16 * m[2] as u16) / 255) as u8,
    ]
}

impl BakedModelTable {
    pub fn new() -> Self {
        let empty = BakedModel {
            faces: [BakedFace::EMPTY; 6],
            flags: 0,
            tint: BakedTint::None,
            mc_entity: -1,
            dh_material_id: 0,
            light_emission: 0,
        };
        let mut by_model = HashMap::new();
        by_model.insert(empty.clone(), 0);
        Self {
            models: vec![empty],
            by_model,
            block_to_model: HashMap::new(),
        }
    }

    /// Drop everything (resource reload or shader-pack `generation` change).
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.len() <= 1
    }

    pub fn model(&self, model_id: u32) -> Option<&BakedModel> {
        self.models.get(model_id as usize)
    }

    pub fn model_id_for_block(&self, block_id: u32) -> u32 {
        self.block_to_model.get(&block_id).copied().unwrap_or(0)
    }

    /// Bake one block state; returns its model id (`0` when it draws nothing). Identical bakes
    /// share one id (`ModelFactory` deduplication).
    pub fn bake(&mut self, req: &BakeRequest, sprites: &dyn SpriteSource) -> u32 {
        let model = bake_model(req, sprites);
        let id = match model {
            None => 0,
            Some(m) => match self.by_model.get(&m) {
                Some(&id) => id,
                None => {
                    let id = self.models.len() as u32;
                    self.by_model.insert(m.clone(), id);
                    self.models.push(m);
                    id
                }
            },
        };
        self.block_to_model.insert(req.block_id, id);
        id
    }

    /// Mesher metadata for a block id (empty for unknown ids).
    pub fn meta_for(&self, block_id: u32) -> ModelMeta {
        let id = self.model_id_for_block(block_id);
        if id == 0 {
            return ModelMeta::EMPTY;
        }
        let m = &self.models[id as usize];
        let mut exists = 0u8;
        let mut occ = 0u8;
        for f in Face::ALL {
            let face = &m.faces[f as usize];
            if face.exists {
                exists |= 1 << f.id();
            }
            if face.occludes {
                occ |= 1 << f.id();
            }
        }
        ModelMeta {
            model_id: id,
            flags: m.flags,
            face_exists: exists,
            face_occludes: occ,
            light_emission: m.light_emission,
        }
    }

    pub fn gpu_models(&self) -> Vec<GpuModelEntry> {
        self.models
            .iter()
            .map(|m| {
                let mut e = GpuModelEntry {
                    flags: m.flags as u32,
                    mc_entity: m.mc_entity,
                    dh_material_id: m.dh_material_id as u32,
                    light_emission: m.light_emission as u32,
                    ..Default::default()
                };
                for (i, f) in m.faces.iter().enumerate() {
                    e.face_avg[i] = pack_rgba(f.avg);
                    e.face_flags[i] = f.gpu_flags();
                    e.face_rect[i] = f.gpu_rect();
                }
                e
            })
            .collect()
    }

    /// Texel buffer: `(model * 6 + face) * FACE_TEXELS` is a face's first texel.
    pub fn gpu_texels(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.models.len() * 6 * FACE_TEXELS);
        for m in &self.models {
            for f in &m.faces {
                out.extend_from_slice(&f.texels);
            }
        }
        out
    }
}

impl ModelLookup for BakedModelTable {
    fn meta(&self, block_id: u32) -> ModelMeta {
        self.meta_for(block_id)
    }
}

fn fluid_quads(kind: FluidKind) -> Vec<BakeQuad> {
    let (still, flow) = match kind {
        FluidKind::Water => ("minecraft:block/water_still", "minecraft:block/water_flow"),
        FluidKind::Lava => ("minecraft:block/lava_still", "minecraft:block/lava_flow"),
    };
    let mut out = Vec::new();
    for face in Face::ALL {
        let vertical = face.axis() == 0;
        let (a0, a1) = (0.0f32, 1.0f32);
        let plane = if face.is_positive() { 1.0 } else { 0.0 };
        let corner = |a: f32, b: f32| -> [f32; 3] {
            match face.axis() {
                0 => [a, plane, b],
                1 => [a, b, plane],
                _ => [plane, a, b],
            }
        };
        let positions = [
            corner(a0, a0),
            corner(a1, a0),
            corner(a1, a1),
            corner(a0, a1),
        ];
        // The flow sprite is two blocks wide: vanilla maps one block to half of it.
        let s = if vertical { 1.0 } else { 0.5 };
        out.push(BakeQuad {
            positions,
            uvs: [[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]],
            sprite: (if vertical { still } else { flow }).to_string(),
            facing: face,
            tint_index: 0,
        });
    }
    out
}

/// Rasterise one face of the request into the 4×4 grid.
#[allow(clippy::too_many_arguments)]
fn bake_face(
    face: Face,
    quads: &[&BakeQuad],
    sprites: &mut dyn FnMut(&str) -> Option<SpritePixels>,
    flat: [u8; 3],
    tinted: bool,
    fixed_mul: Option<[u8; 3]>,
    translucent: bool,
    water_alpha: Option<u8>,
) -> BakedFace {
    if quads.is_empty() {
        return BakedFace::EMPTY;
    }
    // Nearest quad first (Voxy's depth test keeps the closest fragment).
    struct Q<'a> {
        q: &'a BakeQuad,
        depth: f32,
        bbox: [f32; 4],
        det: f32,
        c0: (f32, f32),
        e1: (f32, f32),
        e3: (f32, f32),
        tinted: bool,
    }
    let mut qs: Vec<Q> = Vec::new();
    for &q in quads {
        let c: Vec<(f32, f32, f32)> = q.positions.iter().map(|&p| face_coords(face, p)).collect();
        let (mut a0, mut a1, mut b0, mut b1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        let mut d = 0.0f32;
        for &(a, b, depth_axis) in &c {
            a0 = a0.min(a);
            a1 = a1.max(a);
            b0 = b0.min(b);
            b1 = b1.max(b);
            d += depth_axis;
        }
        d /= 4.0;
        let depth = if face.is_positive() { 1.0 - d } else { d };
        let e1 = (c[1].0 - c[0].0, c[1].1 - c[0].1);
        let e3 = (c[3].0 - c[0].0, c[3].1 - c[0].1);
        let det = e1.0 * e3.1 - e1.1 * e3.0;
        if det.abs() < 1e-9 || !(a1 > a0 && b1 > b0) {
            continue;
        }
        qs.push(Q {
            q,
            depth,
            bbox: [a0, a1, b0, b1],
            det,
            c0: (c[0].0, c[0].1),
            e1,
            e3,
            tinted: q.tint_index >= 0 && tinted,
        });
    }
    if qs.is_empty() {
        return BakedFace::EMPTY;
    }
    qs.sort_by(|x, y| x.depth.total_cmp(&y.depth));
    let nearest = qs[0].depth;
    // Voxy: a face deeper than -0.1 exists (`offset < -0.1` is empty).
    if nearest < -0.1 {
        return BakedFace::EMPTY;
    }

    let mut cache: HashMap<String, Option<SpritePixels>> = HashMap::new();
    let mut texels = [0u32; FACE_TEXELS];
    let mut mask = 0u16;
    let (mut r_sum, mut g_sum, mut b_sum, mut w_sum) = (0u64, 0u64, 0u64, 0u64);
    let (mut covered, mut tinted_texels, mut opaque_texels) = (0usize, 0usize, 0usize);
    let (mut ra0, mut ra1, mut rb0, mut rb1) = (FACE_GRID, 0usize, FACE_GRID, 0usize);

    for j in 0..FACE_GRID {
        for i in 0..FACE_GRID {
            let (mut r, mut g, mut b, mut a_sum) = (0u32, 0u32, 0u32, 0u32);
            let mut weight = 0u32;
            let mut tint_votes = 0u32;
            let mut taken = 0u32;
            for sj in 0..2 {
                for si in 0..2 {
                    let pa = (i as f32 + (si as f32 + 0.5) / 2.0) / FACE_GRID as f32;
                    let pb = (j as f32 + (sj as f32 + 0.5) / 2.0) / FACE_GRID as f32;
                    for q in &qs {
                        const EPS: f32 = 1e-4;
                        if pa < q.bbox[0] - EPS
                            || pa > q.bbox[1] + EPS
                            || pb < q.bbox[2] - EPS
                            || pb > q.bbox[3] + EPS
                        {
                            continue;
                        }
                        let (da, db) = (pa - q.c0.0, pb - q.c0.1);
                        let s = ((da * q.e3.1 - db * q.e3.0) / q.det).clamp(0.0, 1.0);
                        let t = ((q.e1.0 * db - q.e1.1 * da) / q.det).clamp(0.0, 1.0);
                        let u = q.q.uvs[0][0]
                            + s * (q.q.uvs[1][0] - q.q.uvs[0][0])
                            + t * (q.q.uvs[3][0] - q.q.uvs[0][0]);
                        let v = q.q.uvs[0][1]
                            + s * (q.q.uvs[1][1] - q.q.uvs[0][1])
                            + t * (q.q.uvs[3][1] - q.q.uvs[0][1]);
                        let px = match cache
                            .entry(q.q.sprite.clone())
                            .or_insert_with(|| sprites(&q.q.sprite))
                        {
                            Some(p) => p.pixel(u, v),
                            // Sprite pixels unavailable (`NotReady` before the first resource
                            // generation): fall back to the static block colour.
                            None => [flat[0], flat[1], flat[2], 255],
                        };
                        if px[3] == 0 {
                            continue;
                        }
                        let w = px[3] as u32;
                        r += px[0] as u32 * w;
                        g += px[1] as u32 * w;
                        b += px[2] as u32 * w;
                        a_sum += w;
                        weight += w;
                        taken += 1;
                        if q.tinted {
                            tint_votes += 1;
                        }
                        break;
                    }
                }
            }
            if taken == 0 {
                continue;
            }
            let mut rgb = [(r / weight) as u8, (g / weight) as u8, (b / weight) as u8];
            if let Some(m) = fixed_mul {
                rgb = mix_rgb(rgb, m);
            }
            // Uncovered subsamples count as transparent so rim texels get partial alpha.
            let mut alpha = (a_sum / 4) as u8;
            if let Some(wa) = water_alpha {
                alpha = wa;
            }
            texels[j * FACE_GRID + i] = pack_rgba([rgb[0], rgb[1], rgb[2], alpha]);
            covered += 1;
            if alpha == 255 {
                opaque_texels += 1;
            }
            if tint_votes * 2 >= taken {
                mask |= 1 << (j * FACE_GRID + i);
                tinted_texels += 1;
            }
            ra0 = ra0.min(i);
            ra1 = ra1.max(i);
            rb0 = rb0.min(j);
            rb1 = rb1.max(j);
            r_sum += rgb[0] as u64 * alpha as u64;
            g_sum += rgb[1] as u64 * alpha as u64;
            b_sum += rgb[2] as u64 * alpha as u64;
            w_sum += alpha as u64;
        }
    }
    if covered == 0 {
        return BakedFace::EMPTY;
    }
    let avg_a = (w_sum / covered as u64) as u8;
    let w = w_sum.max(1);
    let avg = [
        (r_sum / w) as u8,
        (g_sum / w) as u8,
        (b_sum / w) as u8,
        avg_a,
    ];

    // Voxy `ModelFactory` metadata rules.
    let mut occludes = !translucent; // translucent faces never occlude
    occludes &= nearest < 0.1; // far-away faces do not occlude
    occludes &= covered as f32 / FACE_TEXELS as f32 > 0.9; // must cover > 90% of the block face
    let can_be_occluded = nearest < 0.3;
    let self_lighting = nearest > 0.01 || translucent;
    // Rect on the 0..16 scale (Voxy `faceSize`, scaled from the grid like its 16-texel size).
    let scale = |v: usize| -> u8 { ((v as f32 / (FACE_GRID - 1) as f32) * 15.0).round() as u8 };
    let rect = [scale(ra0), scale(ra1), scale(rb0), scale(rb1)];
    let area = ((ra1 - ra0 + 1) * (rb1 - rb0 + 1)) as f32;
    let cutout = !translucent
        && (covered as f32 / area < 0.9
            || covered < FACE_TEXELS
            || (avg_a < 255 && opaque_texels < covered));
    let tint_state = if tinted_texels == 0 {
        0
    } else if tinted_texels == covered {
        2
    } else {
        1
    };
    BakedFace {
        exists: true,
        occludes,
        can_be_occluded,
        cutout,
        self_lighting,
        tint_state,
        tint_mask: mask,
        rect,
        depth: ((nearest.max(0.0) * 64.0).round() as u32).min(62) as u8,
        avg,
        texels,
    }
}

fn bake_model(req: &BakeRequest, sprites: &dyn SpriteSource) -> Option<BakedModel> {
    let name = req.name.as_str();
    let mut source = tint_source_for_block(name);
    let fluid_q;
    let quads: &[BakeQuad] = match req.fluid {
        Some(kind) => {
            if kind == FluidKind::Water {
                source = TintSource::Water;
            }
            fluid_q = fluid_quads(kind);
            &fluid_q
        }
        None => &req.quads,
    };
    if quads.is_empty() {
        return None;
    }
    let translucent = req.translucent || req.fluid == Some(FluidKind::Water);
    let flat = crate::colors::block_color(name, "minecraft:plains", 0);
    let (tinted, fixed_mul, baked_tint) = match source {
        TintSource::None => (false, None, BakedTint::None),
        TintSource::Grass => (true, None, BakedTint::Grass),
        TintSource::Foliage => (true, None, BakedTint::Foliage),
        TintSource::Water => (true, None, BakedTint::Water),
        // Constant colours are premultiplied into the texels like Voxy's constant tint.
        TintSource::Fixed(c) => (false, Some(c), BakedTint::None),
    };
    let water_alpha = (req.fluid == Some(FluidKind::Water) || is_water_block(name))
        .then(|| crate::resource_colors::water_still_rgba()[3]);

    let mut fetch = |s: &str| sprites.sprite_pixels(s).ok();
    let mut faces = [BakedFace::EMPTY; 6];
    for f in Face::ALL {
        let fq: Vec<&BakeQuad> = quads.iter().filter(|q| q.facing == f).collect();
        faces[f as usize] = bake_face(
            f,
            &fq,
            &mut fetch,
            flat,
            tinted,
            fixed_mul,
            translucent,
            water_alpha,
        );
    }
    if faces.iter().all(|f| !f.exists) {
        return None;
    }
    let fully_opaque = faces.iter().all(|f| f.occludes);
    let mut flags = 0u8;
    if matches!(
        baked_tint,
        BakedTint::Grass | BakedTint::Foliage | BakedTint::Water
    ) {
        flags |= MODEL_BIOME_COLOURED;
    }
    if translucent {
        flags |= MODEL_TRANSLUCENT;
        flags |= MODEL_CULLS_SAME;
    }
    if req.fluid.is_some() {
        flags |= MODEL_FLUID;
    }
    if fully_opaque {
        flags |= MODEL_FULLY_OPAQUE;
    }
    Some(BakedModel {
        faces,
        flags,
        tint: baked_tint,
        mc_entity: req.mc_entity.unwrap_or(-1),
        dh_material_id: classify_dh_material(name),
        light_emission: req.light_emission.min(15),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sprites(HashMap<String, SpritePixels>);

    impl SpriteSource for Sprites {
        fn sprite_pixels(&self, sprite: &str) -> Result<SpritePixels, SpriteError> {
            self.0.get(sprite).cloned().ok_or(SpriteError::NoSuchSprite)
        }
    }

    fn solid(rgba: [u8; 4]) -> SpritePixels {
        SpritePixels {
            width: 2,
            height: 2,
            rgba: rgba.repeat(4),
        }
    }

    fn cube_quads(sprite: &str, tint_index: i32) -> Vec<BakeQuad> {
        let mut out = Vec::new();
        for face in Face::ALL {
            let plane = if face.is_positive() { 1.0 } else { 0.0 };
            let c = |a: f32, b: f32| -> [f32; 3] {
                match face.axis() {
                    0 => [a, plane, b],
                    1 => [a, b, plane],
                    _ => [plane, a, b],
                }
            };
            out.push(BakeQuad {
                positions: [c(0.0, 0.0), c(1.0, 0.0), c(1.0, 1.0), c(0.0, 1.0)],
                uvs: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                sprite: sprite.into(),
                facing: face,
                tint_index,
            });
        }
        out
    }

    fn req(id: u32, name: &str, quads: Vec<BakeQuad>) -> BakeRequest {
        BakeRequest {
            block_id: id,
            name: name.into(),
            quads,
            fluid: None,
            translucent: false,
            mc_entity: None,
            light_emission: 0,
        }
    }

    fn sprites(entries: &[(&str, [u8; 4])]) -> Sprites {
        Sprites(
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), solid(*v)))
                .collect(),
        )
    }

    #[test]
    fn stone_cube_bakes_opaque_occluding_faces_with_pack_id_and_material() {
        let s = sprites(&[("minecraft:block/stone", [125, 125, 125, 255])]);
        let mut t = BakedModelTable::new();
        let mut r = req(
            1,
            "minecraft:stone",
            cube_quads("minecraft:block/stone", -1),
        );
        r.mc_entity = Some(10020);
        let id = t.bake(&r, &s);
        assert_eq!(id, 1);
        let m = t.model(id).unwrap();
        assert!(
            m.faces
                .iter()
                .all(|f| f.exists && f.occludes && f.tint_state == 0)
        );
        assert_eq!(m.faces[Face::PosY as usize].avg, [125, 125, 125, 255]);
        assert_eq!(m.mc_entity, 10020);
        assert_eq!(
            m.dh_material_id,
            crate::colors::classify_dh_material("minecraft:stone")
        );
        let meta = t.meta_for(1);
        assert!(meta.fully_opaque() && !meta.translucent() && meta.face_occludes(Face::NegX));
        let gpu = t.gpu_models();
        assert_eq!(gpu[1].mc_entity, 10020);
        assert_eq!(gpu[1].dh_material_id, m.dh_material_id as u32);
        assert_eq!(t.gpu_texels().len(), 2 * 6 * FACE_TEXELS);
    }

    #[test]
    fn missing_pack_id_is_minus_one_and_identical_bakes_dedupe() {
        let s = sprites(&[("a", [10, 20, 30, 255])]);
        let mut t = BakedModelTable::new();
        let a = t.bake(&req(1, "minecraft:stone", cube_quads("a", -1)), &s);
        let b = t.bake(&req(2, "minecraft:stone", cube_quads("a", -1)), &s);
        assert_eq!(a, b);
        assert_eq!(t.len(), 2);
        assert_eq!(t.model(a).unwrap().mc_entity, -1);
        assert_eq!(t.model_id_for_block(2), a);
        assert!(t.meta_for(99).is_empty());
    }

    #[test]
    fn grass_top_is_fully_tinted_and_side_overlay_partially() {
        let mut quads = cube_quads("minecraft:block/dirt", -1);
        // Replace +Y with a tinted grass top.
        for q in &mut quads {
            if q.facing == Face::PosY {
                q.sprite = "minecraft:block/grass_top".into();
                q.tint_index = 0;
            }
        }
        // Overlay on +X covering the upper half (b in 0.5..1), tint index 0.
        quads.push(BakeQuad {
            positions: [
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            uvs: [[0.0, 0.5], [1.0, 0.5], [1.0, 0.0], [0.0, 0.0]],
            sprite: "minecraft:block/grass_side_overlay".into(),
            facing: Face::PosX,
            tint_index: 0,
        });
        // Vanilla overlays sit a hair in front of the base face so they win the depth test.
        let n = quads.len() - 1;
        for p in &mut quads[n].positions {
            p[0] = 1.02;
        }
        let s = sprites(&[
            ("minecraft:block/dirt", [134, 96, 67, 255]),
            ("minecraft:block/grass_top", [150, 150, 150, 255]),
            ("minecraft:block/grass_side_overlay", [150, 150, 150, 255]),
        ]);
        let mut t = BakedModelTable::new();
        let id = t.bake(&req(5, "minecraft:grass_block", quads), &s);
        let m = t.model(id).unwrap();
        assert_eq!(m.tint, BakedTint::Grass);
        assert_eq!(m.flags & MODEL_BIOME_COLOURED, MODEL_BIOME_COLOURED);
        assert_eq!(m.faces[Face::PosY as usize].tint_state, 2);
        assert_eq!(m.faces[Face::NegY as usize].tint_state, 0);
        // The overlay covers the upper half (Y) of +X: partially tinted.
        let side = &m.faces[Face::PosX as usize];
        assert_eq!(side.tint_state, 1);
        assert_eq!(side.tint_mask, 0xCCCC);
        assert_eq!(t.meta_for(5).biome_coloured(), true);
    }

    #[test]
    fn water_is_translucent_biome_tinted_with_alpha_180_and_never_occludes() {
        let s = sprites(&[
            ("minecraft:block/water_still", [177, 177, 177, 255]),
            ("minecraft:block/water_flow", [177, 177, 177, 255]),
        ]);
        let mut t = BakedModelTable::new();
        let mut r = req(7, "minecraft:water", Vec::new());
        r.fluid = Some(FluidKind::Water);
        let id = t.bake(&r, &s);
        let m = t.model(id).unwrap();
        assert_eq!(m.flags & MODEL_FLUID, MODEL_FLUID);
        assert_eq!(m.tint, BakedTint::Water);
        for f in &m.faces {
            assert!(f.exists && !f.occludes);
            assert_eq!(f.tint_state, 2);
            assert_eq!(f.texels[0] >> 24, 180);
        }
        let meta = t.meta_for(7);
        assert!(meta.translucent() && meta.fluid() && meta.culls_same() && !meta.fully_opaque());
        assert_eq!(m.dh_material_id, crate::colors::DH_BLOCK_WATER);
    }

    #[test]
    fn slab_top_does_not_occlude_and_records_depth_and_rect() {
        // Bottom slab: +Y face at y = 0.5 covering the whole block, +X face only lower half.
        let mut quads = Vec::new();
        let c = |a: f32, b: f32, y: f32| [a, y, b];
        quads.push(BakeQuad {
            positions: [
                c(0.0, 0.0, 0.5),
                c(1.0, 0.0, 0.5),
                c(1.0, 1.0, 0.5),
                c(0.0, 1.0, 0.5),
            ],
            uvs: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            sprite: "p".into(),
            facing: Face::PosY,
            tint_index: -1,
        });
        quads.push(BakeQuad {
            positions: [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.5, 1.0],
                [1.0, 0.5, 0.0],
            ],
            uvs: [[0.0, 1.0], [1.0, 1.0], [1.0, 0.5], [0.0, 0.5]],
            sprite: "p".into(),
            facing: Face::PosX,
            tint_index: -1,
        });
        let s = sprites(&[("p", [200, 100, 50, 255])]);
        let mut t = BakedModelTable::new();
        let id = t.bake(&req(3, "minecraft:oak_slab", quads), &s);
        let m = t.model(id).unwrap();
        let top = &m.faces[Face::PosY as usize];
        assert!(
            top.exists && !top.occludes,
            "depth 0.5 is farther than Voxy's 0.1"
        );
        assert_eq!(top.depth, 32);
        let side = &m.faces[Face::PosX as usize];
        assert!(side.exists && !side.occludes);
        // The slab covers the lower half only: Y is the first tangent of an X face.
        assert_eq!(side.rect[0], 0);
        assert!(side.rect[1] < 15);
        assert_eq!(side.rect[3], 15);
        assert!(!m.faces[Face::NegZ as usize].exists);
        assert!(t.meta_for(3).has_face(Face::PosY) && !t.meta_for(3).has_face(Face::PosZ));
    }

    #[test]
    fn sprite_not_ready_falls_back_to_static_colour_and_empty_models_bake_to_zero() {
        let s = Sprites(HashMap::new());
        let mut t = BakedModelTable::new();
        let id = t.bake(&req(1, "minecraft:stone", cube_quads("missing", -1)), &s);
        assert_ne!(id, 0);
        let flat = crate::colors::block_color("minecraft:stone", "minecraft:plains", 0);
        let avg = t.model(id).unwrap().faces[0].avg;
        assert_eq!([avg[0], avg[1], avg[2]], flat);
        assert_eq!(t.bake(&req(2, "minecraft:barrier", Vec::new()), &s), 0);
    }

    #[test]
    fn gpu_layouts_are_pod_sized() {
        assert_eq!(std::mem::size_of::<GpuModelEntry>(), 112);
        assert_eq!(std::mem::size_of::<GpuBiomeTint>(), 16);
        let t = BakedModelTable::new();
        assert!(!bytemuck::cast_slice::<GpuModelEntry, u8>(&t.gpu_models()).is_empty());
    }

    #[test]
    fn biome_tint_rows_pack_rgba() {
        let tab = BiomeTintTable::build(&["minecraft:plains", "minecraft:swamp"], |_| 0);
        let rows = tab.gpu_rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].water, pack_rgba([97, 123, 100, 255]));
    }
}
