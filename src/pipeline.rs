//! Live Voxy-style octree pipeline: ingest (G1) -> mip propagation -> bakery -> 32³ greedy mesh ->
//! `render_bridge` upload queue. The server thread ingests chunk snapshots and marks sections dirty;
//! the client tick bakes block models and meshes the dirty sections into the GPU arena. Single
//! player shares this state between both sides; a remote client fills the same store from
//! `network::octree_sync`.

use crate::bakery::{
    BakeQuad, BakeRequest, BakedModel, BakedModelTable, BakedTint, BiomeTintTable, FluidKind,
    MODEL_TRANSLUCENT, SpriteError, SpritePixels, SpriteSource, default_translucent, pack_rgba,
};
use crate::gpu_arena::{BUCKETS, BuiltSection, MODEL_WORDS, SectionAabb};
use crate::mesh::{SectionQuads, mesh_octree_section};
use crate::octree::{Face, NeighborBorders, OctreeSection, OctreeSectionPos};
use crate::octree_ingest::{OctreeIngestStore, ingest_chunk_snapshot};
use crate::render_bridge::submit_built_section;
use mod_api::{BlockState, ChunkSnapshot, ClientHost, Direction, ServerHost};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{LazyLock, Mutex};

/// Block models baked per client tick (each bake reads sprite pixels on the CPU).
const BAKES_PER_TICK: usize = 64;
/// Sections meshed per client tick.
const MESHES_PER_TICK: usize = 16;

/// Octree voxel ids: `0` is air, `n` is `states[n - 1]`; biome ids index `biomes`.
#[derive(Default)]
pub struct Mapper {
    states: Vec<BlockState>,
    names: Vec<String>,
    by_state: HashMap<BlockState, u32>,
    biomes: Vec<String>,
    by_biome: HashMap<String, u16>,
}

impl Mapper {
    fn block(&mut self, host: &dyn ServerHost, state: BlockState) -> u32 {
        if let Some(&id) = self.by_state.get(&state) {
            return id;
        }
        let name = state.block().id(host).unwrap_or("minecraft:stone");
        let id = if matches!(
            name,
            "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air"
        ) {
            0
        } else {
            self.states.push(state);
            self.names.push(name.to_owned());
            self.states.len() as u32
        };
        self.by_state.insert(state, id);
        id
    }

    fn biome(&mut self, key: &str) -> u16 {
        if let Some(&id) = self.by_biome.get(key) {
            return id;
        }
        let id = self.biomes.len() as u16;
        self.biomes.push(key.to_owned());
        self.by_biome.insert(key.to_owned(), id);
        id
    }

    /// `Mapper.getBlockStateOpacity`: translucent blocks let light through, the rest block it.
    fn opacity(&self, id: u32) -> u8 {
        match id.checked_sub(1).and_then(|i| self.names.get(i as usize)) {
            Some(name) if default_translucent(name) => 1,
            Some(_) => 15,
            None => 0,
        }
    }
}

#[derive(Default)]
struct Pipeline {
    stores: HashMap<String, OctreeIngestStore>,
    mapper: Mapper,
    models: BakedModelTable,
    baked: usize,
    models_dirty: bool,
    dirty: HashMap<String, VecDeque<OctreeSectionPos>>,
    queued: HashMap<String, HashSet<OctreeSectionPos>>,
}

static PIPELINE: LazyLock<Mutex<Pipeline>> = LazyLock::new(Default::default);

impl Pipeline {
    fn queue(&mut self, dim: &str, positions: impl IntoIterator<Item = OctreeSectionPos>) {
        let set = self.queued.entry(dim.to_owned()).or_default();
        let queue = self.dirty.entry(dim.to_owned()).or_default();
        for pos in positions {
            if set.insert(pos) {
                queue.push_back(pos);
            }
        }
    }
}

/// G1: ingest a loaded chunk's snapshot (server thread). Returns whether the chunk was accepted.
pub fn ingest_loaded_chunk(
    host: &dyn ServerHost,
    dim: &str,
    chunk: [i32; 2],
    snapshot: &ChunkSnapshot,
) -> bool {
    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;
    let store = p.stores.entry(dim.to_owned()).or_default();
    let mapper = &mut p.mapper;
    let accepted = ingest_chunk_snapshot(
        store,
        chunk,
        snapshot,
        &mut |state| mapper.block(host, state),
        &mut |biome| mapper.biome(biome),
    );
    if accepted {
        let mapper = &p.mapper;
        let touched = p
            .stores
            .get_mut(dim)
            .expect("store inserted above")
            .propagate_dirty(&|id| mapper.opacity(id));
        p.queue(dim, touched);
    }
    accepted
}

/// Drops every section of `dim` (level unload).
pub fn clear_dimension(dim: &str) {
    let mut p = PIPELINE.lock().unwrap();
    p.stores.remove(dim);
    p.dirty.remove(dim);
    p.queued.remove(dim);
}

/// A resource reload or shader-pack change invalidates every baked model: bake them all again and
/// mesh everything once the models exist.
pub fn invalidate_models() {
    let mut p = PIPELINE.lock().unwrap();
    p.models.clear();
    p.baked = 0;
    p.models_dirty = true;
    let dims: Vec<String> = p.stores.keys().cloned().collect();
    for dim in dims {
        let positions = p.stores[&dim].section_positions();
        p.queue(&dim, positions);
    }
}

const fn face_of(direction: Direction) -> Face {
    match direction {
        Direction::Down => Face::NegY,
        Direction::Up => Face::PosY,
        Direction::North => Face::NegZ,
        Direction::South => Face::PosZ,
        Direction::West => Face::NegX,
        Direction::East => Face::PosX,
    }
}

/// `RenderAssets::atlas_sprite_pixels` behind the bakery's by-name sprite lookup.
struct AtlasSprites<'a> {
    host: &'a dyn ClientHost,
}

impl SpriteSource for AtlasSprites<'_> {
    fn sprite_pixels(&self, sprite: &str) -> Result<SpritePixels, SpriteError> {
        let sprite_ref = self
            .host
            .texture_sprites(&[sprite])
            .pop()
            .flatten()
            .ok_or(SpriteError::NoSuchSprite)?;
        let pixels = self
            .host
            .render_assets()
            .atlas_sprite_pixels(sprite_ref)
            .map_err(|_| SpriteError::NoSuchSprite)?;
        Ok(SpritePixels {
            width: pixels.width,
            height: pixels.height,
            rgba: pixels.rgba,
        })
    }
}

fn bake_request(host: &dyn ClientHost, block_id: u32, name: &str, state: BlockState) -> BakeRequest {
    let assets = host.render_assets();
    let fluid = match name {
        "minecraft:water" => Some(FluidKind::Water),
        "minecraft:lava" => Some(FluidKind::Lava),
        _ => None,
    };
    let mut quads = Vec::new();
    let mut light_emission = 0;
    if fluid.is_none()
        && let Some(key) = assets
            .block_state_models(state)
            .ok()
            .and_then(|keys| keys.iter().next())
        && let Ok(view) = assets.baked_model(key)
    {
        for q in view.quads() {
            light_emission = light_emission.max(q.light_emission);
            quads.push(BakeQuad {
                positions: *q.positions,
                uvs: *q.uvs,
                sprite: q.sprite.to_owned(),
                facing: face_of(q.facing),
                tint_index: q.tint_index,
            });
        }
    }
    BakeRequest {
        block_id,
        name: name.to_owned(),
        quads,
        fluid,
        translucent: default_translucent(name),
        mc_entity: assets.shader_pack_block_id(state),
        light_emission,
    }
}

/// One model as the 16 words `voxy_quads.vsh/fsh` read: `[0..6]` atlas tile origins (unused: tile
/// size 0 selects the flat colour), `[6..12]` per-face average colour, `[12]` flags (bit 0
/// translucent, bit 1 face shading, bit 2 cutout, bits 4..8 tint kind), `[13]` constant tint,
/// `[14]` tile size, `[15]` block light emission.
fn model_words(m: &BakedModel) -> [u32; MODEL_WORDS as usize] {
    let mut w = [0u32; MODEL_WORDS as usize];
    for (i, face) in m.faces.iter().enumerate() {
        w[6 + i] = pack_rgba(face.avg);
    }
    let tint_kind = match m.tint {
        BakedTint::None => 0,
        BakedTint::Grass => 1,
        BakedTint::Foliage => 2,
        BakedTint::Water => 3,
    };
    let cutout = m.faces.iter().any(|f| f.cutout);
    w[12] = u32::from(m.flags & MODEL_TRANSLUCENT != 0)
        | 2
        | (u32::from(cutout) << 2)
        | (tint_kind << 4);
    w[15] = u32::from(m.light_emission);
    w
}

/// Buckets a mesh for the arena: bucket 0 translucent, 1 double-sided (unused), `2 + face` opaque.
fn built_section(quads: &SectionQuads) -> BuiltSection {
    let mut by_bucket: [Vec<_>; BUCKETS] = Default::default();
    by_bucket[0] = quads.translucent.clone();
    for q in &quads.opaque {
        by_bucket[2 + q.face() as usize].push(*q);
    }
    let mut counts = [0u16; BUCKETS];
    let mut all = Vec::with_capacity(quads.quad_count() as usize);
    for (i, bucket) in by_bucket.into_iter().enumerate() {
        counts[i] = bucket.len() as u16;
        all.extend(bucket);
    }
    let size = |i: usize| quads.aabb_max[i].saturating_sub(quads.aabb_min[i]).clamp(1, 32);
    BuiltSection {
        level: quads.pos.detail,
        pos: [quads.pos.x, quads.pos.y, quads.pos.z],
        aabb: SectionAabb {
            offset: quads.aabb_min.map(|v| v.min(31)),
            size: [size(0), size(1), size(2)],
        },
        quads: all,
        counts,
    }
}

/// Client tick: bake pending block models, upload the model and biome tables when they grew, then
/// mesh dirty sections of `dim` into the GPU arena queue.
pub fn client_tick(host: &mut dyn ClientHost, dim: &str) {
    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;

    let sprites = AtlasSprites { host };
    let mut baked_now = 0;
    while p.baked < p.mapper.states.len() && baked_now < BAKES_PER_TICK {
        let (state, id) = (p.mapper.states[p.baked], p.baked as u32 + 1);
        let request = bake_request(host, id, &p.mapper.names[p.baked], state);
        p.models.bake(&request, &sprites);
        p.baked += 1;
        baked_now += 1;
        p.models_dirty = true;
    }
    if p.models_dirty && p.baked == p.mapper.states.len() {
        p.models_dirty = false;
        let words: Vec<u32> = (0..p.models.len())
            .filter_map(|i| p.models.model(i as u32))
            .flat_map(model_words)
            .collect();
        let biomes: Vec<&str> = p.mapper.biomes.iter().map(String::as_str).collect();
        let tints = BiomeTintTable::build(&biomes, |_| 0x3F76E4).gpu_rows();
        let tint_words: Vec<u32> = tints
            .iter()
            .flat_map(|t| [t.grass, t.foliage, t.water])
            .collect();
        crate::render_bridge::queue_model_tables(words, tint_words);
    }
    if p.baked < p.mapper.states.len() {
        return;
    }

    for _ in 0..MESHES_PER_TICK {
        let Some(pos) = p.dirty.get_mut(dim).and_then(VecDeque::pop_front) else {
            break;
        };
        if let Some(set) = p.queued.get_mut(dim) {
            set.remove(&pos);
        }
        let Some(store) = p.stores.get(dim) else {
            continue;
        };
        let key = (dim.to_owned(), pos.detail, [pos.x, pos.y, pos.z]);
        let Some(section) = store.section(pos) else {
            submit_built_section(key, empty_section(pos));
            continue;
        };
        let neighbors: [Option<&OctreeSection>; 6] =
            Face::ALL.map(|face| store.section(pos.neighbor(face)));
        let borders = NeighborBorders::from_neighbors(neighbors);
        let quads = mesh_octree_section(section, &borders, &p.models);
        submit_built_section(key, built_section(&quads));
    }
}

fn empty_section(pos: OctreeSectionPos) -> BuiltSection {
    BuiltSection {
        level: pos.detail,
        pos: [pos.x, pos.y, pos.z],
        aabb: SectionAabb::FULL,
        quads: Vec::new(),
        counts: [0; BUCKETS],
    }
}
