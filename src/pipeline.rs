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
use crate::network::octree_sync::{ClientEvent, MappingEntry, OctreeSectionDto, ServerSource};
use crate::octree::Voxel;
use crate::octree::{Face, NeighborBorders, OctreeSection, OctreeSectionPos};
use crate::octree_ingest::DistantColumn;
use crate::octree_ingest::{OctreeIngestStore, ingest_chunk_snapshot, ingest_saved_chunk};
use crate::render_bridge::submit_built_section;
use crate::storage::{DEFAULT_OCTREE_COMPRESSION, LodIoWorker, OctreeMapping};
use mod_api::{
    BlockKey, BlockState, ChunkSnapshot, ClientHost, Direction, Host, HostBlockStates, PlayerRef,
    ServerHost, TerrainColumn,
};
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::{LazyLock, Mutex};

/// Block models baked per client tick (each bake reads sprite pixels on the CPU).
const BAKES_PER_TICK: usize = 64;
/// Sections meshed per client tick.
const MESHES_PER_TICK: usize = 32;

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
    fn block(&mut self, host: &dyn Host, state: BlockState) -> u32 {
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

    /// The octree id of `key`'s first (default) state; air and unknown blocks are `0`.
    fn block_key(&mut self, host: &dyn Host, key: BlockKey) -> u32 {
        match host
            .block_states()
            .states_of(key)
            .and_then(|r| r.iter().next())
        {
            Some(state) => self.block(host, state),
            None => 0,
        }
    }

    /// The octree id of the block called `name`. A file or a server names blocks, not states, so
    /// a block's default state stands for all of its states.
    fn block_named(&mut self, host: &dyn Host, name: &str) -> u32 {
        match host.block_key(name) {
            Some(key) => self.block_key(host, key),
            None => 0,
        }
    }

    /// Names of the ids `section` uses, for a save or a network message.
    fn mapping_of(&self, section: &OctreeSection) -> (Vec<MappingEntry>, OctreeMapping) {
        let mut entries = Vec::new();
        let mut mapping = OctreeMapping::default();
        for voxel in section.to_rle().palette {
            let (block, biome) = (voxel.block_id(), voxel.biome_id());
            if let Some(name) = block
                .checked_sub(1)
                .and_then(|i| self.names.get(i as usize))
                && mapping.blocks.insert(block, name.clone()).is_none()
            {
                entries.push(MappingEntry {
                    kind: 0,
                    id: block,
                    name: name.clone(),
                });
            }
            if let Some(name) = self.biomes.get(biome as usize)
                && mapping.biomes.insert(biome, name.clone()).is_none()
            {
                entries.push(MappingEntry {
                    kind: 1,
                    id: u32::from(biome),
                    name: name.clone(),
                });
            }
        }
        (entries, mapping)
    }

    /// Translate a section whose ids come from another mapping (a file, a server) into this
    /// session's ids by name.
    fn remap(
        &mut self,
        host: &dyn Host,
        section: &OctreeSection,
        blocks: &BTreeMap<u32, String>,
        biomes: &BTreeMap<u16, String>,
    ) -> OctreeSection {
        let mut block_ids: HashMap<u32, u32> = HashMap::new();
        let mut biome_ids: HashMap<u16, u16> = HashMap::new();
        let dense: Vec<Voxel> = section
            .to_dense()
            .into_iter()
            .map(|v| {
                let block = match v.block_id() {
                    0 => 0,
                    id => *block_ids.entry(id).or_insert_with(|| {
                        blocks.get(&id).map_or(0, |n| self.block_named(host, n))
                    }),
                };
                let biome = *biome_ids
                    .entry(v.biome_id())
                    .or_insert_with(|| biomes.get(&v.biome_id()).map_or(0, |n| self.biome(n)));
                if block == 0 {
                    Voxel::air_with_light(v.sky_light(), v.block_light())
                } else {
                    Voxel::new(block, biome, v.sky_light(), v.block_light())
                }
            })
            .collect();
        let mut out = OctreeSection::from_dense(section.pos, &dense);
        out.revision = section.revision;
        out
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
    /// Sections currently uploaded to the GPU arena (strictly leaf sections: never both parent
    /// and child for the same volume).
    rendered: HashMap<String, HashSet<OctreeSectionPos>>,
    /// Sections changed since their last save.
    unsaved: HashMap<String, HashSet<OctreeSectionPos>>,
    /// Sections already asked of disk or of the server, per dimension.
    requested: HashMap<String, HashSet<OctreeSectionPos>>,
}

static PIPELINE: LazyLock<Mutex<Pipeline>> = LazyLock::new(Default::default);

/// Returns `true` when `pos` is the active leaf detail level for its volume: `pos` exists and is
/// non-empty in `store`, and none of its finer octree descendants exist and are non-empty.
/// Ancestors stay in `store` for persistence and coarse fallback, but are never drawn alongside
/// their children (matching Voxy's hierarchical traversal).
pub(crate) fn is_render_leaf(store: &OctreeIngestStore, pos: OctreeSectionPos) -> bool {
    let Some(section) = store.section(pos) else {
        return false;
    };
    if section.is_empty() {
        return false;
    }
    !has_non_empty_descendant(store, pos)
}

fn has_non_empty_descendant(store: &OctreeIngestStore, pos: OctreeSectionPos) -> bool {
    let Some(kids) = pos.children() else {
        return false;
    };
    for child in kids {
        if store.section(child).is_some_and(|s| !s.is_empty()) {
            return true;
        }
        if child.detail > 0 && has_non_empty_descendant(store, child) {
            return true;
        }
    }
    false
}

impl Pipeline {
    fn queue(&mut self, dim: &str, positions: impl IntoIterator<Item = OctreeSectionPos>) {
        let rendered = self.rendered.get(dim);
        let set = self.queued.entry(dim.to_owned()).or_default();
        let queue = self.dirty.entry(dim.to_owned()).or_default();
        let mut push = |pos: OctreeSectionPos| {
            if set.insert(pos) {
                queue.push_back(pos);
            }
        };
        for pos in positions {
            push(pos);
            // Evict any currently-rendered ancestor superseded by a finer section, and re-mesh
            // any currently-rendered same-detail neighbour so touching border walls are culled.
            if let Some(live) = rendered {
                let mut anc = pos.parent();
                while let Some(a) = anc {
                    if live.contains(&a) {
                        push(a);
                    }
                    anc = a.parent();
                }
                for face in Face::ALL {
                    let nb = pos.neighbor(face);
                    if live.contains(&nb) {
                        push(nb);
                    }
                }
            }
        }
    }
}

/// Which source a snapshot came from: G1 (live chunk) or G3 (saved Anvil chunk).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotSource {
    Loaded,
    Saved,
}

/// Ingest one chunk snapshot into `dim`'s octree store (server thread) and queue the touched
/// sections for meshing. Returns whether the chunk was accepted (a higher-priority source owns it
/// otherwise).
pub fn ingest_chunk(
    host: &dyn ServerHost,
    dim: &str,
    chunk: [i32; 2],
    snapshot: &ChunkSnapshot,
    source: SnapshotSource,
) -> bool {
    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;
    let store = p.stores.entry(dim.to_owned()).or_default();
    let mapper = std::cell::RefCell::new(&mut p.mapper);
    let ingest = match source {
        SnapshotSource::Loaded => ingest_chunk_snapshot,
        SnapshotSource::Saved => ingest_saved_chunk,
    };
    let accepted = ingest(
        store,
        chunk,
        snapshot,
        &mut |state| mapper.borrow_mut().block(host, state),
        &mut |biome| mapper.borrow_mut().biome(biome),
    );
    if accepted {
        let mapper = &p.mapper;
        let touched = p
            .stores
            .get_mut(dim)
            .expect("store inserted above")
            .propagate_dirty(&|id| mapper.opacity(id));
        p.unsaved
            .entry(dim.to_owned())
            .or_default()
            .extend(touched.iter().copied());
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
    p.rendered.remove(dim);
    p.unsaved.remove(dim);
    p.requested.remove(dim);
}

/// A resource reload, shader-pack change or `/ds remesh` invalidates every baked model: bake them
/// all again and mesh everything once the models exist. Returns how many sections were queued.
pub fn invalidate_models() -> usize {
    let mut p = PIPELINE.lock().unwrap();
    p.models.clear();
    p.baked = 0;
    p.models_dirty = true;
    let dims: Vec<String> = p.stores.keys().cloned().collect();
    let mut queued = 0;
    for dim in dims {
        let positions = p.stores[&dim].section_positions();
        queued += positions.len();
        p.queue(&dim, positions);
    }
    queued
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

fn bake_request(
    host: &dyn ClientHost,
    block_id: u32,
    name: &str,
    state: BlockState,
) -> BakeRequest {
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

/// Side of the baked face atlas: one 1024² dynamic texture (the mod's whole 4 MiB budget).
pub const ATLAS_SIDE: usize = 1024;
/// Texels per face side (`bakery::FACE_GRID`).
const TILE: usize = crate::bakery::FACE_GRID;
/// Faces the atlas holds; later models draw with the flat per-face colour (tile size 0).
const ATLAS_FACES: usize = (ATLAS_SIDE / TILE) * (ATLAS_SIDE / TILE);

/// One model as the 16 words `voxy_quads.vsh/fsh` read: `[0..6]` atlas tile origins (`x | y << 16`
/// in atlas pixels), `[6..12]` per-face average colour, `[12]` flags (bit 0 translucent, bit 1 face
/// shading, bit 2 cutout, bits 4..8 tint kind), `[13]` constant tint, `[14]` tile size
/// (`w | h << 16`, 0 = flat colour), `[15]` block light emission. Faces of model `index` are
/// painted into `atlas` (RGBA8, `ATLAS_SIDE²`) when they fit.
fn model_words(index: usize, m: &BakedModel, atlas: &mut [u8]) -> [u32; MODEL_WORDS as usize] {
    let mut w = [0u32; MODEL_WORDS as usize];
    let textured = (index + 1) * 6 <= ATLAS_FACES;
    for (i, face) in m.faces.iter().enumerate() {
        w[6 + i] = pack_rgba(face.avg);
        if textured {
            let slot = index * 6 + i;
            let (tx, ty) = (
                (slot % (ATLAS_SIDE / TILE)) * TILE,
                (slot / (ATLAS_SIDE / TILE)) * TILE,
            );
            w[i] = (tx as u32) | ((ty as u32) << 16);
            for (n, texel) in face.texels.iter().enumerate() {
                let (x, y) = (tx + n % TILE, ty + n / TILE);
                let at = (y * ATLAS_SIDE + x) * 4;
                atlas[at..at + 4].copy_from_slice(&texel.to_le_bytes());
            }
        }
    }
    if textured {
        w[14] = (TILE as u32) | ((TILE as u32) << 16);
    }
    let tint_kind = match m.tint {
        BakedTint::None => 0,
        BakedTint::Grass => 1,
        BakedTint::Foliage => 2,
        BakedTint::Water => 3,
    };
    let cutout = m.faces.iter().any(|f| f.cutout);
    let mut face_tints = 0u32;
    for (i, face) in m.faces.iter().enumerate() {
        face_tints |= (u32::from(face.tint_state) & 3) << (16 + i * 2);
    }
    w[12] = u32::from(m.flags & MODEL_TRANSLUCENT != 0)
        | 2
        | (u32::from(cutout) << 2)
        | (tint_kind << 4)
        | (u32::from(m.dh_material_id & 0xF) << 8)
        | face_tints;
    w[15] = m.mc_entity as u32;
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
    let size = |i: usize| {
        quads.aabb_max[i]
            .saturating_sub(quads.aabb_min[i])
            .clamp(1, 32)
    };
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
        let mut atlas = vec![0u8; ATLAS_SIDE * ATLAS_SIDE * 4];
        let words: Vec<u32> = (0..p.models.len())
            .filter_map(|i| p.models.model(i as u32).map(|m| (i, m)))
            .flat_map(|(i, m)| model_words(i, m, &mut atlas))
            .collect();
        let biomes: Vec<&str> = p.mapper.biomes.iter().map(String::as_str).collect();
        let tints = BiomeTintTable::build(&biomes, |_| 0x3F76E4).gpu_rows();
        let tint_words: Vec<u32> = tints
            .iter()
            .flat_map(|t| [t.grass, t.foliage, t.water])
            .collect();
        crate::render_bridge::queue_model_tables(words, tint_words, atlas);
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
        if !is_render_leaf(store, pos) {
            if p.rendered
                .get_mut(dim)
                .is_some_and(|live| live.remove(&pos))
            {
                submit_built_section(key, empty_section(pos));
            }
            continue;
        }
        let Some(section) = store.section(pos) else {
            continue;
        };
        let neighbors: [Option<&OctreeSection>; 6] =
            Face::ALL.map(|face| store.section(pos.neighbor(face)));
        let borders = NeighborBorders::from_neighbors(neighbors);
        let quads = mesh_octree_section(section, &borders, &p.models);
        let built = built_section(&quads);
        if built.is_empty() {
            if p.rendered
                .get_mut(dim)
                .is_some_and(|live| live.remove(&pos))
            {
                submit_built_section(key, empty_section(pos));
            }
        } else {
            p.rendered.entry(dim.to_owned()).or_default().insert(pos);
            submit_built_section(key, built);
        }
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

/// G4: ingest one finished distant-generation job. `cells` are `(cx, cz)` cells of the
/// `64 x 64` grid whose block origin is `origin`, each `1 << detail` blocks wide; `samples` run in
/// the same order. A sample covering `2^detail` blocks is one voxel of a level-`detail` section.
pub fn ingest_distant(
    host: &dyn ServerHost,
    dim: &str,
    detail: u8,
    origin: (i32, i32),
    cells: &[(usize, usize)],
    samples: &[Option<TerrainColumn>],
) {
    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;
    let step = 1i32 << detail;
    let columns: Vec<DistantColumn<'_, BlockKey>> = cells
        .iter()
        .zip(samples)
        .filter_map(|(&(cx, cz), sample)| {
            let col = sample.as_ref()?;
            Some(DistantColumn {
                x: origin.0 + cx as i32 * step,
                z: origin.1 + cz as i32 * step,
                height: col.height,
                water: col.water,
                block: col.block,
                biome: &col.biome,
            })
        })
        .collect();
    if columns.is_empty() {
        return;
    }
    let water = p.mapper.block_named(host, "minecraft:water");
    let store = p.stores.entry(dim.to_owned()).or_default();
    let mapper = std::cell::RefCell::new(&mut p.mapper);
    let mut block = |key| mapper.borrow_mut().block_key(host, key);
    let mut biome = |name: &str| mapper.borrow_mut().biome(name);
    let touched = if detail == 0 {
        crate::octree_ingest::ingest_distant_columns(
            store, &columns, water, &mut block, &mut biome,
        );
        let mapper = &p.mapper;
        p.stores
            .get_mut(dim)
            .expect("store inserted above")
            .propagate_dirty(&|id| mapper.opacity(id))
    } else {
        store.ingest_coarse_columns(detail, &columns, water, &mut block, &mut biome)
    };
    p.unsaved
        .entry(dim.to_owned())
        .or_default()
        .extend(touched.iter().copied());
    p.queue(dim, touched);
}

/// Hand up to `limit` changed sections of every dimension to the I/O worker (it coalesces newer
/// saves of one section). Returns how many were queued.
pub fn flush_saves(io: &LodIoWorker, world_dir: &Path, limit: usize) -> usize {
    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;
    let mut queued = 0;
    for (dim, set) in p.unsaved.iter_mut() {
        let Some(store) = p.stores.get(dim) else {
            set.clear();
            continue;
        };
        while queued < limit {
            let Some(&pos) = set.iter().next() else { break };
            set.remove(&pos);
            let Some(section) = store.section(pos).filter(|s| !s.is_empty()) else {
                continue;
            };
            let (_, mapping) = p.mapper.mapping_of(section);
            io.save_octree_section(world_dir.to_owned(), dim.clone(), section.clone(), mapping);
            queued += 1;
        }
    }
    queued
}

/// Ask disk (single player and server) for stored sections around `center` that the store does not
/// hold yet: every level from fine to coarse, `radius` sections out, at most `budget` requests.
pub fn request_stored_sections(
    io: &LodIoWorker,
    world_dir: &Path,
    dim: &str,
    center: [f64; 3],
    radius: i32,
    budget: usize,
) {
    const Y_SECTIONS: std::ops::RangeInclusive<i32> = -2..=9;
    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;
    let requested = p.requested.entry(dim.to_owned()).or_default();
    let store = p.stores.get(dim);
    let mut left = budget;
    for level in 0..=crate::octree::MAX_OCTREE_DETAIL {
        let edge = (crate::octree::SECTION_EDGE as f64) * f64::from(1u32 << level);
        let (cx, cz) = (
            (center[0] / edge).floor() as i32,
            (center[2] / edge).floor() as i32,
        );
        for sx in cx - radius..=cx + radius {
            for sz in cz - radius..=cz + radius {
                for sy in (Y_SECTIONS.start() >> level)..=(Y_SECTIONS.end() >> level) {
                    let pos = OctreeSectionPos::new(level, sx, sy, sz);
                    if left == 0 {
                        return;
                    }
                    if store.is_some_and(|s| s.section(pos).is_some()) || !requested.insert(pos) {
                        continue;
                    }
                    io.load_octree_section(world_dir.to_owned(), dim.to_owned(), pos);
                    left -= 1;
                }
            }
        }
    }
}

/// Adopt sections the I/O worker finished loading, translating their ids to this session's.
pub fn adopt_loaded(host: &dyn Host, io: &LodIoWorker) {
    let loaded = io.drain_octree_responses();
    if loaded.is_empty() {
        return;
    }
    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;
    for l in loaded {
        let section = p
            .mapper
            .remap(host, &l.section, &l.mapping.blocks, &l.mapping.biomes);
        let pos = section.pos;
        if p.stores
            .entry(l.dimension.clone())
            .or_default()
            .adopt_section(section)
        {
            p.queue(&l.dimension, [pos]);
        }
    }
}

/// Client side of `octree_sync`: adopt what the server sent and ask for sections around `center`
/// that the store lacks (at most `budget` new requests; a cached timestamp is not tracked).
pub fn client_sync(
    host: &mut dyn ClientHost,
    dim: &str,
    center: [f64; 3],
    radius: i32,
    budget: usize,
) {
    let events = crate::network::octree_sync::drain_client_events();
    {
        let mut guard = PIPELINE.lock().unwrap();
        let p = &mut *guard;
        for event in events {
            let (level, section, mapping) = match event {
                ClientEvent::Section {
                    level,
                    section,
                    mapping,
                }
                | ClientEvent::PartialUpdate {
                    level,
                    section,
                    mapping,
                } => (level, section, mapping),
                ClientEvent::Lost { level, pos } | ClientEvent::Failed { level, pos, .. } => {
                    // Ask again later.
                    if let Some(r) = p.requested.get_mut(&level) {
                        r.remove(&pos);
                    }
                    continue;
                }
                ClientEvent::Config(_) | ClientEvent::UpToDate { .. } => continue,
            };
            let blocks = mapping
                .iter()
                .filter(|e| e.kind == 0)
                .map(|e| (e.id, e.name.clone()))
                .collect();
            let biomes = mapping
                .iter()
                .filter(|e| e.kind == 1)
                .map(|e| (e.id as u16, e.name.clone()))
                .collect();
            let section = p.mapper.remap(host, &section, &blocks, &biomes);
            let pos = section.pos;
            let store = p.stores.entry(level.clone()).or_default();
            // A newer server copy replaces ours.
            store.replace_section(section);
            p.queue(&level, [pos]);
        }
    }
    let mut want = Vec::new();
    {
        let mut guard = PIPELINE.lock().unwrap();
        let p = &mut *guard;
        let requested = p.requested.entry(dim.to_owned()).or_default();
        let store = p.stores.get(dim);
        let edge = crate::octree::SECTION_EDGE as f64;
        let (cx, cz) = (
            (center[0] / edge).floor() as i32,
            (center[2] / edge).floor() as i32,
        );
        'scan: for sx in cx - radius..=cx + radius {
            for sz in cz - radius..=cz + radius {
                for sy in -2..=9 {
                    let pos = OctreeSectionPos::new(0, sx, sy, sz);
                    if store.is_some_and(|s| s.section(pos).is_some()) || !requested.insert(pos) {
                        continue;
                    }
                    want.push(pos);
                    if want.len() >= budget {
                        break 'scan;
                    }
                }
            }
        }
    }
    for pos in want {
        crate::network::octree_sync::client_send_request(host, dim, pos, None);
    }
}

static PLAYER_XZ: Mutex<Vec<(PlayerRef, (i32, i32))>> = Mutex::new(Vec::new());

/// The server tick records where each player is, for the sync distance checks.
pub fn set_player_positions(positions: Vec<(PlayerRef, (i32, i32))>) {
    *PLAYER_XZ.lock().unwrap() = positions;
}

/// What `octree_sync` serves from: the pipeline's own sections, named by block and biome.
struct PipelineSource;

impl ServerSource<PlayerRef> for PipelineSource {
    fn section_timestamp(&self, level: &str, pos: OctreeSectionPos) -> Option<i64> {
        let p = PIPELINE.lock().unwrap();
        p.stores.get(level)?.section(pos).map(|s| s.revision as i64)
    }

    fn section_dto(&self, level: &str, pos: OctreeSectionPos) -> Option<OctreeSectionDto> {
        let p = PIPELINE.lock().unwrap();
        let section = p.stores.get(level)?.section(pos)?;
        let (entries, _) = p.mapper.mapping_of(section);
        OctreeSectionDto::from_section(section, &entries, DEFAULT_OCTREE_COMPRESSION, 0).ok()
    }

    fn player_block_xz(&self, player: PlayerRef) -> Option<(i32, i32)> {
        PLAYER_XZ
            .lock()
            .unwrap()
            .iter()
            .find(|(p, _)| *p == player)
            .map(|(_, xz)| *xz)
    }
}

/// Serve octree sections to connected players (call once when the server starts).
pub fn install_sync_source() {
    crate::network::octree_sync::install_server_source(Arc::new(PipelineSource));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_selection_never_renders_parent_alongside_non_empty_child() {
        let mut store = OctreeIngestStore::new();
        let parent_pos = OctreeSectionPos::new(1, 0, 2, 0);
        let mut parent = OctreeSection::new(parent_pos);
        parent.set(0, 0, 0, Voxel::new(1, 0, 15, 0));
        store.adopt_section(parent);
        assert!(
            is_render_leaf(&store, parent_pos),
            "coarse section with no children is a render leaf"
        );

        let child_pos = parent_pos.children().unwrap()[0];
        let mut child = OctreeSection::new(child_pos);
        child.set(0, 0, 0, Voxel::new(1, 0, 15, 0));
        store.adopt_section(child);
        assert!(
            is_render_leaf(&store, child_pos),
            "finer child becomes the render leaf"
        );
        assert!(
            !is_render_leaf(&store, parent_pos),
            "parent is no longer rendered once a finer child exists"
        );
    }
}
