//! Live Voxy-style octree pipeline: ingest (G1) -> mip propagation -> bakery -> 32³ greedy mesh ->
//! `render_bridge` upload queue. The server thread ingests chunk snapshots and marks sections dirty;
//! the client tick bakes block models and meshes the dirty sections into the GPU arena. Single
//! player shares this state between both sides; a remote client fills the same store from
//! `network::octree_sync`.

use crate::bakery::{
    BakeQuad, BakeRequest, BakedModel, BakedModelTable, BakedTint, BiomeTintTable, FluidKind,
    MODEL_TRANSLUCENT, SpriteError, SpritePixels, SpriteSource, build_atlas_mips,
    default_translucent, pack_rgba,
};
use crate::gpu_arena::{BUCKETS, BuiltSection, MODEL_WORDS, SectionAabb};
use crate::mesh::{SectionQuads, mesh_octree_section_masked};
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
/// Maximum dirty sections handed off to the background mesher thread per client tick.
const MESHES_PER_TICK: usize = 16;
/// Backpressure cap on the background mesher queue so RAM stays bounded while flying fast.
const MAX_PENDING_MESH_JOBS: usize = 64;

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

    /// The octree id of `key`'s vanilla default state (`Block.defaultBlockState`, e.g.
    /// `grass_block[snowy=false]` rather than ordinal 0 `snowy=true`); air and unknown blocks
    /// are `0`.
    fn block_key(&mut self, host: &dyn Host, key: BlockKey) -> u32 {
        match host.block_state_default(key).or_else(|| {
            host.block_states()
                .states_of(key)
                .and_then(|r| r.iter().next())
        }) {
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

/// Compact immutable snapshot of `ModelMeta` by block id, shared with the background mesher
/// thread so meshing never blocks the main render thread or holds `PIPELINE`.
#[derive(Default)]
struct MetaTable(Vec<crate::bakery::ModelMeta>);

impl crate::bakery::ModelLookup for MetaTable {
    fn meta(&self, block_id: u32) -> crate::bakery::ModelMeta {
        self.0
            .get(block_id as usize)
            .copied()
            .unwrap_or(crate::bakery::ModelMeta::EMPTY)
    }
}

struct MeshJob {
    dim: String,
    pos: OctreeSectionPos,
    epoch: u64,
    section: OctreeSection,
    borders: NeighborBorders,
    covered: u8,
    metas: Arc<MetaTable>,
}

type MeshDone = (String, OctreeSectionPos, u64, BuiltSection);

struct MeshWorker {
    queue: Arc<(Mutex<VecDeque<MeshJob>>, std::sync::Condvar)>,
    done: Arc<Mutex<Vec<MeshDone>>>,
}

impl MeshWorker {
    fn new() -> Self {
        let queue = Arc::new((
            Mutex::new(VecDeque::<MeshJob>::new()),
            std::sync::Condvar::new(),
        ));
        let done = Arc::new(Mutex::new(Vec::<MeshDone>::new()));
        let q_bg = Arc::clone(&queue);
        let d_bg = Arc::clone(&done);
        let _ = std::thread::Builder::new()
            .name("dh-mesher".to_owned())
            .spawn(move || {
                loop {
                    let job = {
                        let (lock, cv) = &*q_bg;
                        let mut q = lock.lock().unwrap();
                        while q.is_empty() {
                            q = cv.wait(q).unwrap();
                        }
                        q.pop_front().unwrap()
                    };
                    let quads = mesh_octree_section_masked(
                        &job.section,
                        &job.borders,
                        &*job.metas,
                        job.covered,
                    );
                    let built = built_section(&quads);
                    d_bg.lock()
                        .unwrap()
                        .push((job.dim, job.pos, job.epoch, built));
                }
            });
        Self { queue, done }
    }

    fn pending_len(&self) -> usize {
        self.queue.0.lock().unwrap().len()
    }

    fn push(&self, job: MeshJob) {
        let (lock, cv) = &*self.queue;
        lock.lock().unwrap().push_back(job);
        cv.notify_one();
    }

    fn clear_dim(&self, dim: &str) {
        self.queue.0.lock().unwrap().retain(|j| j.dim != dim);
        self.done.lock().unwrap().retain(|(d, _, _, _)| d != dim);
    }

    fn drain_done(&self) -> Vec<MeshDone> {
        std::mem::take(&mut *self.done.lock().unwrap())
    }
}

static MESH_WORKER: LazyLock<MeshWorker> = LazyLock::new(MeshWorker::new);

#[derive(Default)]
struct Pipeline {
    stores: HashMap<String, OctreeIngestStore>,
    mapper: Mapper,
    models: BakedModelTable,
    meta_snapshot: Arc<MetaTable>,
    baked: usize,
    models_dirty: bool,
    dirty: HashMap<String, VecDeque<OctreeSectionPos>>,
    queued: HashMap<String, HashSet<OctreeSectionPos>>,
    /// Monotonic epoch per `(dim, pos)` so stale off-thread mesh results are dropped before upload.
    next_epoch: u64,
    section_epoch: HashMap<(String, OctreeSectionPos), u64>,
    /// Sections currently uploaded to the GPU arena (strictly leaf sections: never both parent
    /// and child for the same volume).
    rendered: HashMap<String, HashSet<OctreeSectionPos>>,
    /// Sections changed since their last save.
    unsaved: HashMap<String, HashSet<OctreeSectionPos>>,
    /// Sections already asked of disk or of the server, per dimension.
    requested: HashMap<String, HashSet<OctreeSectionPos>>,
}

static PIPELINE: LazyLock<Mutex<Pipeline>> = LazyLock::new(Default::default);

/// 8-bit mask of child octants of `pos` that already have a rendered (uploaded to the GPU arena)
/// finer section or descendant. Used when meshing a parent so an octant is never masked out
/// before its replacement child is actually drawn (preventing X-ray pop-in holes).
fn covered_rendered_mask(rendered: &HashSet<OctreeSectionPos>, pos: OctreeSectionPos) -> u8 {
    let Some(kids) = pos.children() else {
        return 0;
    };
    let mut mask = 0u8;
    for (ci, child) in kids.into_iter().enumerate() {
        if rendered.contains(&child)
            || (child.detail > 0 && covered_rendered_mask(rendered, child) != 0)
        {
            mask |= 1u8 << ci;
        }
    }
    mask
}

/// 8-bit mask of child octants of `pos` (indexed by `child_index(cx, cy, cz)`) that already have
/// a non-empty finer section or descendant in `store`. Covered octants are masked out when meshing
/// `pos` so parent and child never overlap, while uncovered octants still draw so LOD ring
/// boundaries never open sky holes or flat stone ceilings.
#[cfg(test)]
pub(crate) fn covered_child_mask(store: &OctreeIngestStore, pos: OctreeSectionPos) -> u8 {
    let Some(kids) = pos.children() else {
        return 0;
    };
    let mut mask = 0u8;
    for (ci, child) in kids.into_iter().enumerate() {
        if store.section(child).is_some_and(|s| !s.is_empty())
            || (child.detail > 0 && has_non_empty_descendant(store, child))
        {
            mask |= 1u8 << ci;
        }
    }
    mask
}

/// Returns `true` when `pos` exists, is non-empty, and has no finer non-empty descendants.
#[cfg(test)]
pub(crate) fn is_render_leaf(store: &OctreeIngestStore, pos: OctreeSectionPos) -> bool {
    let Some(section) = store.section(pos) else {
        return false;
    };
    !section.is_empty() && covered_child_mask(store, pos) == 0
}

#[cfg(test)]
fn has_non_empty_descendant(store: &OctreeIngestStore, pos: OctreeSectionPos) -> bool {
    covered_child_mask(store, pos) != 0
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

/// Which source a snapshot came from: G1 (live chunk), G3 (saved Anvil chunk), or G4 (detached
/// full-chunk generation with trees/features).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotSource {
    Loaded,
    Saved,
    Distant,
}

/// Writes one chunk snapshot into `dim`'s store without propagating mips or queueing meshes.
/// Returns whether the chunk was accepted; the caller runs [`propagate_and_queue`] afterwards.
fn ingest_unpropagated(
    p: &mut Pipeline,
    host: &dyn ServerHost,
    dim: &str,
    chunk: [i32; 2],
    snapshot: &ChunkSnapshot,
    source: SnapshotSource,
) -> bool {
    let cave_cull = (source == SnapshotSource::Distant && dim != "minecraft:the_nether")
        .then(|| (60, p.mapper.block_named(host, "minecraft:stone")));
    let store = p.stores.entry(dim.to_owned()).or_default();
    let mapper = std::cell::RefCell::new(&mut p.mapper);
    match source {
        SnapshotSource::Loaded => ingest_chunk_snapshot(
            store,
            chunk,
            snapshot,
            &mut |state| mapper.borrow_mut().block(host, state),
            &mut |biome| mapper.borrow_mut().biome(biome),
        ),
        SnapshotSource::Saved => ingest_saved_chunk(
            store,
            chunk,
            snapshot,
            &mut |state| mapper.borrow_mut().block(host, state),
            &mut |biome| mapper.borrow_mut().biome(biome),
        ),
        SnapshotSource::Distant => crate::octree_ingest::ingest_chunk_culled(
            store,
            chunk,
            snapshot,
            crate::octree_ingest::IngestSource::Distant,
            cave_cull,
            &mut |state| mapper.borrow_mut().block(host, state),
            &mut |biome| mapper.borrow_mut().biome(biome),
        ),
    }
}

/// Propagates every dirty section of `dim`'s store up the octree once, then marks the touched
/// sections unsaved and queues them for meshing.
fn propagate_and_queue(p: &mut Pipeline, dim: &str) {
    let mapper = &p.mapper;
    let touched = p
        .stores
        .get_mut(dim)
        .expect("store inserted by ingest")
        .propagate_dirty(&|id| mapper.opacity(id));
    p.unsaved
        .entry(dim.to_owned())
        .or_default()
        .extend(touched.iter().copied());
    p.queue(dim, touched);
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
    let accepted = ingest_unpropagated(p, host, dim, chunk, snapshot, source);
    if accepted {
        propagate_and_queue(p, dim);
    }
    accepted
}

/// Ingest several chunk snapshots (one finished distant job) under a single `PIPELINE` lock and
/// run `propagate_dirty` once for all accepted chunks instead of once per chunk. Returns how many
/// chunks were accepted.
pub fn ingest_chunks_batch(
    host: &dyn ServerHost,
    dim: &str,
    chunks: &[([i32; 2], &ChunkSnapshot)],
    source: SnapshotSource,
) -> usize {
    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;
    let mut accepted = 0;
    for &(chunk, snapshot) in chunks {
        if ingest_unpropagated(p, host, dim, chunk, snapshot, source) {
            accepted += 1;
        }
    }
    if accepted > 0 {
        propagate_and_queue(p, dim);
    }
    accepted
}

/// Drops every section of `dim` (level unload).
pub fn clear_dimension(dim: &str) {
    MESH_WORKER.clear_dim(dim);
    let mut p = PIPELINE.lock().unwrap();
    p.stores.remove(dim);
    p.dirty.remove(dim);
    p.queued.remove(dim);
    p.section_epoch.retain(|(d, _), _| d != dim);
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

/// Side of the baked face atlas: one 2048² dynamic texture, 128 x 128 face tiles of 16 texels
/// (16 384 faces = 2 730 full six-face models).
pub const ATLAS_SIDE: usize = 2048;
/// Mip levels of the atlas (`0..=4`: 2048, 1024, 512, 256, 128 texels wide, i.e. face tiles of 16,
/// 8, 4, 2 and 1 texels). `voxy_quads.fsh` samples them with `textureGrad`.
pub const ATLAS_MIP_LEVELS: u32 = 5;
/// Texels per face side (`bakery::FACE_GRID`).
const TILE: usize = crate::bakery::FACE_GRID;
/// Faces the atlas holds; later models draw with the flat per-face colour (tile size 0).
const ATLAS_FACES: usize = (ATLAS_SIDE / TILE) * (ATLAS_SIDE / TILE);
// Every mip level must keep whole tiles (`build_atlas_mips` never mixes neighbouring faces).
const _: () = assert!(TILE >= 1 << (ATLAS_MIP_LEVELS - 1));

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
    for (i, mut bucket) in by_bucket.into_iter().enumerate() {
        if bucket.len() > u16::MAX as usize {
            bucket.truncate(u16::MAX as usize);
        }
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
    // Non-blocking lock: if the server thread is ingesting a distant batch, skip this tick rather
    // than stalling the main render thread.
    let Ok(mut guard) = PIPELINE.try_lock() else {
        return;
    };
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
        let mut metas = Vec::with_capacity(p.mapper.states.len() + 1);
        for id in 0..=p.mapper.states.len() {
            metas.push(p.models.meta_for(id as u32));
        }
        p.meta_snapshot = Arc::new(MetaTable(metas));
        let atlas_mips = build_atlas_mips(&atlas, ATLAS_SIDE, TILE, ATLAS_MIP_LEVELS);
        crate::render_bridge::queue_model_tables(words, tint_words, atlas_mips);
    }
    if p.baked < p.mapper.states.len() {
        return;
    }

    let mut remesh_ancestors: Vec<(String, OctreeSectionPos)> = Vec::new();
    for (done_dim, pos, epoch, built) in MESH_WORKER.drain_done() {
        if p.section_epoch.get(&(done_dim.clone(), pos)).copied() != Some(epoch) {
            continue;
        }
        let has_quads = !built.is_empty();
        let key = (done_dim.clone(), pos.detail, [pos.x, pos.y, pos.z]);
        submit_built_section(key, built);
        if !has_quads {
            // The matching epoch is the newest job for `pos`: an empty result retires the entry.
            p.section_epoch.remove(&(done_dim.clone(), pos));
        }
        let changed = if has_quads {
            p.rendered.entry(done_dim.clone()).or_default().insert(pos)
        } else {
            p.rendered
                .get_mut(&done_dim)
                .is_some_and(|live| live.remove(&pos))
        };
        if changed {
            let mut anc = pos.parent();
            while let Some(a) = anc {
                let should_requeue = if has_quads {
                    p.rendered
                        .get(&done_dim)
                        .is_some_and(|live| live.contains(&a))
                } else {
                    p.stores
                        .get(&done_dim)
                        .is_some_and(|s| s.section(a).is_some_and(|sec| !sec.is_empty()))
                };
                if should_requeue {
                    remesh_ancestors.push((done_dim.clone(), a));
                }
                anc = a.parent();
            }
        }
    }
    for (anc_dim, anc_pos) in remesh_ancestors {
        p.queue(&anc_dim, [anc_pos]);
    }

    let metas = Arc::clone(&p.meta_snapshot);
    let mut pending = MESH_WORKER.pending_len();
    for _ in 0..MESHES_PER_TICK {
        if pending >= MAX_PENDING_MESH_JOBS {
            break;
        }
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
        let Some(section) = store.section(pos).filter(|s| !s.is_empty()) else {
            // Removing the entry drops any in-flight mesh result for `pos` (no epoch matches) and
            // keeps `section_epoch` bounded by the live sections.
            p.section_epoch.remove(&(dim.to_owned(), pos));
            if p.rendered
                .get_mut(dim)
                .is_some_and(|live| live.remove(&pos))
            {
                submit_built_section(key, empty_section(pos));
            }
            continue;
        };
        let covered = p
            .rendered
            .get(dim)
            .map_or(0, |live| covered_rendered_mask(live, pos));
        if covered == 0xFF {
            p.section_epoch.remove(&(dim.to_owned(), pos));
            if p.rendered
                .get_mut(dim)
                .is_some_and(|live| live.remove(&pos))
            {
                submit_built_section(key, empty_section(pos));
            }
            continue;
        }
        let neighbors: [Option<&OctreeSection>; 6] =
            Face::ALL.map(|face| store.section(pos.neighbor(face)));
        let borders = NeighborBorders::from_neighbors(neighbors);
        let section_clone = section.clone();
        p.next_epoch = p.next_epoch.wrapping_add(1);
        let epoch = p.next_epoch;
        p.section_epoch.insert((dim.to_owned(), pos), epoch);
        MESH_WORKER.push(MeshJob {
            dim: dim.to_owned(),
            pos,
            epoch,
            section: section_clone,
            borders,
            covered,
            metas: Arc::clone(&metas),
        });
        pending += 1;
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
    let stone = p.mapper.block_named(host, "minecraft:stone");
    let store = p.stores.entry(dim.to_owned()).or_default();
    let mapper = std::cell::RefCell::new(&mut p.mapper);
    let mut block = |key| mapper.borrow_mut().block_key(host, key);
    let mut biome = |name: &str| mapper.borrow_mut().biome(name);
    let touched = if detail == 0 {
        crate::octree_ingest::ingest_distant_columns(
            store, &columns, water, stone, &mut block, &mut biome,
        );
        let mapper = &p.mapper;
        p.stores
            .get_mut(dim)
            .expect("store inserted above")
            .propagate_dirty(&|id| mapper.opacity(id))
    } else {
        store.ingest_coarse_columns(detail, &columns, water, stone, &mut block, &mut biome)
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
            // A clear must supersede any pending save; only the I/O worker knows whether
            // there is a disk slot to rewrite.
            let Some(section) = store.section(pos) else {
                continue;
            };
            let (_, mapping) = p.mapper.mapping_of(section);
            io.save_octree_section(world_dir.to_owned(), dim.clone(), section.clone(), mapping);
            queued += 1;
        }
    }
    queued
}

/// Level-0 section rows (`y`) the sweeps cover; level `n` covers them shifted right by `n`.
const Y_SECTIONS: std::ops::RangeInclusive<i32> = -2..=9;

/// The sections the GPU traversal asked for (`render_bridge::drain_gpu_node_requests`: finer
/// children of visible nodes) that the store lacks and nobody asked for yet, at most `budget`,
/// marked as requested. Positions outside the vertical range the sweeps cover are ignored.
fn take_gpu_requested(
    store: Option<&OctreeIngestStore>,
    requested: &mut HashSet<OctreeSectionPos>,
    budget: usize,
) -> Vec<OctreeSectionPos> {
    let mut out = Vec::new();
    for (detail, [x, y, z]) in crate::render_bridge::drain_gpu_node_requests() {
        if out.len() >= budget {
            break;
        }
        if detail > crate::octree::MAX_OCTREE_DETAIL
            || !(Y_SECTIONS.start() >> detail..=Y_SECTIONS.end() >> detail).contains(&y)
        {
            continue;
        }
        let pos = OctreeSectionPos::new(detail, x, y, z);
        if store.is_some_and(|s| s.section(pos).is_some()) || !requested.insert(pos) {
            continue;
        }
        out.push(pos);
    }
    out
}

/// Ask disk (single player and server) for stored sections around `center` that the store does not
/// hold yet: first what the GPU traversal asked for, then every level from fine to coarse,
/// `radius` sections out, at most `budget` requests.
pub fn request_stored_sections(
    io: &LodIoWorker,
    world_dir: &Path,
    dim: &str,
    center: [f64; 3],
    radius: i32,
    budget: usize,
) {
    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;
    let requested = p.requested.entry(dim.to_owned()).or_default();
    let store = p.stores.get(dim);
    let mut left = budget;
    for pos in take_gpu_requested(store, requested, left) {
        io.load_octree_section(world_dir.to_owned(), dim.to_owned(), pos);
        left -= 1;
    }
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
        want.extend(take_gpu_requested(store, requested, budget));
        let edge = crate::octree::SECTION_EDGE as f64;
        let (cx, cz) = (
            (center[0] / edge).floor() as i32,
            (center[2] / edge).floor() as i32,
        );
        'scan: for sx in cx - radius..=cx + radius {
            for sz in cz - radius..=cz + radius {
                for sy in -2..=9 {
                    if want.len() >= budget {
                        break 'scan;
                    }
                    let pos = OctreeSectionPos::new(0, sx, sy, sz);
                    if store.is_some_and(|s| s.section(pos).is_some()) || !requested.insert(pos) {
                        continue;
                    }
                    want.push(pos);
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
