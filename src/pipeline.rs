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
use crate::lod::SectionPos;
use crate::mesh::{SectionQuads, mesh_octree_section};
use crate::network::octree_sync::{ClientEvent, MappingEntry, OctreeSectionDto, ServerSource};
use crate::octree::Voxel;
use crate::octree::{Face, NeighborBorders, OctreeSection, OctreeSectionPos};
use crate::octree_ingest::{
    DistantColumn, DistantSample, OctreeIngestStore, ingest_chunk_snapshot, ingest_saved_chunk,
};
use crate::render_bridge::submit_built_section;
use crate::storage::{DEFAULT_OCTREE_COMPRESSION, LodIoWorker, OctreeLoaded, OctreeMapping};
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
    complete: bool,
    borders: NeighborBorders,
    metas: Arc<MetaTable>,
}

type MeshDone = (String, OctreeSectionPos, u64, BuiltSection);

impl MeshJob {
    fn build(self) -> MeshDone {
        let quads = mesh_octree_section(&self.section, &self.borders, &*self.metas);
        (
            self.dim,
            self.pos,
            self.epoch,
            built_section(&quads, self.complete),
        )
    }
}

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
                    let done = job.build();
                    d_bg.lock().unwrap().push(done);
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

    fn clear_all(&self) {
        self.queue.0.lock().unwrap().clear();
        self.done.lock().unwrap().clear();
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
    /// Sections meshed with border inputs, including empty results. Changes to adjacent
    /// sections can expose faces and invalidate a previously empty mesh.
    meshed: HashMap<String, HashSet<OctreeSectionPos>>,
    /// Sections changed since their last save.
    unsaved: HashMap<String, HashSet<OctreeSectionPos>>,
    /// Sections already asked of disk or of the server, per dimension.
    requested: HashMap<String, HashSet<OctreeSectionPos>>,
}

static PIPELINE: LazyLock<Mutex<Pipeline>> = LazyLock::new(Default::default);

impl Pipeline {
    fn clear_dimension(&mut self, dim: &str) {
        self.stores.remove(dim);
        self.forget_gpu_residency(dim);
        self.unsaved.remove(dim);
        self.requested.remove(dim);
    }

    fn forget_gpu_residency(&mut self, dim: &str) {
        self.dirty.remove(dim);
        self.queued.remove(dim);
        self.meshed.remove(dim);
        // Keep next_epoch so a job already popped by the worker cannot match on return.
        self.section_epoch.retain(|(d, _), _| d != dim);
    }

    fn clear_all(&mut self) {
        self.stores.clear();
        self.dirty.clear();
        self.queued.clear();
        self.section_epoch.clear();
        self.meshed.clear();
        self.unsaved.clear();
        self.requested.clear();
        // Keep next_epoch so a mesh still in flight cannot match a reopened world's section.
    }

    fn flush_saves(&mut self, io: &LodIoWorker, world_dir: &Path, limit: usize) -> usize {
        let mut queued = 0;
        for (dim, set) in self.unsaved.iter_mut() {
            let Some(store) = self.stores.get(dim) else {
                set.clear();
                continue;
            };
            while queued < limit {
                let Some(&pos) = set.iter().next() else { break };
                set.remove(&pos);
                let complete = store.is_section_complete(pos);
                let (section, mapping) = match store.section(pos) {
                    Some(section) => (section.clone(), self.mapper.mapping_of(section).1),
                    None if complete => (OctreeSection::new(pos), OctreeMapping::default()),
                    None => continue,
                };
                io.save_octree_section(
                    world_dir.to_owned(),
                    dim.clone(),
                    section,
                    mapping,
                    complete,
                );
                queued += 1;
            }
        }
        queued
    }

    /// Collect up to `budget` unrequested octree disk loads in desired-leaf order, probing
    /// missing leaves and their coarser ancestor chain through `MAX_OCTREE_DETAIL` so saved
    /// coarser sections survive movement or quality increases. Advance `cursor` only after
    /// every cell and ancestor probe of the current leaf is stored or requested.
    fn collect_stored_requests_for_leaves(
        &mut self,
        dim: &str,
        leaves: &[SectionPos],
        cursor: &mut usize,
        budget: usize,
    ) -> Vec<OctreeSectionPos> {
        if budget == 0 || *cursor >= leaves.len() {
            return Vec::new();
        }
        const Y_SECTIONS: std::ops::RangeInclusive<i32> = -2..=9;
        let requested = self.requested.entry(dim.to_owned()).or_default();
        let store = self.stores.get(dim);
        let mut out = Vec::new();
        while let Some(leaf) = leaves.get(*cursor) {
            let level = leaf.detail.min(crate::octree::MAX_OCTREE_DETAIL);
            let span = 2i32 << (leaf.detail - level);
            let (sx0, sz0) = (leaf.x * span, leaf.z * span);
            for sx in sx0..sx0 + span {
                for sz in sz0..sz0 + span {
                    for sy in (Y_SECTIONS.start() >> level)..=(Y_SECTIONS.end() >> level) {
                        let mut cur = Some(OctreeSectionPos::new(level, sx, sy, sz));
                        while let Some(candidate) = cur {
                            if store.is_some_and(|s| s.section(candidate).is_some()) {
                                break;
                            }
                            if !requested.contains(&candidate) {
                                if out.len() >= budget {
                                    return out;
                                }
                                requested.insert(candidate);
                                out.push(candidate);
                            }
                            cur = candidate.parent();
                        }
                    }
                }
            }
            *cursor += 1;
        }
        out
    }

    fn adopt_loaded_entries(
        &mut self,
        loaded: Vec<OctreeLoaded>,
        is_loaded: impl Fn(&str) -> bool,
        mut remap: impl FnMut(&mut Mapper, &OctreeSection, &OctreeMapping) -> OctreeSection,
    ) {
        for l in loaded {
            if !is_loaded(&l.dimension) {
                continue;
            }
            let section = remap(&mut self.mapper, &l.section, &l.mapping);
            let pos = section.pos;
            let store = self.stores.entry(l.dimension.clone()).or_default();
            if store.adopt_section(section, l.complete) {
                self.queue(&l.dimension, [pos]);
            }
        }
    }

    fn requeue_dimension(&mut self, dim: &str) -> usize {
        let Some(positions) = self
            .stores
            .get(dim)
            .map(OctreeIngestStore::section_positions)
        else {
            return 0;
        };
        let count = positions.len();
        self.queue(dim, positions);
        count
    }

    /// Queues every stored section of every dimension; returns how many.
    fn queue_all(&mut self) -> usize {
        let dims: Vec<String> = self.stores.keys().cloned().collect();
        dims.into_iter()
            .map(|dim| self.requeue_dimension(&dim))
            .sum()
    }

    /// The GPU arena is gone, so nothing counts as uploaded: drop residency and pre-reset worker
    /// epochs (their results are discarded), then mesh every stored section again.
    fn reset_arena_state(&mut self) -> usize {
        self.meshed.clear();
        self.section_epoch.clear();
        self.dirty.clear();
        self.queued.clear();
        self.models_dirty = true;
        self.queue_all()
    }

    fn queue(&mut self, dim: &str, positions: impl IntoIterator<Item = OctreeSectionPos>) {
        let meshed = self.meshed.get(dim);
        let set = self.queued.entry(dim.to_owned()).or_default();
        let queue = self.dirty.entry(dim.to_owned()).or_default();
        let mut push = |pos: OctreeSectionPos| {
            if set.insert(pos) {
                queue.push_back(pos);
            }
        };
        for pos in positions {
            push(pos);
            // Re-mesh same-detail neighbours so touching border walls are culled or exposed.
            if let Some(live) = meshed {
                for face in Face::ALL {
                    let nb = pos.neighbor(face);
                    if live.contains(&nb) {
                        push(nb);
                    }
                }
            }
        }
    }

    fn dispatch_meshes(
        &mut self,
        dim: &str,
        mut pending: usize,
        mut submit: impl FnMut(crate::render_bridge::SectionKey, BuiltSection),
        mut enqueue: impl FnMut(MeshJob),
    ) {
        let metas = Arc::clone(&self.meta_snapshot);
        for _ in 0..MESHES_PER_TICK {
            if pending >= MAX_PENDING_MESH_JOBS {
                break;
            }
            let Some(pos) = self.dirty.get_mut(dim).and_then(VecDeque::pop_front) else {
                break;
            };
            if let Some(set) = self.queued.get_mut(dim) {
                set.remove(&pos);
            }
            let Some(store) = self.stores.get(dim) else {
                continue;
            };
            let complete = store.is_section_complete(pos);
            let Some(terrain) = store.section(pos).filter(|s| !s.is_empty()) else {
                self.section_epoch.remove(&(dim.to_owned(), pos));
                if let Some(live) = self.meshed.get_mut(dim) {
                    live.remove(&pos);
                }
                let key = (dim.to_owned(), pos.detail, [pos.x, pos.y, pos.z]);
                // Incomplete air withdraws any previous empty proof for this position.
                submit(key, empty_section(pos, complete));
                continue;
            };
            let neighbors: [Option<&OctreeSection>; 6] =
                Face::ALL.map(|face| store.section(pos.neighbor(face)));
            let borders = NeighborBorders::from_neighbors(neighbors);
            let section = terrain.clone();
            self.next_epoch = self.next_epoch.wrapping_add(1);
            let epoch = self.next_epoch;
            self.section_epoch.insert((dim.to_owned(), pos), epoch);
            enqueue(MeshJob {
                dim: dim.to_owned(),
                pos,
                epoch,
                section,
                complete,
                borders,
                metas: Arc::clone(&metas),
            });
            pending += 1;
        }
    }

    fn apply_mesh_completions(
        &mut self,
        active_dim: Option<&str>,
        completions: Vec<MeshDone>,
        mut submit: impl FnMut(crate::render_bridge::SectionKey, BuiltSection),
    ) {
        for (done_dim, pos, epoch, built) in completions {
            if self.section_epoch.get(&(done_dim.clone(), pos)).copied() != Some(epoch) {
                continue;
            }
            if active_dim != Some(done_dim.as_str()) {
                self.section_epoch.remove(&(done_dim, pos));
                continue;
            }
            let empty = built.is_empty();
            let key = (done_dim.clone(), pos.detail, [pos.x, pos.y, pos.z]);
            submit(key, built);
            if empty {
                self.section_epoch.remove(&(done_dim.clone(), pos));
            }
            self.meshed.entry(done_dim).or_default().insert(pos);
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
            dim != "minecraft:the_nether", // Cull underground only for non-Nether distant chunks.
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
    let mut p = PIPELINE.lock().unwrap();
    p.clear_dimension(dim);
    MESH_WORKER.clear_dim(dim);
}

/// Retires a departing client's meshes while keeping the integrated server's octree and disk state.
pub fn forget_gpu_residency(dim: &str) {
    let mut p = PIPELINE.lock().unwrap();
    p.forget_gpu_residency(dim);
    MESH_WORKER.clear_dim(dim);
}

/// Queues the entering dimension's retained sections for upload; returns how many.
pub fn requeue_dimension(dim: &str) -> usize {
    PIPELINE.lock().unwrap().requeue_dimension(dim)
}

/// Drops the closed world's sections, player positions and queued mesh work.
pub fn clear_all() {
    {
        let mut p = PIPELINE.lock().unwrap();
        p.clear_all();
        MESH_WORKER.clear_all();
    }
    PLAYER_XZ.lock().unwrap().clear();
}

/// A resource reload, shader-pack change or `/ds remesh` invalidates every baked model: bake them
/// all again and mesh everything once the models exist. Returns how many sections were queued.
pub fn invalidate_models() -> usize {
    let mut p = PIPELINE.lock().unwrap();
    p.models.clear();
    p.baked = 0;
    p.models_dirty = true;
    p.queue_all()
}

/// The GPU arena was lost and its sections are gone: forget what was uploaded and mesh every
/// stored section again. Returns how many sections were queued.
pub(crate) fn reset_arena_state() -> usize {
    PIPELINE.lock().unwrap().reset_arena_state()
}

/// A new arena has empty model, tint and atlas tables: send them again without re-baking.
pub(crate) fn mark_models_dirty() {
    PIPELINE.lock().unwrap().models_dirty = true;
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
fn built_section(quads: &SectionQuads, complete: bool) -> BuiltSection {
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
        complete,
    }
}

/// Client tick: bake pending block models, upload the model and biome tables when they grew, then
/// mesh dirty sections of the active level into the GPU arena queue.
pub fn client_tick(host: &mut dyn ClientHost, active_dim: Option<&str>) {
    // Non-blocking lock: if the server thread is ingesting a distant batch, skip this tick rather
    // than stalling the main render thread.
    let Ok(mut guard) = PIPELINE.try_lock() else {
        return;
    };
    let p = &mut *guard;
    let Some(dim) = active_dim else {
        p.apply_mesh_completions(None, MESH_WORKER.drain_done(), |_, _| {});
        return;
    };

    if let Some(bounds) = p
        .stores
        .get(dim)
        .and_then(OctreeIngestStore::y_bounds)
        .or_else(|| {
            let (min_y, height) = host.level(dim)?.height(host)?;
            Some((min_y, min_y + height))
        })
    {
        crate::render_bridge::set_dimension_y_bounds(dim, bounds);
    }

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
        crate::render_bridge::queue_model_tables(words, tint_words, atlas);
    }
    if p.baked < p.mapper.states.len() {
        return;
    }

    p.apply_mesh_completions(Some(dim), MESH_WORKER.drain_done(), submit_built_section);

    p.dispatch_meshes(
        dim,
        MESH_WORKER.pending_len(),
        submit_built_section,
        |job| {
            MESH_WORKER.push(job);
        },
    );
}

fn empty_section(pos: OctreeSectionPos, complete: bool) -> BuiltSection {
    BuiltSection {
        level: pos.detail,
        pos: [pos.x, pos.y, pos.z],
        aabb: SectionAabb::FULL,
        quads: Vec::new(),
        counts: [0; BUCKETS],
        complete,
    }
}

/// Whether a finished job leaves more schedulable work at its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DistantCoverage {
    Complete,
    /// At least one column has no surface sample or usable authoritative reduction.
    Unavailable,
    Incomplete,
}

impl Pipeline {
    fn coarse_coverage(&self, dim: &str, pos: SectionPos) -> [[u64; 16]; 4] {
        self.stores
            .get(dim)
            .map_or([[0; 16]; 4], |store| store.coarse_coverage(pos))
    }

    fn ingest_coarse_job<'a, B>(
        &mut self,
        dim: &str,
        pos: SectionPos,
        columns: impl IntoIterator<Item = DistantSample<'a, B>>,
        materials: (u32, u32),
        mut resolve_block: impl FnMut(&mut Mapper, B) -> u32,
    ) -> DistantCoverage {
        let store = self.stores.entry(dim.to_owned()).or_default();
        let coverage =
            (pos.detail <= crate::octree::MAX_OCTREE_DETAIL).then(|| store.coarse_coverage(pos));
        // The worker may also have sampled cells solely to refill the legacy row, or
        // disk adoption may have covered them since dispatch. Preserve their octree bytes.
        let columns = columns.into_iter().filter(|column| {
            coverage.as_ref().is_none_or(|bits| {
                let (x, z) = match column {
                    DistantSample::Surface(col) => (col.x, col.z),
                    DistantSample::NoSurface(x, z) => (*x, *z),
                };
                let (cx, cz) = pos.local_cell(x, z);
                !coarse_cell_covered(bits, cx, cz)
            })
        });
        let mapper = std::cell::RefCell::new(&mut self.mapper);
        let unavailable = store.ingest_coarse_columns(
            pos.detail,
            columns,
            materials,
            &mut |key| resolve_block(&mut mapper.borrow_mut(), key),
            &mut |name| mapper.borrow_mut().biome(name),
            &|id| mapper.borrow().opacity(id),
        );
        let complete = store.coarse_coverage(pos) == [[u64::MAX; 16]; 4];
        propagate_and_queue(self, dim);
        if complete {
            DistantCoverage::Complete
        } else if unavailable {
            DistantCoverage::Unavailable
        } else {
            DistantCoverage::Incomplete
        }
    }
}

pub(crate) fn coarse_coverage(dim: &str, pos: SectionPos) -> [[u64; 16]; 4] {
    PIPELINE.lock().unwrap().coarse_coverage(dim, pos)
}

pub(crate) fn coarse_cell_covered(coverage: &[[u64; 16]; 4], x: usize, z: usize) -> bool {
    use crate::octree::SECTION_EDGE;

    let stack = x / SECTION_EDGE + 2 * (z / SECTION_EDGE);
    let bit = x % SECTION_EDGE + (z % SECTION_EDGE) * SECTION_EDGE;
    coverage[stack][bit / 64] & (1 << (bit % 64)) != 0
}

/// G4: ingest the returned prefix of one finished distant-generation job. Each cell covers
/// `2^detail` blocks and maps to one octree voxel at that detail.
pub(crate) fn ingest_distant(
    host: &dyn ServerHost,
    dim: &str,
    pos: SectionPos,
    cells: &[(usize, usize)],
    samples: &[Option<TerrainColumn>],
) -> DistantCoverage {
    // Detail 0 retains its existing snapshot/surface scheduler and completion behavior.
    if pos.detail == 0 && !samples.iter().any(Option::is_some) {
        return DistantCoverage::Complete;
    }

    let mut guard = PIPELINE.lock().unwrap();
    let p = &mut *guard;
    let origin = pos.block_origin();
    let step = pos.cell_size();
    let columns = cells.iter().zip(samples).map(|(&(cx, cz), sample)| {
        let x = origin.0 + cx as i32 * step;
        let z = origin.1 + cz as i32 * step;
        match sample {
            Some(col) => DistantSample::Surface(DistantColumn {
                x,
                z,
                height: col.height,
                water: col.water,
                block: col.block,
                biome: &col.biome,
            }),
            None => DistantSample::NoSurface(x, z),
        }
    });
    let water = p.mapper.block_named(host, "minecraft:water");
    let stone = p.mapper.block_named(host, "minecraft:stone");
    let store = p.stores.entry(dim.to_owned()).or_default();
    if store.y_bounds().is_none()
        && let Some((min_y, height)) = host.level(dim).and_then(|level| level.height(host))
    {
        store.set_y_bounds((min_y, min_y + height));
    }
    if pos.detail > 0 {
        return p.ingest_coarse_job(dim, pos, columns, (water, stone), |mapper, key| {
            mapper.block_key(host, key)
        });
    }

    let columns: Vec<_> = columns
        .filter_map(|column| match column {
            DistantSample::Surface(col) => Some(col),
            DistantSample::NoSurface(..) => None,
        })
        .collect();
    let mapper = std::cell::RefCell::new(&mut p.mapper);
    crate::octree_ingest::ingest_distant_columns(
        store,
        &columns,
        water,
        stone,
        &mut |key| mapper.borrow_mut().block_key(host, key),
        &mut |name| mapper.borrow_mut().biome(name),
    );
    propagate_and_queue(p, dim);
    DistantCoverage::Complete
}

/// Hand up to `limit` changed sections of every dimension to the I/O worker (it coalesces newer
/// saves of one section). Returns how many were queued.
pub fn flush_saves(io: &LodIoWorker, world_dir: &Path, limit: usize) -> usize {
    PIPELINE.lock().unwrap().flush_saves(io, world_dir, limit)
}

/// Ask disk for stored octree sections covering the desired leaves across the full configured
/// render distance, paced by `cursor` and bounded by `budget`.
pub fn request_stored_sections_for_leaves(
    io: &LodIoWorker,
    world_dir: &Path,
    dim: &str,
    leaves: &[SectionPos],
    cursor: &mut usize,
    budget: usize,
) -> usize {
    if budget == 0 {
        return 0;
    }
    let positions = PIPELINE
        .lock()
        .unwrap()
        .collect_stored_requests_for_leaves(dim, leaves, cursor, budget);
    let count = positions.len();
    for pos in positions {
        io.load_octree_section(world_dir.to_owned(), dim.to_owned(), pos);
    }
    count
}

/// Adopt sections the I/O worker finished loading, translating their ids to this session's.
pub fn adopt_loaded(host: &dyn ServerHost, io: &LodIoWorker) {
    let loaded = io.drain_octree_responses();
    if loaded.is_empty() {
        return;
    }
    PIPELINE.lock().unwrap().adopt_loaded_entries(
        loaded,
        |dim| host.level(dim).is_some(),
        |mapper, section, mapping| mapper.remap(host, section, &mapping.blocks, &mapping.biomes),
    );
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
    use crate::config::DhConfig;
    use crate::gpu_arena::{GpuNode, test_node_tree, unpack_section_pos};
    use crate::lod::{DimensionLodStore, SECTION_SIZE};
    use crate::octree::MAX_OCTREE_DETAIL;
    use crate::octree_ingest::{ChunkVoxelSource, IngestSource};

    struct FixtureChunk {
        sections: usize,
        bottom_block: u32,
    }

    impl ChunkVoxelSource for FixtureChunk {
        type Block = u32;

        fn min_y(&self) -> i32 {
            0
        }
        fn section_count(&self) -> usize {
            self.sections
        }
        fn section_uniform_block(&self, index: usize) -> Option<u32> {
            Some(if index == 0 { self.bottom_block } else { 0 })
        }
        fn block(&self, _: usize, _: i32, _: usize) -> Option<u32> {
            unreachable!("fixture slabs are uniform")
        }
        fn biome(&self, _: usize, _: i32, _: usize) -> Option<&str> {
            None
        }
        fn light(&self, _: usize, _: i32, _: usize) -> (u8, u8) {
            (15, 0)
        }
    }

    fn coarse_job_columns(leaf: SectionPos) -> Vec<DistantColumn<'static, u32>> {
        let (x, z) = leaf.block_origin();
        let step = leaf.cell_size();
        (0..SECTION_SIZE as i32)
            .flat_map(|cz| {
                (0..SECTION_SIZE as i32).map(move |cx| DistantColumn {
                    x: x + cx * step,
                    z: z + cz * step,
                    height: 6,
                    water: None,
                    block: 1,
                    biome: "minecraft:plains",
                })
            })
            .collect()
    }

    fn fixture_node(nodes: &[GpuNode], pos: OctreeSectionPos) -> &GpuNode {
        nodes
            .iter()
            .find(|n| unpack_section_pos([n.pos0, n.pos1]) == (pos.detail, [pos.x, pos.y, pos.z]))
            .unwrap()
    }

    #[test]
    fn leaf_disk_reload_covers_full_render_distance_with_cursor_and_backpressure() {
        let dim = "minecraft:overworld";
        let mut leaves =
            DimensionLodStore::desired_sections_coarse_to_fine(0.0, 0.0, DhConfig::default());
        leaves.extend([SectionPos::new(5, -3, 2), SectionPos::new(6, 3, -3)]);
        let mut p = Pipeline::default();
        let mut cursor = 0;
        assert!(
            p.collect_stored_requests_for_leaves(dim, &leaves, &mut cursor, 0)
                .is_empty()
        );
        assert_eq!(cursor, 0);
        assert!(p.requested.is_empty());

        let mut expected = HashSet::new();
        for &leaf in &leaves {
            let (x, z) = leaf.block_origin();
            let width = leaf.block_span();
            for detail in leaf.detail.min(MAX_OCTREE_DETAIL)..=MAX_OCTREE_DETAIL {
                let edge = OctreeSectionPos::new(detail, 0, 0, 0).block_span();
                for sx in x.div_euclid(edge)..=(x + width - 1).div_euclid(edge) {
                    for sz in z.div_euclid(edge)..=(z + width - 1).div_euclid(edge) {
                        for sy in (-64i32).div_euclid(edge)..=319i32.div_euclid(edge) {
                            expected.insert(OctreeSectionPos::new(detail, sx, sy, sz));
                        }
                    }
                }
            }
        }
        let mut seen = HashSet::new();
        while cursor < leaves.len() {
            let previous = cursor;
            let batch = p.collect_stored_requests_for_leaves(dim, &leaves, &mut cursor, 20);
            assert!(batch.len() <= 20);
            assert!(cursor > previous || !batch.is_empty());
            if seen.is_empty() {
                let (x, z) = leaves[0].block_origin();
                assert_eq!(batch.len(), 20);
                assert_eq!(batch[0], OctreeSectionPos::new(0, x / 32, -2, z / 32));
                assert_eq!(cursor, 0, "the first leaf is still being drained");
            }
            for pos in batch {
                assert!(seen.insert(pos), "duplicate request for {pos:?}");
            }
        }
        assert_eq!(cursor, leaves.len());
        assert!(
            seen.iter()
                .any(|p| p.detail == 0 && p.x.abs().max(p.z.abs()) > 8)
        );
        assert_eq!(seen, expected);

        // Isolate one fine cell's chain: every other fine cell of this leaf is resident.
        let leaf = [SectionPos::new(0, 0, 0)];
        let fine = OctreeSectionPos::new(0, 0, -2, 0);
        for (resident, want) in [(fine, vec![]), (fine.parent().unwrap(), vec![fine])] {
            let mut p = Pipeline::default();
            let store = p.stores.entry(dim.to_owned()).or_default();
            for &pos in &expected {
                if pos.detail == 0
                    && (0..2).contains(&pos.x)
                    && (0..2).contains(&pos.z)
                    && pos != fine
                {
                    store.adopt_section(OctreeSection::new(pos), true);
                }
            }
            store.adopt_section(OctreeSection::new(resident), true);
            let mut cursor = 0;
            assert_eq!(
                p.collect_stored_requests_for_leaves(dim, &leaf, &mut cursor, 20),
                want
            );
            assert_eq!(cursor, leaf.len());
        }
    }

    #[test]
    fn leaf_disk_reload_recovers_saved_coarse_d1_section_under_high_quality_d0_leaves() {
        use crate::storage::{RegionCache, fresh_tmp_dir};

        let cfg = DhConfig::default();
        let (x, z) = (384, 0);
        let desired = |config| DimensionLodStore::desired_sections_coarse_to_fine(0.0, 0.0, config);
        assert!(desired(cfg).contains(&SectionPos::from_block(1, x, z)));
        let leaves = desired(DhConfig {
            horizontal_quality: 2,
            ..cfg
        });
        assert!(leaves.contains(&SectionPos::from_block(0, x, z)));

        let dim = "minecraft:overworld";
        let dir = fresh_tmp_dir("reload-coarse-section");
        let saved = OctreeSectionPos::new(1, 6, 1, 0);
        let voxel = Voxel::new(1, 0, 15, 0);
        let section = OctreeSection::filled(saved, voxel);
        RegionCache::default()
            .save_octree_section(
                &dir,
                dim,
                &section,
                &OctreeMapping::default(),
                true,
                DEFAULT_OCTREE_COMPRESSION,
            )
            .unwrap();
        let mut p = Pipeline::default();
        let mut cache = RegionCache::default();
        let mut cursor = 0;
        let found: Vec<_> = p
            .collect_stored_requests_for_leaves(dim, &leaves, &mut cursor, usize::MAX)
            .into_iter()
            .filter_map(|pos| cache.load_octree_section(&dir, dim, pos))
            .map(|(section, _, _)| (section.pos, section.get(0, 0, 0)))
            .collect();
        drop(cache);
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(cursor, leaves.len());
        assert_eq!(found, [(saved, voxel)]);
    }

    #[test]
    fn flush_saves_before_clear_dimension_persists_dirty_octree_section() {
        use crate::storage::{RegionCache, fresh_tmp_dir};

        let dir = fresh_tmp_dir("unload-flush");
        let io = LodIoWorker::spawn();
        let mut p = Pipeline::default();
        let pos = OctreeSectionPos::new(0, 1, 2, 3);
        let voxel = Voxel::new(1, 0, 15, 0);
        let mut section = OctreeSection::new(pos);
        section.set(0, 0, 0, voxel);
        let (saved, discarded) = ("test:saved", "test:discarded");
        for dim in [saved, discarded] {
            let store = p.stores.entry(dim.into()).or_default();
            store.adopt_section(section.clone(), dim == saved);
            p.unsaved.insert(dim.into(), HashSet::from([pos]));
        }
        p.clear_dimension(discarded);
        let queued = p.flush_saves(&io, &dir, usize::MAX);
        p.clear_dimension(saved);
        io.flush();
        drop(io);
        let [kept, lost] = [saved, discarded]
            .map(|dim| RegionCache::default().load_octree_section(&dir, dim, pos));
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(queued, 1);
        let (kept, _, complete) = kept.unwrap();
        assert_eq!(kept.get(0, 0, 0), voxel);
        assert!(complete);
        assert!(lost.is_none());
    }

    #[test]
    fn flush_saves_replaces_complete_terrain_with_partially_sampled_air() {
        use crate::storage::{RegionCache, fresh_tmp_dir};

        let dim = "test:partial-air";
        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let dir = fresh_tmp_dir("partial-air-overwrite");
        let mut cache = RegionCache::default();
        cache
            .save_octree_section(
                &dir,
                dim,
                &mesh_fixture(pos),
                &OctreeMapping::default(),
                true,
                DEFAULT_OCTREE_COMPRESSION,
            )
            .unwrap();
        let (old, _, complete) = cache.load_octree_section(&dir, dim, pos).unwrap();
        drop(cache);
        let mut p = Pipeline::default();
        let store = p.stores.entry(dim.into()).or_default();
        assert!(store.adopt_section(old, complete));
        assert!(store.is_section_complete(pos));
        assert!(crate::octree_ingest::ingest_chunk_snapshot(
            store,
            [0, 0],
            &FixtureChunk {
                sections: 1,
                bottom_block: 0
            },
            &mut |b| b,
            &mut |_| 0,
        ));
        let air = Voxel::air_with_light(15, 0);
        assert!(store.section(pos).unwrap().is_empty());
        assert_eq!(store.section(pos).unwrap().get(3, 3, 3), air);
        assert!(!store.is_section_complete(pos));
        propagate_and_queue(&mut p, dim);
        let io = LodIoWorker::spawn();
        assert!(p.flush_saves(&io, &dir, usize::MAX) > 0);
        io.flush();
        drop(io);
        let (saved, _, complete) = RegionCache::default()
            .load_octree_section(&dir, dim, pos)
            .unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let mut reopened = OctreeIngestStore::new();
        assert!(reopened.adopt_section(saved, complete));
        assert!(reopened.section(pos).unwrap().is_empty());
        assert_eq!(reopened.section(pos).unwrap().get(3, 3, 3), air);
        assert!(!reopened.is_section_complete(pos));
    }

    fn mesh_fixture(pos: OctreeSectionPos) -> OctreeSection {
        let mut terrain = OctreeSection::new(pos);
        terrain.set(3, 3, 3, Voxel::new(1, 0, 15, 0));
        terrain
    }

    fn drain_fixture_meshes(p: &mut Pipeline, dim: &str) -> Vec<BuiltSection> {
        use crate::bakery::ModelMeta;
        p.meta_snapshot = Arc::new(MetaTable(vec![
            ModelMeta::EMPTY,
            ModelMeta {
                model_id: 1,
                face_exists: 0x3f,
                face_occludes: 0x3f,
                ..ModelMeta::EMPTY
            },
            ModelMeta {
                model_id: 2,
                face_exists: 0x3f,
                face_occludes: 0x3f,
                ..ModelMeta::EMPTY
            },
        ]));
        let mut submitted = Vec::new();
        while !p.dirty[dim].is_empty() {
            let mut done = Vec::new();
            p.dispatch_meshes(
                dim,
                0,
                |_, built| submitted.push(built),
                |job| done.push(job.build()),
            );
            p.apply_mesh_completions(Some(dim), done, |_, built| submitted.push(built));
        }
        submitted
    }

    #[test]
    fn cache_only_reopen_hands_off_to_terrain_child_with_seven_air_siblings() {
        use crate::storage::{RegionCache, fresh_tmp_dir};

        let dim = "test:cache-only";
        let parent = OctreeSectionPos::new(1, 0, 0, 0);
        let kids = parent.children().unwrap();
        let terrain = kids[0];
        let positions = [parent].into_iter().chain(kids);
        let dir = fresh_tmp_dir("cache-only-handoff");

        // Represent air replacing a previous nonempty slot as well as new empty records.
        RegionCache::default()
            .save_octree_section(
                &dir,
                dim,
                &mesh_fixture(kids[7]),
                &OctreeMapping::default(),
                true,
                DEFAULT_OCTREE_COMPRESSION,
            )
            .unwrap();
        let mut p = Pipeline::default();
        let store = p.stores.entry(dim.into()).or_default();
        for pos in positions.clone() {
            let section = if pos == parent || pos == terrain {
                mesh_fixture(pos)
            } else {
                OctreeSection::new(pos)
            };
            assert!(store.adopt_section(section, true));
        }
        p.unsaved.insert(dim.into(), positions.clone().collect());
        {
            let io = LodIoWorker::spawn();
            assert_eq!(p.flush_saves(&io, &dir, usize::MAX), 9);
            io.flush();
        }

        let mut reopened = Pipeline::default();
        let io = LodIoWorker::spawn();
        for pos in positions {
            io.load_octree_section(dir.clone(), dim.into(), pos);
        }
        io.flush();
        let loaded = io.drain_octree_responses();
        drop(io);
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(loaded.len(), 9);
        let overwritten = loaded.iter().find(|l| l.section.pos == kids[7]).unwrap();
        assert!(overwritten.complete && overwritten.section.is_empty());
        reopened.adopt_loaded_entries(loaded, |_| true, |_, section, _| section.clone());
        assert_eq!(reopened.dirty[dim].len(), 9);
        let sections = drain_fixture_meshes(&mut reopened, dim);
        assert_eq!(
            sections
                .iter()
                .filter(|s| s.is_empty() && s.complete)
                .count(),
            7
        );
        let nodes = test_node_tree(&sections, None);
        let node = nodes
            .iter()
            .find(|n| unpack_section_pos([n.pos0, n.pos1]) == (1, [0; 3]))
            .unwrap();
        assert_ne!(node.section_id, GpuNode::NONE);
        assert_eq!(node.child_mask, 1);
        let child = nodes[node.child_base as usize];
        assert_ne!(child.section_id, GpuNode::NONE);
        assert_eq!(unpack_section_pos([child.pos0, child.pos1]), (0, [0; 3]));
    }

    #[test]
    fn coarse_children_do_not_certify_stale_parent_bytes() {
        use crate::storage::{RegionCache, fresh_tmp_dir};

        let dim = "test:stale-parent";
        let bounds = (0, 192);
        let parent = OctreeSectionPos::new(2, 0, 0, 0);
        let upper_air = OctreeSectionPos::new(1, 0, 2, 0);
        let mut p = Pipeline::default();
        let store = p.stores.entry(dim.into()).or_default();
        assert!(crate::octree_ingest::ingest_chunk(
            store,
            [0, 0],
            &FixtureChunk {
                sections: 12,
                bottom_block: 1,
            },
            IngestSource::Distant,
            &mut |b| b,
            &mut |_| 0,
        ));
        propagate_and_queue(&mut p, dim);
        assert!(p.unsaved[dim].contains(&parent));
        assert!(p.queued[dim].contains(&parent));
        let earlier = p.stores[dim].section(parent).unwrap().clone();
        assert!(!earlier.is_empty());
        assert!(!p.stores[dim].is_section_complete(parent));

        let target = SectionPos::from_block(1, 0, 0);
        let store = p.stores.get_mut(dim).unwrap();
        store.ingest_coarse_columns(
            1,
            coarse_job_columns(target)
                .into_iter()
                .map(DistantSample::Surface),
            (0, 1),
            &mut |b| b,
            &mut |_| 0,
            &|_| 15,
        );
        assert_eq!(store.y_bounds(), Some(bounds));
        for child in parent.children().unwrap() {
            assert!(store.is_section_complete(child));
        }
        assert!(store.section(upper_air).is_none());
        assert!(store.is_section_complete(upper_air));
        assert!(!store.is_section_complete(parent));
        assert_eq!(store.section(parent).unwrap().revision, earlier.revision);
        propagate_and_queue(&mut p, dim);

        let dir = fresh_tmp_dir("stale-parent");
        let io = LodIoWorker::spawn();
        p.flush_saves(&io, &dir, usize::MAX);
        io.flush();
        drop(io);
        let mut cache = RegionCache::default();
        let (saved, _, complete) = cache.load_octree_section(&dir, dim, parent).unwrap();
        assert_eq!(saved.to_dense(), earlier.to_dense());
        assert!(!complete);
        let (air, _, complete) = cache.load_octree_section(&dir, dim, upper_air).unwrap();
        assert!(air.is_empty() && complete);
        drop(cache);
        std::fs::remove_dir_all(dir).unwrap();

        let sections = drain_fixture_meshes(&mut p, dim);
        let stale = sections
            .iter()
            .find(|s| s.level == 2 && s.pos == [0; 3])
            .unwrap();
        assert!(!stale.is_empty() && !stale.complete);
        let nodes = test_node_tree(&sections, Some(bounds));
        let node = fixture_node(&nodes, parent);
        assert_ne!(node.section_id, GpuNode::NONE);
        assert_ne!(node.child_base, GpuNode::NONE);
        assert_eq!(node.child_mask, 0x0f);
    }

    fn adopted_d1_leaf(dim: &str, target: SectionPos) -> Pipeline {
        let mut p = Pipeline::default();
        let store = p.stores.entry(dim.to_owned()).or_default();
        store.set_y_bounds((0, 64));
        for dz in 0..2 {
            for dx in 0..2 {
                let pos = OctreeSectionPos::new(1, target.x * 2 + dx, 0, target.z * 2 + dz);
                assert!(store.adopt_section(mesh_fixture(pos), true));
            }
        }
        assert_eq!(p.coarse_coverage(dim, target), [[u64::MAX; 16]; 4]);
        p
    }

    fn single_distant_target(target: SectionPos) -> crate::DesiredCache {
        crate::DesiredCache {
            center: [0; 2],
            config: DhConfig::default(),
            list: Arc::from([target]),
            disk_cursor: 0,
            octree_disk_cursor: 0,
            sample_cursor: 0,
        }
    }

    #[test]
    fn scheduler_requests_invalidated_legacy_cells_without_overwriting_adopted_octree() {
        use crate::{DistantTarget, gen_worker::GenOutcome, lod::SECTION_CELLS};

        let dim = "test:invalidated-legacy";
        let target = SectionPos::from_block(1, 384, 0);
        let mut p = adopted_d1_leaf(dim, target);
        let octree_pos = OctreeSectionPos::new(1, 6, 0, 0);
        let mut cache = single_distant_target(target);
        // The state left by invalidate_lod_section: no legacy row or sampled marker and
        // a rewound cursor, but the adopted octree leaf is still complete.
        let mut legacy = DimensionLodStore::default();
        let Some((pos, DistantTarget::Builtin(cells))) = crate::pick_distant_target(
            &mut cache,
            &mut legacy,
            None,
            |_| false,
            |pos| p.coarse_coverage(dim, pos),
        ) else {
            panic!("expected an invalidated legacy row to be sampled again");
        };
        assert_eq!(pos, target);
        assert_eq!(cells.len(), SECTION_CELLS);
        for (i, &cell) in cells.iter().enumerate() {
            assert_eq!(cell, (i % SECTION_SIZE, i / SECTION_SIZE));
        }

        let before = p.stores[dim].section(octree_pos).unwrap().to_dense();
        let columns = coarse_job_columns(pos);
        let samples = cells
            .iter()
            .map(|&(cx, cz)| DistantSample::Surface(columns[cx + cz * SECTION_SIZE]));
        let coverage = p.ingest_coarse_job(dim, pos, samples, (0, 1), |_, b| b);
        assert_eq!(coverage, DistantCoverage::Complete);
        assert_eq!(
            p.stores[dim].section(octree_pos).unwrap().to_dense(),
            before
        );
        assert_eq!(p.coarse_coverage(dim, target), [[u64::MAX; 16]; 4]);
        assert!(crate::record_distant_completion(
            &mut legacy,
            pos,
            GenOutcome::Complete,
            coverage,
            false,
        ));
        assert!(!legacy.sampled_distant_sections.contains(&pos));
    }

    #[test]
    fn scheduler_checks_only_the_legacy_row_for_a_builtin_d5_target() {
        use crate::lod::{LodColumn, SECTION_CELLS, SOURCE_DISTANT_SURFACE};
        use crate::{DistantTarget, gen_worker::GenOutcome};

        let dim = "test:invalidated-d5-legacy";
        let target = SectionPos::from_block(5, 6144, 0);
        let mut p = Pipeline::default();
        p.stores
            .entry(dim.into())
            .or_default()
            .set_y_bounds((0, 64));
        p.ingest_coarse_job(
            dim,
            target,
            coarse_job_columns(target)
                .into_iter()
                .map(DistantSample::Surface),
            (0, 1),
            |_, b| b,
        );
        assert_eq!(p.coarse_coverage(dim, target), [[u64::MAX; 16]; 4]);

        let mut legacy = DimensionLodStore::default();
        let pick = |legacy: &mut DimensionLodStore, p: &Pipeline| {
            crate::pick_distant_target(
                &mut single_distant_target(target),
                legacy,
                None,
                |_| false,
                |pos| p.coarse_coverage(dim, pos),
            )
        };
        // Invalidation leaves no legacy row or sampled marker, but keeps the octree masks.
        let Some((pos, DistantTarget::Builtin(cells))) = pick(&mut legacy, &p) else {
            panic!("expected an invalidated D5 legacy row to be sampled again");
        };
        assert_eq!(pos, target);
        assert_eq!(cells.len(), SECTION_CELLS);
        for (i, &cell) in cells.iter().enumerate() {
            assert_eq!(cell, (i % SECTION_SIZE, i / SECTION_SIZE));
        }

        let empty_octree = Pipeline::default();
        assert_eq!(empty_octree.coarse_coverage(dim, target), [[0; 16]; 4]);
        // A successful worker result is terminal at D5 even without octree proof.
        assert!(!crate::record_distant_completion(
            &mut legacy,
            target,
            GenOutcome::Complete,
            DistantCoverage::Incomplete,
            false,
        ));
        assert!(legacy.sampled_distant_sections.remove(&target));

        let row = legacy.get_or_create_mut(target);
        let column = LodColumn {
            solid_top_y: 12,
            solid_bottom_y: 0,
            source: SOURCE_DISTANT_SURFACE,
            ..LodColumn::EMPTY
        };
        for (cx, cz) in cells {
            assert!(row.set_cell(cx, cz, column));
        }
        assert!(pick(&mut legacy, &empty_octree).is_none());
        assert!(legacy.sampled_distant_sections.contains(&target));
    }

    #[test]
    fn scheduler_checks_only_the_legacy_row_for_an_eligible_custom_generator() {
        use crate::lod::{LodColumn, SOURCE_DISTANT_SURFACE};
        use crate::{
            DistantTarget,
            api::{LodGenOutput, LodGenRequest, LodWorldGenerator},
        };

        struct D1Generator;
        impl LodWorldGenerator for D1Generator {
            fn detail_range(&self) -> std::ops::RangeInclusive<u8> {
                1..=1
            }

            fn generate(&self, _: &LodGenRequest, _: &mut dyn FnMut(LodGenOutput)) {}
        }

        let dim = "test:custom-generator";
        let target = SectionPos::from_block(1, 384, 0);
        let p = adopted_d1_leaf(dim, target);
        let mut state = crate::DhState::default();
        state
            .world_generators
            .insert(dim.into(), Arc::new(D1Generator));
        let generator = &state.world_generators[dim];
        let custom = || Some((generator, generator.detail_range()));
        let mut cache = single_distant_target(target);
        let mut legacy = DimensionLodStore::default();
        let Some((pos, DistantTarget::Custom(selected))) = crate::pick_distant_target(
            &mut cache,
            &mut legacy,
            custom(),
            |_| false,
            |pos| p.coarse_coverage(dim, pos),
        ) else {
            panic!("octree coverage must not satisfy the registered generator");
        };
        assert_eq!(pos, target);
        assert!(Arc::ptr_eq(&selected, generator));

        let row = legacy.get_or_create_mut(target);
        let column = LodColumn {
            solid_top_y: 12,
            solid_bottom_y: 0,
            source: SOURCE_DISTANT_SURFACE,
            ..LodColumn::EMPTY
        };
        for cz in 0..SECTION_SIZE {
            for cx in 0..SECTION_SIZE {
                assert!(row.set_cell(cx, cz, column));
            }
        }
        let empty_octree = Pipeline::default();
        assert!(
            crate::pick_distant_target(
                &mut cache,
                &mut legacy,
                custom(),
                |_| false,
                |pos| empty_octree.coarse_coverage(dim, pos),
            )
            .is_none()
        );
    }

    #[test]
    fn scheduler_completes_mixed_ring_despite_partial_or_full_legacy_row() {
        use crate::gen_worker::GenOutcome;
        use crate::lod::{LodColumn, SOURCE_DISTANT_SURFACE};

        let dim = "test:mixed-ring";
        let bounds = (0, 64);
        let parent = OctreeSectionPos::new(3, 1, 0, 0);
        let [fine, coarse, ..] = parent.children().unwrap();
        let target = SectionPos::from_block(1, 384, 0);
        let cached_cell = target.local_cell(400, 0);

        for full_legacy_row in [false, true] {
            let mut state = crate::DhState::default();
            let cache = crate::ensure_desired_cache(
                &mut state,
                (dim.into(), 0),
                [0.0, 64.0, 0.0],
                DhConfig::default(),
            );
            let leaves = &cache.list;
            cache.sample_cursor = leaves.iter().position(|&pos| pos == target).unwrap();
            assert!(!leaves.contains(&SectionPos::from_block(1, 256, 0)));

            let mut legacy = DimensionLodStore::default();
            let row = legacy.get_or_create_mut(target);
            let legacy_voxel = LodColumn {
                solid_top_y: 12,
                solid_bottom_y: 0,
                source: SOURCE_DISTANT_SURFACE,
                ..LodColumn::EMPTY
            };
            for z in 0..SECTION_SIZE {
                for x in 0..SECTION_SIZE {
                    if full_legacy_row || (x, z) == cached_cell {
                        assert!(row.set_cell(x, z, legacy_voxel));
                    }
                }
            }
            assert_eq!(row.get(cached_cell.0, cached_cell.1), legacy_voxel);
            assert_eq!(
                row.populated as usize,
                if full_legacy_row { 4096 } else { 1 }
            );

            let mut p = Pipeline::default();
            let store = p.stores.entry(dim.into()).or_default();
            assert!(store.adopt_section(mesh_fixture(parent), true));
            for sibling in &parent.children().unwrap()[2..4] {
                assert!(store.adopt_section(OctreeSection::new(*sibling), true));
            }
            // Populate the fine leaves chosen by the scheduler, not D1 stand-ins. Only one
            // contains terrain; the air leaves still prove their part of the branch.
            for x in [256, 320] {
                for z in [0, 64] {
                    let leaf = SectionPos::from_block(0, x, z);
                    assert!(leaves.contains(&leaf));
                    for dx in 0..2 {
                        for dz in 0..2 {
                            for sy in 0..2 {
                                let pos = OctreeSectionPos::new(0, x / 32 + dx, sy, z / 32 + dz);
                                let section = if (x, z, dx, dz, sy) == (256, 0, 0, 0, 0) {
                                    mesh_fixture(pos)
                                } else {
                                    OctreeSection::new(pos)
                                };
                                assert!(store.adopt_section(section, true));
                            }
                        }
                    }
                }
            }
            assert!(crate::octree_ingest::ingest_chunk_snapshot(
                store,
                [24, 0],
                &FixtureChunk {
                    sections: 4,
                    bottom_block: 2
                },
                &mut |b| b,
                &mut |_| 0,
            ));
            assert_eq!(store.y_bounds(), Some(bounds));
            propagate_and_queue(&mut p, dim);
            let overlap = coarse.children().unwrap()[0];
            let at_live = |store: &OctreeIngestStore| store.section(overlap).unwrap().get(0, 0, 0);
            let live_voxel = at_live(&p.stores[dim]);
            assert_eq!(live_voxel.block_id(), 2);
            p.requeue_dimension(dim);
            let nodes = test_node_tree(&drain_fixture_meshes(&mut p, dim), Some(bounds));
            let node = fixture_node(&nodes, parent);
            assert_ne!(node.section_id, GpuNode::NONE);
            assert_eq!((node.child_base, node.child_mask), (GpuNode::NONE, 0));

            let (pos, crate::DistantTarget::Builtin(cells)) = crate::pick_distant_target(
                cache,
                &mut legacy,
                None,
                |_| false,
                |pos| p.coarse_coverage(dim, pos),
            )
            .unwrap() else {
                panic!("expected a built-in job");
            };
            assert_eq!(pos, target);
            assert_eq!(cells.len(), SECTION_SIZE * SECTION_SIZE);
            assert!(cells.contains(&cached_cell));

            // Deliver the whole scheduled job normally. Dev 2 can return None even for
            // cells intersecting loaded terrain; the octree must resolve them from D0.
            let columns = coarse_job_columns(pos);
            let samples = cells.into_iter().map(|(cx, cz)| {
                let col = columns[cx + cz * SECTION_SIZE];
                if [col.x >> 4, col.z >> 4] == [24, 0] {
                    DistantSample::NoSurface(col.x, col.z)
                } else {
                    DistantSample::Surface(col)
                }
            });
            let coverage = p.ingest_coarse_job(dim, pos, samples, (0, 1), |_, b| b);
            assert_eq!(coverage, DistantCoverage::Complete);
            assert!(!crate::record_distant_completion(
                &mut legacy,
                pos,
                GenOutcome::Complete,
                coverage,
                true,
            ));
            assert!(legacy.sampled_distant_sections.contains(&pos));

            let store = &p.stores[dim];
            assert_eq!(store.voxel_at(384, 0, 0).block_id(), 2, "authoritative D0");
            assert_eq!(at_live(store), live_voxel, "authoritative D1 reduction");
            assert_eq!(
                store.section(overlap).unwrap().get(8, 3, 0).block_id(),
                1,
                "legacy cell dispatched"
            );
            assert_eq!(p.coarse_coverage(dim, pos), [[u64::MAX; 16]; 4]);
            for child in &coarse.children().unwrap()[..4] {
                assert!(
                    store.is_section_complete(*child),
                    "all 1024 columns at {child:?}"
                );
            }

            p.requeue_dimension(dim);
            let nodes = test_node_tree(&drain_fixture_meshes(&mut p, dim), Some(bounds));
            let node = fixture_node(&nodes, parent);
            assert_ne!(node.section_id, GpuNode::NONE);
            assert_ne!(node.child_base, GpuNode::NONE);
            assert_eq!(node.child_mask, 0b11);
            let [fine_branch, coarse_branch] = &nodes[node.child_base as usize..][..2] else {
                unreachable!()
            };
            assert_eq!(fine_branch, fixture_node(&nodes, fine));
            assert_eq!(fine_branch.section_id, GpuNode::NONE);
            assert_ne!(fine_branch.child_base, GpuNode::NONE);
            assert_ne!(
                nodes[fine_branch.child_base as usize].child_base,
                GpuNode::NONE
            );
            assert_eq!(coarse_branch, fixture_node(&nodes, coarse));
            assert_ne!(coarse_branch.child_base, GpuNode::NONE);
            assert_ne!(fixture_node(&nodes, overlap).section_id, GpuNode::NONE);
        }
    }

    #[test]
    fn incomplete_coarse_jobs_retry_missing_cells_but_no_surface_is_terminal() {
        use crate::gen_worker::GenOutcome;

        let dim = "test:no-surface";
        let target = SectionPos::from_block(1, 384, 0);
        let overlap = OctreeSectionPos::new(1, 6, 0, 0);
        let cached = Voxel::new(2, 0, 15, 0);
        let mut p = Pipeline::default();
        let store = p.stores.entry(dim.into()).or_default();
        store.set_y_bounds((0, 64));
        assert!(store.adopt_section(OctreeSection::filled(overlap, cached), false));

        let mut state = crate::DhState::default();
        let cache = crate::ensure_desired_cache(
            &mut state,
            (dim.into(), 0),
            [0.0, 64.0, 0.0],
            DhConfig::default(),
        );
        cache.sample_cursor = cache.list.iter().position(|&pos| pos == target).unwrap();
        let mut legacy = DimensionLodStore::default();
        let pick =
            |cache: &mut crate::DesiredCache, legacy: &mut DimensionLodStore, p: &Pipeline| {
                crate::pick_distant_target(
                    cache,
                    legacy,
                    None,
                    |_| false,
                    |pos| p.coarse_coverage(dim, pos),
                )
            };

        let (pos, crate::DistantTarget::Builtin(cells)) = pick(cache, &mut legacy, &p).unwrap()
        else {
            panic!("expected a built-in job");
        };
        assert_eq!(pos, target);
        let columns = coarse_job_columns(pos);
        let (x, z) = cells[0];
        let first = columns[x + z * SECTION_SIZE];
        let coverage =
            p.ingest_coarse_job(dim, pos, [DistantSample::Surface(first)], (0, 1), |_, b| b);
        assert_eq!(coverage, DistantCoverage::Incomplete);
        assert!(!crate::record_distant_completion(
            &mut legacy,
            pos,
            GenOutcome::Cancelled,
            coverage,
            false,
        ));
        assert!(crate::record_distant_completion(
            &mut legacy,
            pos,
            GenOutcome::Error,
            coverage,
            false,
        ));
        assert!(crate::record_distant_completion(
            &mut legacy,
            pos,
            GenOutcome::Complete,
            coverage,
            true,
        ));
        assert!(!legacy.sampled_distant_sections.contains(&pos));

        // As in the worker result adapter, the first surface sample fills both stores.
        legacy.get_or_create_mut(pos).set_cell(
            x,
            z,
            crate::lod::LodColumn {
                solid_top_y: first.height as i16,
                solid_bottom_y: 0,
                source: crate::lod::SOURCE_DISTANT_SURFACE,
                ..crate::lod::LodColumn::EMPTY
            },
        );
        let (retry, crate::DistantTarget::Builtin(remaining)) =
            pick(cache, &mut legacy, &p).unwrap()
        else {
            panic!("expected a built-in retry");
        };
        assert_eq!(retry, target);
        assert_eq!(remaining.as_slice(), &cells[1..]);
        let samples = remaining.into_iter().map(|(cx, cz)| {
            let col = columns[cx + cz * SECTION_SIZE];
            if (col.x, col.z) == (400, 0) {
                DistantSample::NoSurface(col.x, col.z)
            } else {
                DistantSample::Surface(col)
            }
        });
        let coverage = p.ingest_coarse_job(dim, retry, samples, (0, 1), |_, b| b);
        assert_eq!(coverage, DistantCoverage::Unavailable);
        assert!(!crate::record_distant_completion(
            &mut legacy,
            retry,
            GenOutcome::Complete,
            coverage,
            true,
        ));
        let store = &p.stores[dim];
        assert_eq!(store.section(overlap).unwrap().get(8, 3, 0), cached);
        assert!(!store.is_section_complete(overlap));
        assert_ne!(
            pick(cache, &mut legacy, &p).map(|(pos, _)| pos),
            Some(target)
        );
    }

    #[test]
    fn clear_all_retires_all_world_maps_and_keeps_process_state() {
        let mut p = Pipeline {
            next_epoch: 7,
            baked: 2,
            models_dirty: true,
            ..Default::default()
        };
        let pos = OctreeSectionPos::new(0, 1, 2, 3);
        for dim in ["test:a", "test:b"] {
            p.stores.insert(dim.into(), OctreeIngestStore::default());
            p.queue(dim, [pos]);
            p.section_epoch.insert((dim.into(), pos), 7);
            p.meshed.insert(dim.into(), HashSet::from([pos]));
            p.unsaved.insert(dim.into(), HashSet::from([pos]));
            p.requested.insert(dim.into(), HashSet::from([pos]));
        }
        p.clear_all();
        assert!(p.stores.is_empty());
        assert!(p.dirty.is_empty());
        assert!(p.queued.is_empty());
        assert!(p.section_epoch.is_empty());
        assert!(p.meshed.is_empty());
        assert!(p.unsaved.is_empty());
        assert!(p.requested.is_empty());
        assert_eq!((p.next_epoch, p.baked, p.models_dirty), (7, 2, true));
    }

    #[test]
    fn adopt_loaded_entries_drops_unloaded_dimension_without_recreating_store() {
        let (unloaded, loaded) = ("test:unloaded", "test:loaded");
        let pos = OctreeSectionPos::new(0, 1, 2, 3);
        let entries = [unloaded, loaded].map(|dim| OctreeLoaded {
            dimension: dim.into(),
            section: OctreeSection::new(pos),
            mapping: OctreeMapping::default(),
            complete: dim == unloaded,
        });
        let mut p = Pipeline::default();
        let mut remaps = 0;
        p.adopt_loaded_entries(
            entries.into(),
            |dim| dim == loaded,
            |_, section, _| {
                remaps += 1;
                section.clone()
            },
        );
        assert_eq!(remaps, 1);
        assert!(!p.stores.contains_key(unloaded));
        assert!(!p.dirty.contains_key(unloaded));
        assert!(!p.queued.contains_key(unloaded));
        assert!(p.stores[loaded].section(pos).is_some());
        assert!(!p.stores[loaded].is_section_complete(pos));
        assert_eq!(p.dirty[loaded], VecDeque::from([pos]));
        assert_eq!(p.queued[loaded], HashSet::from([pos]));
    }

    #[test]
    fn forget_gpu_residency_and_requeue_dimension_preserve_store_and_requeue() {
        let (overworld, nether) = ("minecraft:overworld", "minecraft:the_nether");
        let parent = OctreeSectionPos::new(1, 0, 2, 0);
        let child = parent.children().unwrap()[0];
        let nether_pos = OctreeSectionPos::new(0, 1, 2, 3);
        let missing = OctreeSectionPos::new(0, 6, 7, 8);
        let voxel = Voxel::new(1, 0, 15, 0);
        let mut p = Pipeline {
            next_epoch: 10,
            ..Default::default()
        };
        for (dim, positions, live, epoch) in [
            (overworld, &[child, parent][..], child, 9),
            (nether, &[nether_pos][..], nether_pos, 10),
        ] {
            let store = p.stores.entry(dim.to_owned()).or_default();
            for &pos in positions {
                store.adopt_section(OctreeSection::filled(pos, voxel), true);
            }
            p.queue(dim, positions.iter().copied());
            p.unsaved
                .insert(dim.to_owned(), positions.iter().copied().collect());
            p.requested.insert(dim.to_owned(), HashSet::from([missing]));
            p.meshed.insert(dim.to_owned(), HashSet::from([live]));
            p.section_epoch.insert((dim.to_owned(), live), epoch);
        }

        p.forget_gpu_residency(overworld);

        assert!(!p.dirty.contains_key(overworld));
        assert!(!p.queued.contains_key(overworld));
        assert!(!p.meshed.contains_key(overworld));
        assert_eq!(
            p.section_epoch,
            HashMap::from([((nether.to_owned(), nether_pos), 10)])
        );
        assert_eq!(p.next_epoch, 10);
        for (dim, expected) in [
            (overworld, HashSet::from([child, parent])),
            (nether, HashSet::from([nether_pos])),
        ] {
            let store = &p.stores[dim];
            assert_eq!(
                store
                    .section_positions()
                    .into_iter()
                    .collect::<HashSet<_>>(),
                expected
            );
            for &pos in &expected {
                assert_eq!(store.section(pos).unwrap().get(0, 0, 0), voxel);
            }
            assert_eq!(p.unsaved[dim], expected);
            assert_eq!(p.requested[dim], HashSet::from([missing]));
        }
        assert_eq!(p.dirty[nether], VecDeque::from([nether_pos]));
        assert_eq!(p.queued[nether], HashSet::from([nether_pos]));
        assert_eq!(p.meshed[nether], HashSet::from([nether_pos]));

        assert_eq!(p.requeue_dimension(overworld), 2);
        assert_eq!(p.dirty[overworld].len(), 2);
        assert_eq!(
            p.dirty[overworld].iter().copied().collect::<HashSet<_>>(),
            HashSet::from([child, parent])
        );
        assert_eq!(p.queued[overworld], HashSet::from([child, parent]));
        assert!(!p.meshed.contains_key(overworld));
        assert_eq!(p.requeue_dimension("minecraft:the_end"), 0);
    }

    fn nonempty_mesh_done(dim: &str, pos: OctreeSectionPos, epoch: u64) -> MeshDone {
        let mut built = empty_section(pos, false);
        built.quads.push(crate::gpu_arena::PackedQuad::default());
        built.counts[0] = 1;
        (dim.to_owned(), pos, epoch, built)
    }

    #[test]
    fn apply_mesh_completions_keeps_parent_and_child_without_requeue() {
        let overworld = "minecraft:overworld";
        let parent = OctreeSectionPos::new(1, 0, 2, 0);
        let child = parent.children().unwrap()[0];
        let key = |pos: OctreeSectionPos| (overworld.to_owned(), pos.detail, [pos.x, pos.y, pos.z]);
        let voxel = Voxel::new(1, 0, 15, 0);

        for separate_ticks in [true, false] {
            let mut p = Pipeline {
                next_epoch: 2,
                section_epoch: HashMap::from([
                    ((overworld.to_owned(), child), 1),
                    ((overworld.to_owned(), parent), 2),
                ]),
                ..Pipeline::default()
            };
            let store = p.stores.entry(overworld.to_owned()).or_default();
            for pos in [child, parent] {
                assert!(store.adopt_section(OctreeSection::filled(pos, voxel), true));
            }
            let child_done = nonempty_mesh_done(overworld, child, 1);
            let parent_done = nonempty_mesh_done(overworld, parent, 2);
            let mut submitted = Vec::new();
            if separate_ticks {
                p.apply_mesh_completions(Some(overworld), vec![child_done], |key, _| {
                    submitted.push(key)
                });
                assert_eq!(submitted, [key(child)]);
                assert_eq!(p.meshed[overworld], HashSet::from([child]));
                assert!(p.dirty.is_empty());
                assert!(p.queued.is_empty());

                p.apply_mesh_completions(Some(overworld), vec![parent_done], |key, _| {
                    submitted.push(key)
                });
            } else {
                p.apply_mesh_completions(
                    Some(overworld),
                    vec![child_done, parent_done],
                    |key, _| submitted.push(key),
                );
            }
            assert_eq!(submitted, [key(child), key(parent)]);
            assert_eq!(p.meshed[overworld], HashSet::from([child, parent]));
            assert!(p.dirty.is_empty());
            assert!(p.queued.is_empty());
        }
    }

    #[test]
    fn apply_mesh_completions_drops_inactive_dimension_and_no_active_level() {
        let (overworld, nether, end) = (
            "minecraft:overworld",
            "minecraft:the_nether",
            "minecraft:the_end",
        );
        let [pos_ow, pos_ne, pos_end, pos_stale] =
            [1, 2, 3, 4].map(|x| OctreeSectionPos::new(0, x, 0, 0));
        let mut p = Pipeline {
            section_epoch: HashMap::from([
                ((overworld.to_owned(), pos_ow), 1),
                ((nether.to_owned(), pos_ne), 2),
                ((end.to_owned(), pos_end), 3),
                ((nether.to_owned(), pos_stale), 9),
            ]),
            ..Default::default()
        };
        let mut submitted = Vec::new();
        p.apply_mesh_completions(
            Some(nether),
            vec![
                nonempty_mesh_done(overworld, pos_ow, 1),
                nonempty_mesh_done(nether, pos_ne, 2),
                nonempty_mesh_done(nether, pos_stale, 8),
            ],
            |key, _| submitted.push(key),
        );
        assert_eq!(
            submitted,
            [(
                nether.to_owned(),
                pos_ne.detail,
                [pos_ne.x, pos_ne.y, pos_ne.z]
            )]
        );
        let meshed = HashMap::from([(nether.to_owned(), HashSet::from([pos_ne]))]);
        assert_eq!(p.meshed, meshed);
        assert!(
            !p.section_epoch
                .contains_key(&(overworld.to_owned(), pos_ow))
        );
        assert_eq!(p.section_epoch.get(&(nether.to_owned(), pos_ne)), Some(&2));
        assert_eq!(
            p.section_epoch.get(&(nether.to_owned(), pos_stale)),
            Some(&9)
        );

        p.apply_mesh_completions(None, vec![nonempty_mesh_done(end, pos_end, 3)], |key, _| {
            submitted.push(key)
        });
        assert_eq!(submitted.len(), 1);
        assert_eq!(p.meshed, meshed);
        assert!(!p.section_epoch.contains_key(&(end.to_owned(), pos_end)));
    }

    #[test]
    fn arena_reset_clears_meshed_and_epochs_before_requeuing() {
        let dim = "minecraft:overworld";
        let parent_pos = OctreeSectionPos::new(1, 0, 2, 0);
        let child_pos = parent_pos.children().unwrap()[0];
        let mut store = OctreeIngestStore::new();
        for pos in [child_pos, parent_pos] {
            let mut section = OctreeSection::new(pos);
            section.set(0, 0, 0, Voxel::new(1, 0, 15, 0));
            store.adopt_section(section, true);
        }
        let mut p = Pipeline::default();
        p.stores.insert(dim.to_owned(), store);
        p.meshed
            .entry(dim.to_owned())
            .or_default()
            .insert(child_pos);
        p.section_epoch.insert((dim.to_owned(), parent_pos), 7);
        p.dirty
            .entry(dim.to_owned())
            .or_default()
            .push_back(child_pos);

        assert_eq!(p.reset_arena_state(), 2);

        assert!(p.meshed.is_empty());
        assert!(p.section_epoch.is_empty());
        assert!(p.models_dirty);
        let queued = &p.queued[dim];
        assert!(queued.contains(&child_pos) && queued.contains(&parent_pos));
        assert_eq!(p.dirty[dim].len(), 2);
    }
}
