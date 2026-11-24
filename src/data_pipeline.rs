//! Stage 2b off-thread data pipeline wired to Sunrise's G1–G4 general mod API primitives:
//!
//! - **G1 (`#5233` / PR `#5339`)**: `ChunkRef::snapshot(host)` -> zero-copy `ChunkSnapshot`
//!   (`Send + Sync`), full-height multi-layer extraction with 3D quart biomes and real sky/block
//!   light on **both client and server** (including multiplayer client-side LOD capture).
//! - **G2 (`#5234` / PR `#5340`)**: `LevelRef::request_chunk` / `drain_chunk_requests` /
//!   `ChunkRequestRef::cancel` non-blocking full-chunk load/generation queue.
//! - **G3 (`#5235` / PR `#5341`)**: `LevelRef::saved_chunk_reader` -> thread-safe
//!   `SavedChunkReader` (`region_chunks`, `read`) background import of previously explored Anvil
//!   `.mca` regions without loading chunks into `World`.
//! - **G4 (`#5236` / PR `#5347`)**: `LevelRef::detached_generator` -> thread-safe
//!   `DetachedChunkGenerator` (`generate_chunk(pos, GenerationStage)`, `sample_surface_columns`)
//!   for off-thread full distant terrain/carvers/features/light generation across Overworld,
//!   Nether, End, and custom dimensions.
//!
//! Everything here is built on a pure `ChunkSnapshotLike` trait abstraction so:
//! 1. All extraction, worker-thread batching, region-import scanning, and detached generation
//!    scheduling compiles and passes unit tests immediately on the `f8aa80e` SDK.
//! 2. Enabling `--features sunrise-gaps-g1-g4` (once `#5339`–`#5341` + `#5347` hit a Sunrise Dev
//!    SDK) flips on the direct `mod_api::ChunkSnapshot`, `mod_api::SavedChunkReader`,
//!    `mod_api::DetachedChunkGenerator`, and `LevelRef::request_chunk` adapters with zero
//!    rewriting of the DH pipeline.

use crate::colors::{biome_water_color, block_color, is_skippable_for_lod, is_water_block};
use crate::config::DhConfig;
use crate::lod::{
    ColumnSample, ExtraSpan, LodColumn, SOURCE_DISTANT_SURFACE, SectionPos, extract_column_spans,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

/// Provenance priority for chunks imported from saved Anvil regions (G3): higher than distant
/// surface estimates, lower than freshly live-loaded chunks.
pub const SOURCE_SAVED_REGION: u8 = 2;

/// Target worldgen depth for G4 detached generation, mirroring `mod_api::GenerationStage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DistantGenStage {
    Surface,
    Carvers,
    Features,
    Light,
}

impl DistantGenStage {
    pub fn from_config_mode(mode: u32) -> Self {
        match mode {
            0 => Self::Surface,
            1 => Self::Carvers,
            2 => Self::Features,
            _ => Self::Light,
        }
    }
}

/// Abstract read-only view over a 16x16 full-height chunk snapshot (mirrors `mod_api::ChunkSnapshot`).
///
/// Implemented directly for `mod_api::ChunkSnapshot` under `--features sunrise-gaps-g1-g4` and for
/// synthetic test fixtures in unit tests.
pub trait SnapshotSource: Send + Sync + 'static {
    fn pos(&self) -> [i32; 2];
    fn min_y(&self) -> i32;
    fn height(&self) -> i32;
    fn section_count(&self) -> usize;
    fn has_light(&self) -> bool;
    fn section_is_empty(&self, section_index: usize) -> bool;
    fn section_y(&self, section_index: usize) -> i32;
    fn block_id_at(&self, lx: u8, y: i32, lz: u8) -> Option<&str>;
    fn light_at(&self, lx: u8, y: i32, lz: u8) -> (u8, u8);
    fn biome_id_at(&self, lx: u8, y: i32, lz: u8) -> &str;
}

/// Extracted 16x16 LOD chunk columns ready to merge into `DimensionLodStore` at `detail = 0`.
#[derive(Clone, Debug)]
pub struct ExtractedChunkColumns {
    pub dimension: String,
    pub chunk_pos: [i32; 2],
    pub columns: Vec<(LodColumn, Vec<ExtraSpan>)>,
    pub affected_sections: Vec<SectionPos>,
}

/// Extract all 256 multi-layer LOD columns from a full-height immutable chunk snapshot off-thread.
///
/// Uses section-level `section_is_empty()` skipping so air-only vertical sections cost O(1),
/// reads real sky/block light when `snapshot.has_light()` (falling back to open-sky daylight for
/// pre-light detached stages), and samples 3D quart biomes per layer.
pub fn extract_chunk_snapshot<S: SnapshotSource>(
    dimension: &str,
    snapshot: &S,
    cfg: DhConfig,
    source_priority: u8,
) -> ExtractedChunkColumns {
    let [cx, cz] = snapshot.pos();
    let bx0 = cx << 4;
    let bz0 = cz << 4;
    let min_y = snapshot.min_y();
    let max_y = min_y + snapshot.height();
    let has_light = snapshot.has_light();
    let has_ceiling = dimension == "minecraft:the_nether";
    let max_extra = cfg.max_extra_spans();

    // Pre-collect non-empty section Y ranges from top down so air sections are skipped in O(1).
    let mut non_empty_ranges: Vec<(i32, i32)> = Vec::with_capacity(snapshot.section_count());
    for si in (0..snapshot.section_count()).rev() {
        if !snapshot.section_is_empty(si) {
            let sy0 = (snapshot.section_y(si) << 4).max(min_y);
            let sy1 = (sy0 + 16).min(max_y);
            if sy1 > sy0 {
                non_empty_ranges.push((sy0, sy1));
            }
        }
    }

    let mut columns = Vec::with_capacity(256);
    let mut samples: Vec<ColumnSample> = Vec::with_capacity(64);

    for lz in 0u8..16 {
        for lx in 0u8..16 {
            samples.clear();
            let mut seen_solid_above = false;

            for &(sy0, sy1) in &non_empty_ranges {
                for y in (sy0..sy1).rev() {
                    let Some(block_id) = snapshot.block_id_at(lx, y, lz) else {
                        continue;
                    };
                    if is_skippable_for_lod(block_id) {
                        continue;
                    }
                    let is_fluid = is_water_block(block_id);
                    let biome_id = snapshot.biome_id_at(lx, y, lz);
                    let rgb = if is_fluid {
                        biome_water_color(biome_id, 0)
                    } else {
                        block_color(block_id, biome_id, 0)
                    };
                    let (sky, block) = if has_light {
                        // Sample light from the block directly above the top face when in bounds
                        // (matches vanilla surface illumination; solid blocks themselves hold 0).
                        if y + 1 < max_y {
                            snapshot.light_at(lx, y + 1, lz)
                        } else {
                            (15, 0)
                        }
                    } else if has_ceiling {
                        (0, 6)
                    } else if !seen_solid_above {
                        (15, 0)
                    } else {
                        (4, 0)
                    };
                    if !is_fluid {
                        seen_solid_above = true;
                    }
                    samples.push(ColumnSample {
                        y,
                        rgb,
                        sky_light: sky,
                        block_light: block,
                        is_fluid,
                    });
                }
            }

            let extracted = extract_column_spans(
                &samples,
                source_priority,
                max_extra,
                cfg.cave_culling_enabled,
                cfg.cave_culling_height,
                has_ceiling,
            );
            columns.push(extracted);
        }
    }

    // One 16x16 chunk falls inside a single 64x64 detail-0 section (16 divides 64).
    let sec_pos = SectionPos::from_block(0, bx0, bz0);
    ExtractedChunkColumns {
        dimension: dimension.to_owned(),
        chunk_pos: [cx, cz],
        columns,
        affected_sections: vec![sec_pos],
    }
}

/// Abstract thread-safe saved Anvil chunk reader (mirrors G3 `mod_api::SavedChunkReader`).
pub trait SavedRegionSource: Send + Sync + 'static {
    type Snapshot: SnapshotSource;
    fn region_chunks(&self, rx: i32, rz: i32) -> Result<Vec<[i32; 2]>, String>;
    fn read(&self, pos: [i32; 2]) -> Result<Option<Self::Snapshot>, String>;
}

/// Abstract thread-safe detached world generator (mirrors G4 `mod_api::DetachedChunkGenerator`).
pub trait DetachedGeneratorSource: Send + Sync + 'static {
    type Snapshot: SnapshotSource;
    fn generate_chunk(
        &self,
        pos: [i32; 2],
        stage: DistantGenStage,
    ) -> Result<Self::Snapshot, String>;
}

enum DataJob {
    ExtractSnapshot {
        dimension: String,
        snapshot: Box<dyn SnapshotSource>,
        cfg: DhConfig,
        source_priority: u8,
    },
    ImportSavedRegion {
        run: Box<dyn FnOnce() -> Vec<ExtractedChunkColumns> + Send + 'static>,
    },
    DetachedGenerate {
        run: Box<dyn FnOnce() -> Option<ExtractedChunkColumns> + Send + 'static>,
    },
}

/// Background multi-purpose worker pool for G1 snapshot extraction, G3 saved region import, and
/// G4 detached worldgen. Keeps all heavy CPU work completely off the client/server tick threads.
pub struct DataPipelineWorker {
    tx: Sender<DataJob>,
    rx: Mutex<Receiver<ExtractedChunkColumns>>,
    imported_regions: Mutex<HashMap<String, HashSet<[i32; 2]>>>,
    inflight_detached: Mutex<HashMap<String, HashSet<[i32; 2]>>>,
}

impl DataPipelineWorker {
    pub fn spawn() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<DataJob>();
        let (res_tx, res_rx) = mpsc::channel::<ExtractedChunkColumns>();
        let shared_rx = Arc::new(Mutex::new(job_rx));

        let threads = thread::available_parallelism()
            .map(|n| (n.get() / 2).clamp(1, 4))
            .unwrap_or(2);

        for idx in 0..threads {
            let job_rx = Arc::clone(&shared_rx);
            let res_tx = res_tx.clone();
            let _ = thread::Builder::new()
                .name(format!("dh-data-worker-{idx}"))
                .spawn(move || {
                    loop {
                        let job = {
                            let guard = job_rx.lock().unwrap();
                            guard.recv()
                        };
                        let Ok(job) = job else {
                            break;
                        };
                        match job {
                            DataJob::ExtractSnapshot {
                                dimension,
                                snapshot,
                                cfg,
                                source_priority,
                            } => {
                                let extracted = extract_dyn_snapshot(
                                    &dimension,
                                    snapshot.as_ref(),
                                    cfg,
                                    source_priority,
                                );
                                let _ = res_tx.send(extracted);
                            }
                            DataJob::ImportSavedRegion { run } => {
                                for extracted in run() {
                                    let _ = res_tx.send(extracted);
                                }
                            }
                            DataJob::DetachedGenerate { run } => {
                                if let Some(extracted) = run() {
                                    let _ = res_tx.send(extracted);
                                }
                            }
                        }
                    }
                });
        }

        Self {
            tx: job_tx,
            rx: Mutex::new(res_rx),
            imported_regions: Mutex::new(HashMap::new()),
            inflight_detached: Mutex::new(HashMap::new()),
        }
    }

    /// Enqueue a G1 zero-copy chunk snapshot for background multi-layer column extraction.
    pub fn submit_snapshot<S: SnapshotSource>(
        &self,
        dimension: String,
        snapshot: S,
        cfg: DhConfig,
        source_priority: u8,
    ) {
        let _ = self.tx.send(DataJob::ExtractSnapshot {
            dimension,
            snapshot: Box::new(snapshot),
            cfg,
            source_priority,
        });
    }

    /// Schedule one saved Anvil region `(rx, rz)` for G3 background import if not already imported.
    pub fn schedule_saved_region<R: SavedRegionSource>(
        &self,
        dimension: String,
        region: [i32; 2],
        reader: Arc<R>,
        cfg: DhConfig,
    ) -> bool {
        {
            let mut imported = self.imported_regions.lock().unwrap();
            let set = imported.entry(dimension.clone()).or_default();
            if !set.insert(region) {
                return false;
            }
        }
        let dim_for_job = dimension.clone();
        let run = Box::new(move || {
            let Ok(chunks) = reader.region_chunks(region[0], region[1]) else {
                return Vec::new();
            };
            let mut out = Vec::with_capacity(chunks.len());
            for pos in chunks {
                if let Ok(Some(snap)) = reader.read(pos) {
                    out.push(extract_chunk_snapshot(
                        &dim_for_job,
                        &snap,
                        cfg,
                        SOURCE_SAVED_REGION,
                    ));
                }
            }
            out
        });
        let _ = self.tx.send(DataJob::ImportSavedRegion { run });
        true
    }

    /// Schedule one distant chunk for G4 off-thread detached world generation.
    pub fn schedule_detached_chunk<G: DetachedGeneratorSource>(
        &self,
        dimension: String,
        chunk_pos: [i32; 2],
        stage: DistantGenStage,
        generator: Arc<G>,
        cfg: DhConfig,
    ) -> bool {
        {
            let mut inflight = self.inflight_detached.lock().unwrap();
            let set = inflight.entry(dimension.clone()).or_default();
            if !set.insert(chunk_pos) {
                return false;
            }
        }
        let dim_for_job = dimension.clone();
        let run = Box::new(move || {
            let snap = generator.generate_chunk(chunk_pos, stage).ok()?;
            Some(extract_chunk_snapshot(
                &dim_for_job,
                &snap,
                cfg,
                SOURCE_DISTANT_SURFACE,
            ))
        });
        let _ = self.tx.send(DataJob::DetachedGenerate { run });
        true
    }

    /// Drain extracted chunk batches completed by background workers.
    pub fn drain_finished(&self) -> Vec<ExtractedChunkColumns> {
        let mut out = Vec::new();
        let guard = self.rx.lock().unwrap();
        while let Ok(item) = guard.try_recv() {
            out.push(item);
        }
        drop(guard);
        if !out.is_empty() {
            let mut inflight = self.inflight_detached.lock().unwrap();
            for item in &out {
                if let Some(set) = inflight.get_mut(&item.dimension) {
                    set.remove(&item.chunk_pos);
                }
            }
        }
        out
    }

    pub fn clear_dimension(&self, dimension: &str) {
        self.imported_regions.lock().unwrap().remove(dimension);
        self.inflight_detached.lock().unwrap().remove(dimension);
    }
}

fn extract_dyn_snapshot(
    dimension: &str,
    snapshot: &dyn SnapshotSource,
    cfg: DhConfig,
    source_priority: u8,
) -> ExtractedChunkColumns {
    let [cx, cz] = snapshot.pos();
    let bx0 = cx << 4;
    let bz0 = cz << 4;
    let min_y = snapshot.min_y();
    let max_y = min_y + snapshot.height();
    let has_light = snapshot.has_light();
    let has_ceiling = dimension == "minecraft:the_nether";
    let max_extra = cfg.max_extra_spans();

    let mut non_empty_ranges: Vec<(i32, i32)> = Vec::with_capacity(snapshot.section_count());
    for si in (0..snapshot.section_count()).rev() {
        if !snapshot.section_is_empty(si) {
            let sy0 = (snapshot.section_y(si) << 4).max(min_y);
            let sy1 = (sy0 + 16).min(max_y);
            if sy1 > sy0 {
                non_empty_ranges.push((sy0, sy1));
            }
        }
    }

    let mut columns = Vec::with_capacity(256);
    let mut samples: Vec<ColumnSample> = Vec::with_capacity(64);
    for lz in 0u8..16 {
        for lx in 0u8..16 {
            samples.clear();
            let mut seen_solid_above = false;
            for &(sy0, sy1) in &non_empty_ranges {
                for y in (sy0..sy1).rev() {
                    let Some(block_id) = snapshot.block_id_at(lx, y, lz) else {
                        continue;
                    };
                    if is_skippable_for_lod(block_id) {
                        continue;
                    }
                    let is_fluid = is_water_block(block_id);
                    let biome_id = snapshot.biome_id_at(lx, y, lz);
                    let rgb = if is_fluid {
                        biome_water_color(biome_id, 0)
                    } else {
                        block_color(block_id, biome_id, 0)
                    };
                    let (sky, block) = if has_light {
                        if y + 1 < max_y {
                            snapshot.light_at(lx, y + 1, lz)
                        } else {
                            (15, 0)
                        }
                    } else if has_ceiling {
                        (0, 6)
                    } else if !seen_solid_above {
                        (15, 0)
                    } else {
                        (4, 0)
                    };
                    if !is_fluid {
                        seen_solid_above = true;
                    }
                    samples.push(ColumnSample {
                        y,
                        rgb,
                        sky_light: sky,
                        block_light: block,
                        is_fluid,
                    });
                }
            }
            columns.push(extract_column_spans(
                &samples,
                source_priority,
                max_extra,
                cfg.cave_culling_enabled,
                cfg.cave_culling_height,
                has_ceiling,
            ));
        }
    }

    ExtractedChunkColumns {
        dimension: dimension.to_owned(),
        chunk_pos: [cx, cz],
        columns,
        affected_sections: vec![SectionPos::from_block(0, bx0, bz0)],
    }
}

/// Bounded scheduler for G2 non-blocking server chunk requests (`LevelRef::request_chunk`).
///
/// Tracks in-flight chunk coordinates per dimension so DH never overwhelms the server streaming
/// queue and cleanly cancels out-of-range requests when the player moves or changes dimensions.
#[derive(Default)]
pub struct AsyncChunkRequestQueue {
    pub max_inflight: usize,
    pending_order: VecDeque<[i32; 2]>,
    inflight: HashSet<[i32; 2]>,
}

impl AsyncChunkRequestQueue {
    pub fn new(max_inflight: usize) -> Self {
        Self {
            max_inflight: max_inflight.max(1),
            pending_order: VecDeque::new(),
            inflight: HashSet::new(),
        }
    }

    pub fn can_submit(&self) -> bool {
        self.inflight.len() < self.max_inflight
    }

    pub fn mark_submitted(&mut self, pos: [i32; 2]) -> bool {
        if self.inflight.len() >= self.max_inflight || !self.inflight.insert(pos) {
            return false;
        }
        self.pending_order.push_back(pos);
        true
    }

    pub fn mark_completed(&mut self, pos: [i32; 2]) {
        self.inflight.remove(&pos);
    }

    pub fn inflight_len(&self) -> usize {
        self.inflight.len()
    }

    /// Return in-flight chunks farther than `max_chunk_dist` from `center_chunk` so callers can
    /// invoke `ChunkRequestRef::cancel(host)`.
    pub fn stale_positions(
        &mut self,
        center_chunk: [i32; 2],
        max_chunk_dist: i32,
    ) -> Vec<[i32; 2]> {
        let mut stale = Vec::new();
        self.inflight.retain(|&p| {
            let dx = (p[0] - center_chunk[0]).abs();
            let dz = (p[1] - center_chunk[1]).abs();
            if dx > max_chunk_dist || dz > max_chunk_dist {
                stale.push(p);
                false
            } else {
                true
            }
        });
        stale
    }
}

/// Apply an extracted 16x16 chunk batch into `DimensionLodStore` at `detail = 0`, returning all
/// modified `SectionPos` keys ready for quadtree propagation and meshing.
pub fn apply_extracted_chunk(
    store: &mut crate::lod::DimensionLodStore,
    batch: ExtractedChunkColumns,
) -> Vec<SectionPos> {
    let [cx, cz] = batch.chunk_pos;
    let bx0 = cx << 4;
    let bz0 = cz << 4;
    let sec_pos = SectionPos::from_block(0, bx0, bz0);
    let mut updated = false;
    {
        let sec = store.get_or_create_mut(sec_pos);
        for (idx, (col, spans)) in batch.columns.into_iter().enumerate() {
            if col.is_empty() {
                continue;
            }
            let lx = (idx % 16) as i32;
            let lz = (idx / 16) as i32;
            let (cell_x, cell_z) = sec_pos.local_cell(bx0 + lx, bz0 + lz);
            if sec.set_cell_with_spans(cell_x, cell_z, col, spans) {
                updated = true;
            }
        }
    }
    if updated { vec![sec_pos] } else { Vec::new() }
}

/// Direct adapters onto Sunrise's G1–G4 `mod_api` types (`#5339`, `#5340`, `#5341`, `#5347`).
///
/// Activated automatically once a Dev SDK includes `#5339`–`#5341` + `#5347` by enabling the
/// `sunrise-gaps-g1-g4` Cargo feature (or removing the cfg gate after the SDK bump).
#[cfg(feature = "sunrise-gaps-g1-g4")]
pub mod sunrise_adapters {
    use super::*;
    use crate::lod::SOURCE_LOADED_CHUNK;
    use mod_api::{
        ChunkRequestRef, ChunkSnapshot, DetachedChunkGenerator, GenerationStage, Host, LevelRef,
        SavedChunkReader, ServerHost,
    };

    /// Wraps a `mod_api::ChunkSnapshot` plus an immutable block-id lookup table resolved once on
    /// the tick thread so off-thread workers never need a `&dyn Host`.
    pub struct ModChunkSnapshot {
        snapshot: ChunkSnapshot,
        block_ids: Arc<HashMap<u32, String>>,
    }

    impl ModChunkSnapshot {
        pub fn new(snapshot: ChunkSnapshot, block_ids: Arc<HashMap<u32, String>>) -> Self {
            Self {
                snapshot,
                block_ids,
            }
        }

        pub fn capture_loaded(
            host: &dyn Host,
            level: LevelRef,
            chunk_pos: [i32; 2],
            block_ids: Arc<HashMap<u32, String>>,
        ) -> Option<Self> {
            let chunk = level.chunk(host, chunk_pos)?;
            let snapshot = chunk.snapshot(host)?;
            Some(Self::new(snapshot, block_ids))
        }
    }

    impl SnapshotSource for ModChunkSnapshot {
        fn pos(&self) -> [i32; 2] {
            self.snapshot.pos()
        }
        fn min_y(&self) -> i32 {
            self.snapshot.min_y()
        }
        fn height(&self) -> i32 {
            self.snapshot.height()
        }
        fn section_count(&self) -> usize {
            self.snapshot.section_count()
        }
        fn has_light(&self) -> bool {
            self.snapshot.has_light()
        }
        fn section_is_empty(&self, section_index: usize) -> bool {
            self.snapshot
                .section(section_index)
                .map(|s| s.is_empty())
                .unwrap_or(true)
        }
        fn section_y(&self, section_index: usize) -> i32 {
            self.snapshot
                .section(section_index)
                .map(|s| s.section_y())
                .unwrap_or(0)
        }
        fn block_id_at(&self, lx: u8, y: i32, lz: u8) -> Option<&str> {
            let state = self.snapshot.block_state(lx, y, lz)?;
            self.block_ids.get(&state.block().raw()).map(String::as_str)
        }
        fn light_at(&self, lx: u8, y: i32, lz: u8) -> (u8, u8) {
            self.snapshot.light(lx, y, lz).unwrap_or((15, 0))
        }
        fn biome_id_at(&self, lx: u8, y: i32, lz: u8) -> &str {
            self.snapshot
                .biome(lx, y, lz)
                .map(|k| k.id())
                .unwrap_or("minecraft:plains")
        }
    }

    pub struct ModSavedRegionSource {
        reader: SavedChunkReader,
        block_ids: Arc<HashMap<u32, String>>,
    }

    impl ModSavedRegionSource {
        pub fn new(reader: SavedChunkReader, block_ids: Arc<HashMap<u32, String>>) -> Self {
            Self { reader, block_ids }
        }
    }

    impl SavedRegionSource for ModSavedRegionSource {
        type Snapshot = ModChunkSnapshot;
        fn region_chunks(&self, rx: i32, rz: i32) -> Result<Vec<[i32; 2]>, String> {
            self.reader.region_chunks(rx, rz).map_err(|e| e.to_string())
        }
        fn read(&self, pos: [i32; 2]) -> Result<Option<Self::Snapshot>, String> {
            self.reader
                .read(pos)
                .map(|opt| opt.map(|s| ModChunkSnapshot::new(s, Arc::clone(&self.block_ids))))
                .map_err(|e| e.to_string())
        }
    }

    pub struct ModDetachedGeneratorSource {
        generator: DetachedChunkGenerator,
        block_ids: Arc<HashMap<u32, String>>,
    }

    impl ModDetachedGeneratorSource {
        pub fn new(
            generator: DetachedChunkGenerator,
            block_ids: Arc<HashMap<u32, String>>,
        ) -> Self {
            Self {
                generator,
                block_ids,
            }
        }
    }

    impl DetachedGeneratorSource for ModDetachedGeneratorSource {
        type Snapshot = ModChunkSnapshot;
        fn generate_chunk(
            &self,
            pos: [i32; 2],
            stage: DistantGenStage,
        ) -> Result<Self::Snapshot, String> {
            let api_stage = match stage {
                DistantGenStage::Surface => GenerationStage::Surface,
                DistantGenStage::Carvers => GenerationStage::Carvers,
                DistantGenStage::Features => GenerationStage::Features,
                DistantGenStage::Light => GenerationStage::Light,
            };
            let snap = self
                .generator
                .generate_chunk(pos, api_stage)
                .map_err(|e| e.to_string())?;
            Ok(ModChunkSnapshot::new(snap, Arc::clone(&self.block_ids)))
        }
    }

    /// Server-side driver for G2 (`LevelRef::request_chunk` + `drain_chunk_requests`).
    #[derive(Default)]
    pub struct ServerChunkRequestDriver {
        queue: AsyncChunkRequestQueue,
        handles: HashMap<[i32; 2], ChunkRequestRef>,
    }

    impl ServerChunkRequestDriver {
        pub fn new(max_inflight: usize) -> Self {
            Self {
                queue: AsyncChunkRequestQueue::new(max_inflight),
                handles: HashMap::new(),
            }
        }

        pub fn submit(
            &mut self,
            host: &mut dyn ServerHost,
            level: LevelRef,
            pos: [i32; 2],
        ) -> bool {
            if !self.queue.can_submit() {
                return false;
            }
            if let Ok(handle) = level.request_chunk(host, pos) {
                self.queue.mark_submitted(pos);
                self.handles.insert(pos, handle);
                true
            } else {
                false
            }
        }

        pub fn drain_into_worker(
            &mut self,
            host: &mut dyn ServerHost,
            level: LevelRef,
            dimension: &str,
            worker: &DataPipelineWorker,
            block_ids: &Arc<HashMap<u32, String>>,
            cfg: DhConfig,
        ) -> usize {
            let completed = level.drain_chunk_requests(host);
            let mut accepted = 0usize;
            for item in completed {
                self.queue.mark_completed(item.pos);
                self.handles.remove(&item.pos);
                if let Ok(snap) = item.result {
                    worker.submit_snapshot(
                        dimension.to_owned(),
                        ModChunkSnapshot::new(snap, Arc::clone(block_ids)),
                        cfg,
                        SOURCE_LOADED_CHUNK,
                    );
                    accepted += 1;
                }
            }
            accepted
        }

        pub fn cancel_far(
            &mut self,
            host: &mut dyn ServerHost,
            center_chunk: [i32; 2],
            max_chunk_dist: i32,
        ) {
            for pos in self.queue.stale_positions(center_chunk, max_chunk_dist) {
                if let Some(handle) = self.handles.remove(&pos) {
                    let _ = handle.cancel(host);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod::{DimensionLodStore, SOURCE_LOADED_CHUNK};

    struct MockSnapshot {
        pos: [i32; 2],
        has_light: bool,
        biome: String,
        surface_y: i32,
        water_y: Option<i32>,
        overhang_y: Option<i32>,
    }

    impl SnapshotSource for MockSnapshot {
        fn pos(&self) -> [i32; 2] {
            self.pos
        }
        fn min_y(&self) -> i32 {
            -64
        }
        fn height(&self) -> i32 {
            384
        }
        fn section_count(&self) -> usize {
            24
        }
        fn has_light(&self) -> bool {
            self.has_light
        }
        fn section_is_empty(&self, section_index: usize) -> bool {
            let sy0 = -64 + (section_index as i32) * 16;
            let sy1 = sy0 + 16;
            let top = self.overhang_y.or(self.water_y).unwrap_or(self.surface_y);
            sy0 > top || sy1 <= self.surface_y - 4
        }
        fn section_y(&self, section_index: usize) -> i32 {
            -4 + section_index as i32
        }
        fn block_id_at(&self, _lx: u8, y: i32, _lz: u8) -> Option<&str> {
            if let Some(oy) = self.overhang_y
                && (oy - 1..=oy).contains(&y)
            {
                return Some("minecraft:stone");
            }
            if let Some(wy) = self.water_y
                && y > self.surface_y
                && y <= wy
            {
                return Some("minecraft:water");
            }
            if y == self.surface_y {
                return Some("minecraft:grass_block");
            }
            if y < self.surface_y && y >= self.surface_y - 3 {
                return Some("minecraft:dirt");
            }
            None
        }
        fn light_at(&self, _lx: u8, y: i32, _lz: u8) -> (u8, u8) {
            if y > self.surface_y { (14, 2) } else { (0, 7) }
        }
        fn biome_id_at(&self, _lx: u8, _y: i32, _lz: u8) -> &str {
            &self.biome
        }
    }

    struct MockSavedRegion {
        chunks: HashMap<[i32; 2], i32>,
    }

    impl SavedRegionSource for MockSavedRegion {
        type Snapshot = MockSnapshot;
        fn region_chunks(&self, _rx: i32, _rz: i32) -> Result<Vec<[i32; 2]>, String> {
            Ok(self.chunks.keys().copied().collect())
        }
        fn read(&self, pos: [i32; 2]) -> Result<Option<Self::Snapshot>, String> {
            Ok(self.chunks.get(&pos).map(|&surface_y| MockSnapshot {
                pos,
                has_light: true,
                biome: "minecraft:forest".to_owned(),
                surface_y,
                water_y: None,
                overhang_y: None,
            }))
        }
    }

    struct MockDetachedGen;

    impl DetachedGeneratorSource for MockDetachedGen {
        type Snapshot = MockSnapshot;
        fn generate_chunk(
            &self,
            pos: [i32; 2],
            stage: DistantGenStage,
        ) -> Result<Self::Snapshot, String> {
            Ok(MockSnapshot {
                pos,
                has_light: stage == DistantGenStage::Light,
                biome: "minecraft:taiga".to_owned(),
                surface_y: 72,
                water_y: Some(74),
                overhang_y: None,
            })
        }
    }

    #[test]
    fn g1_snapshot_extraction_captures_overhang_light_and_applies_to_store() {
        let snap = MockSnapshot {
            pos: [2, -1],
            has_light: true,
            biome: "minecraft:plains".to_owned(),
            surface_y: 64,
            water_y: None,
            overhang_y: Some(90),
        };
        let cfg = DhConfig::default();
        let extracted =
            extract_chunk_snapshot("minecraft:overworld", &snap, cfg, SOURCE_LOADED_CHUNK);
        assert_eq!(extracted.columns.len(), 256);
        let (primary, extra) = &extracted.columns[0];
        assert_eq!(primary.solid_top_y, 91);
        assert_eq!(LodColumn::sky_light(primary.solid_rgba), 14);
        assert_eq!(LodColumn::block_light(primary.solid_rgba), 2);
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].top_y, 65);

        let mut store = DimensionLodStore::default();
        let dirty = apply_extracted_chunk(&mut store, extracted);
        assert_eq!(dirty.len(), 1);
        let sec = store.sections.get(&dirty[0]).unwrap();
        assert_eq!(sec.populated, 256);
    }

    #[test]
    fn g3_saved_region_and_g4_detached_gen_run_on_background_worker() {
        let worker = DataPipelineWorker::spawn();
        let cfg = DhConfig::default();

        let mut chunks = HashMap::new();
        chunks.insert([1, 2], 70);
        let reader = Arc::new(MockSavedRegion { chunks });
        assert!(worker.schedule_saved_region(
            "minecraft:overworld".to_owned(),
            [0, 0],
            Arc::clone(&reader),
            cfg,
        ));
        // Duplicate region schedule is deduplicated.
        assert!(!worker.schedule_saved_region(
            "minecraft:overworld".to_owned(),
            [0, 0],
            reader,
            cfg,
        ));

        let detached_gen = Arc::new(MockDetachedGen);
        assert!(worker.schedule_detached_chunk(
            "minecraft:overworld".to_owned(),
            [3, 4],
            DistantGenStage::Light,
            detached_gen,
            cfg,
        ));

        let mut received = Vec::new();
        for _ in 0..100 {
            received.extend(worker.drain_finished());
            if received.len() >= 2 {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(received.len(), 2);
        assert!(received.iter().any(|b| b.chunk_pos == [1, 2]));
        let gen_batch = received.iter().find(|b| b.chunk_pos == [3, 4]).unwrap();
        assert_eq!(gen_batch.columns[0].0.fluid_top_y, 75);
    }

    #[test]
    fn g2_request_queue_bounds_inflight_and_cancels_stale_chunks() {
        let mut q = AsyncChunkRequestQueue::new(2);
        assert!(q.mark_submitted([0, 0]));
        assert!(q.mark_submitted([100, 100]));
        assert!(!q.mark_submitted([1, 1]));
        let stale = q.stale_positions([0, 0], 16);
        assert_eq!(stale, vec![[100, 100]]);
        assert_eq!(q.inflight_len(), 1);
        assert!(q.mark_submitted([1, 1]));
    }
}
