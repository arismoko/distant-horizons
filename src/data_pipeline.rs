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
//!   for off-thread full distant terrain/carvers/features/light generation AND fast coarse outer
//!   ring sampling across Overworld, Nether, End, and custom dimensions.
//!
//! Everything here is built on pure `SnapshotSource` / `SavedRegionSource` /
//! `DetachedGeneratorSource` trait abstractions so:
//! 1. All extraction, worker-thread batching, region-import ordering, and detached generation
//!    scheduling compiles and passes unit tests immediately on the `f8aa80e` SDK.
//! 2. Enabling `--features sunrise-gaps-g1-g4` (once `#5339`–`#5341` + `#5347` hit a Sunrise Dev
//!    SDK) activates the direct `mod_api` adapters wired into `DistantHorizonsMod` hooks with zero
//!    rewriting of the DH pipeline.

use crate::colors::{
    biome_water_color, block_color, classify_dh_material, is_skippable_for_lod, is_water_block,
};
use crate::config::DhConfig;
use crate::lod::{
    ColumnSample, DimensionLodStore, ExtraSpan, LodColumn, SECTION_SIZE, SOURCE_DETACHED_GEN,
    SOURCE_DISTANT_SURFACE, SOURCE_SAVED_REGION, SectionPos, extract_column_spans,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

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
    /// Optional map-color fallback (RGB) from `BlockStates::map_color_rgba` for modded blocks not
    /// in DH's built-in palette table.
    fn block_fallback_rgb_at(&self, _lx: u8, _y: i32, _lz: u8) -> Option<[u8; 3]> {
        None
    }
    fn light_at(&self, lx: u8, y: i32, lz: u8) -> (u8, u8);
    fn biome_id_at(&self, lx: u8, y: i32, lz: u8) -> &str;
}

/// One surface column returned by G4 `DetachedChunkGenerator::sample_surface_columns`.
#[derive(Clone, Debug)]
pub struct DetachedSurfaceSample {
    pub solid_top_y: i32,
    pub fluid_top_y: Option<i32>,
    pub surface_block_id: String,
    pub biome_id: String,
    pub fallback_rgb: Option<[u8; 3]>,
}

/// Extracted 16x16 LOD chunk columns ready to merge into `DimensionLodStore` at `detail = 0`.
#[derive(Clone, Debug)]
pub struct ExtractedChunkColumns {
    pub dimension: String,
    pub chunk_pos: [i32; 2],
    pub columns: Vec<(LodColumn, Vec<ExtraSpan>)>,
    pub affected_sections: Vec<SectionPos>,
}

/// Extracted coarse section cells produced off-thread via G4 `sample_surface_columns`.
#[derive(Clone, Debug)]
pub struct ExtractedCoarseSection {
    pub dimension: String,
    pub section_pos: SectionPos,
    pub cells: Vec<(usize, usize, LodColumn)>,
}

/// Result produced by a background `DataPipelineWorker` job.
#[derive(Clone, Debug)]
pub enum PipelineBatch {
    Chunk(ExtractedChunkColumns),
    CoarseSection(ExtractedCoarseSection),
}

/// Extract all 256 multi-layer LOD columns from a full-height immutable chunk snapshot off-thread.
///
/// Uses section-level `section_is_empty()` skipping so air-only vertical sections cost O(1),
/// reads real sky/block light when `snapshot.has_light()` (falling back to open-sky daylight for
/// pre-light detached stages), and uses `block_fallback_rgb_at` for modded custom blocks.
pub fn extract_chunk_snapshot<S: SnapshotSource + ?Sized>(
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
                    } else if let Some(custom_rgb) = snapshot.block_fallback_rgb_at(lx, y, lz)
                        && !block_id.starts_with("minecraft:")
                    {
                        custom_rgb
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
                        material_id: classify_dh_material(block_id),
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
    fn sample_surface_columns(
        &self,
        columns: &[(i32, i32)],
    ) -> Result<Vec<Option<DetachedSurfaceSample>>, String>;
}

/// Returns Anvil region coordinates `(rx, rz)` intersecting the player's DH render distance, sorted
/// nearest-first from `(cam_x, cam_z)`.
pub fn nearby_regions_sorted(cam_x: f64, cam_z: f64, render_distance_chunks: i32) -> Vec<[i32; 2]> {
    let cam_cx = (cam_x.floor() as i32) >> 4;
    let cam_cz = (cam_z.floor() as i32) >> 4;
    let r_chunks = render_distance_chunks.max(16);
    let min_rx = (cam_cx - r_chunks) >> 5;
    let max_rx = (cam_cx + r_chunks) >> 5;
    let min_rz = (cam_cz - r_chunks) >> 5;
    let max_rz = (cam_cz + r_chunks) >> 5;
    let mut regions = Vec::new();
    for rz in min_rz..=max_rz {
        for rx in min_rx..=max_rx {
            regions.push([rx, rz]);
        }
    }
    regions.sort_by_key(|&[rx, rz]| {
        let center_x = (rx << 5) + 16;
        let center_z = (rz << 5) + 16;
        let dx = (center_x - cam_cx) as i64;
        let dz = (center_z - cam_cz) as i64;
        dx * dx + dz * dz
    });
    regions
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
    CoarseSample {
        run: Box<dyn FnOnce() -> Option<ExtractedCoarseSection> + Send + 'static>,
    },
}

/// Background multi-purpose worker pool for G1 snapshot extraction, G3 saved region import, and
/// G4 detached worldgen + off-thread coarse surface sampling.
pub struct DataPipelineWorker {
    tx: Sender<DataJob>,
    rx: Mutex<Receiver<PipelineBatch>>,
    imported_regions: Mutex<HashMap<String, HashSet<[i32; 2]>>>,
    inflight_detached: Mutex<HashMap<String, HashSet<[i32; 2]>>>,
    inflight_coarse: Mutex<HashMap<String, HashSet<SectionPos>>>,
}

impl DataPipelineWorker {
    pub fn spawn() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<DataJob>();
        let (res_tx, res_rx) = mpsc::channel::<PipelineBatch>();
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
                                let extracted = extract_chunk_snapshot(
                                    &dimension,
                                    snapshot.as_ref(),
                                    cfg,
                                    source_priority,
                                );
                                let _ = res_tx.send(PipelineBatch::Chunk(extracted));
                            }
                            DataJob::ImportSavedRegion { run } => {
                                for extracted in run() {
                                    let _ = res_tx.send(PipelineBatch::Chunk(extracted));
                                }
                            }
                            DataJob::DetachedGenerate { run } => {
                                if let Some(extracted) = run() {
                                    let _ = res_tx.send(PipelineBatch::Chunk(extracted));
                                }
                            }
                            DataJob::CoarseSample { run } => {
                                if let Some(coarse) = run() {
                                    let _ = res_tx.send(PipelineBatch::CoarseSection(coarse));
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
            inflight_coarse: Mutex::new(HashMap::new()),
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
                SOURCE_DETACHED_GEN,
            ))
        });
        let _ = self.tx.send(DataJob::DetachedGenerate { run });
        true
    }

    /// Schedule one coarse outer-ring section (`detail >= 2`) for fast off-thread G4
    /// `sample_surface_columns` sampling (4096 cells per section, zero tick-thread work).
    pub fn schedule_coarse_section<G: DetachedGeneratorSource>(
        &self,
        dimension: String,
        section_pos: SectionPos,
        generator: Arc<G>,
    ) -> bool {
        {
            let mut inflight = self.inflight_coarse.lock().unwrap();
            let set = inflight.entry(dimension.clone()).or_default();
            if !set.insert(section_pos) {
                return false;
            }
        }
        let dim_for_job = dimension.clone();
        let run = Box::new(move || {
            let (ox, oz) = section_pos.block_origin();
            let step = section_pos.cell_size();
            let half = step >> 1;
            let mut coords = Vec::with_capacity(SECTION_SIZE * SECTION_SIZE);
            for cz in 0..SECTION_SIZE {
                for cx in 0..SECTION_SIZE {
                    coords.push((
                        ox + (cx as i32) * step + half,
                        oz + (cz as i32) * step + half,
                    ));
                }
            }
            let samples = generator.sample_surface_columns(&coords).ok()?;
            let mut cells = Vec::with_capacity(samples.len());
            for (idx, sample) in samples.into_iter().enumerate() {
                let Some(s) = sample else {
                    continue;
                };
                let cx = idx % SECTION_SIZE;
                let cz = idx / SECTION_SIZE;
                let rgb = if let Some(custom_rgb) = s.fallback_rgb
                    && !s.surface_block_id.starts_with("minecraft:")
                {
                    custom_rgb
                } else {
                    block_color(&s.surface_block_id, &s.biome_id, 0)
                };
                let fluid_rgb = biome_water_color(&s.biome_id, 0);
                let col = LodColumn::from_solid_and_fluid(
                    s.solid_top_y,
                    (s.solid_top_y - 16).max(-64),
                    rgb,
                    s.fluid_top_y,
                    fluid_rgb,
                    15,
                    0,
                    SOURCE_DISTANT_SURFACE,
                )
                .with_material(classify_dh_material(&s.surface_block_id));
                cells.push((cx, cz, col));
            }
            Some(ExtractedCoarseSection {
                dimension: dim_for_job,
                section_pos,
                cells,
            })
        });
        let _ = self.tx.send(DataJob::CoarseSample { run });
        true
    }

    pub fn inflight_detached_count(&self, dimension: &str) -> usize {
        self.inflight_detached
            .lock()
            .unwrap()
            .get(dimension)
            .map_or(0, HashSet::len)
    }

    pub fn inflight_coarse_count(&self, dimension: &str) -> usize {
        self.inflight_coarse
            .lock()
            .unwrap()
            .get(dimension)
            .map_or(0, HashSet::len)
    }

    /// Drain extracted chunk and coarse-section batches completed by background workers.
    pub fn drain_finished(&self) -> Vec<PipelineBatch> {
        let mut out = Vec::new();
        let guard = self.rx.lock().unwrap();
        while let Ok(item) = guard.try_recv() {
            out.push(item);
        }
        drop(guard);
        if !out.is_empty() {
            let mut detached = self.inflight_detached.lock().unwrap();
            let mut coarse = self.inflight_coarse.lock().unwrap();
            for item in &out {
                match item {
                    PipelineBatch::Chunk(c) => {
                        if let Some(set) = detached.get_mut(&c.dimension) {
                            set.remove(&c.chunk_pos);
                        }
                    }
                    PipelineBatch::CoarseSection(s) => {
                        if let Some(set) = coarse.get_mut(&s.dimension) {
                            set.remove(&s.section_pos);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn clear_dimension(&self, dimension: &str) {
        self.imported_regions.lock().unwrap().remove(dimension);
        self.inflight_detached.lock().unwrap().remove(dimension);
        self.inflight_coarse.lock().unwrap().remove(dimension);
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
    store: &mut DimensionLodStore,
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

/// Apply an off-thread coarse-section surface sample batch into `DimensionLodStore`.
pub fn apply_coarse_section(
    store: &mut DimensionLodStore,
    batch: ExtractedCoarseSection,
) -> Option<SectionPos> {
    let sec = store.get_or_create_mut(batch.section_pos);
    let mut updated = false;
    for (cx, cz, col) in batch.cells {
        if sec.set_cell(cx, cz, col) {
            updated = true;
        }
    }
    store.sampled_distant_sections.insert(batch.section_pos);
    updated.then_some(batch.section_pos)
}

/// Direct adapters onto Sunrise's G1–G4 `mod_api` types (`#5339`, `#5340`, `#5341`, `#5347`).
///
/// Activated once a Dev SDK includes `#5339`–`#5341` + `#5347` by enabling the
/// `sunrise-gaps-g1-g4` Cargo feature (or removing the cfg gate after the SDK bump).
#[cfg(feature = "sunrise-gaps-g1-g4")]
pub mod sunrise_adapters {
    use super::*;
    use crate::lod::SOURCE_LOADED_CHUNK;
    use mod_api::{
        BlockKey, ChunkRequestRef, ChunkSnapshot, DetachedChunkGenerator, GenerationStage, Host,
        LevelRef, SavedChunkReader, ServerHost,
    };

    /// Immutable per-block lookup table built once on the tick thread from `Host` so background
    /// worker threads never lock or touch `Host`, while still resolving exact vanilla & modded
    /// block IDs and `BlockStates::map_color_rgba` fallback colors.
    #[derive(Clone, Debug)]
    pub struct BlockVisualEntry {
        pub id: String,
        pub fallback_rgb: Option<[u8; 3]>,
    }

    pub type BlockVisualTable = Arc<HashMap<BlockKey, BlockVisualEntry>>;

    /// Snapshot all registered blocks (`host.block_keys()`) and their default map colors once.
    pub fn build_block_visual_table(host: &dyn Host) -> BlockVisualTable {
        let states = host.block_states();
        let mut map = HashMap::new();
        for key in host.block_keys() {
            let Some(id) = host.block_key_id(key) else {
                continue;
            };
            let fallback_rgb = states
                .states_of(key)
                .and_then(|range| range.first())
                .and_then(|st| states.map_color(st))
                .and_then(|packed| states.map_color_rgba(packed))
                .filter(|rgba| rgba[3] > 0)
                .map(|rgba| [rgba[0], rgba[1], rgba[2]]);
            map.insert(
                key,
                BlockVisualEntry {
                    id: id.to_owned(),
                    fallback_rgb,
                },
            );
        }
        Arc::new(map)
    }

    /// Wraps a `mod_api::ChunkSnapshot` plus an immutable `BlockVisualTable`.
    pub struct ModChunkSnapshot {
        snapshot: ChunkSnapshot,
        blocks: BlockVisualTable,
    }

    impl ModChunkSnapshot {
        pub fn new(snapshot: ChunkSnapshot, blocks: BlockVisualTable) -> Self {
            Self { snapshot, blocks }
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
            self.blocks.get(&state.block()).map(|e| e.id.as_str())
        }
        fn block_fallback_rgb_at(&self, lx: u8, y: i32, lz: u8) -> Option<[u8; 3]> {
            let state = self.snapshot.block_state(lx, y, lz)?;
            self.blocks.get(&state.block())?.fallback_rgb
        }
        fn light_at(&self, lx: u8, y: i32, lz: u8) -> (u8, u8) {
            self.snapshot.light(lx, y, lz).unwrap_or((15, 0))
        }
        fn biome_id_at(&self, lx: u8, y: i32, lz: u8) -> &str {
            self.snapshot
                .biome(lx, y, lz)
                .and_then(|id| self.snapshot.biome_name(id))
                .unwrap_or("minecraft:plains")
        }
    }

    pub struct ModSavedRegionSource {
        reader: SavedChunkReader,
        blocks: BlockVisualTable,
    }

    impl ModSavedRegionSource {
        pub fn new(reader: SavedChunkReader, blocks: BlockVisualTable) -> Self {
            Self { reader, blocks }
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
                .map(|opt| opt.map(|s| ModChunkSnapshot::new(s, Arc::clone(&self.blocks))))
                .map_err(|e| e.to_string())
        }
    }

    pub struct ModDetachedGeneratorSource {
        generator: DetachedChunkGenerator,
        blocks: BlockVisualTable,
    }

    impl ModDetachedGeneratorSource {
        pub fn new(generator: DetachedChunkGenerator, blocks: BlockVisualTable) -> Self {
            Self { generator, blocks }
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
            Ok(ModChunkSnapshot::new(snap, Arc::clone(&self.blocks)))
        }

        fn sample_surface_columns(
            &self,
            columns: &[(i32, i32)],
        ) -> Result<Vec<Option<DetachedSurfaceSample>>, String> {
            let cols = self
                .generator
                .sample_surface_columns(columns)
                .map_err(|e| e.to_string())?;
            Ok(cols
                .into_iter()
                .map(|opt| {
                    opt.map(|c| DetachedSurfaceSample {
                        solid_top_y: c.surface_y,
                        fluid_top_y: c.water_y,
                        surface_block_id: c.surface_block,
                        biome_id: c.biome,
                        fallback_rgb: None,
                    })
                })
                .collect())
        }
    }

    /// Per-dimension handles cached on `level_load` for G3 saved-region import and G4 detached
    /// worldgen without repeatedly re-acquiring handles every tick.
    pub struct DimensionGapHandles {
        pub saved_reader: Option<Arc<ModSavedRegionSource>>,
        pub detached_gen: Option<Arc<ModDetachedGeneratorSource>>,
        pub chunk_requests: ServerChunkRequestDriver,
    }

    impl Default for DimensionGapHandles {
        fn default() -> Self {
            Self {
                saved_reader: None,
                detached_gen: None,
                chunk_requests: ServerChunkRequestDriver::new(16),
            }
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
            blocks: &BlockVisualTable,
            cfg: DhConfig,
        ) -> usize {
            let Some(completed) = level.drain_chunk_requests(host) else {
                return 0;
            };
            let mut accepted = 0usize;
            for item in completed {
                self.queue.mark_completed(item.pos);
                self.handles.remove(&item.pos);
                if let Ok(snap) = item.result {
                    worker.submit_snapshot(
                        dimension.to_owned(),
                        ModChunkSnapshot::new(snap, Arc::clone(blocks)),
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
    use crate::lod::SOURCE_LOADED_CHUNK;

    struct MockSnapshot {
        pos: [i32; 2],
        has_light: bool,
        biome: String,
        surface_block: String,
        fallback_rgb: Option<[u8; 3]>,
        surface_y: i32,
        water_y: Option<i32>,
        overhang_y: Option<i32>,
    }

    impl MockSnapshot {
        fn grass(pos: [i32; 2], surface_y: i32) -> Self {
            Self {
                pos,
                has_light: true,
                biome: "minecraft:plains".to_owned(),
                surface_block: "minecraft:grass_block".to_owned(),
                fallback_rgb: None,
                surface_y,
                water_y: None,
                overhang_y: None,
            }
        }
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
                return Some(&self.surface_block);
            }
            if y < self.surface_y && y >= self.surface_y - 3 {
                return Some("minecraft:dirt");
            }
            None
        }
        fn block_fallback_rgb_at(&self, _lx: u8, y: i32, _lz: u8) -> Option<[u8; 3]> {
            (y == self.surface_y).then_some(self.fallback_rgb).flatten()
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
                biome: "minecraft:forest".to_owned(),
                ..MockSnapshot::grass(pos, surface_y)
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
                has_light: stage == DistantGenStage::Light,
                biome: "minecraft:taiga".to_owned(),
                water_y: Some(74),
                ..MockSnapshot::grass(pos, 72)
            })
        }

        fn sample_surface_columns(
            &self,
            columns: &[(i32, i32)],
        ) -> Result<Vec<Option<DetachedSurfaceSample>>, String> {
            Ok(columns
                .iter()
                .map(|_| {
                    Some(DetachedSurfaceSample {
                        solid_top_y: 68,
                        fluid_top_y: None,
                        surface_block_id: "minecraft:sand".to_owned(),
                        biome_id: "minecraft:desert".to_owned(),
                        fallback_rgb: None,
                    })
                })
                .collect())
        }
    }

    #[test]
    fn g1_snapshot_extraction_captures_overhang_light_and_modded_block_color() {
        let snap = MockSnapshot {
            surface_block: "examplemod:violet_crystal".to_owned(),
            fallback_rgb: Some([170, 50, 220]),
            overhang_y: Some(90),
            ..MockSnapshot::grass([2, -1], 64)
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
        assert_eq!(
            [extra[0].rgba[0], extra[0].rgba[1], extra[0].rgba[2]],
            [170, 50, 220]
        );

        let mut store = DimensionLodStore::default();
        let dirty = apply_extracted_chunk(&mut store, extracted);
        assert_eq!(dirty.len(), 1);
        let sec = store.sections.get(&dirty[0]).unwrap();
        assert_eq!(sec.populated, 256);
    }

    #[test]
    fn provenance_priority_keeps_loaded_over_saved_over_detached_over_surface() {
        const {
            assert!(SOURCE_LOADED_CHUNK > SOURCE_SAVED_REGION);
            assert!(SOURCE_SAVED_REGION > SOURCE_DETACHED_GEN);
            assert!(SOURCE_DETACHED_GEN > SOURCE_DISTANT_SURFACE);
        }

        let cfg = DhConfig::default();
        let mut store = DimensionLodStore::default();
        let detached = extract_chunk_snapshot(
            "minecraft:overworld",
            &MockSnapshot::grass([0, 0], 60),
            cfg,
            SOURCE_DETACHED_GEN,
        );
        apply_extracted_chunk(&mut store, detached);
        let saved = extract_chunk_snapshot(
            "minecraft:overworld",
            &MockSnapshot::grass([0, 0], 70),
            cfg,
            SOURCE_SAVED_REGION,
        );
        apply_extracted_chunk(&mut store, saved);
        let sec_pos = SectionPos::new(0, 0, 0);
        assert_eq!(store.sections[&sec_pos].get(0, 0).solid_top_y, 71);

        // Detached gen must not overwrite saved region data.
        let weaker = extract_chunk_snapshot(
            "minecraft:overworld",
            &MockSnapshot::grass([0, 0], 50),
            cfg,
            SOURCE_DETACHED_GEN,
        );
        apply_extracted_chunk(&mut store, weaker);
        assert_eq!(store.sections[&sec_pos].get(0, 0).solid_top_y, 71);

        // Loaded chunk overrides saved region data.
        let loaded = extract_chunk_snapshot(
            "minecraft:overworld",
            &MockSnapshot::grass([0, 0], 82),
            cfg,
            SOURCE_LOADED_CHUNK,
        );
        apply_extracted_chunk(&mut store, loaded);
        assert_eq!(store.sections[&sec_pos].get(0, 0).solid_top_y, 83);
    }

    #[test]
    fn g3_saved_region_g4_detached_and_coarse_sample_run_on_background_worker() {
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
            Arc::clone(&detached_gen),
            cfg,
        ));
        let coarse_pos = SectionPos::new(2, 1, 0);
        assert!(worker.schedule_coarse_section(
            "minecraft:overworld".to_owned(),
            coarse_pos,
            detached_gen,
        ));

        let mut received = Vec::new();
        for _ in 0..100 {
            received.extend(worker.drain_finished());
            if received.len() >= 3 {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(received.len(), 3);
        let mut store = DimensionLodStore::default();
        let mut saw_saved = false;
        let mut saw_detached = false;
        let mut saw_coarse = false;
        for batch in received {
            match batch {
                PipelineBatch::Chunk(c) if c.chunk_pos == [1, 2] => {
                    saw_saved = true;
                    apply_extracted_chunk(&mut store, c);
                }
                PipelineBatch::Chunk(c) if c.chunk_pos == [3, 4] => {
                    saw_detached = true;
                    assert_eq!(c.columns[0].0.fluid_top_y, 75);
                    apply_extracted_chunk(&mut store, c);
                }
                PipelineBatch::CoarseSection(s) if s.section_pos == coarse_pos => {
                    saw_coarse = true;
                    assert_eq!(s.cells.len(), SECTION_SIZE * SECTION_SIZE);
                    assert_eq!(apply_coarse_section(&mut store, s), Some(coarse_pos));
                }
                _ => {}
            }
        }
        assert!(saw_saved && saw_detached && saw_coarse);
        assert_eq!(
            store.sections[&coarse_pos].populated,
            (SECTION_SIZE * SECTION_SIZE) as u16
        );
    }

    #[test]
    fn nearby_regions_sorted_places_camera_region_first() {
        let regions = nearby_regions_sorted(20.0, 20.0, 64);
        assert!(!regions.is_empty());
        assert_eq!(regions[0], [0, 0]);
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
