//! Background worker pool for off-thread distant terrain generation via
//! `mod_api::DetachedChunkGenerator`.
//!
//! Worker count follows Distant Horizons' `EDhApiThreadPreset` (`ThreadPresetConfigEventHandler`):
//! `ceil(available_processors * percent)` clamped to `1..=available_processors`, default
//! `Balanced` (50%). [`GenWorkerPool::reconfigure`] applies a preset live.
//!
//! Already-saved Anvil chunks are ingested first through `mod_api::SavedChunkReader` (G3), and
//! only columns without a saved chunk fall back to `DetachedChunkGenerator` (G4); neither loads
//! anything into the live world.
//!
//! Keeps all noise, surface-rule, and biome sampling off the server tick thread.
//! Finished section batches are drained in O(1) during `level_tick_post`, and
//! `server_stopping` cancels and joins all worker threads cleanly.

use crate::config::ThreadPreset;
use crate::lod::SectionPos;
use mod_api::{
    ChunkSnapshot, DetachedChunkGenerator, GenerationStage, SavedChunkReader, TerrainColumn,
};
use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Matches DH `GeneratedFullDataSourceProvider.MAX_RETRIEVAL_REQUESTS_PER_THREAD` (line 67,
/// `getMaxRetrievalQueueCount()` line 206: `20 * numberOfThreads`).
const MAX_RETRIEVAL_REQUESTS_PER_THREAD: usize = 20;

/// Cap on finished `ChunkSnapshot`s buffered in `done` (4 detail-0 jobs of 16 chunks). At or
/// above it no new detail-0 job is dispatched, so a busy server thread cannot let full 3D
/// snapshots pile up: `done` stays below `MAX_DONE_CHUNKS` plus what the already dispatched
/// detail-0 jobs (at most `2 * workers`) can still add.
pub const MAX_DONE_CHUNKS: usize = 64;

/// Whether dispatching another detail-0 job must wait: `done_chunks` snapshots are already
/// buffered, or `unfinished_jobs` detail-0 jobs (queued or running) already fill two per worker.
fn detail0_dispatch_paused(done_chunks: usize, unfinished_jobs: usize, workers: usize) -> bool {
    done_chunks >= MAX_DONE_CHUNKS || unfinished_jobs >= workers.max(1) * 2
}

/// Columns sampled per `sample_surface_columns` call inside a worker thread so
/// world-exit / invalidation cancellation is observed promptly between chunks.
const WORKER_CHUNK_COLS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenOutcome {
    Complete,
    Cancelled,
    Error,
}

struct GenJob {
    job_id: u64,
    global_epoch: u64,
    dim_epoch: u64,
    dimension: String,
    pos: SectionPos,
    generator: DetachedChunkGenerator,
    saved_reader: Option<SavedChunkReader>,
    /// When set on a `detail == 0` job, unsaved chunks run `generate_chunk(chunk, stage)` so
    /// near/mid distant terrain gets real vanilla trees, foliage, and surface features.
    feature_stage: Option<GenerationStage>,
    cells: Vec<(usize, usize)>,
}

pub struct GenResult {
    pub job_id: u64,
    pub global_epoch: u64,
    pub dim_epoch: u64,
    pub dimension: String,
    pub pos: SectionPos,
    pub cells: Vec<(usize, usize)>,
    pub samples: Vec<Option<TerrainColumn>>,
    /// Chunks read from saved Anvil storage (G3). Their columns are absent from `cells`.
    pub saved_chunks: Vec<([i32; 2], ChunkSnapshot)>,
    /// Chunks generated at `Features`/`Light` stage (G4 full 3D chunks with trees).
    pub generated_chunks: Vec<([i32; 2], ChunkSnapshot)>,
    pub outcome: GenOutcome,
}

#[derive(Clone, Copy, Default)]
struct DimensionEpoch {
    epoch: u64,
    unloading: bool,
}

struct Shared {
    queue: Mutex<VecDeque<GenJob>>,
    cv: Condvar,
    in_flight: Mutex<HashMap<(String, SectionPos), u64>>,
    dim_epochs: Mutex<HashMap<String, DimensionEpoch>>,
    done: Mutex<Vec<GenResult>>,
    global_epoch: AtomicU64,
    next_job_id: AtomicU64,
    stop: AtomicBool,
    /// Desired worker count; workers with `idx >= target_threads` exit.
    target_threads: AtomicUsize,
    /// `ThreadPreset` discriminant (`threadRunTimeRatio` is derived from it).
    preset: AtomicU8,
}

pub struct GenWorkerPool {
    shared: Arc<Shared>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    max_in_flight: AtomicUsize,
}

/// Logical processor count (DH `Runtime.availableProcessors()`).
fn logical_cpus() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .max(1)
}

impl GenWorkerPool {
    pub fn new() -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            in_flight: Mutex::new(HashMap::new()),
            dim_epochs: Mutex::new(HashMap::new()),
            done: Mutex::new(Vec::new()),
            global_epoch: AtomicU64::new(1),
            next_job_id: AtomicU64::new(1),
            stop: AtomicBool::new(false),
            target_threads: AtomicUsize::new(ThreadPreset::default().thread_count(logical_cpus())),
            preset: AtomicU8::new(ThreadPreset::default() as u8),
        });
        Self {
            shared,
            handles: Mutex::new(Vec::new()),
            max_in_flight: AtomicUsize::new(0),
        }
    }

    /// Spawn worker threads if not already running. Returns the number of live workers.
    pub fn ensure_started(&self) -> usize {
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        if !handles.is_empty() {
            return handles.len();
        }
        self.shared.stop.store(false, Ordering::Release);
        let target = self.shared.target_threads.load(Ordering::Acquire);
        spawn_workers(&self.shared, &mut handles, target);
        let spawned = handles.len();
        self.max_in_flight.store(
            spawned * MAX_RETRIEVAL_REQUESTS_PER_THREAD,
            Ordering::Release,
        );
        spawned
    }

    /// Apply a DH thread preset. Spawns extra workers or retires the highest-indexed ones
    /// (they finish their current batch, then exit) and rescales the retrieval cap
    /// (`GeneratedFullDataSourceProvider.getMaxRetrievalQueueCount()` = 20 * threads).
    pub fn reconfigure(&self, preset: ThreadPreset) {
        let target = preset.thread_count(logical_cpus());
        if self.shared.preset.load(Ordering::Acquire) == preset as u8
            && self.worker_count() == target
        {
            return;
        }
        self.shared.preset.store(preset as u8, Ordering::Release);
        self.shared.target_threads.store(target, Ordering::Release);
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        if handles.is_empty() {
            return; // not running; `ensure_started` picks the new target up
        }
        if handles.len() > target {
            // Dropping a `JoinHandle` detaches; the worker sees `idx >= target` and exits.
            handles.truncate(target);
            self.shared.cv.notify_all();
        } else {
            spawn_workers(&self.shared, &mut handles, target);
        }
        self.max_in_flight.store(
            handles.len() * MAX_RETRIEVAL_REQUESTS_PER_THREAD,
            Ordering::Release,
        );
    }

    /// Number of live worker threads.
    pub fn worker_count(&self) -> usize {
        self.handles.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn has_capacity(&self) -> bool {
        let cap = self.max_in_flight.load(Ordering::Acquire);
        if cap == 0 {
            return false;
        }
        let set = self
            .shared
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        set.len() < cap
    }

    /// Backpressure for detail-0 jobs, the only ones that carry `ChunkSnapshot`s.
    fn detail0_paused(&self) -> bool {
        let (done_chunks, done_jobs) = {
            let done = self.shared.done.lock().unwrap_or_else(|e| e.into_inner());
            (
                done.iter()
                    .map(|r| r.saved_chunks.len() + r.generated_chunks.len())
                    .sum::<usize>(),
                done.iter().filter(|r| r.pos.detail == 0).count(),
            )
        };
        // `in_flight` holds queued, running and finished-but-undrained jobs alike.
        let in_flight_jobs = self
            .shared
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .filter(|(_, p)| p.detail == 0)
            .count();
        detail0_dispatch_paused(
            done_chunks,
            in_flight_jobs.saturating_sub(done_jobs),
            self.worker_count(),
        )
    }

    pub fn is_in_flight(&self, dim: &str, pos: SectionPos) -> bool {
        let set = self
            .shared
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        set.iter().any(|((d, p), _)| d == dim && *p == pos)
    }

    /// Enqueue a distant section for off-thread sampling. Returns `true` if queued,
    /// `false` if no workers are running, the pool is at capacity, or this section is in flight.
    pub fn enqueue(
        &self,
        dimension: String,
        pos: SectionPos,
        generator: DetachedChunkGenerator,
        saved_reader: Option<SavedChunkReader>,
        feature_stage: Option<GenerationStage>,
        cells: Vec<(usize, usize)>,
    ) -> bool {
        if cells.is_empty() {
            return false;
        }
        if self.ensure_started() == 0 {
            return false;
        }
        if pos.detail == 0 && self.detail0_paused() {
            return false;
        }
        let cap = self.max_in_flight.load(Ordering::Acquire);
        let job_id = self.shared.next_job_id.fetch_add(1, Ordering::Relaxed);
        let dim_epoch = {
            let epochs = self
                .shared
                .dim_epochs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let epoch = epochs.get(&dimension).copied().unwrap_or_default();
            if epoch.unloading {
                return false;
            }
            epoch.epoch
        };
        {
            let mut in_flight = self
                .shared
                .in_flight
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if in_flight.len() >= cap {
                return false;
            }
            let key = (dimension.clone(), pos);
            if in_flight.contains_key(&key) {
                return false;
            }
            in_flight.insert(key, job_id);
        }
        let global_epoch = self.shared.global_epoch.load(Ordering::Acquire);
        let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.push_back(GenJob {
            job_id,
            global_epoch,
            dim_epoch,
            dimension,
            pos,
            generator,
            saved_reader,
            feature_stage,
            cells,
        });
        self.shared.cv.notify_one();
        true
    }

    /// Drain completed results for `dim` matching the current global epoch, dimension epoch,
    /// and `in_flight` job ID, clearing their `in_flight` entries safely.
    pub fn drain_for_dim(&self, dim: &str) -> Vec<GenResult> {
        let current_global = self.shared.global_epoch.load(Ordering::Acquire);
        let current_dim = {
            let epochs = self
                .shared
                .dim_epochs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            epochs.get(dim).map_or(0, |entry| entry.epoch)
        };

        let mut done = self.shared.done.lock().unwrap_or_else(|e| e.into_inner());
        if done.is_empty() {
            return Vec::new();
        }
        let mut candidates = Vec::new();
        let mut keep = Vec::new();
        let mut stale_keys = Vec::new();
        for res in done.drain(..) {
            if res.global_epoch != current_global {
                stale_keys.push((res.dimension, res.pos, res.job_id));
            } else if res.dimension == dim {
                if res.dim_epoch == current_dim {
                    candidates.push(res);
                } else {
                    stale_keys.push((res.dimension, res.pos, res.job_id));
                }
            } else {
                keep.push(res);
            }
        }
        *done = keep;
        drop(done);

        let mut accepted = Vec::with_capacity(candidates.len());
        {
            let mut in_flight = self
                .shared
                .in_flight
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for (d, p, id) in stale_keys {
                let key = (d, p);
                if in_flight.get(&key) == Some(&id) {
                    in_flight.remove(&key);
                }
            }
            for res in candidates {
                let key = (res.dimension.clone(), res.pos);
                if in_flight.get(&key) == Some(&res.job_id) {
                    in_flight.remove(&key);
                    accepted.push(res);
                }
            }
        }
        accepted
    }

    /// Cancel queued, running, and completed work for a single invalidated section.
    pub fn invalidate_section(&self, dim: &str, pos: SectionPos) {
        self.shared
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|(d, p), _| !(d == dim && *p == pos));
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|job| !(job.dimension == dim && job.pos == pos));
        self.shared
            .done
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|res| !(res.dimension == dim && res.pos == pos));
    }

    pub fn dimension_allowed(&self, dim: &str) -> bool {
        self.shared
            .dim_epochs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(dim)
            .is_none_or(|entry| !entry.unloading)
    }

    pub fn allow_dimension(&self, dim: &str) {
        self.shared
            .dim_epochs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(dim.to_owned())
            .or_default()
            .unloading = false;
    }

    /// Cancel queued, running, and completed work for this level and reject more until load.
    pub fn reset_dimension(&self, dim: &str) {
        {
            let mut epochs = self
                .shared
                .dim_epochs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let entry = epochs.entry(dim.to_owned()).or_default();
            entry.epoch = entry.epoch.wrapping_add(1);
            entry.unloading = true;
        }
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|job| job.dimension != dim);
        self.shared
            .done
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|res| res.dimension != dim);
        self.shared
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|(d, _), _| d != dim);
    }

    /// Cancel all queued and running work and join every worker thread on world close.
    pub fn shutdown_and_join(&self) {
        self.shared.global_epoch.fetch_add(1, Ordering::Release);
        self.shared.stop.store(true, Ordering::Release);
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.shared
            .done
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.shared
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.shared.cv.notify_all();

        let to_join: Vec<JoinHandle<()>> = {
            let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *handles)
        };
        for handle in to_join {
            let _ = handle.join();
        }
        self.max_in_flight.store(0, Ordering::Release);
        self.shared.stop.store(false, Ordering::Release);
    }
}

impl Default for GenWorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

fn is_job_valid(shared: &Shared, job: &GenJob) -> bool {
    if shared.stop.load(Ordering::Acquire) {
        return false;
    }
    if shared.global_epoch.load(Ordering::Acquire) != job.global_epoch {
        return false;
    }
    let dim_ok = {
        let epochs = shared.dim_epochs.lock().unwrap_or_else(|e| e.into_inner());
        epochs
            .get(&job.dimension)
            .map_or(job.dim_epoch == 0, |entry| {
                !entry.unloading && entry.epoch == job.dim_epoch
            })
    };
    if !dim_ok {
        return false;
    }
    let in_flight = shared.in_flight.lock().unwrap_or_else(|e| e.into_inner());
    in_flight.get(&(job.dimension.clone(), job.pos)) == Some(&job.job_id)
}

fn clear_in_flight_if_matching(shared: &Shared, dim: &str, pos: SectionPos, job_id: u64) {
    let mut in_flight = shared.in_flight.lock().unwrap_or_else(|e| e.into_inner());
    let key = (dim.to_owned(), pos);
    if in_flight.get(&key) == Some(&job_id) {
        in_flight.remove(&key);
    }
}

fn spawn_workers(shared: &Arc<Shared>, handles: &mut Vec<JoinHandle<()>>, target: usize) {
    while handles.len() < target {
        let idx = handles.len();
        let shared_clone = Arc::clone(shared);
        match thread::Builder::new()
            .name(format!("ds-worldgen-{idx}"))
            .spawn(move || worker_loop(shared_clone, idx))
        {
            Ok(handle) => handles.push(handle),
            Err(_) => break,
        }
    }
}

fn worker_loop(shared: Arc<Shared>, idx: usize) {
    loop {
        let job = {
            let mut q = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            while q.is_empty()
                && !shared.stop.load(Ordering::Acquire)
                && idx < shared.target_threads.load(Ordering::Acquire)
            {
                q = shared.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
            if shared.stop.load(Ordering::Acquire)
                || idx >= shared.target_threads.load(Ordering::Acquire)
            {
                return;
            }
            q.pop_front()
        };
        let Some(job) = job else {
            continue;
        };

        if !is_job_valid(&shared, &job) {
            clear_in_flight_if_matching(&shared, &job.dimension, job.pos, job.job_id);
            continue;
        }

        let busy_start = std::time::Instant::now();
        let (ox, oz) = job.pos.block_origin();
        let step = job.pos.cell_size();
        let half = step >> 1;

        let run_result = catch_unwind(AssertUnwindSafe(|| {
            let mut samples = Vec::with_capacity(job.cells.len());
            let mut saved_chunks: Vec<([i32; 2], ChunkSnapshot)> = Vec::new();
            let mut generated_chunks: Vec<([i32; 2], ChunkSnapshot)> = Vec::new();
            let mut outcome = GenOutcome::Complete;
            let mut query = Vec::with_capacity(WORKER_CHUNK_COLS);
            let mut fallback_cells: Vec<(usize, usize)> = Vec::with_capacity(job.cells.len());

            // G3 + G4 full 3D chunks for detail-0 (4x4 chunks): read saved Anvil chunks first,
            // then generate unsaved chunks at `feature_stage` (`Features`/`Light`) so near/mid
            // terrain carries real vanilla trees and foliage. Coarser rings stay on fast column
            // sampling.
            if job.pos.detail == 0 && (job.saved_reader.is_some() || job.feature_stage.is_some()) {
                let mut chunk_state: HashMap<[i32; 2], bool> = HashMap::new();
                for &(cx, cz) in &job.cells {
                    let chunk = [(ox + cx as i32) >> 4, (oz + cz as i32) >> 4];
                    let covered = match chunk_state.get(&chunk) {
                        Some(&known) => known,
                        None => {
                            if !is_job_valid(&shared, &job) {
                                return (
                                    samples,
                                    saved_chunks,
                                    generated_chunks,
                                    Vec::new(),
                                    GenOutcome::Cancelled,
                                );
                            }
                            let mut found = false;
                            if let Some(reader) = job.saved_reader.as_ref()
                                && let Ok(Some(snap)) = reader.read(chunk)
                            {
                                saved_chunks.push((chunk, snap));
                                found = true;
                            }
                            if !found
                                && let Some(stage) = job.feature_stage
                                && let Ok(snap) = job.generator.generate_chunk(chunk, stage)
                            {
                                generated_chunks.push((chunk, snap));
                                found = true;
                            }
                            chunk_state.insert(chunk, found);
                            found
                        }
                    };
                    if !covered {
                        fallback_cells.push((cx, cz));
                    }
                }
            } else {
                fallback_cells.extend_from_slice(&job.cells);
            }

            // G4: everything not covered by a 3D snapshot is sampled by the detached generator.
            for chunk in fallback_cells.chunks(WORKER_CHUNK_COLS) {
                if !is_job_valid(&shared, &job) {
                    outcome = GenOutcome::Cancelled;
                    break;
                }
                query.clear();
                for &(cx, cz) in chunk {
                    query.push((
                        ox + (cx as i32) * step + half,
                        oz + (cz as i32) * step + half,
                    ));
                }
                match job.generator.sample_surface_columns(&query) {
                    Ok(mut batch) => samples.append(&mut batch),
                    Err(_) => {
                        outcome = GenOutcome::Error;
                        break;
                    }
                }
            }
            (
                samples,
                saved_chunks,
                generated_chunks,
                fallback_cells,
                outcome,
            )
        }));

        let (samples, saved_chunks, generated_chunks, cells, outcome) = match run_result {
            Ok(parts) => parts,
            Err(_) => (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                GenOutcome::Error,
            ),
        };

        // DH `threadRunTimeRatio`: idle for `busy * (1/ratio - 1)` so the thread runs `ratio` of
        // the time (`MinimalImpact` = 0.5, every other preset 1.0).
        let ratio = ThreadPreset::from_u8(shared.preset.load(Ordering::Acquire)).run_time_ratio();
        if ratio < 1.0 {
            thread::sleep(busy_start.elapsed().mul_f64(1.0 / ratio - 1.0));
        }

        if outcome == GenOutcome::Cancelled {
            clear_in_flight_if_matching(&shared, &job.dimension, job.pos, job.job_id);
            continue;
        }

        shared
            .done
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(GenResult {
                job_id: job.job_id,
                global_epoch: job.global_epoch,
                dim_epoch: job.dim_epoch,
                dimension: job.dimension,
                pos: job.pos,
                cells,
                samples,
                saved_chunks,
                generated_chunks,
                outcome,
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_level_rejects_generation_until_reloaded() {
        let pool = GenWorkerPool::new();
        assert!(pool.dimension_allowed("unloading"));
        pool.reset_dimension("unloading");
        assert!(!pool.dimension_allowed("unloading"));
        assert!(pool.dimension_allowed("sibling"));
        pool.allow_dimension("unloading");
        assert!(pool.dimension_allowed("unloading"));
        assert_eq!(pool.shared.dim_epochs.lock().unwrap()["unloading"].epoch, 1);
    }

    #[test]
    fn done_chunk_cap_pauses_detail0_dispatch() {
        assert!(!detail0_dispatch_paused(0, 0, 4));
        assert!(!detail0_dispatch_paused(MAX_DONE_CHUNKS - 1, 0, 4));
        assert!(detail0_dispatch_paused(MAX_DONE_CHUNKS, 0, 4));
        assert!(detail0_dispatch_paused(MAX_DONE_CHUNKS + 16, 0, 4));
    }

    #[test]
    fn unfinished_detail0_jobs_are_bounded_by_two_per_worker() {
        assert!(!detail0_dispatch_paused(0, 7, 4));
        assert!(detail0_dispatch_paused(0, 8, 4));
        // No workers still counts as one, so the cap never divides to zero.
        assert!(detail0_dispatch_paused(0, 2, 0));
    }

    #[test]
    fn a_full_done_buffer_is_at_most_four_full_jobs() {
        assert_eq!(MAX_DONE_CHUNKS / 16, 4);
    }
}
