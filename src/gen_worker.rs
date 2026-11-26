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

struct Shared<Job = GenJob> {
    queue: Mutex<VecDeque<Job>>,
    cv: Condvar,
    in_flight: Mutex<HashMap<(String, SectionPos), u64>>,
    dim_epochs: Mutex<HashMap<String, u64>>,
    done: Mutex<Vec<GenResult>>,
    global_epoch: AtomicU64,
    next_job_id: AtomicU64,
    stop: AtomicBool,
    /// Workers with `idx >= target_threads` park before taking another job.
    /// Changes are made under `queue` so a growing target cannot miss a waiter.
    target_threads: AtomicUsize,
    /// `ThreadPreset` discriminant (`threadRunTimeRatio` is derived from it).
    preset: AtomicU8,
}

pub struct GenWorkerPool {
    shared: Arc<Shared>,
    /// A slot keeps its original worker until world close, even above the active target.
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

impl<Job> Shared<Job> {
    fn new(target: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            in_flight: Mutex::new(HashMap::new()),
            dim_epochs: Mutex::new(HashMap::new()),
            done: Mutex::new(Vec::new()),
            global_epoch: AtomicU64::new(1),
            next_job_id: AtomicU64::new(1),
            stop: AtomicBool::new(false),
            target_threads: AtomicUsize::new(target),
            preset: AtomicU8::new(ThreadPreset::default() as u8),
        }
    }

    fn set_worker_target(&self, target: usize) {
        let previous = self.target_threads.load(Ordering::Acquire);
        if target != previous {
            let _queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            self.target_threads.store(target, Ordering::Release);
            if target > previous {
                self.cv.notify_all();
            }
        }
    }

    fn push_job(&self, job: Job) {
        let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.push_back(job);
        // A single notification might go only to a parked slot. Wake all when work first
        // arrives; eligible workers keep checking the queue until it is empty again.
        if q.len() == 1 {
            self.cv.notify_all();
        }
    }

    fn wait_for_job(&self, idx: usize, #[cfg(test)] mut before_wait: impl FnMut()) -> Option<Job> {
        let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        while !self.stop.load(Ordering::Acquire)
            && (q.is_empty() || idx >= self.target_threads.load(Ordering::Acquire))
        {
            #[cfg(test)]
            before_wait();
            q = self.cv.wait(q).unwrap_or_else(|e| e.into_inner());
        }
        if self.stop.load(Ordering::Acquire) {
            return None;
        }
        q.pop_front()
    }
}

impl GenWorkerPool {
    pub fn new() -> Self {
        let shared = Arc::new(Shared::new(
            ThreadPreset::default().thread_count(logical_cpus()),
        ));
        Self {
            shared,
            handles: Mutex::new(Vec::new()),
            max_in_flight: AtomicUsize::new(0),
        }
    }

    /// Spawn worker threads if not already running. Returns the number eligible for new work.
    pub fn ensure_started(&self) -> usize {
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        let target = self.shared.target_threads.load(Ordering::Acquire);
        if handles.is_empty() {
            spawn_workers(&self.shared, &mut handles, target);
            self.max_in_flight.store(
                handles.len() * MAX_RETRIEVAL_REQUESTS_PER_THREAD,
                Ordering::Release,
            );
        }
        target.min(handles.len())
    }

    /// Apply a DH thread preset. Workers above the target finish their current batch, then
    /// park in their original slots. Growing reuses those slots before spawning extra workers.
    /// The retrieval cap follows the active count (DH's `getMaxRetrievalQueueCount()`).
    pub fn reconfigure(&self, preset: ThreadPreset) {
        self.apply_preset(preset, preset.thread_count(logical_cpus()));
    }

    fn apply_preset(&self, preset: ThreadPreset, target: usize) {
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        self.shared.preset.store(preset as u8, Ordering::Release);
        self.shared.set_worker_target(target);
        if handles.is_empty() {
            return; // not running; `ensure_started` picks the new target up
        }
        spawn_workers(&self.shared, &mut handles, target);
        self.max_in_flight.store(
            target.min(handles.len()) * MAX_RETRIEVAL_REQUESTS_PER_THREAD,
            Ordering::Release,
        );
    }

    /// Number of worker threads eligible for new work.
    pub fn worker_count(&self) -> usize {
        let handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        self.shared
            .target_threads
            .load(Ordering::Acquire)
            .min(handles.len())
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
    pub(crate) fn enqueue(
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
            epochs.get(&dimension).copied().unwrap_or(0)
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
        self.shared.push_job(GenJob {
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
            epochs.get(dim).copied().unwrap_or(0)
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

    /// Cancel queued, running, and completed work for this level.
    pub fn reset_dimension(&self, dim: &str) {
        {
            let mut epochs = self
                .shared
                .dim_epochs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let entry = epochs.entry(dim.to_owned()).or_default();
            *entry = entry.wrapping_add(1);
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
        self.shutdown_and_join_inner(
            #[cfg(test)]
            || {},
        );
    }

    fn shutdown_and_join_inner(&self, #[cfg(test)] before_join: impl FnOnce()) {
        // Keep start and resize out until every slot has joined. Workers never take this lock.
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.global_epoch.fetch_add(1, Ordering::Release);
            self.shared.stop.store(true, Ordering::Release);
            queue.clear();
            self.shared.cv.notify_all();
        }
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
        #[cfg(test)]
        before_join();
        for handle in handles.drain(..) {
            let _ = handle.join();
        }
        // A worker finishing its final sample can publish after the initial clear.
        self.shared
            .done
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
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
        epochs.get(&job.dimension).copied().unwrap_or(0) == job.dim_epoch
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
    while let Some(job) = shared.wait_for_job(
        idx,
        #[cfg(test)]
        || {},
    ) {
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
    use std::sync::mpsc;

    #[test]
    fn trimmed_slot_parks_with_work_and_resumes_on_growth() {
        let shared = Arc::new(Shared::new(2));
        shared.push_job(41);
        let (started, running) = mpsc::channel();
        let (release, held) = mpsc::channel();
        let (parking, parked) = mpsc::channel();
        let (popped, result) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let worker = thread::spawn(move || {
            started.send(worker_shared.wait_for_job(1, || {})).unwrap();
            held.recv().unwrap();
            let mut parking = Some(parking);
            let job = worker_shared.wait_for_job(1, || {
                if let Some(parking) = parking.take() {
                    parking.send(()).unwrap();
                }
            });
            popped.send(job).unwrap();
        });

        let deadline = std::time::Duration::from_millis(500);
        let first_job = running.recv_timeout(deadline);
        shared.set_worker_target(1);
        shared.push_job(42);
        release.send(()).unwrap();

        // The worker signals under the queue lock just before waiting. Growth must acquire
        // that same lock, so it cannot notify before the worker has entered the wait.
        let did_park = parked.recv_timeout(deadline);
        let queued_while_trimmed = shared.queue.lock().unwrap().front().copied();
        shared.set_worker_target(2);
        let job = result.recv_timeout(deadline);
        // Wake the worker for cleanup even if the growth notification regresses.
        {
            let _queue = shared.queue.lock().unwrap();
            shared.stop.store(true, Ordering::Release);
            shared.cv.notify_all();
        }
        worker.join().unwrap();
        assert_eq!(first_job, Ok(Some(41)));
        assert_eq!(did_park, Ok(()));
        assert_eq!(queued_while_trimmed, Some(42));
        assert_eq!(job, Ok(Some(42)));
    }

    #[test]
    fn busy_slot_survives_preset_changes_and_is_joined_on_close() {
        let pool = GenWorkerPool::new();
        pool.apply_preset(ThreadPreset::Balanced, 1);
        assert_eq!(pool.ensure_started(), 1);

        let (entered, busy) = mpsc::channel();
        let (release, held) = mpsc::channel();
        let (completed, finished) = mpsc::channel();
        // Stand in for a generator call already in progress at index 1. No SDK handles
        // are needed to exercise the actual ownership, resize and shutdown paths.
        let upper = thread::spawn(move || {
            entered.send(()).unwrap();
            held.recv().unwrap();
            completed.send(()).unwrap();
        });
        let upper_id = upper.thread().id();
        pool.handles.lock().unwrap().push(upper);
        busy.recv().unwrap();

        let snapshot = || {
            let active = pool.ensure_started();
            let workers = pool.worker_count();
            let cap = pool.max_in_flight.load(Ordering::Acquire);
            let handles = pool.handles.lock().unwrap();
            let upper = handles.get(1).map(|h| h.thread().id());
            (active, workers, cap, handles.len(), upper)
        };
        pool.apply_preset(ThreadPreset::IPaidForTheWholeCpu, 2);
        pool.apply_preset(ThreadPreset::Balanced, 1);
        let shrunk = snapshot();
        pool.apply_preset(ThreadPreset::IPaidForTheWholeCpu, 2);
        let grown = snapshot();
        // Close at the smaller preset: the still-busy upper slot must be joined too.
        pool.apply_preset(ThreadPreset::Balanced, 1);
        let epoch = pool.shared.global_epoch.load(Ordering::Acquire);
        let (stopping, start_excluded) = thread::scope(|scope| {
            let close = scope.spawn(|| pool.shutdown_and_join());
            let queue = pool.shared.queue.lock().unwrap();
            drop(
                pool.shared
                    .cv
                    .wait_while(queue, |_| {
                        pool.shared.global_epoch.load(Ordering::Acquire) == epoch
                    })
                    .unwrap(),
            );
            let stopping = pool.shared.stop.load(Ordering::Acquire);
            // ensure_started must still be excluded while the busy slot cannot return.
            let start_excluded = matches!(
                pool.handles.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            );
            release.send(()).unwrap();
            close.join().unwrap();
            (stopping, start_excluded)
        });

        assert_eq!(
            shrunk,
            (1, 1, MAX_RETRIEVAL_REQUESTS_PER_THREAD, 2, Some(upper_id))
        );
        assert_eq!(
            grown,
            (
                2,
                2,
                2 * MAX_RETRIEVAL_REQUESTS_PER_THREAD,
                2,
                Some(upper_id)
            )
        );
        assert!(stopping && start_excluded);
        assert_eq!(finished.try_recv(), Ok(()));
        assert!(pool.handles.lock().unwrap().is_empty());
        assert!(!pool.has_capacity());
    }

    #[test]
    fn close_discards_a_batch_completed_after_cancellation() {
        let pool = GenWorkerPool::new();
        let (release, held) = mpsc::channel();
        let (completed, finished) = mpsc::channel();
        let shared = Arc::clone(&pool.shared);
        let epoch = shared.global_epoch.load(Ordering::Acquire);
        let worker = thread::spawn(move || {
            held.recv().unwrap();
            assert!(shared.stop.load(Ordering::Acquire));
            shared.done.lock().unwrap().push(GenResult {
                job_id: 1,
                global_epoch: epoch,
                dim_epoch: 0,
                dimension: "closing".into(),
                pos: SectionPos::new(0, 0, 0),
                cells: vec![(0, 0)],
                samples: vec![None],
                saved_chunks: Vec::new(),
                generated_chunks: Vec::new(),
                outcome: GenOutcome::Complete,
            });
            completed.send(()).unwrap();
        });
        pool.handles.lock().unwrap().push(worker);

        // Release the final batch only after the pre-join clears and stop publication.
        pool.shutdown_and_join_inner(|| release.send(()).unwrap());
        assert_eq!(finished.try_recv(), Ok(()));
        // Inspect the buffer directly: drain_for_dim would hide stale output via its epoch check.
        assert!(pool.shared.done.lock().unwrap().is_empty());
    }

    #[test]
    fn cancelling_one_level_increments_only_its_epoch() {
        let pool = GenWorkerPool::new();
        pool.reset_dimension("sibling");
        pool.reset_dimension("unloading");
        pool.reset_dimension("unloading");
        let epochs = pool.shared.dim_epochs.lock().unwrap();
        assert_eq!(epochs["unloading"], 2);
        assert_eq!(epochs["sibling"], 1);
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
