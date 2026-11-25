//! Background worker pool for off-thread distant terrain generation via
//! `mod_api::DetachedChunkGenerator`.
//!
//! Default worker count follows Distant Horizons' `EDhApiThreadPreset.BALANCED`
//! (`ThreadPresetConfigEventHandler.getDefaultThreadCount()` -> `getThreadCountByPercent(0.5)`:
//! `ceil(available_processors * 0.5)` clamped to `1..=available_processors`).
//!
//! Keeps all noise, surface-rule, and biome sampling off the server tick thread.
//! Finished section batches are drained in O(1) during `level_tick_post`, and
//! `server_stopping` cancels and joins all worker threads cleanly.

use crate::lod::SectionPos;
use mod_api::{DetachedChunkGenerator, TerrainColumn};
use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

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
    pub outcome: GenOutcome,
}

struct Shared {
    queue: Mutex<VecDeque<GenJob>>,
    cv: Condvar,
    in_flight: Mutex<HashMap<(String, SectionPos), u64>>,
    dim_epochs: Mutex<HashMap<String, u64>>,
    done: Mutex<Vec<GenResult>>,
    global_epoch: AtomicU64,
    next_job_id: AtomicU64,
    stop: AtomicBool,
}

pub struct GenWorkerPool {
    shared: Arc<Shared>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    max_in_flight: AtomicUsize,
}

/// Matches DH `ThreadPresetConfigEventHandler.getDefaultThreadCount()` (`BALANCED` = 50% of
/// logical processors, rounded up, clamped to `1..=total`).
fn default_worker_count() -> usize {
    let total = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .max(1);
    total.div_ceil(2).clamp(1, total)
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
        let target = default_worker_count();
        for idx in 0..target {
            let shared_clone = Arc::clone(&self.shared);
            if let Ok(handle) = thread::Builder::new()
                .name(format!("ds-worldgen-{idx}"))
                .spawn(move || worker_loop(shared_clone))
            {
                handles.push(handle);
            }
        }
        let spawned = handles.len();
        self.max_in_flight.store(spawned * 4, Ordering::Release);
        spawned
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
        cells: Vec<(usize, usize)>,
    ) -> bool {
        if cells.is_empty() {
            return false;
        }
        if self.ensure_started() == 0 {
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
        let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.push_back(GenJob {
            job_id,
            global_epoch,
            dim_epoch,
            dimension,
            pos,
            generator,
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

    /// Cancel queued, running, and completed work for a single unloading dimension by bumping
    /// its per-dimension epoch.
    pub fn reset_dimension(&self, dim: &str) {
        {
            let mut epochs = self
                .shared
                .dim_epochs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let entry = epochs.entry(dim.to_owned()).or_insert(0);
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
            .retain(|(d, _)| d != dim);
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

fn worker_loop(shared: Arc<Shared>) {
    loop {
        let job = {
            let mut q = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            while q.is_empty() && !shared.stop.load(Ordering::Acquire) {
                q = shared.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
            if shared.stop.load(Ordering::Acquire) {
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

        let (ox, oz) = job.pos.block_origin();
        let step = job.pos.cell_size();
        let half = step >> 1;

        let run_result = catch_unwind(AssertUnwindSafe(|| {
            let mut samples = Vec::with_capacity(job.cells.len());
            let mut outcome = GenOutcome::Complete;
            let mut query = Vec::with_capacity(WORKER_CHUNK_COLS);

            for chunk in job.cells.chunks(WORKER_CHUNK_COLS) {
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
            (samples, outcome)
        }));

        let (samples, outcome) = match run_result {
            Ok(pair) => pair,
            Err(_) => (Vec::new(), GenOutcome::Error),
        };

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
                cells: job.cells,
                samples,
                outcome,
            });
    }
}
