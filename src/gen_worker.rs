//! Background worker pool for off-thread distant terrain generation via
//! `mod_api::DetachedChunkGenerator`.
//!
//! Keeps all noise, surface-rule, and biome sampling off the server tick thread.
//! Finished section batches are drained in O(1) during `level_tick_post`.

use crate::lod::SectionPos;
use mod_api::{DetachedChunkGenerator, TerrainColumn};
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

/// Columns sampled per `sample_surface_columns` call inside a worker thread so
/// world-exit / epoch cancellation is observed promptly between chunks.
const WORKER_CHUNK_COLS: usize = 256;

struct GenJob {
    epoch: u64,
    dimension: String,
    pos: SectionPos,
    generator: DetachedChunkGenerator,
    cells: Vec<(usize, usize)>,
}

pub struct GenResult {
    pub epoch: u64,
    pub dimension: String,
    pub pos: SectionPos,
    pub cells: Vec<(usize, usize)>,
    pub samples: Vec<Option<TerrainColumn>>,
    /// True if every chunk succeeded without cancellation or generator error.
    pub complete: bool,
}

struct Shared {
    queue: Mutex<VecDeque<GenJob>>,
    cv: Condvar,
    in_flight: Mutex<HashSet<(String, SectionPos)>>,
    done: Mutex<Vec<GenResult>>,
    epoch: AtomicU64,
}

pub struct GenWorkerPool {
    shared: Arc<Shared>,
    max_in_flight: usize,
}

impl GenWorkerPool {
    pub fn spawn() -> Self {
        let workers = thread::available_parallelism()
            .map(|n| (n.get() / 2).clamp(1, 4))
            .unwrap_or(2);
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            in_flight: Mutex::new(HashSet::new()),
            done: Mutex::new(Vec::new()),
            epoch: AtomicU64::new(1),
        });

        for idx in 0..workers {
            let shared_clone = Arc::clone(&shared);
            let _ = thread::Builder::new()
                .name(format!("ds-worldgen-{idx}"))
                .spawn(move || worker_loop(shared_clone));
        }

        Self {
            shared,
            max_in_flight: workers * 4,
        }
    }

    pub fn has_capacity(&self) -> bool {
        self.shared
            .in_flight
            .lock()
            .map(|set| set.len() < self.max_in_flight)
            .unwrap_or(false)
    }

    pub fn is_in_flight(&self, dim: &str, pos: SectionPos) -> bool {
        self.shared
            .in_flight
            .lock()
            .map(|set| set.iter().any(|(d, p)| d == dim && *p == pos))
            .unwrap_or(false)
    }

    /// Enqueue a distant section for off-thread sampling. Returns `true` if queued,
    /// `false` if the pool is at capacity or this section is already in flight.
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
        {
            let mut in_flight = self.shared.in_flight.lock().unwrap();
            if in_flight.len() >= self.max_in_flight {
                return false;
            }
            if !in_flight.insert((dimension.clone(), pos)) {
                return false;
            }
        }
        let epoch = self.shared.epoch.load(Ordering::Relaxed);
        let mut q = self.shared.queue.lock().unwrap();
        q.push_back(GenJob {
            epoch,
            dimension,
            pos,
            generator,
            cells,
        });
        self.shared.cv.notify_one();
        true
    }

    /// Drain completed results for `dim` matching the current epoch, clearing their
    /// `in_flight` entries (and discarding any stale-epoch results).
    pub fn drain_for_dim(&self, dim: &str) -> Vec<GenResult> {
        let current_epoch = self.shared.epoch.load(Ordering::Relaxed);
        let mut done = self.shared.done.lock().unwrap();
        if done.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut keep = Vec::new();
        let mut clear_keys = Vec::new();
        for res in done.drain(..) {
            if res.epoch != current_epoch {
                clear_keys.push((res.dimension, res.pos));
            } else if res.dimension == dim {
                clear_keys.push((res.dimension.clone(), res.pos));
                out.push(res);
            } else {
                keep.push(res);
            }
        }
        *done = keep;
        drop(done);

        if !clear_keys.is_empty() {
            let mut in_flight = self.shared.in_flight.lock().unwrap();
            for key in clear_keys {
                in_flight.remove(&key);
            }
        }
        out
    }

    /// Cancel all queued and running work across all dimensions (world start/stop).
    pub fn reset_all(&self) {
        self.shared.epoch.fetch_add(1, Ordering::Relaxed);
        self.shared.queue.lock().unwrap().clear();
        self.shared.done.lock().unwrap().clear();
        self.shared.in_flight.lock().unwrap().clear();
    }

    /// Drop queued/finished work for a single unloading dimension.
    pub fn reset_dimension(&self, dim: &str) {
        self.shared
            .queue
            .lock()
            .unwrap()
            .retain(|job| job.dimension != dim);
        self.shared
            .done
            .lock()
            .unwrap()
            .retain(|res| res.dimension != dim);
        self.shared
            .in_flight
            .lock()
            .unwrap()
            .retain(|(d, _)| d != dim);
    }
}

fn worker_loop(shared: Arc<Shared>) {
    loop {
        let job = {
            let mut q = shared.queue.lock().unwrap();
            while q.is_empty() {
                q = shared.cv.wait(q).unwrap();
            }
            q.pop_front()
        };
        let Some(job) = job else {
            continue;
        };

        if shared.epoch.load(Ordering::Relaxed) != job.epoch {
            shared
                .in_flight
                .lock()
                .unwrap()
                .remove(&(job.dimension, job.pos));
            continue;
        }

        let (ox, oz) = job.pos.block_origin();
        let step = job.pos.cell_size();
        let half = step >> 1;

        let mut samples = Vec::with_capacity(job.cells.len());
        let mut complete = true;
        let mut query = Vec::with_capacity(WORKER_CHUNK_COLS);

        for chunk in job.cells.chunks(WORKER_CHUNK_COLS) {
            if shared.epoch.load(Ordering::Relaxed) != job.epoch {
                complete = false;
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
                    complete = false;
                    break;
                }
            }
        }

        shared.done.lock().unwrap().push(GenResult {
            epoch: job.epoch,
            dimension: job.dimension,
            pos: job.pos,
            cells: job.cells,
            samples,
            complete,
        });
    }
}
