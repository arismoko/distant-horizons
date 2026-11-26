//! Ingestion of server-side terrain sources into 32³ [`OctreeSection`]s.
//!
//! Sources, in priority order (a lower source never overwrites a higher one in the same chunk):
//! - G1 [`IngestSource::Loaded`]: a loaded chunk's `ChunkSnapshot` (full 3D),
//! - G3 [`IngestSource::Saved`]: a saved Anvil chunk read through `SavedChunkReader`, which hands
//!   back the same `ChunkSnapshot` type and therefore takes the same section path,
//! - G4 [`IngestSource::Distant`]: detached Features/Light snapshots.
//! - [`IngestSource::Estimate`]: detached surface and water columns for unseen terrain.
//!
//! Layout references (Distant Horizons `FullDataSourceV2`, under
//! `coreSubProjects/core/src/main/java/com/seibel/distanthorizons/core/dataObjects/fullData/sources/`):
//! - `FullDataSourceV2.java:77` `WIDTH = 64` columns per DH data source; Voxy-style octree
//!   sections are 32³ instead (`octree::SECTION_EDGE`), so a DH source maps onto 2×2 sections
//!   horizontally and the 16×16×16 vanilla chunk sections used here nest exactly 2×2×2 inside one.
//! - `FullDataSourceV2.java:605-616` `downsampleFromOneDetailLevelBelow`: a 2×2 block of input
//!   columns becomes one parent column; parents are rebuilt from their children. The 3D
//!   equivalent is `mip_voxels`, used to reduce authored patches up to `MAX_OCTREE_DETAIL`.
//! - `FullDataSourceV2.java:1105-1161` (`updateFromOneDetailLevelAbove`) only fills a column that
//!   is still empty, so lower-detail data never replaces real data; the source priority here
//!   keeps the same guarantee for chunk-level provenance.
//!
//! A uniform-air vanilla section maps to one aligned octant. Open sky is stored as lit air
//! (`Voxel::air_with_light(15, 0)`), distinct from the dark/unwritten `Voxel::AIR` around it,
//! without allocating voxel indices. `ChunkEdit::capture` emits lit air only when every column
//! above is still open, and dark `Voxel::AIR` otherwise. In `ChunkEdit::apply`, a uniform patch
//! with the value `Voxel::AIR` clears only its aligned region in an existing section and never
//! creates one. A uniform lit-air patch can create a missing section.

use crate::octree::{
    OctreeSection, OctreeSectionPos, SECTION_EDGE, Voxel, child_index, mip_voxels,
};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

/// Edge length of a vanilla chunk section in blocks.
const VANILLA_SECTION: usize = 16;
/// Vanilla biome storage is a 4×4×4 quart grid per section.
const QUART: usize = 4;
/// World-Y floor for subsurface fill in surface estimates. For detached snapshots outside
/// the Nether, intake skips only whole sections below this floor; it does not fill caves.
pub const DISTANT_SOLID_FLOOR: i32 = 0;

/// Where a chunk's voxels came from. The derived order is the priority order: `Loaded` wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IngestSource {
    /// Surface estimates, below all captured snapshots.
    Estimate,
    /// G4 detached Features/Light snapshots.
    Distant,
    /// G3 `SavedChunkReader` Anvil chunk.
    Saved,
    /// G1 live `ChunkSnapshot`.
    Loaded,
}

/// Read access to one chunk's voxels. Implemented for `mod_api::ChunkSnapshot` below and by test
/// doubles. `y` is an absolute world Y; `lx`/`lz` are chunk-local `0..16`.
pub trait ChunkVoxelSource {
    /// Opaque block handle resolved to a Mapper block id by the caller's closure.
    type Block: Copy + Eq + Hash;

    fn min_y(&self) -> i32;
    /// Number of 16-tall vanilla sections starting at `min_y`.
    fn section_count(&self) -> usize;
    /// `Some(block)` when section `index` is a single uniform block (air included).
    fn section_uniform_block(&self, index: usize) -> Option<Self::Block>;
    fn block(&self, lx: usize, y: i32, lz: usize) -> Option<Self::Block>;
    fn biome(&self, lx: usize, y: i32, lz: usize) -> Option<&str>;
    /// `(sky, block)` light at the position.
    fn light(&self, lx: usize, y: i32, lz: usize) -> (u8, u8);
}

// `min_y` and `section_count` give the snapshot's vertical extent. `section` and
// `SectionSnapshot::uniform_block` read uniform states without scanning voxels. `block_state`,
// `biome` and `light` use chunk-local X/Z and world Y. The adapter checks `has_light` and
// falls back to `(15, 0)` when no light sample is available.
impl ChunkVoxelSource for mod_api::ChunkSnapshot {
    type Block = mod_api::BlockState;

    fn min_y(&self) -> i32 {
        mod_api::ChunkSnapshot::min_y(self)
    }

    fn section_count(&self) -> usize {
        mod_api::ChunkSnapshot::section_count(self)
    }

    fn section_uniform_block(&self, index: usize) -> Option<Self::Block> {
        self.section(index)
            .and_then(|section| section.uniform_block())
    }

    fn block(&self, lx: usize, y: i32, lz: usize) -> Option<Self::Block> {
        self.block_state(lx as u8, y, lz as u8)
    }

    fn biome(&self, lx: usize, y: i32, lz: usize) -> Option<&str> {
        mod_api::ChunkSnapshot::biome(self, lx as u8, y, lz as u8).map(|k| k.identifier().as_str())
    }

    fn light(&self, lx: usize, y: i32, lz: usize) -> (u8, u8) {
        if self.has_light() {
            mod_api::ChunkSnapshot::light(self, lx as u8, y, lz as u8).unwrap_or((15, 0))
        } else {
            (15, 0)
        }
    }
}

/// One G4 surface sample at an absolute block column.
#[derive(Clone, Copy, Debug)]
pub struct DistantColumn {
    pub x: i32,
    pub z: i32,
    /// Highest solid block Y.
    pub height: i32,
    /// Water surface Y when the column is submerged.
    pub water: Option<i32>,
    pub block: u32,
    pub biome: u16,
}

type EstimateColumns = HashSet<(OctreeSectionPos, usize, usize)>;

/// Detail-0 sections, chunk provenance and the dirty set awaiting mip propagation.
#[derive(Default)]
pub struct OctreeIngestStore {
    sections: HashMap<OctreeSectionPos, OctreeSection>,
    chunk_sources: HashMap<[i32; 2], IngestSource>,
    /// Section Ys holding G4 voxels per block column, so a re-sample can clear its old column.
    distant_column_ys: HashMap<(i32, i32), BTreeSet<i32>>,
    dirty: HashSet<OctreeSectionPos>,
    // Resident records have cache priority over fresh estimates, including their air.
    restored: HashSet<OctreeSectionPos>,
}

impl OctreeIngestStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn section(&self, pos: OctreeSectionPos) -> Option<&OctreeSection> {
        self.sections.get(&pos)
    }

    /// Every stored section position, for a full re-mesh.
    pub fn section_positions(&self) -> Vec<OctreeSectionPos> {
        self.sections.keys().copied().collect()
    }

    pub fn source_of(&self, chunk: [i32; 2]) -> Option<IngestSource> {
        self.chunk_sources.get(&chunk).copied()
    }

    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    /// World voxel at detail 0 (air when the section is absent).
    pub fn voxel_at(&self, x: i32, y: i32, z: i32) -> Voxel {
        let edge = SECTION_EDGE as i32;
        let pos = OctreeSectionPos::new(
            0,
            x.div_euclid(edge),
            y.div_euclid(edge),
            z.div_euclid(edge),
        );
        self.sections.get(&pos).map_or(Voxel::AIR, |s| {
            s.get(
                x.rem_euclid(edge) as usize,
                y.rem_euclid(edge) as usize,
                z.rem_euclid(edge) as usize,
            )
        })
    }

    /// `true` (and records provenance) when `src` may write into `chunk`.
    fn accept(&mut self, chunk: [i32; 2], src: IngestSource) -> bool {
        match self.chunk_sources.get(&chunk) {
            Some(&existing) if existing > src => false,
            _ => {
                self.chunk_sources.insert(chunk, src);
                true
            }
        }
    }

    fn write(&mut self, pos: OctreeSectionPos, f: impl FnOnce(&mut OctreeSection)) {
        let section = self
            .sections
            .entry(pos)
            .or_insert_with(|| OctreeSection::new(pos));
        let before = section.revision;
        f(section);
        if section.revision != before {
            self.dirty.insert(pos);
        }
    }

    /// Insert a section loaded from disk or the network when nothing newer holds its position.
    /// Not marked dirty: the caller queues it for meshing.
    pub fn adopt_section(&mut self, section: OctreeSection) -> bool {
        match self.sections.entry(section.pos) {
            std::collections::hash_map::Entry::Occupied(_) => false,
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(section);
                true
            }
        }
    }

    pub(crate) fn is_cached(&self, pos: OctreeSectionPos) -> bool {
        self.restored.contains(&pos)
    }

    pub(crate) fn take_dirty(&mut self) -> HashSet<OctreeSectionPos> {
        std::mem::take(&mut self.dirty)
    }

    pub(crate) fn adopt_cached(&mut self, section: OctreeSection) {
        self.restored.insert(section.pos);
        self.replace_section(section);
    }

    /// Replace a section with a newer copy from the server. Not marked dirty.
    pub fn replace_section(&mut self, section: OctreeSection) {
        self.sections.insert(section.pos, section);
    }

    /// Evict resident sections except protected edits. Chunk source priority is per input chunk,
    /// not per section; only section-specific records for actually evicted keys are forgotten.
    pub(crate) fn evict_sections_except(
        &mut self,
        protected: Option<&HashSet<OctreeSectionPos>>,
    ) -> usize {
        let mut evicted = HashSet::new();
        self.sections.retain(|pos, _| {
            if protected.is_some_and(|set| set.contains(pos)) {
                return true;
            }
            evicted.insert(*pos);
            false
        });
        self.dirty.retain(|pos| !evicted.contains(pos));
        self.restored.retain(|pos| !evicted.contains(pos));
        let edge = SECTION_EDGE as i32;
        self.distant_column_ys.retain(|&(x, z), ys| {
            let (sx, sz) = (x.div_euclid(edge), z.div_euclid(edge));
            ys.retain(|&sy| !evicted.contains(&OctreeSectionPos::new(0, sx, sy, sz)));
            !ys.is_empty()
        });
        evicted.len()
    }

    /// Slots a direct estimate can touch, including the previous surface span on a re-sample.
    pub(crate) fn estimate_positions(
        &self,
        detail: u8,
        x: i32,
        z: i32,
        height: i32,
        water: Option<i32>,
        mut need: impl FnMut(OctreeSectionPos),
    ) {
        let sx = x >> (5 + detail);
        let sz = z >> (5 + detail);
        let bottom = height.min(DISTANT_SOLID_FLOOR) >> detail;
        let top = height.max(water.unwrap_or(height)) >> detail;
        let ys = (bottom >> 5)..=((top + SECTION_EDGE as i32) >> 5);
        let previous = self.distant_column_ys.get(&(x, z)).filter(|_| detail == 0);
        for sy in previous.into_iter().flatten().copied().chain(ys) {
            let pos = OctreeSectionPos::new(detail, sx, sy, sz);
            need(pos);
            if detail == 0 {
                std::iter::successors(pos.parent(), |p| p.parent()).for_each(&mut need);
            }
        }
    }

    /// Propagate just the touched estimate columns. Never re-mip a whole resident child over
    /// untouched parent columns, and never replace a cached parent with a fresh estimate.
    pub(crate) fn propagate_estimates(
        &mut self,
        mut columns: EstimateColumns,
        readable: &dyn Fn(OctreeSectionPos) -> bool,
        opacity: &dyn Fn(u32) -> u8,
    ) {
        while !columns.is_empty() {
            let parents: HashSet<_> = columns
                .into_iter()
                .filter_map(|(child, x, z)| {
                    let parent = child.parent()?;
                    Some((
                        parent,
                        ((child.x as usize & 1) << 4) | (x >> 1),
                        ((child.z as usize & 1) << 4) | (z >> 1),
                        child.y & 1,
                    ))
                })
                .collect();
            columns = HashSet::new();
            for (parent, x, z, y_half) in parents {
                if !readable(parent) || self.is_cached(parent) {
                    continue;
                }
                let child = OctreeSectionPos::new(
                    parent.detail - 1,
                    parent.x * 2 + (x >> 4) as i32,
                    parent.y * 2 + y_half,
                    parent.z * 2 + (z >> 4) as i32,
                );
                let Some(section) = self.sections.get(&child) else {
                    continue;
                };
                let (cx, cz) = ((x & 15) * 2, (z & 15) * 2);
                let values: [Voxel; 16] = std::array::from_fn(|y| {
                    mip_voxels(
                        std::array::from_fn(|i| {
                            section.get(cx + (i >> 2), y * 2 + ((i >> 1) & 1), cz + (i & 1))
                        }),
                        opacity,
                    )
                });
                if !self.sections.contains_key(&parent) && values.iter().all(|v| v.is_air()) {
                    continue;
                }
                self.write(parent, |section| {
                    for (y, v) in values.into_iter().enumerate() {
                        section.set(x, (y_half as usize * 16) + y, z, v);
                    }
                });
                columns.insert((parent, x, z));
            }
        }
    }
}

/// A captured vanilla octant. Its extent is authored even when every voxel is air.
/// This is an input edit, not a resident section; it can wait for any number of independent slot probes.
pub(crate) struct ChunkEdit {
    chunk: [i32; 2],
    source: IngestSource,
    octants: Vec<(i32, ChunkOctant)>,
}

enum ChunkOctant {
    Uniform(Voxel),
    Dense(Box<[Voxel]>),
}

impl ChunkEdit {
    /// Convert the immutable snapshot once; retain no raw snapshots in the I/O wait queue.
    pub(crate) fn capture<S: ChunkVoxelSource>(
        chunk: [i32; 2],
        snap: &S,
        source: IngestSource,
        skip_below_floor: bool,
        block_id: &mut dyn FnMut(S::Block) -> u32,
        biome_id: &mut dyn FnMut(&str) -> u16,
    ) -> Self {
        let mut octants = Vec::with_capacity(snap.section_count());
        let mut open = [true; VANILLA_SECTION * VANILLA_SECTION];
        for index in (0..snap.section_count()).rev() {
            let y0 = snap.min_y() + (index * VANILLA_SECTION) as i32;
            if skip_below_floor && y0 + VANILLA_SECTION as i32 <= DISTANT_SOLID_FLOOR {
                continue;
            }
            let uniform_id = snap.section_uniform_block(index).map(&mut *block_id);
            if uniform_id == Some(0) {
                let air = if open.iter().all(|&o| o) {
                    Voxel::air_with_light(15, 0)
                } else {
                    Voxel::AIR
                };
                octants.push((y0, ChunkOctant::Uniform(air)));
                continue;
            }

            let mut biomes = [0u16; QUART * QUART * QUART];
            for qy in 0..QUART {
                for qz in 0..QUART {
                    for qx in 0..QUART {
                        biomes[(qy * QUART + qz) * QUART + qx] = snap
                            .biome(qx * 4, y0 + (qy * 4) as i32, qz * 4)
                            .map_or(0, &mut *biome_id);
                    }
                }
            }
            let mut data = vec![Voxel::AIR; VANILLA_SECTION.pow(3)].into_boxed_slice();
            for ly in (0..VANILLA_SECTION).rev() {
                let y = y0 + ly as i32;
                for lz in 0..VANILLA_SECTION {
                    for lx in 0..VANILLA_SECTION {
                        let id = match uniform_id {
                            Some(id) => id,
                            None => snap.block(lx, y, lz).map_or(0, &mut *block_id),
                        };
                        let (mut sky, blk) = snap.light(lx, y, lz);
                        let column_open = &mut open[lz * VANILLA_SECTION + lx];
                        if id == 0 {
                            if *column_open {
                                sky = 15;
                            }
                        } else {
                            *column_open = false;
                        }
                        let voxel = if id == 0 {
                            Voxel::air_with_light(sky, blk)
                        } else {
                            let biome =
                                biomes[(((ly >> 2) * QUART) + (lz >> 2)) * QUART + (lx >> 2)];
                            Voxel::new(id, biome, sky, blk)
                        };
                        data[(ly * VANILLA_SECTION + lz) * VANILLA_SECTION + lx] = voxel;
                    }
                }
            }
            octants.push((y0, ChunkOctant::Dense(data)));
        }
        Self {
            chunk,
            source,
            octants,
        }
    }

    pub(crate) fn positions(&self) -> impl Iterator<Item = OctreeSectionPos> + '_ {
        self.octants.iter().flat_map(move |&(y, _)| {
            let pos = OctreeSectionPos::new(0, self.chunk[0] >> 1, y >> 5, self.chunk[1] >> 1);
            std::iter::successors(Some(pos), |pos| pos.parent())
        })
    }

    /// The same aligned 16³ input supplies a complete 8³/4³/2³/1³ patch in its ancestors.
    /// Never re-mip unknown siblings (or depend on loading another slot to author this one).
    pub(crate) fn apply(
        self,
        store: &mut OctreeIngestStore,
        readable: &dyn Fn(OctreeSectionPos) -> bool,
        max_detail: u8,
        opacity: &dyn Fn(u32) -> u8,
    ) -> bool {
        if !store.accept(self.chunk, self.source) {
            return false;
        }
        for (y, mut voxels) in self.octants {
            let origin = [self.chunk[0] * 16, y, self.chunk[1] * 16];
            for detail in 0..=max_detail {
                let side = VANILLA_SECTION >> detail;
                let [x, y, z] = origin.map(|v| v >> detail);
                let pos = OctreeSectionPos::new(detail, x >> 5, y >> 5, z >> 5);
                let [ox, oy, oz] = [x, y, z].map(|v| (v & 31) as usize);
                let known = store.sections.contains_key(&pos);
                if readable(pos)
                    && (self.source > IngestSource::Distant || !store.is_cached(pos))
                    && (known
                        || match &voxels {
                            ChunkOctant::Uniform(v) => *v != Voxel::AIR,
                            ChunkOctant::Dense(v) => detail == 0 || v.iter().any(|v| !v.is_air()),
                        })
                {
                    store.write(pos, |section| match &voxels {
                        ChunkOctant::Uniform(v) if detail == 0 => {
                            section.fill_octant(child_index(ox >> 4, oy >> 4, oz >> 4), *v);
                        }
                        ChunkOctant::Uniform(v) => {
                            for dz in 0..side {
                                for dx in 0..side {
                                    section.fill_column(ox + dx, oz + dz, oy, oy + side, *v);
                                }
                            }
                        }
                        ChunkOctant::Dense(data) => {
                            for dy in 0..side {
                                for dz in 0..side {
                                    for dx in 0..side {
                                        section.set(
                                            ox + dx,
                                            oy + dy,
                                            oz + dz,
                                            data[(dy * side + dz) * side + dx],
                                        );
                                    }
                                }
                            }
                        }
                    });
                }
                if detail == max_detail {
                    break;
                }
                voxels = match voxels {
                    ChunkOctant::Uniform(v) => ChunkOctant::Uniform(mip_voxels([v; 8], opacity)),
                    ChunkOctant::Dense(data) => {
                        let next = side / 2;
                        let mut mip = Vec::with_capacity(next.pow(3));
                        for dy in 0..next {
                            for dz in 0..next {
                                for dx in 0..next {
                                    let kids = std::array::from_fn(|i| {
                                        let x = dx * 2 + (i >> 2);
                                        let y = dy * 2 + ((i >> 1) & 1);
                                        let z = dz * 2 + (i & 1);
                                        data[(y * side + z) * side + x]
                                    });
                                    mip.push(mip_voxels(kids, opacity));
                                }
                            }
                        }
                        ChunkOctant::Dense(mip.into_boxed_slice())
                    }
                };
            }
        }
        true
    }
}

/// A surface estimate is one column of voxels at its selected detail. Only detail zero
/// re-samples clear their old spans and participate in the existing upward estimate mips.
pub(crate) fn ingest_estimates(
    store: &mut OctreeIngestStore,
    detail: u8,
    columns: &[DistantColumn],
    water_block: u32,
    subsurface_block: u32,
    readable: &dyn Fn(OctreeSectionPos) -> bool,
) -> (usize, EstimateColumns) {
    let edge = SECTION_EDGE as i32;
    let mut written = 0;
    let mut changed = HashSet::new();
    for col in columns {
        let chunk = [col.x >> 4, col.z >> 4];
        if col.block == 0
            || store
                .source_of(chunk)
                .is_some_and(|s| s > IngestSource::Estimate)
        {
            continue;
        }
        let (vx, vz) = (col.x >> detail, col.z >> detail);
        let (sx, sz) = (vx.div_euclid(edge), vz.div_euclid(edge));
        let (lx, lz) = (vx.rem_euclid(edge) as usize, vz.rem_euclid(edge) as usize);
        let col_key = (col.x, col.z);
        if detail == 0
            && let Some(ys) = store.distant_column_ys.remove(&col_key)
        {
            for sy in ys {
                let pos = OctreeSectionPos::new(detail, sx, sy, sz);
                if readable(pos) && !store.is_cached(pos) {
                    store.write(pos, |s| s.fill_column(lx, lz, 0, SECTION_EDGE, Voxel::AIR));
                    changed.insert((pos, lx, lz));
                }
            }
        }
        let height = col.height >> detail;
        let bottom = height.min(DISTANT_SOLID_FLOOR >> detail);
        let water_top = col.water.map(|y| y >> detail).filter(|&y| y > height);
        let sky_top = water_top.unwrap_or(height) + edge;
        let mut fill = |start: i32, end: i32, voxel: Voxel| {
            if start >= end {
                return;
            }
            for sy in start.div_euclid(edge)..=(end - 1).div_euclid(edge) {
                let pos = OctreeSectionPos::new(detail, sx, sy, sz);
                if !readable(pos) || store.is_cached(pos) {
                    continue;
                }
                let y0 = sy * edge;
                store.write(pos, |section| {
                    section.fill_column(
                        lx,
                        lz,
                        (start - y0).max(0) as usize,
                        (end - y0).min(edge) as usize,
                        voxel,
                    );
                });
                if detail == 0 {
                    changed.insert((pos, lx, lz));
                }
            }
        };
        let subsurface = if subsurface_block == 0 {
            col.block
        } else {
            subsurface_block
        };
        fill(bottom, height, Voxel::new(subsurface, col.biome, 15, 0));
        fill(height, height + 1, Voxel::new(col.block, col.biome, 15, 0));
        if let Some(w) = water_top {
            for y in (height + 1)..=w {
                let sky = 15u8.saturating_sub((w - y).clamp(0, 15) as u8);
                fill(y, y + 1, Voxel::new(water_block, col.biome, sky, 0));
            }
        }
        fill(
            water_top.unwrap_or(height) + 1,
            sky_top + 1,
            Voxel::air_with_light(15, 0),
        );
        if detail == 0 {
            store
                .distant_column_ys
                .entry(col_key)
                .or_default()
                .extend(bottom.div_euclid(edge)..=sky_top.div_euclid(edge));
            store.accept(chunk, IngestSource::Estimate);
        }
        written += 1;
    }
    (written, changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::MAX_OCTREE_DETAIL;
    use std::cell::Cell;

    const STONE: u32 = 1;
    const LOG: u32 = 2;
    const LEAVES: u32 = 3;
    const WATER: u32 = 4;
    const DIRT: u32 = 5;

    #[test]
    fn selective_eviction_forgets_only_the_removed_section_bookkeeping() {
        let mut store = OctreeIngestStore::default();
        let kept = OctreeSectionPos::new(0, -1, 0, 0);
        let dropped = OctreeSectionPos::new(0, -1, 1, 0);
        for pos in [kept, dropped] {
            store.adopt_cached(OctreeSection::new(pos));
            store.dirty.insert(pos);
        }
        store
            .distant_column_ys
            .insert((-1, 0), BTreeSet::from([0, 1]));
        store.distant_column_ys.insert((-2, 0), BTreeSet::from([1]));
        store.distant_column_ys.insert((0, 0), BTreeSet::from([1]));
        store.chunk_sources.insert([-1, 0], IngestSource::Loaded);

        assert_eq!(store.evict_sections_except(Some(&HashSet::from([kept]))), 1);
        assert_eq!(store.section_positions(), vec![kept]);
        assert_eq!(store.dirty, HashSet::from([kept]));
        assert_eq!(store.restored, HashSet::from([kept]));
        assert_eq!(store.distant_column_ys[&(-1, 0)], BTreeSet::from([0]));
        assert!(!store.distant_column_ys.contains_key(&(-2, 0)));
        assert_eq!(store.distant_column_ys[&(0, 0)], BTreeSet::from([1]));
        assert_eq!(store.source_of([-1, 0]), Some(IngestSource::Loaded));
    }

    /// 4 sections tall chunk (y 0..64) backed by a sparse map.
    struct MockChunk {
        blocks: HashMap<(usize, i32, usize), u32>,
        block_calls: Cell<usize>,
        light: (u8, u8),
    }

    impl MockChunk {
        fn new() -> Self {
            Self {
                blocks: HashMap::new(),
                block_calls: Cell::new(0),
                light: (15, 0),
            }
        }
        fn put(&mut self, x: usize, y: i32, z: usize, b: u32) {
            self.blocks.insert((x, y, z), b);
        }
    }

    impl ChunkVoxelSource for MockChunk {
        type Block = u32;
        fn min_y(&self) -> i32 {
            0
        }
        fn section_count(&self) -> usize {
            4
        }
        fn section_uniform_block(&self, index: usize) -> Option<u32> {
            let (lo, hi) = (index as i32 * 16, index as i32 * 16 + 16);
            let any = self.blocks.keys().any(|&(_, y, _)| y >= lo && y < hi);
            if any { None } else { Some(0) }
        }
        fn block(&self, lx: usize, y: i32, lz: usize) -> Option<u32> {
            self.block_calls.set(self.block_calls.get() + 1);
            self.blocks.get(&(lx, y, lz)).copied()
        }
        fn biome(&self, _: usize, _: i32, _: usize) -> Option<&str> {
            Some("minecraft:plains")
        }
        fn light(&self, _: usize, _: i32, _: usize) -> (u8, u8) {
            self.light
        }
    }

    fn ingest(
        store: &mut OctreeIngestStore,
        chunk: [i32; 2],
        c: &MockChunk,
        src: IngestSource,
    ) -> bool {
        ChunkEdit::capture(chunk, c, src, false, &mut |b| b, &mut |_| 7).apply(
            store,
            &|_| true,
            0,
            &|_| 15,
        )
    }

    fn column(x: i32, z: i32, height: i32, water: Option<i32>, block: u32) -> DistantColumn {
        DistantColumn {
            x,
            z,
            height,
            water,
            block,
            biome: 7,
        }
    }

    #[test]
    fn canopy_air_gap_and_ground_survive() {
        let mut c = MockChunk::new();
        for y in 0..40 {
            for x in 0..16 {
                for z in 0..16 {
                    c.put(x, y, z, STONE);
                }
            }
        }
        for y in 40..44 {
            c.put(8, y, 8, LOG);
        }
        for y in 46..49 {
            for x in 6..=10 {
                for z in 6..=10 {
                    c.put(x, y, z, LEAVES);
                }
            }
        }
        let mut store = OctreeIngestStore::new();
        assert!(ingest(&mut store, [0, 0], &c, IngestSource::Loaded));

        assert_eq!(store.voxel_at(8, 10, 8).block_id(), STONE);
        assert_eq!(store.voxel_at(8, 41, 8).block_id(), LOG);
        // Air gap between trunk top and canopy, away from the trunk column.
        assert!(store.voxel_at(6, 44, 8).is_air());
        assert!(store.voxel_at(6, 45, 8).is_air());
        assert_eq!(store.voxel_at(6, 47, 8).block_id(), LEAVES);
        // Air above the canopy carries sky light.
        assert_eq!(store.voxel_at(8, 55, 8).sky_light(), 15);
        // Biome comes from the quart grid.
        assert_eq!(store.voxel_at(8, 10, 8).biome_id(), 7);
        // Chunk (0,0) occupies the +x/+z corner half of octree section (0, y, 0).
        assert!(store.section(OctreeSectionPos::new(0, 0, 0, 0)).is_some());
        assert!(store.section(OctreeSectionPos::new(0, 0, 1, 0)).is_some());
    }

    #[test]
    fn detached_air_below_an_overhang_keeps_snapshot_light() {
        let mut chunk = MockChunk::new();
        chunk.put(0, 55, 0, STONE);
        chunk.light = (4, 7);
        let mut store = OctreeIngestStore::new();
        ChunkEdit::capture(
            [0, 0],
            &chunk,
            IngestSource::Distant,
            true,
            &mut |b| b,
            &mut |_| 7,
        )
        .apply(&mut store, &|_| true, 0, &|_| 15);
        assert_eq!(store.voxel_at(0, 55, 0).block_id(), STONE);
        assert_eq!(store.voxel_at(0, 54, 0), Voxel::air_with_light(4, 7));
        assert_eq!(store.voxel_at(0, 56, 0), Voxel::air_with_light(15, 7));
    }

    #[test]
    fn negative_chunk_maps_to_floor_section() {
        let mut c = MockChunk::new();
        c.put(3, 5, 4, STONE);
        let mut store = OctreeIngestStore::new();
        assert!(ingest(&mut store, [-1, -1], &c, IngestSource::Loaded));
        assert_eq!(store.voxel_at(-16 + 3, 5, -16 + 4).block_id(), STONE);
        assert!(store.section(OctreeSectionPos::new(0, -1, 0, -1)).is_some());
    }

    #[test]
    fn first_and_repeated_open_sky_ingest_keep_light_without_dirtying_on_repeat() {
        use crate::octree::{Face, NeighborBorders};

        let empty = MockChunk::new();
        let mut covered = MockChunk::new();
        covered.put(1, 50, 1, STONE);
        let mut store = OctreeIngestStore::new();
        assert!(ingest(&mut store, [0, 0], &covered, IngestSource::Loaded));
        store.take_dirty();

        // This open-sky chunk is across a section boundary from the covered one.
        let pos = OctreeSectionPos::new(0, -1, 1, 0);
        assert!(ingest(&mut store, [-1, 0], &empty, IngestSource::Loaded));
        assert_eq!(store.dirty_count(), 2);
        let lit = Voxel::air_with_light(15, 0);
        let section = store.section(pos).unwrap();
        assert!(section.is_empty());
        assert_eq!(section.get(31, 0, 0), lit);
        assert_eq!(section.get(31, 0, 16), Voxel::AIR);
        let revision = section.revision;
        let borders = NeighborBorders::from_neighbors(std::array::from_fn(|i| {
            (i == Face::NegX as usize).then_some(section)
        }));
        assert_eq!(borders.get(Face::NegX, 0, 0, 0), Some(lit));
        assert_eq!(borders.get(Face::NegX, 0, 0, 16), Some(Voxel::AIR));
        assert_eq!(store.voxel_at(0, 32, 0), Voxel::AIR);

        store.take_dirty();
        let parent = pos.parent().unwrap();
        // The existing light reducer reads lit input.
        let downsampled = crate::octree::downsample_2x2x2(
            parent,
            &parent.children().unwrap().map(|p| store.section(p)),
            &|_| 15,
        );
        assert_eq!(downsampled.get(24, 16, 0), lit);
        assert!(ingest(&mut store, [-1, 0], &empty, IngestSource::Loaded));
        assert_eq!(empty.block_calls.get(), 0);
        assert_eq!(store.section(pos).unwrap().revision, revision);
        assert_eq!(store.dirty_count(), 0);
        assert!(store.take_dirty().is_empty());
    }

    #[test]
    fn live_snapshot_compacts_equal_sky_octant_from_surface_columns() {
        let mut store = OctreeIngestStore::new();
        let mut columns = Vec::new();
        let mut live = MockChunk::new();
        for z in 0..VANILLA_SECTION {
            for x in 0..VANILLA_SECTION {
                columns.push(column(x as i32, z as i32, 15, None, STONE));
                for y in 0..16 {
                    live.put(x, y, z, STONE);
                }
            }
        }
        ingest_estimates(&mut store, 0, &columns, WATER, STONE, &|_| true).0;
        store.take_dirty();

        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let sky = child_index(0, 1, 0);
        let section = store.section(pos).unwrap();
        assert!(!section.is_uniform_octant(sky));
        let voxels = section.to_dense();
        let revision = section.revision;

        assert!(ingest(&mut store, [0, 0], &live, IngestSource::Loaded));
        let section = store.section(pos).unwrap();
        assert!(section.is_uniform_octant(sky));
        assert_eq!(section.revision, revision);
        assert_eq!(section.to_dense(), voxels);
        // The live fixture has taller sky than the estimate, but did not dirty this section.
        assert!(!store.take_dirty().contains(&pos));

        assert!(ingest(&mut store, [0, 0], &live, IngestSource::Loaded));
        let section = store.section(pos).unwrap();
        assert!(section.is_uniform_octant(sky));
        assert_eq!(section.revision, revision);
        assert_eq!(store.dirty_count(), 0);
    }

    #[test]
    fn uniform_air_clears_stale_blocks_without_sampling_each_voxel() {
        let mut c = MockChunk::new();
        c.put(1, 50, 1, STONE); // only section 3 has data
        let mut store = OctreeIngestStore::new();
        ingest(&mut store, [0, 0], &c, IngestSource::Loaded);
        // One non-empty section scanned; the three air sections cost no per-voxel calls.
        assert_eq!(c.block_calls.get(), VANILLA_SECTION.pow(3));
        assert!(store.section(OctreeSectionPos::new(0, 0, 0, 0)).is_none());
        assert!(store.section(OctreeSectionPos::new(0, 0, 1, 0)).is_some());

        // A later all-air snapshot clears the stale voxel.
        let empty = MockChunk::new();
        ingest(&mut store, [0, 0], &empty, IngestSource::Loaded);
        assert_eq!(store.voxel_at(1, 50, 1), Voxel::air_with_light(15, 0));
        assert_eq!(empty.block_calls.get(), 0);
    }

    #[test]
    fn g1_beats_g3_beats_g4() {
        let mut g1 = MockChunk::new();
        g1.put(2, 10, 2, LOG);
        let mut g3 = MockChunk::new();
        g3.put(2, 10, 2, DIRT);
        let g4 = [column(2, 2, 10, None, STONE)];
        let run_g4 = |store: &mut OctreeIngestStore| {
            ingest_estimates(store, 0, &g4, WATER, STONE, &|_| true).0
        };

        // G4 then G3 then G1: each higher source replaces the lower.
        let mut store = OctreeIngestStore::new();
        assert_eq!(run_g4(&mut store), 1);
        assert_eq!(store.voxel_at(2, 10, 2).block_id(), STONE);
        assert!(ingest(&mut store, [0, 0], &g3, IngestSource::Saved));
        assert_eq!(store.voxel_at(2, 10, 2).block_id(), DIRT);
        assert!(ingest(&mut store, [0, 0], &g1, IngestSource::Loaded));
        assert_eq!(store.voxel_at(2, 10, 2).block_id(), LOG);

        // Lower sources arriving afterwards are refused.
        assert_eq!(run_g4(&mut store), 0);
        assert!(!ingest(&mut store, [0, 0], &g3, IngestSource::Saved));
        assert_eq!(store.voxel_at(2, 10, 2).block_id(), LOG);
        assert_eq!(store.source_of([0, 0]), Some(IngestSource::Loaded));

        // G3 first, then G4 is refused.
        let mut store = OctreeIngestStore::new();
        ingest(&mut store, [0, 0], &g3, IngestSource::Saved);
        assert_eq!(run_g4(&mut store), 0);
        assert_eq!(store.voxel_at(2, 10, 2).block_id(), DIRT);
    }

    #[test]
    fn distant_column_has_water_over_terrain_and_resample_replaces() {
        let mut store = OctreeIngestStore::new();
        let cols = [column(40, 5, 20, Some(24), DIRT)];
        ingest_estimates(&mut store, 0, &cols, WATER, STONE, &|_| true).0;
        assert_eq!(store.voxel_at(40, 20, 5).block_id(), DIRT);
        // Solids carry the surface's sky light: a face beside a leaf or grass voxel reads it.
        assert_eq!(store.voxel_at(40, 20, 5).sky_light(), 15);
        assert_eq!(store.voxel_at(40, 0, 5).block_id(), STONE);
        assert!(store.voxel_at(40, -1, 5).is_air());
        assert_eq!(store.voxel_at(40, 24, 5).block_id(), WATER);
        assert_eq!(store.voxel_at(40, 24, 5).sky_light(), 15);
        assert_eq!(store.voxel_at(40, 21, 5).sky_light(), 12);
        assert_eq!(store.voxel_at(40, 25, 5).sky_light(), 15);

        // Terrain re-sampled lower and dry: old water and stone are gone.
        let cols = [column(40, 5, 18, None, STONE)];
        ingest_estimates(&mut store, 0, &cols, WATER, STONE, &|_| true).0;
        assert_eq!(store.voxel_at(40, 18, 5).block_id(), STONE);
        assert!(store.voxel_at(40, 20, 5).is_air());
        assert!(store.voxel_at(40, 24, 5).is_air());
    }

    #[test]
    fn batched_ingest_propagates_once_without_duplicate_ancestors() {
        let mut c = MockChunk::new();
        c.put(1, 50, 1, STONE);
        let mut store = OctreeIngestStore::new();
        // Chunks 0 and 2 land in sibling sections (x = 0 and x = 1) that share every ancestor.
        for chunk in [[0, 0], [2, 0]] {
            assert!(
                ChunkEdit::capture(
                    chunk,
                    &c,
                    IngestSource::Loaded,
                    false,
                    &mut |b| b,
                    &mut |_| 7
                )
                .apply(&mut store, &|_| true, MAX_OCTREE_DETAIL, &|_| 15)
            );
        }
        let touched = store.take_dirty();
        assert_eq!(store.dirty_count(), 0);
        assert!(touched.contains(&OctreeSectionPos::new(1, 0, 0, 0)));
        assert!(touched.iter().filter(|p| p.detail == 0).count() >= 2);
        for detail in 1..=MAX_OCTREE_DETAIL {
            assert_eq!(
                touched.iter().filter(|p| p.detail == detail).count(),
                1,
                "sibling sections share one ancestor at detail {detail}"
            );
        }
        // A second propagation has nothing left to do.
        assert!(store.take_dirty().is_empty());
    }

    #[test]
    fn dirty_sections_propagate_to_every_parent_level() {
        let mut c = MockChunk::new();
        c.put(1, 1, 1, STONE);
        let mut store = OctreeIngestStore::new();
        ChunkEdit::capture(
            [0, 0],
            &c,
            IngestSource::Loaded,
            false,
            &mut |b| b,
            &mut |_| 7,
        )
        .apply(&mut store, &|_| true, MAX_OCTREE_DETAIL, &|_| 15);
        assert!(store.dirty_count() > 0);
        let touched = store.take_dirty();
        assert_eq!(store.dirty_count(), 0);
        for detail in 1..=MAX_OCTREE_DETAIL {
            assert!(
                touched.iter().any(|p| p.detail == detail),
                "no parent at detail {detail}"
            );
        }
        let top = store
            .section(OctreeSectionPos::new(MAX_OCTREE_DETAIL, 0, 0, 0))
            .expect("top-level parent");
        assert!(!top.is_empty());
    }
}
