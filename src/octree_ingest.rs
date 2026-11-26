//! Ingestion of server-side terrain sources into 32³ [`OctreeSection`]s.
//!
//! Sources, in priority order (a lower source never overwrites a higher one in the same chunk):
//! - G1 [`IngestSource::Loaded`]: a loaded chunk's `ChunkSnapshot` (full 3D),
//! - G3 [`IngestSource::Saved`]: a saved Anvil chunk read through `SavedChunkReader`, which hands
//!   back the same `ChunkSnapshot` type and therefore takes the same section path,
//! - G4 [`IngestSource::Distant`]: `DetachedChunkGenerator::sample_surface_columns` surface and
//!   water columns for terrain nobody has visited.
//!
//! Layout references (Distant Horizons `FullDataSourceV2`, under
//! `coreSubProjects/core/src/main/java/com/seibel/distanthorizons/core/dataObjects/fullData/sources/`):
//! - `FullDataSourceV2.java:77` `WIDTH = 64` columns per DH data source; Voxy-style octree
//!   sections are 32³ instead (`octree::SECTION_EDGE`), so a DH source maps onto 2×2 sections
//!   horizontally and the 16×16×16 vanilla chunk sections used here nest exactly 2×2×2 inside one.
//! - `FullDataSourceV2.java:605-616` `downsampleFromOneDetailLevelBelow`: a 2×2 block of input
//!   columns becomes one parent column; parents are rebuilt from their children. The 3D
//!   equivalent is [`downsample_2x2x2`], which [`OctreeIngestStore::propagate_dirty`] runs up to
//!   `MAX_OCTREE_DETAIL`.
//! - `FullDataSourceV2.java:1105-1161` (`updateFromOneDetailLevelAbove`) only fills a column that
//!   is still empty, so lower-detail data never replaces real data; the source priority here
//!   keeps the same guarantee for chunk-level provenance.
//!
//! Empty (uniform-air) vanilla sections are skipped in O(1) through
//! [`ChunkVoxelSource::section_uniform_block`]; the target region is only cleared when a section
//! already exists there (a block broken since the last ingest must not leave stale voxels).

use crate::octree::{
    MAX_OCTREE_DETAIL, OctreeSection, OctreeSectionPos, SECTION_EDGE, Voxel, downsample_2x2x2,
};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

/// Edge length of a vanilla chunk section in blocks.
const VANILLA_SECTION: usize = 16;
/// Vanilla biome storage is a 4×4×4 quart grid per section.
const QUART: usize = 4;
/// World-Y floor down to which G4 surface columns fill solid subsurface rock so steep mountains
/// never leave hollow sky gaps underneath and coarse LODs never hang stalactites below `y = 0`
/// (`vy = 0` at every octree detail level; greedy meshing culls all interior faces).
pub const DISTANT_SOLID_FLOOR: i32 = 0;

/// Where a chunk's voxels came from. The derived order is the priority order: `Loaded` wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IngestSource {
    /// G4 `DetachedChunkGenerator` surface columns.
    Distant = 1,
    /// G3 `SavedChunkReader` Anvil chunk.
    Saved = 2,
    /// G1 live `ChunkSnapshot`.
    Loaded = 3,
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

// NOTE: the section accessors used here (`section_count`, `section`, `SectionSnapshot::is_empty`
// and `uniform_block`) follow the names in the G1 task brief; the SDK source was not available
// in the authoring box, so they are unverified. Per-voxel access uses the signatures already used
// in `lib.rs::sample_loaded_chunks`.
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
pub struct DistantColumn<'a, B> {
    pub x: i32,
    pub z: i32,
    /// Highest solid block Y.
    pub height: i32,
    /// Water surface Y when the column is submerged.
    pub water: Option<i32>,
    pub block: B,
    pub biome: &'a str,
}

/// Detail-0 sections, chunk provenance and the dirty set awaiting mip propagation.
#[derive(Default)]
pub struct OctreeIngestStore {
    sections: HashMap<OctreeSectionPos, OctreeSection>,
    chunk_sources: HashMap<[i32; 2], IngestSource>,
    /// Section Ys holding G4 voxels per block column, so a re-sample can clear its old column.
    distant_column_ys: HashMap<(i32, i32), BTreeSet<i32>>,
    dirty: HashSet<OctreeSectionPos>,
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

    fn set_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel) {
        let edge = SECTION_EDGE as i32;
        let pos = OctreeSectionPos::new(
            0,
            x.div_euclid(edge),
            y.div_euclid(edge),
            z.div_euclid(edge),
        );
        let (lx, ly, lz) = (
            x.rem_euclid(edge) as usize,
            y.rem_euclid(edge) as usize,
            z.rem_euclid(edge) as usize,
        );
        if voxel.is_air() && voxel == Voxel::AIR && !self.sections.contains_key(&pos) {
            return;
        }
        self.write(pos, |s| {
            s.set(lx, ly, lz, voxel);
        });
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

    /// Replace a section with a newer copy from the server. Not marked dirty.
    pub fn replace_section(&mut self, section: OctreeSection) {
        self.sections.insert(section.pos, section);
    }

    /// G4 at a coarse step: a sample covering `2^level` blocks is exactly one voxel of a level-`level`
    /// section, so it is written there directly (no detail-0 expansion, no mip propagation: the
    /// generator supplies every level itself). Columns whose chunk a loaded or saved source owns
    /// are skipped. Returns the sections written.
    pub fn ingest_coarse_columns<B: Copy + Eq + Hash>(
        &mut self,
        level: u8,
        columns: &[DistantColumn<'_, B>],
        water_block: u32,
        subsurface_block: u32,
        block_id: &mut dyn FnMut(B) -> u32,
        biome_id: &mut dyn FnMut(&str) -> u16,
    ) -> Vec<OctreeSectionPos> {
        let edge = SECTION_EDGE as i32;
        let mut touched = HashSet::new();
        let mut fill_span = |store: &mut Self, vx: i32, vz: i32, y0: i32, y1: i32, voxel: Voxel| {
            if y0 >= y1 {
                return;
            }
            let (sx, sz) = (vx.div_euclid(edge), vz.div_euclid(edge));
            let (lx, lz) = (vx.rem_euclid(edge) as usize, vz.rem_euclid(edge) as usize);
            let sy0 = y0.div_euclid(edge);
            let sy1 = (y1 - 1).div_euclid(edge);
            for sy in sy0..=sy1 {
                let pos = OctreeSectionPos::new(level, sx, sy, sz);
                let sec_y0 = sy * edge;
                let ly0 = (y0 - sec_y0).max(0) as usize;
                let ly1 = (y1 - sec_y0).min(edge) as usize;
                let section = store
                    .sections
                    .entry(pos)
                    .or_insert_with(|| OctreeSection::new(pos));
                section.fill_column(lx, lz, ly0, ly1, voxel);
                touched.insert(pos);
            }
        };
        for col in columns {
            if self
                .chunk_sources
                .get(&[col.x >> 4, col.z >> 4])
                .is_some_and(|&s| s > IngestSource::Distant)
            {
                continue;
            }
            let id = block_id(col.block);
            if id == 0 {
                continue;
            }
            let sub_id = if subsurface_block != 0 {
                subsurface_block
            } else {
                id
            };
            let biome = biome_id(col.biome);
            let (vx, vz) = (col.x >> level, col.z >> level);
            let height = col.height >> level;
            let bottom = height.min(DISTANT_SOLID_FLOOR >> level);
            let water_top = col.water.map(|w| w >> level).filter(|&w| w > height);
            let top = water_top.unwrap_or(height);
            fill_span(
                self,
                vx,
                vz,
                bottom,
                height,
                Voxel::new(sub_id, biome, 15, 0),
            );
            fill_span(
                self,
                vx,
                vz,
                height,
                height + 1,
                Voxel::new(id, biome, 15, 0),
            );
            if let Some(w) = water_top {
                for y in (height + 1)..=w {
                    let sky = 15u8.saturating_sub((w - y).clamp(0, 15) as u8);
                    fill_span(
                        self,
                        vx,
                        vz,
                        y,
                        y + 1,
                        Voxel::new(water_block, biome, sky, 0),
                    );
                }
            }
            // Open sky above the surface (see `ingest_distant_columns`).
            fill_span(
                self,
                vx,
                vz,
                top + 1,
                top + edge + 1,
                Voxel::air_with_light(15, 0),
            );
        }
        touched.into_iter().collect()
    }

    /// Rebuild every ancestor of the dirty detail-0 sections up to `MAX_OCTREE_DETAIL` through
    /// `downsample_2x2x2` (`FullDataSourceV2.java:605-616`). Returns every touched position, the
    /// dirty detail-0 sections first. `opacity` is `Mapper.getBlockStateOpacity`.
    pub fn propagate_dirty(&mut self, opacity: &dyn Fn(u32) -> u8) -> Vec<OctreeSectionPos> {
        let mut level: BTreeSet<OctreeSectionPos> =
            std::mem::take(&mut self.dirty).into_iter().collect();
        let mut touched: Vec<OctreeSectionPos> = level.iter().copied().collect();
        for _ in 1..=MAX_OCTREE_DETAIL {
            let parents: BTreeSet<OctreeSectionPos> =
                level.iter().filter_map(|p| p.parent()).collect();
            for parent in &parents {
                let Some(kids) = parent.children() else {
                    continue;
                };
                let merged = {
                    let refs: [Option<&OctreeSection>; 8] = kids.map(|k| self.sections.get(&k));
                    downsample_2x2x2(*parent, &refs, opacity)
                };
                if merged.is_empty() && !self.sections.contains_key(parent) {
                    continue;
                }
                self.sections.insert(*parent, merged);
            }
            touched.extend(parents.iter().copied());
            level = parents;
        }
        touched
    }
}

/// Ingest one chunk (G1 when `src == Loaded`, G3 when `Saved`) at chunk coordinate `chunk`.
/// Returns `false` without writing when a higher-priority source already owns the chunk.
///
/// `block_id` maps a block handle to its Mapper id (`0` = air), `biome_id` maps a biome
/// identifier. Every 16³ vanilla section lands in exactly one 32³ octree section.
pub fn ingest_chunk<S: ChunkVoxelSource>(
    store: &mut OctreeIngestStore,
    chunk: [i32; 2],
    snap: &S,
    src: IngestSource,
    block_id: &mut dyn FnMut(S::Block) -> u32,
    biome_id: &mut dyn FnMut(&str) -> u16,
) -> bool {
    if !store.accept(chunk, src) {
        return false;
    }
    let edge = SECTION_EDGE as i32;
    let (ox, oz) = (chunk[0] * 16, chunk[1] * 16);
    let (sx, sz) = (ox.div_euclid(edge), oz.div_euclid(edge));
    let (off_x, off_z) = (ox.rem_euclid(edge) as usize, oz.rem_euclid(edge) as usize);

    // Top-down: air above a column's highest block sees the open sky, whatever lighting says (a
    // chunk that is not lit yet reads 0, which draws every face beside it black).
    let mut open = [true; VANILLA_SECTION * VANILLA_SECTION];
    for index in (0..snap.section_count()).rev() {
        let y0 = snap.min_y() + (index * VANILLA_SECTION) as i32;
        let pos = OctreeSectionPos::new(0, sx, y0.div_euclid(edge), sz);
        let off_y = y0.rem_euclid(edge) as usize;

        let uniform = snap.section_uniform_block(index);
        let uniform_id = uniform.map(&mut *block_id);
        if uniform_id == Some(0) {
            // Uniform air: nothing to write; clear only where earlier data exists.
            let sky_open = open.iter().all(|&o| o);
            if sky_open || store.sections.contains_key(&pos) {
                let air = if sky_open {
                    Voxel::air_with_light(15, 0)
                } else {
                    Voxel::AIR
                };
                store.write(pos, |s| {
                    for ly in 0..VANILLA_SECTION {
                        for lz in 0..VANILLA_SECTION {
                            for lx in 0..VANILLA_SECTION {
                                s.set(off_x + lx, off_y + ly, off_z + lz, air);
                            }
                        }
                    }
                });
            }
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

        store.write(pos, |s| {
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
                        s.set(off_x + lx, off_y + ly, off_z + lz, voxel);
                    }
                }
            }
        });
    }
    true
}

/// G1: ingest a loaded chunk's `ChunkSnapshot`.
pub fn ingest_chunk_snapshot<S: ChunkVoxelSource>(
    store: &mut OctreeIngestStore,
    chunk: [i32; 2],
    snap: &S,
    block_id: &mut dyn FnMut(S::Block) -> u32,
    biome_id: &mut dyn FnMut(&str) -> u16,
) -> bool {
    ingest_chunk(store, chunk, snap, IngestSource::Loaded, block_id, biome_id)
}

/// G3: ingest a saved Anvil chunk read through `SavedChunkReader` (also a `ChunkSnapshot`).
pub fn ingest_saved_chunk<S: ChunkVoxelSource>(
    store: &mut OctreeIngestStore,
    chunk: [i32; 2],
    snap: &S,
    block_id: &mut dyn FnMut(S::Block) -> u32,
    biome_id: &mut dyn FnMut(&str) -> u16,
) -> bool {
    ingest_chunk(store, chunk, snap, IngestSource::Saved, block_id, biome_id)
}

/// G4: ingest `DetachedChunkGenerator::sample_surface_columns` results. Each column fills solid
/// subsurface rock (`subsurface_block`, or `col.block` when `0`) from `DISTANT_SOLID_FLOOR` up to
/// `height - 1`, `col.block` at `height`, the water span above it (sky light attenuated one level
/// per block of depth, as vanilla water dampens light by 1), and a section's height of lit air on
/// top so the top face and the coarser levels have a light source. Columns whose chunk is owned by
/// G1 or G3 are skipped. `water_block` is the Mapper id of water. Returns the columns written.
pub fn ingest_distant_columns<B: Copy + Eq + Hash>(
    store: &mut OctreeIngestStore,
    columns: &[DistantColumn<'_, B>],
    water_block: u32,
    subsurface_block: u32,
    block_id: &mut dyn FnMut(B) -> u32,
    biome_id: &mut dyn FnMut(&str) -> u16,
) -> usize {
    let edge = SECTION_EDGE as i32;
    let mut written = 0;
    for col in columns {
        if !store.accept([col.x >> 4, col.z >> 4], IngestSource::Distant) {
            continue;
        }
        let id = block_id(col.block);
        if id == 0 {
            continue;
        }
        let sub_id = if subsurface_block != 0 {
            subsurface_block
        } else {
            id
        };
        let biome = biome_id(col.biome);
        let (sx, sz) = (col.x.div_euclid(edge), col.z.div_euclid(edge));
        let (lx, lz) = (
            col.x.rem_euclid(edge) as usize,
            col.z.rem_euclid(edge) as usize,
        );

        // Drop this column's previous G4 voxels.
        if let Some(ys) = store.distant_column_ys.remove(&(col.x, col.z)) {
            for sy in ys {
                store.write(OctreeSectionPos::new(0, sx, sy, sz), |s| {
                    s.fill_column(lx, lz, 0, SECTION_EDGE, Voxel::AIR);
                });
            }
        }

        let bottom = col.height.min(DISTANT_SOLID_FLOOR);
        let water_top = col.water.filter(|&w| w > col.height);
        let top = water_top.unwrap_or(col.height);
        let mut fill_d0 = |y0: i32, y1: i32, voxel: Voxel| {
            if y0 >= y1 {
                return;
            }
            for sy in y0.div_euclid(edge)..=(y1 - 1).div_euclid(edge) {
                let sec_y0 = sy * edge;
                let ly0 = (y0 - sec_y0).max(0) as usize;
                let ly1 = (y1 - sec_y0).min(edge) as usize;
                store.write(OctreeSectionPos::new(0, sx, sy, sz), |s| {
                    s.fill_column(lx, lz, ly0, ly1, voxel);
                });
            }
        };
        fill_d0(bottom, col.height, Voxel::new(sub_id, biome, 15, 0));
        fill_d0(col.height, col.height + 1, Voxel::new(id, biome, 15, 0));
        if let Some(w) = water_top {
            for y in (col.height + 1)..=w {
                let sky = 15u8.saturating_sub((w - y).clamp(0, 15) as u8);
                fill_d0(y, y + 1, Voxel::new(water_block, biome, sky, 0));
            }
        }
        // Open sky above the surface: coarser levels average the air's light, so one lit voxel
        // would leave distant ground dark.
        let sky_top = top + edge;
        fill_d0(top + 1, sky_top + 1, Voxel::air_with_light(15, 0));

        let ys = store.distant_column_ys.entry((col.x, col.z)).or_default();
        for y in bottom..=sky_top {
            ys.insert(y.div_euclid(edge));
        }
        written += 1;
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const STONE: u32 = 1;
    const LOG: u32 = 2;
    const LEAVES: u32 = 3;
    const WATER: u32 = 4;
    const DIRT: u32 = 5;

    /// 4 sections tall chunk (y 0..64) backed by a sparse map.
    struct MockChunk {
        blocks: HashMap<(usize, i32, usize), u32>,
        block_calls: Cell<usize>,
    }

    impl MockChunk {
        fn new() -> Self {
            Self {
                blocks: HashMap::new(),
                block_calls: Cell::new(0),
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
            (15, 0)
        }
    }

    fn ingest(
        store: &mut OctreeIngestStore,
        chunk: [i32; 2],
        c: &MockChunk,
        src: IngestSource,
    ) -> bool {
        ingest_chunk(store, chunk, c, src, &mut |b| b, &mut |_| 7)
    }

    fn column(
        x: i32,
        z: i32,
        height: i32,
        water: Option<i32>,
        block: u32,
    ) -> DistantColumn<'static, u32> {
        DistantColumn {
            x,
            z,
            height,
            water,
            block,
            biome: "minecraft:plains",
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
    fn negative_chunk_maps_to_floor_section() {
        let mut c = MockChunk::new();
        c.put(3, 5, 4, STONE);
        let mut store = OctreeIngestStore::new();
        assert!(ingest(&mut store, [-1, -1], &c, IngestSource::Loaded));
        assert_eq!(store.voxel_at(-16 + 3, 5, -16 + 4).block_id(), STONE);
        assert!(store.section(OctreeSectionPos::new(0, -1, 0, -1)).is_some());
    }

    #[test]
    fn uniform_air_sections_are_skipped_in_o1() {
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
        assert!(store.voxel_at(1, 50, 1).is_air());
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
            ingest_distant_columns(store, &g4, WATER, STONE, &mut |b| b, &mut |_| 7)
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
        ingest_distant_columns(&mut store, &cols, WATER, STONE, &mut |b| b, &mut |_| 3);
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
        ingest_distant_columns(&mut store, &cols, WATER, STONE, &mut |b| b, &mut |_| 3);
        assert_eq!(store.voxel_at(40, 18, 5).block_id(), STONE);
        assert!(store.voxel_at(40, 20, 5).is_air());
        assert!(store.voxel_at(40, 24, 5).is_air());
    }

    #[test]
    fn dirty_sections_propagate_to_every_parent_level() {
        let mut c = MockChunk::new();
        c.put(1, 1, 1, STONE);
        let mut store = OctreeIngestStore::new();
        ingest(&mut store, [0, 0], &c, IngestSource::Loaded);
        assert!(store.dirty_count() > 0);
        let touched = store.propagate_dirty(&|_| 15);
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
