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

use crate::lod::{SECTION_SIZE, SectionPos};
use crate::octree::{
    MAX_OCTREE_DETAIL, OctreeSection, OctreeSectionPos, SECTION_EDGE, Voxel, child_index,
    downsample_2x2x2_onto, mip_voxels,
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

/// A detached surface query result, including the position when no surface was returned.
pub(crate) enum DistantSample<'a, B> {
    Surface(DistantColumn<'a, B>),
    NoSurface(i32, i32),
}

/// Whether the entire section lies outside the dimension's half-open block-Y range.
pub fn is_out_of_build_height(pos: OctreeSectionPos, bounds: (i32, i32)) -> bool {
    let size = (SECTION_EDGE as i64) << pos.detail;
    outside_y_bounds(i64::from(pos.y) * size, size, bounds)
}

fn outside_y_bounds(y0: i64, size: i64, (min_y, max_y): (i32, i32)) -> bool {
    y0 + size <= i64::from(min_y) || y0 >= i64::from(max_y)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompleteKind {
    Direct,
    Mip,
}

/// Octree sections, chunk provenance and the dirty set awaiting mip propagation.
#[derive(Default)]
pub struct OctreeIngestStore {
    sections: HashMap<OctreeSectionPos, OctreeSection>,
    chunk_sources: HashMap<[i32; 2], IngestSource>,
    /// Sampled 16³ slabs per detail-0 section, in octant order (`child_index`).
    covered_subchunks: HashMap<OctreeSectionPos, u8>,
    y_bounds: Option<(i32, i32)>,
    complete_sections: HashMap<OctreeSectionPos, CompleteKind>,
    /// Coarse jobs track x-fastest resolved columns, independently of 3D slab coverage.
    partial_column_stacks: HashMap<(u8, i32, i32), Box<[u64; 16]>>,
    /// Session-only scheduling state, never persisted to disk. Cleared by
    /// `invalidate_coarse_target` and by dimension clear/unload.
    terminal_coarse_targets: HashSet<SectionPos>,
    /// Section Ys holding G4 voxels per block column, so a re-sample can clear its old column.
    distant_column_ys: HashMap<(i32, i32), BTreeSet<i32>>,
    dirty: HashSet<OctreeSectionPos>,
}

pub(crate) fn coarse_cell_covered(coverage: &[[u64; 16]; 4], x: usize, z: usize) -> bool {
    let stack = x / SECTION_EDGE + 2 * (z / SECTION_EDGE);
    let bit = x % SECTION_EDGE + (z % SECTION_EDGE) * SECTION_EDGE;
    coverage[stack][bit / 64] & (1 << (bit % 64)) != 0
}

impl OctreeIngestStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn section(&self, pos: OctreeSectionPos) -> Option<&OctreeSection> {
        self.sections.get(&pos)
    }

    /// Set the dimension's build height as a half-open range in world blocks.
    pub fn set_y_bounds(&mut self, bounds: (i32, i32)) {
        if self.y_bounds.replace(bounds) == Some(bounds) {
            return;
        }
        let finished: Vec<_> = self
            .partial_column_stacks
            .iter()
            .filter(|(_, words)| words.iter().all(|&word| word == u64::MAX))
            .map(|(&stack, _)| stack)
            .collect();
        for stack in finished {
            self.promote_column_stack(stack);
        }
    }

    fn promote_column_stack(&mut self, stack @ (detail, x, z): (u8, i32, i32)) {
        let mut promote = |pos| {
            self.complete_sections.insert(pos, CompleteKind::Direct);
            self.dirty.insert(pos);
        };
        if let Some((lo, hi)) = self.y_bounds {
            let span = (SECTION_EDGE as i32) << detail;
            for y in lo.div_euclid(span)..=(hi - 1).div_euclid(span) {
                promote(OctreeSectionPos::new(detail, x, y, z));
            }
            self.partial_column_stacks.remove(&stack);
        } else {
            for &pos in self.sections.keys() {
                if (pos.detail, pos.x, pos.z) == stack {
                    promote(pos);
                }
            }
        }
    }

    pub fn y_bounds(&self) -> Option<(i32, i32)> {
        self.y_bounds
    }

    /// Whether this section's own bytes are complete, independent of descendant residency.
    pub fn is_section_complete(&self, pos: OctreeSectionPos) -> bool {
        self.complete_sections.contains_key(&pos)
    }

    fn out_of_build_height_subchunk_mask(&self, pos: OctreeSectionPos) -> u8 {
        let Some(bounds) = self.y_bounds else {
            return 0;
        };
        let y0 = i64::from(pos.y) * SECTION_EDGE as i64;
        let size = VANILLA_SECTION as i64;
        (u8::from(outside_y_bounds(y0, size, bounds)) * 0x0f)
            | (u8::from(outside_y_bounds(y0 + size, size, bounds)) * 0xf0)
    }

    /// Stored sections and meshless proven air, for a full re-mesh.
    pub fn section_positions(&self) -> Vec<OctreeSectionPos> {
        self.sections
            .keys()
            .chain(
                self.complete_sections
                    .keys()
                    .filter(|pos| !self.sections.contains_key(pos)),
            )
            .copied()
            .collect()
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

    fn write(
        &mut self,
        pos: OctreeSectionPos,
        preserve_complete: bool,
        f: impl FnOnce(&mut OctreeSection),
    ) {
        let lost_proof = !preserve_complete && self.complete_sections.remove(&pos).is_some();
        let section = self
            .sections
            .entry(pos)
            .or_insert_with(|| OctreeSection::new(pos));
        let before = section.revision;
        f(section);
        if section.revision != before || lost_proof {
            self.dirty.insert(pos);
        }
    }

    /// Insert a disk record when nothing newer holds its position.
    /// Not marked dirty: the caller queues it for meshing.
    pub fn adopt_section(&mut self, section: OctreeSection, complete: bool) -> bool {
        let pos = section.pos;
        if self.sections.contains_key(&pos)
            || self.complete_sections.contains_key(&pos)
            || (pos.detail >= 1
                && self
                    .partial_column_stacks
                    .get(&(pos.detail, pos.x, pos.z))
                    .is_some_and(|words| words.iter().all(|&word| word == u64::MAX)))
        {
            return false;
        }
        if complete {
            self.complete_sections.insert(pos, CompleteKind::Direct);
        }
        self.sections.insert(pos, section);
        true
    }

    /// Replace a section with a newer copy from the server, which carries no coverage proof.
    /// Not marked dirty: the caller queues it for meshing.
    pub fn replace_section(&mut self, section: OctreeSection) {
        let pos = section.pos;
        self.complete_sections.remove(&pos);
        self.covered_subchunks.remove(&pos);
        self.partial_column_stacks
            .remove(&(pos.detail, pos.x, pos.z));
        self.sections.insert(pos, section);
    }

    pub(crate) fn mark_coarse_target_unavailable(&mut self, pos: SectionPos) {
        self.terminal_coarse_targets.insert(pos);
    }

    pub(crate) fn invalidate_coarse_target(&mut self, pos: SectionPos) -> bool {
        self.terminal_coarse_targets.remove(&pos)
    }

    /// Column coverage of the four horizontal octree stacks under one scheduler leaf,
    /// in x-fastest order. A full-height set of proven rows also covers the stack.
    pub(crate) fn coarse_coverage(&self, pos: SectionPos) -> [[u64; 16]; 4] {
        let detail = pos.detail;
        let span = (SECTION_EDGE as i32) << detail;
        std::array::from_fn(|i| {
            let x = pos.x * 2 + (i % 2) as i32;
            let z = pos.z * 2 + (i / 2) as i32;
            if self.y_bounds.is_some_and(|(lo, hi)| {
                (lo.div_euclid(span)..=(hi - 1).div_euclid(span))
                    .all(|y| self.is_section_complete(OctreeSectionPos::new(detail, x, y, z)))
            }) {
                [u64::MAX; 16]
            } else {
                self.partial_column_stacks
                    .get(&(detail, x, z))
                    .map_or([0; 16], |words| **words)
            }
        })
    }

    pub(crate) fn coarse_cells_needed(&self, pos: SectionPos) -> Vec<(usize, usize)> {
        if self.terminal_coarse_targets.contains(&pos) {
            return Vec::new();
        }
        let coverage = self.coarse_coverage(pos);
        if coverage == [[u64::MAX; 16]; 4] {
            return Vec::new();
        }
        (0..SECTION_SIZE)
            .flat_map(|z| (0..SECTION_SIZE).map(move |x| (x, z)))
            .filter(|&(x, z)| !coarse_cell_covered(&coverage, x, z))
            .collect()
    }

    /// Write each G4 sample as one voxel column at its target detail, without expanding it to
    /// detail 0. A loaded or saved chunk supplies the columns entirely within its footprint,
    /// even when the corresponding G4 sample is absent. Returns whether a column cannot
    /// be sampled from the available sources at this detail.
    pub(crate) fn ingest_coarse_columns<'a, B>(
        &mut self,
        level: u8,
        columns: impl IntoIterator<Item = DistantSample<'a, B>>,
        (water_block, subsurface_block): (u32, u32),
        block_id: &mut dyn FnMut(B) -> u32,
        biome_id: &mut dyn FnMut(&str) -> u16,
        opacity: &dyn Fn(u32) -> u8,
    ) -> bool {
        let mut unavailable = false;
        let edge = SECTION_EDGE as i32;
        let bounds = self
            .y_bounds
            .map(|(lo, hi)| (lo >> level, (hi - 1) >> level));
        // Without build height, limit coverage to rows backed by voxel data. Still clear older
        // terrain in those rows so an updated coarse sample describes the entire column there.
        let mut observed = HashMap::<(i32, i32), (i32, i32)>::new();
        if bounds.is_none() {
            for pos in self.sections.keys().filter(|p| p.detail == level) {
                let lo = pos.y * edge;
                let hi = lo + edge - 1;
                let span = observed.entry((pos.x, pos.z)).or_insert((lo, hi));
                *span = (span.0.min(lo), span.1.max(hi));
            }
        }
        for column in columns {
            let (bx, bz, sample) = match column {
                DistantSample::Surface(col) => (col.x, col.z, Some(col)),
                DistantSample::NoSurface(x, z) => (x, z, None),
            };
            let owned = matches!(
                self.source_of([bx >> 4, bz >> 4]),
                Some(IngestSource::Loaded | IngestSource::Saved)
            );
            let (vx, vz) = (bx >> level, bz >> level);
            let (sx, sz) = (vx.div_euclid(edge), vz.div_euclid(edge));
            let (lx, lz) = (vx.rem_euclid(edge) as usize, vz.rem_euclid(edge) as usize);
            if owned {
                // At supported details the aligned voxel fits in one chunk. Loaded/saved
                // snapshots cover its full height. Reuse the same mip rule on its D0 samples:
                // normal propagation may have retained a cached octant instead of replacing
                // it from a dirty but incomplete D0 section.
                if level > MAX_OCTREE_DETAIL {
                    // The legacy scheduler can request wider footprints than the octree
                    // supports. A single authoritative chunk cannot supply these columns.
                    unavailable = true;
                    continue;
                }
                let Some((lo, hi)) = bounds else { continue };
                let (source_x, source_z) = (bx.div_euclid(edge), bz.div_euclid(edge));
                let x = bx.rem_euclid(edge) as usize >> level;
                let z = bz.rem_euclid(edge) as usize >> level;
                for sy in lo.div_euclid(edge)..=hi.div_euclid(edge) {
                    let pos = OctreeSectionPos::new(level, sx, sy, sz);
                    let base = sy * edge;
                    let first = lo.max(base);
                    let last = hi.min(base + edge - 1);
                    let mut voxels = [Voxel::AIR; SECTION_EDGE];
                    let start = (first - base) as usize;
                    let end = (last - base) as usize + 1;
                    for (y, voxel) in (first..=last).zip(&mut voxels[start..end]) {
                        let by = y << level;
                        let source =
                            OctreeSectionPos::new(0, source_x, by.div_euclid(edge), source_z);
                        if let Some(section) = self.section(source) {
                            *voxel = Self::mip_voxel(
                                section,
                                level,
                                x,
                                by.rem_euclid(edge) as usize >> level,
                                z,
                                opacity,
                            );
                        }
                    }
                    if self.sections.contains_key(&pos)
                        || voxels[start..end].iter().any(|&v| v != Voxel::AIR)
                    {
                        self.write(pos, false, |s| {
                            for (ly, &voxel) in (start..end).zip(&voxels[start..end]) {
                                s.set(lx, ly, lz, voxel);
                            }
                        });
                    }
                }
            } else {
                let Some(col) = sample else {
                    // Dev 2 also returns None for custom terrain without a noise surface.
                    // Preserve existing bytes rather than interpreting this as sampled air.
                    unavailable = true;
                    continue;
                };
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
                let height = col.height >> level;
                let bottom = height.min(DISTANT_SOLID_FLOOR >> level);
                let water_top = col.water.filter(|&w| w > col.height).map(|w| w >> level);
                let top = water_top.unwrap_or(height);
                let sky_top = top + edge;
                let (lo, hi) = bounds.unwrap_or_else(|| {
                    let lo = bottom.div_euclid(edge) * edge;
                    let hi = sky_top.div_euclid(edge) * edge + edge - 1;
                    let span = observed.entry((sx, sz)).or_insert((lo, hi));
                    *span = (span.0.min(lo), span.1.max(hi));
                    *span
                });
                let mut fill_span = |y0: i32, y1: i32, voxel: Voxel| {
                    let (y0, y1) = (y0.max(lo), y1.min(hi + 1));
                    if y0 >= y1 {
                        return;
                    }
                    for sy in y0.div_euclid(edge)..=(y1 - 1).div_euclid(edge) {
                        let pos = OctreeSectionPos::new(level, sx, sy, sz);
                        if voxel == Voxel::AIR && !self.sections.contains_key(&pos) {
                            continue;
                        }
                        let sec_y0 = sy * edge;
                        let ly0 = (y0 - sec_y0).max(0) as usize;
                        let ly1 = (y1 - sec_y0).min(edge) as usize;
                        self.write(pos, false, |s| s.fill_column(lx, lz, ly0, ly1, voxel));
                    }
                };
                fill_span(lo, bottom, Voxel::AIR);
                fill_span(bottom, height, Voxel::new(sub_id, biome, 15, 0));
                // If the floor and water surface share a voxel, show water instead of floor.
                let surface_id = if water_top == Some(height) {
                    water_block
                } else {
                    id
                };
                fill_span(height, height + 1, Voxel::new(surface_id, biome, 15, 0));
                if let Some(w) = water_top {
                    for y in (height + 1)..=w {
                        let sky = 15u8.saturating_sub((w - y).clamp(0, 15) as u8);
                        fill_span(y, y + 1, Voxel::new(water_block, biome, sky, 0));
                    }
                }
                fill_span(top + 1, sky_top + 1, Voxel::air_with_light(15, 0));
                fill_span(sky_top + 1, hi + 1, Voxel::AIR);
            }

            if level <= MAX_OCTREE_DETAIL {
                let stack = (level, sx, sz);
                let words = self.partial_column_stacks.entry(stack).or_default();
                let bit = lx + lz * SECTION_EDGE;
                words[bit / 64] |= 1 << (bit % 64);
                if words.iter().all(|&word| word == u64::MAX) {
                    self.promote_column_stack(stack);
                }
            }
        }
        unavailable
    }

    /// Reduce an aligned footprint of authoritative D0 voxels with the regular mip rule.
    fn mip_voxel(
        section: &OctreeSection,
        detail: u8,
        x: usize,
        y: usize,
        z: usize,
        opacity: &dyn Fn(u32) -> u8,
    ) -> Voxel {
        if detail == 0 {
            return section.get(x, y, z);
        }
        mip_voxels(
            std::array::from_fn(|i| {
                Self::mip_voxel(
                    section,
                    detail - 1,
                    x * 2 + (i >> 2),
                    y * 2 + ((i >> 1) & 1),
                    z * 2 + (i & 1),
                    opacity,
                )
            }),
            opacity,
        )
    }

    /// Rebuild ancestors of dirty detail-0 sections, and return every touched position once.
    /// Direct coarse samples are already at the target detail: do not create new coarse-only
    /// fallback meshes above them, which could hide them while unrelated octants are missing.
    pub fn propagate_dirty(&mut self, opacity: &dyn Fn(u32) -> u8) -> Vec<OctreeSectionPos> {
        let mut touched: BTreeSet<_> = self.dirty.drain().collect();
        let mut dirty: BTreeSet<_> = touched.iter().copied().filter(|p| p.detail == 0).collect();
        for _ in 1..=MAX_OCTREE_DETAIL {
            if dirty.is_empty() {
                break;
            }
            let parents: BTreeSet<_> = dirty.iter().filter_map(|pos| pos.parent()).collect();
            let mut updated = BTreeSet::new();
            for parent in parents {
                let preserve = self.complete_sections.get(&parent) == Some(&CompleteKind::Direct);
                let children = parent.children().unwrap();
                let outside = children.map(|child| {
                    self.y_bounds
                        .is_some_and(|b| is_out_of_build_height(child, b))
                });
                let complete: [bool; 8] =
                    std::array::from_fn(|i| outside[i] || self.is_section_complete(children[i]));
                let use_child: [bool; 8] = std::array::from_fn(|i| {
                    !outside[i] && (!preserve || (dirty.contains(&children[i]) && complete[i]))
                });
                if preserve && !use_child.into_iter().any(|use_it| use_it) {
                    continue; // No complete dirty child can replace any of the fallback bytes.
                }
                let rebuild_all = (0..8).all(|i| outside[i] || (use_child[i] && complete[i]));
                let existing = self.sections.get(&parent);
                let air = existing.map(|_| OctreeSection::new(parent));
                let refs = std::array::from_fn(|i| {
                    if use_child[i] {
                        self.sections
                            .get(&children[i])
                            .or(air.as_ref().filter(|_| complete[i]))
                    } else if outside[i] && (!preserve || rebuild_all) {
                        air.as_ref()
                    } else {
                        None
                    }
                });
                if refs.iter().any(Option::is_some) {
                    let overwrites_sampled_columns = refs
                        .iter()
                        .zip(complete)
                        .any(|(child, complete)| child.is_some() && !complete);
                    let merged = downsample_2x2x2_onto(parent, existing, &refs, opacity);
                    if !merged.is_empty() || existing.is_some() {
                        self.sections.insert(parent, merged);
                    }
                    if overwrites_sampled_columns {
                        // A partial child can replace sampled columns with unknown voxels.
                        self.partial_column_stacks
                            .remove(&(parent.detail, parent.x, parent.z));
                    }
                }
                // Retained direct octants do not acquire a new mip proof.
                if rebuild_all && !outside.iter().all(|&o| o) {
                    self.complete_sections.insert(parent, CompleteKind::Mip);
                } else if !preserve {
                    self.complete_sections.remove(&parent);
                }
                updated.insert(parent);
            }
            touched.extend(updated.iter().copied());
            dirty = updated;
        }
        touched.into_iter().collect()
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
    ingest_chunk_culled(store, chunk, snap, src, false, block_id, biome_id)
}

/// Like [`ingest_chunk`], but skips vanilla sections wholly below [`DISTANT_SOLID_FLOOR`]
/// when `cull_distant_underground` is true. Voxels in the remaining sections are ingested normally.
pub fn ingest_chunk_culled<S: ChunkVoxelSource>(
    store: &mut OctreeIngestStore,
    chunk: [i32; 2],
    snap: &S,
    src: IngestSource,
    cull_distant_underground: bool,
    block_id: &mut dyn FnMut(S::Block) -> u32,
    biome_id: &mut dyn FnMut(&str) -> u16,
) -> bool {
    let count = snap.section_count();
    if count == 0 || !store.accept(chunk, src) {
        return false;
    }
    let edge = SECTION_EDGE as i32;
    let (ox, oz) = (chunk[0] * 16, chunk[1] * 16);
    let (sx, sz) = (ox.div_euclid(edge), oz.div_euclid(edge));
    let (off_x, off_z) = (ox.rem_euclid(edge) as usize, oz.rem_euclid(edge) as usize);
    let min_y = snap.min_y();
    store.set_y_bounds((min_y, min_y + (count * VANILLA_SECTION) as i32));

    // Top-down: air above a column's highest block sees the open sky, whatever lighting says (a
    // chunk that is not lit yet reads 0, which draws every face beside it black).
    let mut open = [true; VANILLA_SECTION * VANILLA_SECTION];
    for index in (0..count).rev() {
        let y0 = min_y + (index * VANILLA_SECTION) as i32;
        if src == IngestSource::Distant
            && cull_distant_underground
            && y0 + VANILLA_SECTION as i32 <= DISTANT_SOLID_FLOOR
        {
            continue;
        }
        let pos = OctreeSectionPos::new(0, sx, y0.div_euclid(edge), sz);
        let off_y = y0.rem_euclid(edge) as usize;
        // Replacing a whole slab preserves the existing proof for the other seven slabs.
        let preserve_complete = store.complete_sections.get(&pos) == Some(&CompleteKind::Direct);
        let newly_complete = if preserve_complete {
            false
        } else {
            let bit = 1u8 << child_index(off_x >> 4, off_y >> 4, off_z >> 4);
            let outside = store.out_of_build_height_subchunk_mask(pos);
            let covered = store.covered_subchunks.entry(pos).or_default();
            *covered |= bit;
            (*covered | outside) == u8::MAX
        };

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
                store.write(pos, preserve_complete, |s| {
                    for ly in 0..VANILLA_SECTION {
                        for lz in 0..VANILLA_SECTION {
                            for lx in 0..VANILLA_SECTION {
                                s.set(off_x + lx, off_y + ly, off_z + lz, air);
                            }
                        }
                    }
                });
            }
        } else {
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

            store.write(pos, preserve_complete, |s| {
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
        if newly_complete {
            store.covered_subchunks.remove(&pos);
            store.complete_sections.insert(pos, CompleteKind::Direct);
            store.dirty.insert(pos);
        } else if !preserve_complete && store.complete_sections.remove(&pos).is_some() {
            // Sampled air may take the no-write path but still replaces part of the proof.
            store.dirty.insert(pos);
        }
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
                store.write(OctreeSectionPos::new(0, sx, sy, sz), false, |s| {
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
                store.write(OctreeSectionPos::new(0, sx, sy, sz), false, |s| {
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

    /// A chunk backed by a sparse map (defaults to y 0..64).
    struct MockChunk {
        min_y: i32,
        section_count: usize,
        blocks: HashMap<(usize, i32, usize), u32>,
        block_calls: Cell<usize>,
    }

    impl MockChunk {
        fn new() -> Self {
            Self {
                min_y: 0,
                section_count: 4,
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
            self.min_y
        }
        fn section_count(&self) -> usize {
            self.section_count
        }
        fn section_uniform_block(&self, index: usize) -> Option<u32> {
            let lo = self.min_y + index as i32 * 16;
            let hi = lo + 16;
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
    fn section_coverage_requires_all_four_chunks_or_adoption() {
        let mut store = OctreeIngestStore::new();
        let pos = OctreeSectionPos::new(0, -1, 0, -1);
        let air = MockChunk::new();
        assert!(ingest(&mut store, [-2, -2], &air, IngestSource::Loaded));
        assert!(store.section(pos).unwrap().is_empty());
        assert!(!store.is_section_complete(pos));

        // One distant column cannot vouch for the rest of its chunk.
        ingest_distant_columns(
            &mut store,
            &[column(-1, -1, 0, None, STONE)],
            WATER,
            STONE,
            &mut |b| b,
            &mut |_| 0,
        );
        for chunk in [[-2, -1], [-1, -2]] {
            assert!(ingest(&mut store, chunk, &air, IngestSource::Loaded));
            assert!(!store.is_section_complete(pos));
        }
        assert!(ingest(&mut store, [-1, -1], &air, IngestSource::Loaded));
        assert!(store.is_section_complete(pos));
        assert!(!store.covered_subchunks.contains_key(&pos));
        assert!(!store.is_section_complete(pos.parent().unwrap()));

        let parent = pos.parent().unwrap();
        for child in parent.children().unwrap() {
            if !store.is_section_complete(child) {
                assert!(store.adopt_section(OctreeSection::new(child), true));
            }
        }
        assert!(!store.is_section_complete(parent));
        assert!(store.propagate_dirty(&|_| 15).contains(&parent));
        assert!(store.is_section_complete(parent));
        store.replace_section(OctreeSection::new(parent));
        assert!(!store.is_section_complete(parent));
    }

    #[test]
    fn persisted_coverage_survives_decode_and_adoption() {
        use crate::storage::{
            DEFAULT_OCTREE_COMPRESSION, OctreeMapping, decode_octree_section, encode_octree_section,
        };

        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let mut snap = MockChunk::new();
        snap.put(0, 0, 0, STONE);
        let mut source = OctreeIngestStore::new();
        for (chunks, expected_complete) in [
            (&[[0, 0]][..], false),
            (&[[1, 0], [0, 1], [1, 1]][..], true),
        ] {
            for &chunk in chunks {
                assert!(ingest(&mut source, chunk, &snap, IngestSource::Loaded));
            }
            let section = source.section(pos).unwrap();
            assert!(!section.is_empty());
            assert_eq!(source.is_section_complete(pos), expected_complete);
            let record = encode_octree_section(
                section,
                &OctreeMapping::default(),
                source.is_section_complete(pos),
                DEFAULT_OCTREE_COMPRESSION,
            )
            .unwrap();
            let (decoded, _, complete) = decode_octree_section(&record).unwrap();
            assert_eq!(decoded.to_dense(), section.to_dense());
            let mut reopened = OctreeIngestStore::new();
            assert!(reopened.adopt_section(decoded, complete));
            assert_eq!(reopened.is_section_complete(pos), expected_complete);
        }
    }

    #[test]
    fn adopted_parent_keeps_cached_octant_until_its_child_is_complete() {
        use crate::storage::{
            DEFAULT_OCTREE_COMPRESSION, OctreeMapping, decode_octree_section, encode_octree_section,
        };

        let parent = OctreeSectionPos::new(1, 0, 0, 0);
        let child = parent.children().unwrap()[0];
        let cached = OctreeSection::filled(parent, Voxel::new(STONE, 7, 15, 0));
        let mapping = OctreeMapping::default();
        let record =
            encode_octree_section(&cached, &mapping, true, DEFAULT_OCTREE_COMPRESSION).unwrap();
        let (section, _, complete) = decode_octree_section(&record).unwrap();
        let mut store = OctreeIngestStore::new();
        assert!(store.adopt_section(section, complete));
        assert_eq!(
            store.complete_sections.get(&parent),
            Some(&CompleteKind::Direct)
        );

        let air = MockChunk::new();
        assert!(ingest(&mut store, [0, 0], &air, IngestSource::Loaded));
        assert!(!store.is_section_complete(child));
        assert!(!store.propagate_dirty(&|_| 15).contains(&parent));
        let kept = store.section(parent).unwrap();
        assert_eq!(kept.revision, cached.revision);
        assert_eq!(kept.to_dense(), cached.to_dense());
        assert!(store.is_section_complete(parent));

        for chunk in [[1, 0], [0, 1]] {
            assert!(ingest(&mut store, chunk, &air, IngestSource::Loaded));
        }
        let mut terrain = MockChunk::new();
        terrain.put(0, 0, 0, DIRT);
        assert!(ingest(&mut store, [1, 1], &terrain, IngestSource::Loaded));
        assert!(store.is_section_complete(child));
        assert!(store.propagate_dirty(&|_| 15).contains(&parent));
        let merged = store.section(parent).unwrap();
        assert!(merged.get(0, 0, 0).is_air());
        assert_eq!(merged.get(8, 0, 8).block_id(), DIRT);
        for (x, z) in [(16, 0), (0, 16), (16, 16)] {
            assert_eq!(merged.get(x, 0, z), cached.get(x, 0, z));
        }
        let record = encode_octree_section(
            merged,
            &mapping,
            store.is_section_complete(parent),
            DEFAULT_OCTREE_COMPRESSION,
        )
        .unwrap();
        let (saved, _, complete) = decode_octree_section(&record).unwrap();
        assert!(complete);
        assert_eq!(saved.to_dense(), merged.to_dense());
    }

    #[test]
    fn all_children_outside_build_height_do_not_complete_mip() {
        let parent = OctreeSectionPos::new(1, 0, 1, 0);
        let children = parent.children().unwrap();
        let bounds = (0, 64);
        let mut store = OctreeIngestStore::new();
        store.set_y_bounds(bounds);
        assert!(
            children
                .iter()
                .all(|&child| is_out_of_build_height(child, bounds))
        );
        store.dirty.extend(children);
        assert!(store.propagate_dirty(&|_| 15).contains(&parent));
        assert_eq!(store.complete_sections.get(&parent), None);
    }

    #[test]
    fn mip_parent_rebuilds_as_soon_as_a_child_becomes_incomplete() {
        let parent = OctreeSectionPos::new(1, 0, 0, 0);
        let kids = parent.children().unwrap();
        let rock = Voxel::new(STONE, 7, 15, 0);
        let mut store = OctreeIngestStore::new();
        for pos in kids {
            assert!(store.adopt_section(OctreeSection::filled(pos, rock), true));
        }
        store.dirty.insert(kids[0]);
        assert!(store.propagate_dirty(&|_| 15).contains(&parent));
        assert_eq!(
            store.complete_sections.get(&parent),
            Some(&CompleteKind::Mip)
        );
        assert_eq!(store.section(parent).unwrap().get(0, 0, 0), rock);

        store.replace_section(OctreeSection::new(kids[0]));
        // Network replacement does not queue mip propagation itself.
        store.dirty.insert(kids[0]);
        assert!(!store.is_section_complete(kids[0]));
        assert!(store.propagate_dirty(&|_| 15).contains(&parent));
        assert!(!store.is_section_complete(parent));
        assert!(store.section(parent).unwrap().get(0, 0, 0).is_air());
        assert_eq!(store.section(parent).unwrap().get(16, 0, 0), rock);
    }

    #[test]
    fn partial_direct_writes_invalidate_the_cached_proof() {
        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let cached = OctreeSection::filled(pos, Voxel::new(STONE, 7, 15, 0));
        let mut store = OctreeIngestStore::new();
        assert!(store.adopt_section(cached, true));
        let columns = [column(0, 0, 5, None, DIRT)];
        assert_eq!(
            ingest_distant_columns(&mut store, &columns, WATER, STONE, &mut |b| b, &mut |_| 7),
            1
        );
        assert!(!store.is_section_complete(pos));

        let coarse = pos.parent().unwrap();
        assert!(store.adopt_section(
            OctreeSection::filled(coarse, Voxel::new(STONE, 7, 15, 0)),
            true
        ));
        store.ingest_coarse_columns(
            1,
            columns.map(DistantSample::Surface),
            (WATER, STONE),
            &mut |b| b,
            &mut |_| 7,
            &|_| 15,
        );
        assert!(!store.is_section_complete(coarse));
    }

    #[test]
    fn coarse_column_keeps_water_when_floor_and_surface_share_a_voxel() {
        let mut store = OctreeIngestStore::new();
        let level = 4;
        assert!(!store.ingest_coarse_columns(
            level,
            [DistantSample::Surface(column(0, 0, 52, Some(62), DIRT))],
            (WATER, STONE),
            &mut |b| b,
            &mut |_| 7,
            &|_| 15,
        ));
        let section = store
            .section(OctreeSectionPos::new(level, 0, 0, 0))
            .unwrap();
        assert_eq!(section.get(0, 2, 0), Voxel::new(STONE, 7, 15, 0));
        assert_eq!(section.get(0, 3, 0), Voxel::new(WATER, 7, 15, 0));
        assert_eq!(section.get(0, 4, 0), Voxel::air_with_light(15, 0));
    }

    #[test]
    fn coarse_column_uses_saved_snapshot_under_a_retained_octant() {
        let mut store = OctreeIngestStore::new();
        let pos = OctreeSectionPos::new(1, 0, 0, 0);
        assert!(store.adopt_section(
            OctreeSection::filled(pos, Voxel::new(STONE, 0, 15, 0)),
            true,
        ));
        let mut saved = MockChunk::new();
        saved.put(0, 0, 0, LOG);
        assert!(ingest(&mut store, [0, 0], &saved, IngestSource::Saved));
        assert!(!store.propagate_dirty(&|_| 15).contains(&pos));
        assert_eq!(store.section(pos).unwrap().get(0, 0, 0).block_id(), STONE);

        store.ingest_coarse_columns(
            1,
            [DistantSample::Surface(column(0, 0, 8, None, DIRT))],
            (WATER, STONE),
            &mut |_| panic!("G4 must not replace the saved column"),
            &mut |_| 0,
            &|_| 15,
        );
        let merged = store.section(pos).unwrap();
        assert_eq!(merged.get(0, 0, 0).block_id(), LOG);
        assert!(merged.get(0, 1, 0).is_air());
        assert_eq!(merged.get(1, 0, 0).block_id(), STONE);
        assert!(!store.is_section_complete(pos));
    }

    #[test]
    fn adopted_coarse_stack_only_covers_cells_when_every_in_bounds_row_is_proven() {
        let mut store = OctreeIngestStore::new();
        let leaf = crate::lod::SectionPos::new(1, -1, 0);
        store.set_y_bounds((-64, 64));
        for (y, expected) in [(-1, [0; 16]), (0, [u64::MAX; 16])] {
            let pos = OctreeSectionPos::new(1, -2, y, 0);
            assert!(store.adopt_section(OctreeSection::new(pos), true));
            assert_eq!(
                store.coarse_coverage(leaf),
                [expected, [0; 16], [0; 16], [0; 16]]
            );
        }
    }

    #[test]
    fn coarse_stack_counts_distinct_columns_and_uses_actual_build_height() {
        let pos = OctreeSectionPos::new(1, -1, 0, -1);
        let upper_air = OctreeSectionPos::new(1, -1, 4, -1);
        let stack = (pos.detail, pos.x, pos.z);
        let cols: Vec<_> = (-32..0)
            .flat_map(|z| (-32..0).map(move |x| column(x * 2, z * 2, 4, None, STONE)))
            .collect();
        let mut store = OctreeIngestStore::new();
        let ingest_coarse = |store: &mut OctreeIngestStore, cols: &[DistantColumn<'_, u32>]| {
            store.ingest_coarse_columns(
                1,
                cols.iter().copied().map(DistantSample::Surface),
                (WATER, STONE),
                &mut |b| b,
                &mut |_| 0,
                &|_| 15,
            );
        };
        ingest_coarse(&mut store, &cols[..1023]);
        ingest_coarse(&mut store, &cols[..1]);
        assert!(store.partial_column_stacks.contains_key(&stack));
        assert!(!store.is_section_complete(pos));
        ingest_coarse(&mut store, &cols[1023..]);
        assert_eq!(
            store.partial_column_stacks.get(&stack).map(|words| **words),
            Some([u64::MAX; 16])
        );
        assert_eq!(
            store.complete_sections.get(&pos),
            Some(&CompleteKind::Direct)
        );
        assert!(!store.is_section_complete(upper_air));
        assert!(!store.adopt_section(OctreeSection::new(pos), true));
        assert!(!store.adopt_section(OctreeSection::new(upper_air), false));
        assert!(!store.propagate_dirty(&|_| 15).contains(&upper_air));

        store.set_y_bounds((-64, 320));
        assert!(!store.partial_column_stacks.contains_key(&stack));
        for y in -1..=4 {
            let row = OctreeSectionPos::new(1, -1, y, -1);
            assert_eq!(
                store.complete_sections.get(&row),
                Some(&CompleteKind::Direct)
            );
            assert!(store.dirty.contains(&row));
            assert!(!store.adopt_section(OctreeSection::new(row), true));
        }
        assert!(store.adopt_section(
            OctreeSection::new(OctreeSectionPos::new(1, -1, 5, -1)),
            true
        ));
        assert!(store.is_section_complete(upper_air));
        assert!(store.section(upper_air).is_none());
        assert!(store.propagate_dirty(&|_| 15).contains(&upper_air));
        assert!(store.section_positions().contains(&upper_air));
    }

    #[test]
    fn skipped_g4_slabs_do_not_complete_underground_sections() {
        let mut snap = MockChunk::new();
        snap.min_y = -64;
        snap.section_count = 24;
        snap.put(0, -32, 0, STONE);
        let pos = OctreeSectionPos::new(0, 0, -1, 0);
        let mut store = OctreeIngestStore::new();
        assert!(ingest(&mut store, [0, 0], &snap, IngestSource::Loaded));
        let neighbors = [[1, 0], [0, 1], [1, 1]];
        for chunk in neighbors {
            assert!(ingest_chunk_culled(
                &mut store,
                chunk,
                &snap,
                IngestSource::Distant,
                true,
                &mut |b| b,
                &mut |_| 0,
            ));
        }
        assert_eq!(store.y_bounds(), Some((-64, 320)));
        assert!(store.is_section_complete(OctreeSectionPos::new(0, 0, 0, 0)));
        assert!(!store.is_section_complete(pos));
        for chunk in neighbors {
            assert!(ingest(&mut store, chunk, &snap, IngestSource::Loaded));
        }
        assert!(store.is_section_complete(pos));
    }

    #[test]
    fn sampled_air_without_voxel_writes_completes_a_boundary_leaf() {
        let mut snap = MockChunk::new();
        snap.min_y = 16;
        snap.section_count = 2;
        snap.put(0, 32, 0, STONE); // air in the lower slab takes the no-write path
        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let mut store = OctreeIngestStore::new();
        for chunk in [[0, 0], [1, 0], [0, 1], [1, 1]] {
            assert!(ingest(&mut store, chunk, &snap, IngestSource::Loaded));
        }
        assert!(store.section(pos).is_none());
        assert!(store.is_section_complete(pos));
        assert!(!store.adopt_section(OctreeSection::new(pos), false));
        assert!(store.propagate_dirty(&|_| 15).contains(&pos));
        assert!(store.section_positions().contains(&pos));

        assert!(ingest(&mut store, [0, 0], &snap, IngestSource::Loaded));
        assert!(store.section(pos).is_none());
        assert!(store.is_section_complete(pos));
        assert!(!store.covered_subchunks.contains_key(&pos));
        assert!(store.propagate_dirty(&|_| 15).is_empty());
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
    fn batched_ingest_propagates_once_without_duplicate_ancestors() {
        let mut c = MockChunk::new();
        c.put(1, 1, 1, STONE);
        let mut store = OctreeIngestStore::new();
        // Chunks 0 and 2 land in sibling sections (x = 0 and x = 1) that share every ancestor.
        assert!(ingest(&mut store, [0, 0], &c, IngestSource::Loaded));
        assert!(ingest(&mut store, [2, 0], &c, IngestSource::Loaded));
        let touched = store.propagate_dirty(&|_| 15);
        assert_eq!(store.dirty_count(), 0);
        let unique: HashSet<_> = touched.iter().copied().collect();
        assert_eq!(unique.len(), touched.len(), "one propagation, no repeats");
        assert!(touched.iter().filter(|p| p.detail == 0).count() >= 2);
        for detail in 1..=MAX_OCTREE_DETAIL {
            assert_eq!(
                touched.iter().filter(|p| p.detail == detail).count(),
                1,
                "sibling sections share one ancestor at detail {detail}"
            );
        }
        // A second propagation has nothing left to do.
        assert!(store.propagate_dirty(&|_| 15).is_empty());
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
