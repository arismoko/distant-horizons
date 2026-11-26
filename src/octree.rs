//! Voxy-style 32³ voxel octree sections for Distant Sunrises.
//!
//! Replaces the single-heightmap `LodColumn` pillar assumption with true 3D voxel sections so
//! tree canopies keep their air gaps and trunks, and overhangs, arches, bridges, caves and water
//! over underwater terrain survive at every detail level.
//!
//! Upstream anchors (`/box/shared/references/voxy`, `me.cortex.voxy.common.world`):
//! - `WorldSection`: a 32×32×32 region (`SECTION_VOLUME = 32*32*32`), data index
//!   `(y << 10) | (z << 5) | x`.
//! - `WorldEngine.MAX_LOD_LAYER = 4` (detail levels `0..=4`) and `getWorldSectionId`
//!   (`lvl << 60 | (y & 0xFF) << 52 | (z & 0xFFFFFF) << 28 | (x & 0xFFFFFF) << 4`).
//! - `other.Mapper`: a voxel is one `long`: block id in 20 bits at bit 27, biome id in 9 bits at
//!   bit 47, light byte in 8 bits at bit 56 (`block_light << 4 | sky_light`). Block id 0 is air.
//! - `other.Mipper.mip` + `voxelization.WorldVoxilizedSectionMipper`: each parent voxel is chosen
//!   from its 2×2×2 children by highest block-state opacity (leaves count as 15), ties going to the
//!   child with the highest `x<<2 | y<<1 | z`; all-air children average their light.
//! - `client.core.rendering.building.RenderDataFactory`: the six neighbour face slices
//!   (`neighboringFaces`) used to cull faces across section borders.

use crate::lod::{EMPTY_Y, LodColumn, LodSection, MAX_DETAIL as LOD_MAX_DETAIL, SECTION_SIZE};
use std::collections::HashMap;

/// Voxels along one edge of a section (`WorldSection`: 32).
pub const SECTION_EDGE: usize = 32;
/// Voxels per section (`WorldSection.SECTION_VOLUME`).
pub const SECTION_VOLUME: usize = SECTION_EDGE * SECTION_EDGE * SECTION_EDGE;
/// Highest octree detail level (`WorldEngine.MAX_LOD_LAYER`).
pub const MAX_OCTREE_DETAIL: u8 = 4;

/// Mapper bit layout (`Mapper.getBlockId/getBiomeId/getLightId`).
const BLOCK_SHIFT: u32 = 27;
const BLOCK_MASK: u64 = (1 << 20) - 1;
const BIOME_SHIFT: u32 = 47;
const BIOME_MASK: u64 = 0x1FF;
const LIGHT_SHIFT: u32 = 56;

/// Linear voxel index inside a section: `(y << 10) | (z << 5) | x`.
#[inline]
pub const fn voxel_index(x: usize, y: usize, z: usize) -> usize {
    (y << 10) | (z << 5) | x
}

/// Face direction ids follow Voxy's `face = (axis << 1) | side` with axis `0=Y, 1=Z, 2=X` and
/// side `1` positive (`RenderDataFactory.Mesher.emitQuad`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Face {
    NegY = 0,
    PosY = 1,
    NegZ = 2,
    PosZ = 3,
    NegX = 4,
    PosX = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::NegY,
        Face::PosY,
        Face::NegZ,
        Face::PosZ,
        Face::NegX,
        Face::PosX,
    ];

    #[inline]
    pub const fn id(self) -> u8 {
        self as u8
    }

    pub const fn from_id(id: u8) -> Option<Face> {
        match id {
            0 => Some(Face::NegY),
            1 => Some(Face::PosY),
            2 => Some(Face::NegZ),
            3 => Some(Face::PosZ),
            4 => Some(Face::NegX),
            5 => Some(Face::PosX),
            _ => None,
        }
    }

    /// Axis index: `0 = Y`, `1 = Z`, `2 = X`.
    #[inline]
    pub const fn axis(self) -> u8 {
        (self as u8) >> 1
    }

    /// World axis index the face is perpendicular to: `0 = X`, `1 = Y`, `2 = Z`.
    #[inline]
    pub const fn world_axis(self) -> usize {
        match self.axis() {
            0 => 1,
            1 => 2,
            _ => 0,
        }
    }

    #[inline]
    pub const fn is_positive(self) -> bool {
        (self as u8) & 1 == 1
    }

    #[inline]
    pub const fn opposite(self) -> Face {
        match self {
            Face::NegY => Face::PosY,
            Face::PosY => Face::NegY,
            Face::NegZ => Face::PosZ,
            Face::PosZ => Face::NegZ,
            Face::NegX => Face::PosX,
            Face::PosX => Face::NegX,
        }
    }

    /// Unit block offset `(dx, dy, dz)` of the neighbour this face looks at.
    #[inline]
    pub const fn offset(self) -> (i32, i32, i32) {
        match self {
            Face::NegY => (0, -1, 0),
            Face::PosY => (0, 1, 0),
            Face::NegZ => (0, 0, -1),
            Face::PosZ => (0, 0, 1),
            Face::NegX => (-1, 0, 0),
            Face::PosX => (1, 0, 0),
        }
    }
}

/// One voxel (`Mapper` long layout): block id (20 bits), biome id (9 bits), light byte
/// (`block_light << 4 | sky_light`). Block id `0` is air; air still carries light.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Voxel(pub u64);

impl Voxel {
    pub const AIR: Voxel = Voxel(0);

    /// Build a voxel; out-of-range ids and light are masked to the Mapper field widths.
    #[inline]
    pub const fn new(block_id: u32, biome_id: u16, sky_light: u8, block_light: u8) -> Voxel {
        let light = (((block_light & 0xF) as u64) << 4) | ((sky_light & 0xF) as u64);
        Voxel(
            ((block_id as u64 & BLOCK_MASK) << BLOCK_SHIFT)
                | ((biome_id as u64 & BIOME_MASK) << BIOME_SHIFT)
                | (light << LIGHT_SHIFT),
        )
    }

    /// Air carrying only light (`Mapper.airWithLight`).
    #[inline]
    pub const fn air_with_light(sky_light: u8, block_light: u8) -> Voxel {
        Voxel::new(0, 0, sky_light, block_light)
    }

    #[inline]
    pub const fn block_id(self) -> u32 {
        ((self.0 >> BLOCK_SHIFT) & BLOCK_MASK) as u32
    }

    #[inline]
    pub const fn biome_id(self) -> u16 {
        ((self.0 >> BIOME_SHIFT) & BIOME_MASK) as u16
    }

    /// Raw light byte (`Mapper.getLightId`): `block_light << 4 | sky_light`.
    #[inline]
    pub const fn light(self) -> u8 {
        (self.0 >> LIGHT_SHIFT) as u8
    }

    #[inline]
    pub const fn sky_light(self) -> u8 {
        self.light() & 0xF
    }

    #[inline]
    pub const fn block_light(self) -> u8 {
        self.light() >> 4
    }

    #[inline]
    pub const fn is_air(self) -> bool {
        self.block_id() == 0
    }

    #[inline]
    pub const fn with_light_byte(self, light: u8) -> Voxel {
        Voxel((self.0 & !(0xFFu64 << LIGHT_SHIFT)) | ((light as u64) << LIGHT_SHIFT))
    }
}

/// Section position: `detail` 0 = 1 block per voxel, each level doubles the voxel size. `x`, `y`,
/// `z` are section coordinates at that detail (`32 << detail` blocks per section).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OctreeSectionPos {
    pub detail: u8,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl OctreeSectionPos {
    pub const fn new(detail: u8, x: i32, y: i32, z: i32) -> Self {
        Self { detail, x, y, z }
    }

    /// Edge length of one voxel in blocks (`1 << detail`).
    pub const fn voxel_size(self) -> i32 {
        1 << self.detail
    }

    /// Edge length of the section in blocks (`32 << detail`).
    pub const fn block_span(self) -> i32 {
        (SECTION_EDGE as i32) << self.detail
    }

    /// World block of the section's minimum corner.
    pub const fn block_origin(self) -> [i32; 3] {
        let s = self.block_span();
        [self.x * s, self.y * s, self.z * s]
    }

    /// Parent at `detail + 1` (`None` past `MAX_OCTREE_DETAIL`).
    pub const fn parent(self) -> Option<Self> {
        if self.detail >= MAX_OCTREE_DETAIL {
            None
        } else {
            Some(Self::new(
                self.detail + 1,
                self.x >> 1,
                self.y >> 1,
                self.z >> 1,
            ))
        }
    }

    /// Eight children at `detail - 1`, indexed `cx | cz << 1 | cy << 2` (`child_index`).
    pub const fn children(self) -> Option<[Self; 8]> {
        if self.detail == 0 {
            return None;
        }
        let d = self.detail - 1;
        let (bx, by, bz) = (self.x << 1, self.y << 1, self.z << 1);
        Some([
            Self::new(d, bx, by, bz),
            Self::new(d, bx + 1, by, bz),
            Self::new(d, bx, by, bz + 1),
            Self::new(d, bx + 1, by, bz + 1),
            Self::new(d, bx, by + 1, bz),
            Self::new(d, bx + 1, by + 1, bz),
            Self::new(d, bx, by + 1, bz + 1),
            Self::new(d, bx + 1, by + 1, bz + 1),
        ])
    }

    /// Neighbour section across `face`.
    pub const fn neighbor(self, face: Face) -> Self {
        let (dx, dy, dz) = face.offset();
        Self::new(self.detail, self.x + dx, self.y + dy, self.z + dz)
    }

    /// Voxy's packed section id (`WorldEngine.getWorldSectionId`).
    pub const fn key(self) -> u64 {
        ((self.detail as u64 & 0xF) << 60)
            | ((self.y as u64 & 0xFF) << 52)
            | ((self.z as u64 & 0xFF_FFFF) << 28)
            | ((self.x as u64 & 0xFF_FFFF) << 4)
    }
}

/// Child slot of a parent section: `cx | cz << 1 | cy << 2`.
#[inline]
pub const fn child_index(cx: usize, cy: usize, cz: usize) -> usize {
    cx | (cz << 1) | (cy << 2)
}

/// Why compressed section data was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OctreeError {
    /// A run names a palette entry that does not exist.
    PaletteIndexOutOfRange,
    /// The runs do not cover exactly `SECTION_VOLUME` voxels.
    BadRunTotal,
    /// The palette is empty.
    EmptyPalette,
}

/// One run of identical voxels in x-fastest linear order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub palette_index: u16,
    /// 1..=32768 voxels.
    pub length: u16,
}

/// Compact run-length-encoded form of a section for storage and network.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RleSection {
    pub pos: OctreeSectionPos,
    pub palette: Vec<Voxel>,
    pub runs: Vec<Run>,
}

const OCTANT_EDGE: usize = SECTION_EDGE / 2;
const OCTANT_VOLUME: usize = OCTANT_EDGE * OCTANT_EDGE * OCTANT_EDGE;

#[derive(Clone, Debug)]
enum OctantIndices {
    Uniform(u16),
    Dense(Box<[u16]>),
}

impl OctantIndices {
    #[inline]
    fn get(&self, offset: usize) -> u16 {
        match self {
            Self::Uniform(pi) => *pi,
            Self::Dense(indices) => indices[offset],
        }
    }

    fn dense_mut(&mut self) -> &mut [u16] {
        if let Self::Uniform(pi) = self {
            *self = Self::Dense(vec![*pi; OCTANT_VOLUME].into_boxed_slice());
        }
        match self {
            Self::Dense(indices) => indices,
            Self::Uniform(_) => unreachable!(),
        }
    }
}

/// A 32×32×32 paletted voxel section. Each aligned 16³ octant can hold a single palette index,
/// even when its neighbours use a different value (e.g. sky-lit air beside unwritten air).
#[derive(Clone, Debug)]
pub struct OctreeSection {
    pub pos: OctreeSectionPos,
    palette: Vec<Voxel>,
    lookup: HashMap<Voxel, u16>,
    octants: [OctantIndices; 8],
    pub revision: u64,
}

impl OctreeSection {
    /// An all-air section.
    pub fn new(pos: OctreeSectionPos) -> Self {
        Self::filled(pos, Voxel::AIR)
    }

    /// A section where every voxel is `voxel`.
    pub fn filled(pos: OctreeSectionPos, voxel: Voxel) -> Self {
        Self {
            pos,
            palette: vec![voxel],
            lookup: HashMap::from([(voxel, 0)]),
            octants: std::array::from_fn(|_| OctantIndices::Uniform(0)),
            revision: 0,
        }
    }

    #[inline]
    fn in_range(x: usize, y: usize, z: usize) -> bool {
        x < SECTION_EDGE && y < SECTION_EDGE && z < SECTION_EDGE
    }

    /// Split an in-range section coordinate into an octant and its x-fastest offset.
    #[inline]
    fn address(x: usize, y: usize, z: usize) -> (usize, usize) {
        let mask = OCTANT_EDGE - 1;
        (
            child_index(x >> 4, y >> 4, z >> 4),
            ((y & mask) * OCTANT_EDGE + (z & mask)) * OCTANT_EDGE + (x & mask),
        )
    }

    #[inline]
    fn coords(idx: usize) -> (usize, usize, usize) {
        (idx & 31, idx >> 10, (idx >> 5) & 31)
    }

    /// Voxel at local `(x, y, z)`; out-of-range coordinates read as air.
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        if !Self::in_range(x, y, z) {
            return Voxel::AIR;
        }
        let (octant, offset) = Self::address(x, y, z);
        self.palette[self.octants[octant].get(offset) as usize]
    }

    fn palette_index(&mut self, voxel: Voxel) -> u16 {
        if let Some(&pi) = self.lookup.get(&voxel) {
            return pi;
        }
        // Overwrites leave unused entries behind. At most SECTION_VOLUME entries can be live.
        if self.palette.len() > SECTION_VOLUME {
            self.compact();
        }
        let pi = self.palette.len() as u16;
        self.palette.push(voxel);
        self.lookup.insert(voxel, pi);
        pi
    }

    /// Write a voxel. Returns `false` (and changes nothing) for out-of-range coordinates or an
    /// unchanged voxel.
    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) -> bool {
        if !Self::in_range(x, y, z) {
            return false;
        }
        let (octant, offset) = Self::address(x, y, z);
        if self.palette[self.octants[octant].get(offset) as usize] == voxel {
            return false;
        }
        let pi = self.palette_index(voxel);
        self.octants[octant].dense_mut()[offset] = pi;
        self.revision += 1;
        true
    }

    /// Replace an aligned 16³ octant. Already-uniform data needs only one comparison; mixed
    /// data counts changed voxels to keep the same revision as individual writes.
    pub(crate) fn fill_octant(&mut self, octant: usize, voxel: Voxel) {
        let changed = match &self.octants[octant] {
            OctantIndices::Uniform(pi) => {
                if self.palette[*pi as usize] == voxel {
                    return;
                }
                OCTANT_VOLUME
            }
            OctantIndices::Dense(indices) => indices
                .iter()
                .filter(|&&pi| self.palette[pi as usize] != voxel)
                .count(),
        };
        // Even an equal dense octant can be collapsed without changing its logical revision.
        self.octants[octant] = OctantIndices::Uniform(self.palette_index(voxel));
        self.revision += changed as u64;
    }

    #[cfg(test)]
    pub(crate) fn is_uniform_octant(&self, octant: usize) -> bool {
        matches!(self.octants[octant], OctantIndices::Uniform(_))
    }

    /// Fill the vertical span `y0..y1` (local, clamped to the section) of column `(x, z)` with one
    /// palette lookup instead of per-voxel `set` overhead.
    pub fn fill_column(&mut self, x: usize, z: usize, y0: usize, y1: usize, voxel: Voxel) {
        let end = y1.min(SECTION_EDGE);
        if x >= SECTION_EDGE || z >= SECTION_EDGE || y0 >= end {
            return;
        }
        let mut pi = None;
        let mut changed = false;
        for (start, stop) in [(y0, end.min(OCTANT_EDGE)), (y0.max(OCTANT_EDGE), end)] {
            if start >= stop {
                continue;
            }
            let (octant, offset) = Self::address(x, start, z);
            if let OctantIndices::Uniform(old) = self.octants[octant]
                && self.palette[old as usize] == voxel
            {
                continue;
            }
            let target = *pi.get_or_insert_with(|| self.palette_index(voxel));
            let indices = self.octants[octant].dense_mut();
            for i in (offset..)
                .step_by(OCTANT_EDGE * OCTANT_EDGE)
                .take(stop - start)
            {
                if indices[i] != target {
                    indices[i] = target;
                    changed = true;
                }
            }
        }
        if changed {
            self.revision += 1;
        }
    }

    /// All 32,768 voxels in linear (`voxel_index`) order.
    pub fn to_dense(&self) -> Vec<Voxel> {
        let mut dense = Vec::with_capacity(SECTION_VOLUME);
        for y in 0..SECTION_EDGE {
            for z in 0..SECTION_EDGE {
                for x in [0, OCTANT_EDGE] {
                    let (octant, offset) = Self::address(x, y, z);
                    match &self.octants[octant] {
                        OctantIndices::Uniform(pi) => {
                            dense.extend(std::iter::repeat_n(
                                self.palette[*pi as usize],
                                OCTANT_EDGE,
                            ));
                        }
                        OctantIndices::Dense(indices) => {
                            dense.extend(
                                indices[offset..offset + OCTANT_EDGE]
                                    .iter()
                                    .map(|&pi| self.palette[pi as usize]),
                            );
                        }
                    }
                }
            }
        }
        dense
    }

    /// Build from 32,768 voxels in linear order (extra voxels ignored, missing ones air).
    pub fn from_dense(pos: OctreeSectionPos, voxels: &[Voxel]) -> Self {
        let mut s = Self::new(pos);
        for i in 0..SECTION_VOLUME {
            let voxel = voxels.get(i).copied().unwrap_or(Voxel::AIR);
            let (x, y, z) = Self::coords(i);
            let (octant, offset) = Self::address(x, y, z);
            if offset == 0 {
                s.octants[octant] = OctantIndices::Uniform(s.palette_index(voxel));
            } else if s.palette[s.octants[octant].get(offset) as usize] != voxel {
                let pi = s.palette_index(voxel);
                s.octants[octant].dense_mut()[offset] = pi;
            }
        }
        s
    }

    /// Number of non-air voxels.
    pub fn non_air_count(&self) -> usize {
        self.octants
            .iter()
            .map(|indices| match indices {
                OctantIndices::Uniform(pi) => {
                    if self.palette[*pi as usize].is_air() {
                        0
                    } else {
                        OCTANT_VOLUME
                    }
                }
                OctantIndices::Dense(indices) => indices
                    .iter()
                    .filter(|&&pi| !self.palette[pi as usize].is_air())
                    .count(),
            })
            .sum()
    }

    /// `true` when every voxel is air (light may still differ).
    pub fn is_empty(&self) -> bool {
        self.non_air_count() == 0
    }

    pub fn palette_len(&self) -> usize {
        self.palette.len()
    }

    /// Drop unused palette entries and collapse uniform octants.
    pub fn compact(&mut self) {
        let mut remap = vec![None; self.palette.len()];
        let mut palette = Vec::new();
        let mut map = |pi: u16| {
            *remap[pi as usize].get_or_insert_with(|| {
                palette.push(self.palette[pi as usize]);
                (palette.len() - 1) as u16
            })
        };
        for octant in &mut self.octants {
            match octant {
                OctantIndices::Uniform(pi) => *pi = map(*pi),
                OctantIndices::Dense(indices) => {
                    let first = indices[0];
                    if indices.iter().all(|&pi| pi == first) {
                        *octant = OctantIndices::Uniform(map(first));
                    } else {
                        indices.iter_mut().for_each(|pi| *pi = map(*pi));
                    }
                }
            }
        }
        self.lookup = palette
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u16))
            .collect();
        self.palette = palette;
    }

    /// Run-length encode in x-fastest order, numbering palette entries by first use.
    pub fn to_rle(&self) -> RleSection {
        let mut remap = vec![None; self.palette.len()];
        let mut palette = Vec::new();
        let mut runs: Vec<Run> = Vec::new();
        for y in 0..SECTION_EDGE {
            for z in 0..SECTION_EDGE {
                for x in [0, OCTANT_EDGE] {
                    let (octant, offset) = Self::address(x, y, z);
                    let mut push = |pi: u16, length: u16| {
                        let palette_index = *remap[pi as usize].get_or_insert_with(|| {
                            palette.push(self.palette[pi as usize]);
                            (palette.len() - 1) as u16
                        });
                        if let Some(last) = runs.last_mut()
                            && last.palette_index == palette_index
                        {
                            last.length += length;
                        } else {
                            runs.push(Run {
                                palette_index,
                                length,
                            });
                        }
                    };
                    match &self.octants[octant] {
                        OctantIndices::Uniform(pi) => push(*pi, OCTANT_EDGE as u16),
                        OctantIndices::Dense(indices) => {
                            for &pi in &indices[offset..offset + OCTANT_EDGE] {
                                push(pi, 1);
                            }
                        }
                    }
                }
            }
        }
        RleSection {
            pos: self.pos,
            palette,
            runs,
        }
    }

    /// Decode a run-length section; malformed data is an `Err`, never a panic.
    pub fn from_rle(rle: &RleSection) -> Result<Self, OctreeError> {
        if rle.palette.is_empty() {
            return Err(OctreeError::EmptyPalette);
        }
        let total: usize = rle.runs.iter().map(|r| r.length as usize).sum();
        if total != SECTION_VOLUME {
            return Err(OctreeError::BadRunTotal);
        }
        if rle
            .runs
            .iter()
            .any(|r| r.palette_index as usize >= rle.palette.len())
        {
            return Err(OctreeError::PaletteIndexOutOfRange);
        }
        if let [run] = rle.runs.as_slice() {
            return Ok(Self::filled(
                rle.pos,
                rle.palette[run.palette_index as usize],
            ));
        }
        let dense: Vec<_> = rle
            .runs
            .iter()
            .flat_map(|r| {
                std::iter::repeat_n(rle.palette[r.palette_index as usize], r.length as usize)
            })
            .collect();
        Ok(Self::from_dense(rle.pos, &dense))
    }
}

/// The six neighbour face slices of one section (`RenderDataFactory.neighboringFaces`): for each
/// face, the 32×32 layer of the *neighbouring* section that touches this section. A missing slice
/// reads as air, as Voxy's empty sections do.
///
/// Slice index per face: `±X` -> `(y << 5) | z`, `±Y` -> `(z << 5) | x`, `±Z` -> `(y << 5) | x`.
#[derive(Clone, Debug, Default)]
pub struct NeighborBorders {
    slabs: [Option<Box<[Voxel]>>; 6],
}

impl NeighborBorders {
    pub fn none() -> Self {
        Self::default()
    }

    #[inline]
    fn slab_index(face: Face, x: usize, y: usize, z: usize) -> usize {
        match face.axis() {
            2 => (y << 5) | z,
            0 => (z << 5) | x,
            _ => (y << 5) | x,
        }
    }

    /// The layer of `neighbor` (the section lying across `face` of the target) that touches the
    /// target: `PosX` neighbour contributes its `x = 0` layer, `NegX` its `x = 31`, and so on.
    pub fn extract_slab(neighbor: &OctreeSection, face: Face) -> Box<[Voxel]> {
        let fixed = if face.is_positive() {
            0
        } else {
            SECTION_EDGE - 1
        };
        let mut out = vec![Voxel::AIR; SECTION_EDGE * SECTION_EDGE];
        for a in 0..SECTION_EDGE {
            for b in 0..SECTION_EDGE {
                let (x, y, z) = match face.axis() {
                    2 => (fixed, a, b),
                    0 => (b, fixed, a),
                    _ => (b, a, fixed),
                };
                out[Self::slab_index(face, x, y, z)] = neighbor.get(x, y, z);
            }
        }
        out.into_boxed_slice()
    }

    pub fn set_slab(&mut self, face: Face, slab: Box<[Voxel]>) {
        if slab.len() == SECTION_EDGE * SECTION_EDGE {
            self.slabs[face as usize] = Some(slab);
        }
    }

    /// Neighbour sections indexed by `Face::id()`.
    pub fn from_neighbors(neighbors: [Option<&OctreeSection>; 6]) -> Self {
        let mut b = Self::none();
        for face in Face::ALL {
            if let Some(n) = neighbors[face as usize] {
                b.set_slab(face, Self::extract_slab(n, face));
            }
        }
        b
    }

    pub fn has(&self, face: Face) -> bool {
        self.slabs[face as usize].is_some()
    }

    /// Voxel across `face` of the border cell whose in-section coordinates are `(x, y, z)`
    /// (the coordinate along the face's axis is ignored). `None` when that border is not loaded.
    pub fn get(&self, face: Face, x: usize, y: usize, z: usize) -> Option<Voxel> {
        self.slabs[face as usize]
            .as_ref()
            .map(|s| s[Self::slab_index(face, x, y, z)])
    }
}

/// `Mipper.mip` with surface-wins tie-breaking: pick the parent voxel from eight children
/// `children[x << 2 | y << 1 | z]`. Primary key is `opacity(block_id)` (`Mapper.getBlockStateOpacity`);
/// among equal-opacity children, the upper layer (`y = 1`) always beats the lower layer (`y = 0`)
/// so surface grass/snow/leaves never lose to underground dirt/stone on slopes, and within that
/// layer the majority block wins (breaking ties deterministically by index).
pub fn mip_voxels(children: [Voxel; 8], opacity: &dyn Fn(u32) -> u8) -> Voxel {
    let mut best_op: i32 = -1;
    let mut best_y: i32 = -1;
    for (idx, v) in children.iter().enumerate() {
        if v.is_air() {
            continue;
        }
        let op = opacity(v.block_id()) as i32;
        let y = ((idx >> 1) & 1) as i32;
        if op > best_op || (op == best_op && y > best_y) {
            best_op = op;
            best_y = y;
        }
    }
    if best_op != -1 {
        // Majority vote among the winning (opacity, y) layer so single-block noise does not
        // override surrounding terrain; ties break toward higher index.
        let mut best_idx = 0usize;
        let mut best_count = 0u8;
        for (idx, v) in children.iter().enumerate() {
            if v.is_air()
                || opacity(v.block_id()) as i32 != best_op
                || ((idx >> 1) & 1) as i32 != best_y
            {
                continue;
            }
            let bid = v.block_id();
            let count = children
                .iter()
                .enumerate()
                .filter(|(j, u)| {
                    !u.is_air()
                        && ((j >> 1) & 1) as i32 == best_y
                        && opacity(u.block_id()) as i32 == best_op
                        && u.block_id() == bid
                })
                .count() as u8;
            if count > best_count || (count == best_count && idx >= best_idx) {
                best_count = count;
                best_idx = idx;
            }
        }
        return children[best_idx];
    }
    let mut block = 0u32;
    let mut sky = 0u32;
    for v in &children {
        // Treat unwritten air (light == 0) as open sky (15) when mipping air light so a mix of
        // lit air and unwritten air never averages down to sky 1..2 and darkens coarse cliffs.
        let l = if v.light() == 0 { 0x0F } else { v.light() };
        block += (l & 0xF0) as u32;
        sky += (l & 0x0F) as u32;
    }
    let block = (block / 8) & 0xF0;
    let sky = sky.div_ceil(8);
    children[7].with_light_byte((block | sky) as u8)
}

/// Multi-level 2×2×2 octree mip: eight child sections at `detail - 1` become one parent section
/// at `detail` (`WorldVoxilizedSectionMipper.mipSection` applied across sections). Children are
/// indexed by `child_index(cx, cy, cz)`; `None` is air.
pub fn downsample_2x2x2(
    parent_pos: OctreeSectionPos,
    children: &[Option<&OctreeSection>; 8],
    opacity: &dyn Fn(u32) -> u8,
) -> OctreeSection {
    downsample_2x2x2_onto(parent_pos, None, children, opacity)
}

/// Like [`downsample_2x2x2`], but when `existing` is `Some`, preserves any octant whose child is
/// `None` so partial detail-0 propagation never wipes coarse terrain in sibling octants.
pub fn downsample_2x2x2_onto(
    parent_pos: OctreeSectionPos,
    existing: Option<&OctreeSection>,
    children: &[Option<&OctreeSection>; 8],
    opacity: &dyn Fn(u32) -> u8,
) -> OctreeSection {
    let dense: Vec<Option<Vec<Voxel>>> = children.iter().map(|c| c.map(|s| s.to_dense())).collect();
    let mut out = existing
        .cloned()
        .unwrap_or_else(|| OctreeSection::new(parent_pos));
    for py in 0..SECTION_EDGE {
        for pz in 0..SECTION_EDGE {
            for px in 0..SECTION_EDGE {
                let ci = child_index(px >> 4, py >> 4, pz >> 4);
                let Some(data) = &dense[ci] else { continue };
                let (bx, by, bz) = ((px & 15) * 2, (py & 15) * 2, (pz & 15) * 2);
                let mut kids = [Voxel::AIR; 8];
                for dx in 0..2 {
                    for dy in 0..2 {
                        for dz in 0..2 {
                            kids[(dx << 2) | (dy << 1) | dz] =
                                data[voxel_index(bx + dx, by + dy, bz + dz)];
                        }
                    }
                }
                out.set(px, py, pz, mip_voxels(kids, opacity));
            }
        }
    }
    out.revision = existing.map_or(1, |e| e.revision.wrapping_add(1));
    out
}

/// What the legacy `LodColumn` knew about one run of terrain, handed to the block-id resolver of
/// `lod_section_to_octree`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodVoxelKind {
    pub rgb: [u8; 3],
    pub material_id: u8,
    pub fluid: bool,
}

/// Adapter from the legacy heightmap `LodSection` to octree sections, so the existing storage,
/// network and generation paths can feed the 3D mesher. Each cell's solid span
/// `[solid_bottom_y, solid_top_y)`, fluid span `[solid_top_y, fluid_top_y)` and every
/// `ExtraSpan` become voxel runs at the section's detail (a voxel covers `1 << detail` blocks).
/// `block_for` maps a kind to a non-zero block id (a zero result is skipped).
pub fn lod_section_to_octree(
    lod: &LodSection,
    block_for: &mut dyn FnMut(LodVoxelKind) -> u32,
) -> Vec<OctreeSection> {
    let detail = lod.pos.detail.min(LOD_MAX_DETAIL);
    let per = SECTION_SIZE / SECTION_EDGE;
    let mut out: HashMap<(i32, i32, i32), OctreeSection> = HashMap::new();
    let edge = SECTION_EDGE as i32;
    let size = 1i32 << detail;

    let put = |out: &mut HashMap<(i32, i32, i32), OctreeSection>,
               cx: usize,
               cz: usize,
               bottom: i32,
               top: i32,
               rgba: [u8; 4],
               kind: LodVoxelKind,
               block_for: &mut dyn FnMut(LodVoxelKind) -> u32| {
        let block = block_for(kind);
        if block == 0 || top <= bottom {
            return;
        }
        let light = rgba[3];
        let voxel = Voxel::new(block, 0, (light >> 4) & 0xF, light & 0xF);
        let v0 = bottom.div_euclid(size);
        let v1 = (top + size - 1).div_euclid(size).max(v0 + 1);
        let ox = lod.pos.x * per as i32 + (cx / SECTION_EDGE) as i32;
        let oz = lod.pos.z * per as i32 + (cz / SECTION_EDGE) as i32;
        for vy in v0..v1 {
            let oy = vy.div_euclid(edge);
            let ly = vy.rem_euclid(edge) as usize;
            let sec = out
                .entry((ox, oy, oz))
                .or_insert_with(|| OctreeSection::new(OctreeSectionPos::new(detail, ox, oy, oz)));
            sec.set(cx % SECTION_EDGE, ly, cz % SECTION_EDGE, voxel);
        }
    };

    for cz in 0..SECTION_SIZE {
        for cx in 0..SECTION_SIZE {
            let col: LodColumn = lod.columns[LodSection::index(cx, cz)];
            if col.is_empty() {
                continue;
            }
            if col.has_solid() {
                let top = col.solid_top_y as i32;
                let bottom = if col.solid_bottom_y == EMPTY_Y || col.solid_bottom_y as i32 >= top {
                    top - size
                } else {
                    col.solid_bottom_y as i32
                };
                put(
                    &mut out,
                    cx,
                    cz,
                    bottom,
                    top,
                    col.solid_rgba,
                    LodVoxelKind {
                        rgb: [col.solid_rgba[0], col.solid_rgba[1], col.solid_rgba[2]],
                        material_id: col.material_id,
                        fluid: false,
                    },
                    &mut *block_for,
                );
            }
            if col.has_fluid() {
                let base = if col.has_solid() {
                    col.solid_top_y as i32
                } else {
                    col.fluid_top_y as i32 - size
                };
                put(
                    &mut out,
                    cx,
                    cz,
                    base,
                    col.fluid_top_y as i32,
                    col.fluid_rgba,
                    LodVoxelKind {
                        rgb: [col.fluid_rgba[0], col.fluid_rgba[1], col.fluid_rgba[2]],
                        material_id: crate::colors::DH_BLOCK_WATER,
                        fluid: true,
                    },
                    &mut *block_for,
                );
            }
            for span in lod.extra_spans_at(cx, cz) {
                put(
                    &mut out,
                    cx,
                    cz,
                    span.bottom_y as i32,
                    span.top_y as i32,
                    span.rgba,
                    LodVoxelKind {
                        rgb: [span.rgba[0], span.rgba[1], span.rgba[2]],
                        material_id: span.material_id,
                        fluid: span.is_fluid,
                    },
                    &mut *block_for,
                );
            }
        }
    }
    let mut v: Vec<OctreeSection> = out.into_values().collect();
    v.sort_by_key(|s| s.pos);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    const STONE: u32 = 1;
    const LOG: u32 = 2;
    const LEAVES: u32 = 3;
    const WATER: u32 = 4;

    fn opacity(id: u32) -> u8 {
        match id {
            LEAVES => 15, // Mapper.StateEntry: leaves forced to 15
            WATER => 1,
            0 => 0,
            _ => 15,
        }
    }

    #[test]
    fn voxel_mapper_layout_roundtrips() {
        let v = Voxel::new(0xABCDE, 0x1FF, 9, 5);
        assert_eq!(v.block_id(), 0xABCDE);
        assert_eq!(v.biome_id(), 0x1FF);
        assert_eq!(v.sky_light(), 9);
        assert_eq!(v.block_light(), 5);
        assert_eq!(v.light(), 0x59);
        assert_eq!((v.0 >> 27) & ((1 << 20) - 1), 0xABCDE);
        assert!(Voxel::air_with_light(15, 0).is_air());
        assert_eq!(Voxel::air_with_light(15, 0).sky_light(), 15);
    }

    #[test]
    fn section_key_matches_voxy_world_section_id() {
        let p = OctreeSectionPos::new(3, -1, 2, 5);
        let expect = (3u64 << 60) | (2u64 << 52) | (5u64 << 28) | (0xFF_FFFFu64 << 4);
        assert_eq!(p.key(), expect);
    }

    #[test]
    fn uniform_then_paletted_storage() {
        let mut s = OctreeSection::new(OctreeSectionPos::new(0, 0, 0, 0));
        assert!(s.is_empty());
        assert!(!s.set(0, 0, 0, Voxel::AIR));
        assert!(s.set(1, 2, 3, Voxel::new(STONE, 1, 15, 0)));
        assert_eq!(s.get(1, 2, 3).block_id(), STONE);
        assert!(s.get(0, 0, 0).is_air());
        assert_eq!(s.non_air_count(), 1);
        assert!(!s.set(32, 0, 0, Voxel::new(STONE, 0, 0, 0)));
        assert!(s.get(40, 0, 0).is_air());
    }

    #[test]
    fn sky_lit_octant_is_compact_beside_unwritten_air() {
        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let mut s = OctreeSection::new(pos);
        let lit = Voxel::air_with_light(15, 0);
        let slot = child_index(1, 1, 0);
        s.fill_octant(slot, lit);
        assert!(
            s.octants
                .iter()
                .all(|o| matches!(o, OctantIndices::Uniform(_)))
        );
        assert_eq!(s.get(16, 16, 0), lit);
        assert_eq!(s.get(15, 16, 0), Voxel::AIR);
        assert_eq!(s.get(16, 15, 0), Voxel::AIR);
        assert_eq!(s.get(16, 16, 16), Voxel::AIR);
        assert!(s.is_empty(), "no geometry just because air is lit");
        let dense = s.to_dense();
        for y in 0..32 {
            for z in 0..32 {
                for x in 0..32 {
                    let want = if x >= 16 && y >= 16 && z < 16 {
                        lit
                    } else {
                        Voxel::AIR
                    };
                    assert_eq!(dense[voxel_index(x, y, z)], want);
                }
            }
        }
        let decoded = OctreeSection::from_rle(&s.to_rle()).unwrap();
        assert_eq!(decoded.to_dense(), dense);
        assert!(
            decoded
                .octants
                .iter()
                .all(|o| matches!(o, OctantIndices::Uniform(_)))
        );

        s.set(20, 20, 0, Voxel::new(STONE, 0, 15, 0));
        s.set(31, 20, 15, Voxel::new(LOG, 0, 15, 0));
        assert!(matches!(s.octants[slot], OctantIndices::Dense(_)));
        let revision = s.revision;
        s.fill_octant(slot, lit);
        assert!(matches!(s.octants[slot], OctantIndices::Uniform(_)));
        assert_eq!(s.revision, revision + 2);
        assert_eq!(s.to_dense(), dense);
        s.fill_octant(slot, lit);
        assert_eq!(s.revision, revision + 2);
    }

    #[test]
    fn rle_roundtrip_and_rejects_bad_data() {
        let mut s = OctreeSection::new(OctreeSectionPos::new(1, 4, -2, 9));
        for y in 0..10 {
            for z in 0..32 {
                for x in 0..32 {
                    s.set(x, y, z, Voxel::new(STONE, 2, 15, 0));
                }
            }
        }
        s.set(5, 20, 5, Voxel::new(LOG, 2, 15, 0));
        let rle = s.to_rle();
        assert!(rle.runs.len() < 10, "runs: {}", rle.runs.len());
        let back = OctreeSection::from_rle(&rle).unwrap();
        assert_eq!(back.to_dense(), s.to_dense());

        let mut bad = rle.clone();
        bad.runs[0].length += 1;
        assert_eq!(
            OctreeSection::from_rle(&bad).unwrap_err(),
            OctreeError::BadRunTotal
        );
        let mut bad = rle.clone();
        bad.runs[0].palette_index = 99;
        assert_eq!(
            OctreeSection::from_rle(&bad).unwrap_err(),
            OctreeError::PaletteIndexOutOfRange
        );
    }

    #[test]
    fn uniform_section_is_one_run() {
        let s = OctreeSection::filled(
            OctreeSectionPos::new(0, 0, 0, 0),
            Voxel::new(STONE, 0, 0, 0),
        );
        let rle = s.to_rle();
        assert_eq!(rle.runs.len(), 1);
        assert_eq!(rle.runs[0].length as usize, SECTION_VOLUME);
    }

    #[test]
    fn mip_prefers_opaque_over_water_and_top_corner_on_ties() {
        let mut kids = [Voxel::AIR; 8];
        kids[0b001] = Voxel::new(WATER, 0, 15, 0);
        kids[0b100] = Voxel::new(STONE, 0, 15, 0);
        assert_eq!(mip_voxels(kids, &opacity).block_id(), STONE);

        let mut kids = [Voxel::AIR; 8];
        kids[0b000] = Voxel::new(STONE, 0, 0, 0);
        kids[0b110] = Voxel::new(LOG, 0, 0, 0);
        assert_eq!(mip_voxels(kids, &opacity).block_id(), LOG);

        // Surface-wins rule: upper Y (dy=1, idx=0b010) always beats lower Y (dy=0, idx=0b101)
        // even when the lower voxel has a higher x index.
        let mut slope = [Voxel::AIR; 8];
        slope[0b101] = Voxel::new(STONE, 0, 15, 0);
        slope[0b010] = Voxel::new(LOG, 0, 15, 0);
        assert_eq!(mip_voxels(slope, &opacity).block_id(), LOG);

        // Majority vote in the upper layer beats a single corner outlier.
        let mut maj = [Voxel::AIR; 8];
        maj[0b010] = Voxel::new(LOG, 0, 15, 0);
        maj[0b011] = Voxel::new(LOG, 0, 15, 0);
        maj[0b110] = Voxel::new(LOG, 0, 15, 0);
        maj[0b111] = Voxel::new(STONE, 0, 15, 0);
        assert_eq!(mip_voxels(maj, &opacity).block_id(), LOG);

        // Water wins over air.
        let mut kids = [Voxel::AIR; 8];
        kids[0b010] = Voxel::new(WATER, 0, 0, 0);
        assert_eq!(mip_voxels(kids, &opacity).block_id(), WATER);
    }

    #[test]
    fn mip_all_air_averages_light() {
        let mut kids = [Voxel::air_with_light(15, 0); 8];
        kids[0] = Voxel::air_with_light(7, 8);
        let m = mip_voxels(kids, &opacity);
        assert!(m.is_air());
        // sky: ceil((7 + 7*15) / 8) = 14, block: ((8 << 4) / 8) & 0xF0 = 0x10
        assert_eq!(m.sky_light(), 14);
        assert_eq!(m.light() & 0xF0, 0x10);
    }

    #[test]
    fn downsample_keeps_canopy_air_gap_and_trunk() {
        // Child (0,0,0): ground y 0..4, trunk at (8,4..12,8), canopy y 12..14 at x 6..11.
        let mut c = OctreeSection::new(OctreeSectionPos::new(0, 0, 0, 0));
        for z in 0..32 {
            for x in 0..32 {
                c.fill_column(x, z, 0, 4, Voxel::new(STONE, 0, 0, 0));
            }
        }
        c.fill_column(8, 8, 4, 12, Voxel::new(LOG, 0, 15, 0));
        for z in 6..11 {
            for x in 6..11 {
                c.fill_column(x, z, 12, 14, Voxel::new(LEAVES, 0, 15, 0));
            }
        }
        let mut kids: [Option<&OctreeSection>; 8] = [None; 8];
        kids[child_index(0, 0, 0)] = Some(&c);
        let parent = downsample_2x2x2(OctreeSectionPos::new(1, 0, 0, 0), &kids, &opacity);
        // Ground: voxel y 0..2. Trunk at parent (4, 2..6, 4). Canopy at parent y 6..7.
        assert_eq!(parent.get(0, 0, 0).block_id(), STONE);
        assert_eq!(parent.get(4, 3, 4).block_id(), LOG);
        assert_eq!(parent.get(4, 6, 4).block_id(), LEAVES);
        // Air gap between ground and canopy away from trunk, and above canopy.
        assert!(parent.get(3, 3, 3).is_air());
        assert!(parent.get(4, 8, 4).is_air());
        // Missing children stay air.
        assert!(parent.get(20, 0, 0).is_air());
        assert_eq!(parent.pos.detail, 1);
    }

    #[test]
    fn border_slabs_extract_touching_layers() {
        let mut east = OctreeSection::new(OctreeSectionPos::new(0, 1, 0, 0));
        east.set(0, 3, 4, Voxel::new(STONE, 0, 0, 0));
        east.set(1, 3, 4, Voxel::new(LOG, 0, 0, 0));
        let mut above = OctreeSection::new(OctreeSectionPos::new(0, 0, 1, 0));
        above.set(7, 0, 9, Voxel::new(WATER, 0, 0, 0));
        let mut n: [Option<&OctreeSection>; 6] = [None; 6];
        n[Face::PosX as usize] = Some(&east);
        n[Face::PosY as usize] = Some(&above);
        let b = NeighborBorders::from_neighbors(n);
        assert!(b.has(Face::PosX) && !b.has(Face::NegX));
        assert_eq!(b.get(Face::PosX, 31, 3, 4).unwrap().block_id(), STONE);
        assert!(b.get(Face::PosX, 31, 3, 5).unwrap().is_air());
        assert_eq!(b.get(Face::PosY, 7, 31, 9).unwrap().block_id(), WATER);
        assert!(b.get(Face::NegZ, 0, 0, 0).is_none());
    }

    #[test]
    fn lod_adapter_places_solid_fluid_and_extra_span_voxels() {
        use crate::lod::{ExtraSpan, SOURCE_LOADED_CHUNK, SectionPos};
        let mut lod = LodSection::new(SectionPos::new(0, 0, 0));
        let col = LodColumn::from_solid_and_fluid(
            70,
            60,
            [100, 100, 100],
            Some(74),
            [10, 20, 200],
            15,
            0,
            SOURCE_LOADED_CHUNK,
        );
        lod.set_cell_with_spans(
            3,
            4,
            col,
            vec![ExtraSpan::new(30, 20, [50, 50, 50], 0, 0, false)],
        );
        let mut next = 10;
        let secs = lod_section_to_octree(&lod, &mut |k| {
            if k.fluid {
                WATER
            } else {
                next += 1;
                next
            }
        });
        // Extra span y 20..30 is in section y=0, solid y 60..70 spans sections y=1 and y=2, water
        // y 70..74 is in section y=2 (64..96).
        let find = |y: i32| secs.iter().find(|s| s.pos.y == y).unwrap();
        assert!(!find(0).get(3, 25, 4).is_air());
        assert!(find(0).get(3, 31, 4).is_air());
        assert!(!find(1).get(3, 28, 4).is_air()); // y = 60
        assert!(find(1).get(3, 27, 4).is_air()); // y = 59
        assert!(!find(2).get(3, 4, 4).is_air() && find(2).get(3, 4, 4).block_id() != WATER); // y = 68
        assert_eq!(find(2).get(3, 6, 4).block_id(), WATER); // y = 70
        assert!(find(2).get(3, 10, 4).is_air()); // y = 74 exclusive
    }
}
