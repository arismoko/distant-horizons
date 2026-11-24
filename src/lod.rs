//! Distant Horizons multi-detail-level LOD section store and quadtree selector.

use crate::config::DhConfig;
use std::collections::{HashMap, HashSet};

/// Width/depth in cells of one LOD section (`64 x 64`, matching DH `FullDataSourceV2.WIDTH`).
pub const SECTION_SIZE: usize = 64;
pub const SECTION_CELLS: usize = SECTION_SIZE * SECTION_SIZE;
pub const MAX_DETAIL: u8 = 4;

/// Sentinel Y value indicating no solid or fluid block in a column.
pub const EMPTY_Y: i16 = i16::MIN;

/// Provenance priority for an LOD column: finer/authoritative sources never get overwritten by
/// coarser generator estimates.
pub const SOURCE_EMPTY: u8 = 0;
pub const SOURCE_DISTANT_SURFACE: u8 = 1;
pub const SOURCE_LOADED_CHUNK: u8 = 2;

/// Unique key for one `64 x 64` cell section at a given detail level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SectionPos {
    pub detail: u8,
    pub x: i32,
    pub z: i32,
}

impl SectionPos {
    pub const fn new(detail: u8, x: i32, z: i32) -> Self {
        Self { detail, x, z }
    }

    /// Block width of a single cell at this detail level (`1 << detail`).
    pub const fn cell_size(self) -> i32 {
        1 << self.detail
    }

    /// Block span of the entire 64-cell section (`64 << detail`).
    pub const fn block_span(self) -> i32 {
        (SECTION_SIZE as i32) << self.detail
    }

    /// World block `(x, z)` of cell `(0, 0)`.
    pub const fn block_origin(self) -> (i32, i32) {
        let span = self.block_span();
        (self.x * span, self.z * span)
    }

    /// World center `(x, z)` in blocks.
    pub fn block_center(self) -> (f64, f64) {
        let (ox, oz) = self.block_origin();
        let half = (self.block_span() as f64) * 0.5;
        (ox as f64 + half, oz as f64 + half)
    }

    /// Section containing world block `(bx, bz)` at `detail`.
    pub const fn from_block(detail: u8, bx: i32, bz: i32) -> Self {
        let shift = 6 + detail as u32;
        Self {
            detail,
            x: bx >> shift,
            z: bz >> shift,
        }
    }

    /// Local cell coordinates `(cx, cz)` in `0..64` for world block `(bx, bz)`.
    pub const fn local_cell(self, bx: i32, bz: i32) -> (usize, usize) {
        let (ox, oz) = self.block_origin();
        let cx = ((bx - ox) >> self.detail) as usize;
        let cz = ((bz - oz) >> self.detail) as usize;
        (cx, cz)
    }

    /// Parent section at `detail + 1`.
    pub const fn parent(self) -> Option<Self> {
        if self.detail >= MAX_DETAIL {
            None
        } else {
            Some(Self {
                detail: self.detail + 1,
                x: self.x >> 1,
                z: self.z >> 1,
            })
        }
    }

    /// Four children at `detail - 1`.
    pub const fn children(self) -> Option<[Self; 4]> {
        if self.detail == 0 {
            return None;
        }
        let d = self.detail - 1;
        let bx = self.x << 1;
        let bz = self.z << 1;
        Some([
            Self::new(d, bx, bz),
            Self::new(d, bx + 1, bz),
            Self::new(d, bx, bz + 1),
            Self::new(d, bx + 1, bz + 1),
        ])
    }

    /// Horizontal distance from camera `(cam_x, cam_z)` to the nearest point on this section's
    /// horizontal footprint.
    pub fn min_horizontal_distance(self, cam_x: f64, cam_z: f64) -> f64 {
        let (ox, oz) = self.block_origin();
        let span = self.block_span() as f64;
        let min_x = ox as f64;
        let max_x = min_x + span;
        let min_z = oz as f64;
        let max_z = min_z + span;
        let dx = if cam_x < min_x {
            min_x - cam_x
        } else if cam_x > max_x {
            cam_x - max_x
        } else {
            0.0
        };
        let dz = if cam_z < min_z {
            min_z - cam_z
        } else if cam_z > max_z {
            cam_z - max_z
        } else {
            0.0
        };
        (dx * dx + dz * dz).sqrt()
    }
}

/// Compact 16-byte representation of one LOD column (solid terrain + optional translucent fluid).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodColumn {
    /// Highest solid block top Y (`EMPTY_Y` when unpopulated).
    pub solid_top_y: i16,
    /// Skirt bottom Y for cliffs/void edges.
    pub solid_bottom_y: i16,
    /// Solid RGB + packed light byte (`(sky_light << 4) | block_light`).
    pub solid_rgba: [u8; 4],
    /// Top Y of translucent fluid (water) above the solid floor (`EMPTY_Y` when dry).
    pub fluid_top_y: i16,
    /// Fluid RGB + packed light byte.
    pub fluid_rgba: [u8; 4],
    /// `SOURCE_EMPTY`, `SOURCE_DISTANT_SURFACE`, or `SOURCE_LOADED_CHUNK`.
    pub source: u8,
    pub _pad: u8,
}

impl Default for LodColumn {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl LodColumn {
    pub const EMPTY: Self = Self {
        solid_top_y: EMPTY_Y,
        solid_bottom_y: EMPTY_Y,
        solid_rgba: [0, 0, 0, 0xF0],
        fluid_top_y: EMPTY_Y,
        fluid_rgba: [0, 0, 0, 0xF0],
        source: SOURCE_EMPTY,
        _pad: 0,
    };

    pub const fn is_empty(self) -> bool {
        self.solid_top_y == EMPTY_Y && self.fluid_top_y == EMPTY_Y
    }

    pub const fn has_solid(self) -> bool {
        self.solid_top_y != EMPTY_Y
    }

    pub const fn has_fluid(self) -> bool {
        self.fluid_top_y != EMPTY_Y
    }

    pub const fn pack_light(sky: u8, block: u8) -> u8 {
        ((sky & 0x0F) << 4) | (block & 0x0F)
    }

    pub const fn sky_light(rgba: [u8; 4]) -> u8 {
        (rgba[3] >> 4) & 0x0F
    }

    pub const fn block_light(rgba: [u8; 4]) -> u8 {
        rgba[3] & 0x0F
    }

    pub fn from_solid_and_fluid(
        solid_top_y: i32,
        solid_bottom_y: i32,
        solid_rgb: [u8; 3],
        fluid_top_y: Option<i32>,
        fluid_rgb: [u8; 3],
        sky_light: u8,
        block_light: u8,
        source: u8,
    ) -> Self {
        let light = Self::pack_light(sky_light, block_light);
        let solid_top = solid_top_y.clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16;
        let solid_bot = solid_bottom_y.clamp(i16::MIN as i32 + 1, solid_top as i32) as i16;
        let fluid_top = match fluid_top_y {
            Some(fy) if fy > solid_top_y => fy.clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16,
            _ => EMPTY_Y,
        };
        Self {
            solid_top_y: solid_top,
            solid_bottom_y: solid_bot,
            solid_rgba: [solid_rgb[0], solid_rgb[1], solid_rgb[2], light],
            fluid_top_y: fluid_top,
            fluid_rgba: [fluid_rgb[0], fluid_rgb[1], fluid_rgb[2], light],
            source,
            _pad: 0,
        }
    }

    pub fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..2].copy_from_slice(&self.solid_top_y.to_le_bytes());
        out[2..4].copy_from_slice(&self.solid_bottom_y.to_le_bytes());
        out[4..8].copy_from_slice(&self.solid_rgba);
        out[8..10].copy_from_slice(&self.fluid_top_y.to_le_bytes());
        out[10..14].copy_from_slice(&self.fluid_rgba);
        out[14] = self.source;
        out[15] = 0;
        out
    }

    pub fn from_bytes(b: [u8; 16]) -> Self {
        Self {
            solid_top_y: i16::from_le_bytes([b[0], b[1]]),
            solid_bottom_y: i16::from_le_bytes([b[2], b[3]]),
            solid_rgba: [b[4], b[5], b[6], b[7]],
            fluid_top_y: i16::from_le_bytes([b[8], b[9]]),
            fluid_rgba: [b[10], b[11], b[12], b[13]],
            source: b[14],
            _pad: 0,
        }
    }
}

/// One `64 x 64` cell LOD section in memory.
#[derive(Clone, Debug)]
pub struct LodSection {
    pub pos: SectionPos,
    pub columns: Vec<LodColumn>,
    pub populated: u16,
    pub revision: u64,
    pub dirty_mesh: bool,
    pub dirty_save: bool,
}

impl LodSection {
    pub fn new(pos: SectionPos) -> Self {
        Self {
            pos,
            columns: vec![LodColumn::EMPTY; SECTION_CELLS],
            populated: 0,
            revision: 0,
            dirty_mesh: false,
            dirty_save: false,
        }
    }

    pub fn from_columns(pos: SectionPos, columns: Vec<LodColumn>) -> Self {
        let populated = columns.iter().filter(|c| !c.is_empty()).count() as u16;
        Self {
            pos,
            columns,
            populated,
            revision: 1,
            dirty_mesh: populated > 0,
            dirty_save: false,
        }
    }

    #[inline]
    pub fn index(cx: usize, cz: usize) -> usize {
        cz * SECTION_SIZE + cx
    }

    #[inline]
    pub fn get(&self, cx: usize, cz: usize) -> LodColumn {
        self.columns[Self::index(cx, cz)]
    }

    /// Sets cell `(cx, cz)` if `col.source >= existing.source`. Returns `true` if modified.
    pub fn set_cell(&mut self, cx: usize, cz: usize, col: LodColumn) -> bool {
        if cx >= SECTION_SIZE || cz >= SECTION_SIZE || col.is_empty() {
            return false;
        }
        let idx = Self::index(cx, cz);
        let cur = self.columns[idx];
        if cur.source > col.source {
            return false;
        }
        if cur == col {
            return false;
        }
        if cur.is_empty() {
            self.populated = self.populated.saturating_add(1);
        }
        self.columns[idx] = col;
        self.revision = self.revision.wrapping_add(1);
        self.dirty_mesh = true;
        self.dirty_save = true;
        true
    }

    /// Downsamples a 2x2 block of cells `(cx..cx+2, cz..cz+2)` into one parent cell.
    pub fn downsample_2x2(&self, cx: usize, cz: usize) -> LodColumn {
        let samples = [
            self.get(cx, cz),
            self.get(cx + 1, cz),
            self.get(cx, cz + 1),
            self.get(cx + 1, cz + 1),
        ];
        downsample_quad(samples)
    }
}

/// Combine 4 child LOD columns into one parent column by averaging surface height and selecting
/// representative surface color (preferring loaded-chunk data and water where >= 2 cells are wet).
pub fn downsample_quad(samples: [LodColumn; 4]) -> LodColumn {
    let mut solid_count = 0i32;
    let mut top_sum = 0i32;
    let mut bot_min = i16::MAX;
    let mut r_sum = 0u32;
    let mut g_sum = 0u32;
    let mut b_sum = 0u32;
    let mut sky_max = 0u8;
    let mut blk_max = 0u8;
    let mut best_source = SOURCE_EMPTY;

    let mut fluid_count = 0i32;
    let mut fluid_top_max = EMPTY_Y;
    let mut fr_sum = 0u32;
    let mut fg_sum = 0u32;
    let mut fb_sum = 0u32;

    for col in samples {
        if col.is_empty() {
            continue;
        }
        best_source = best_source.max(col.source);
        if col.has_solid() {
            solid_count += 1;
            top_sum += col.solid_top_y as i32;
            bot_min = bot_min.min(col.solid_bottom_y);
            r_sum += col.solid_rgba[0] as u32;
            g_sum += col.solid_rgba[1] as u32;
            b_sum += col.solid_rgba[2] as u32;
            sky_max = sky_max.max(LodColumn::sky_light(col.solid_rgba));
            blk_max = blk_max.max(LodColumn::block_light(col.solid_rgba));
        }
        if col.has_fluid() {
            fluid_count += 1;
            fluid_top_max = fluid_top_max.max(col.fluid_top_y);
            fr_sum += col.fluid_rgba[0] as u32;
            fg_sum += col.fluid_rgba[1] as u32;
            fb_sum += col.fluid_rgba[2] as u32;
        }
    }

    if solid_count == 0 && fluid_count == 0 {
        return LodColumn::EMPTY;
    }

    let light = LodColumn::pack_light(sky_max, blk_max);
    let (solid_top_y, solid_bottom_y, solid_rgba) = if solid_count > 0 {
        let avg_top = (top_sum / solid_count).clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16;
        let bot = bot_min.min(avg_top);
        (
            avg_top,
            bot,
            [
                (r_sum / solid_count as u32) as u8,
                (g_sum / solid_count as u32) as u8,
                (b_sum / solid_count as u32) as u8,
                light,
            ],
        )
    } else {
        (EMPTY_Y, EMPTY_Y, [0, 0, 0, light])
    };

    let (fluid_top_y, fluid_rgba) = if fluid_count >= 2 && fluid_top_max > solid_top_y {
        (
            fluid_top_max,
            [
                (fr_sum / fluid_count as u32) as u8,
                (fg_sum / fluid_count as u32) as u8,
                (fb_sum / fluid_count as u32) as u8,
                light,
            ],
        )
    } else {
        (EMPTY_Y, [0, 0, 0, light])
    };

    LodColumn {
        solid_top_y,
        solid_bottom_y,
        solid_rgba,
        fluid_top_y,
        fluid_rgba,
        source: best_source,
        _pad: 0,
    }
}

/// Per-dimension LOD section hierarchy and sampling state.
#[derive(Default)]
pub struct DimensionLodStore {
    pub sections: HashMap<SectionPos, LodSection>,
    pub disk_checked: HashSet<SectionPos>,
    pub sampled_distant_sections: HashSet<SectionPos>,
    pub generated_columns: u64,
}

impl DimensionLodStore {
    pub fn get_or_create_mut(&mut self, pos: SectionPos) -> &mut LodSection {
        self.sections
            .entry(pos)
            .or_insert_with(|| LodSection::new(pos))
    }

    /// Propagate dirty detail `d` updates up to parent detail `d + 1`..=`MAX_DETAIL`.
    pub fn propagate_up(&mut self, start_pos: SectionPos) -> Vec<SectionPos> {
        let mut updated = Vec::new();
        let mut cur_pos = start_pos;
        while let Some(parent_pos) = cur_pos.parent() {
            let Some(child_snapshot) = self.sections.get(&cur_pos).map(|s| s.columns.clone())
            else {
                break;
            };
            let child_tmp = LodSection {
                pos: cur_pos,
                columns: child_snapshot,
                populated: 0,
                revision: 0,
                dirty_mesh: false,
                dirty_save: false,
            };
            let qx = (cur_pos.x & 1) as usize * (SECTION_SIZE / 2);
            let qz = (cur_pos.z & 1) as usize * (SECTION_SIZE / 2);
            let parent = self.get_or_create_mut(parent_pos);
            let mut changed = false;
            for pz in 0..(SECTION_SIZE / 2) {
                for px in 0..(SECTION_SIZE / 2) {
                    let down = child_tmp.downsample_2x2(px * 2, pz * 2);
                    if !down.is_empty() && parent.set_cell(qx + px, qz + pz, down) {
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
            updated.push(parent_pos);
            cur_pos = parent_pos;
        }
        updated
    }

    /// Returns the desired LOD sections covering the player's render distance via top-down
    /// quadtree traversal. If a node should subdivide but its children aren't ready yet,
    /// `ready_meshes` lets the parent render as a seamless fallback until children arrive.
    pub fn select_visible_sections(
        cam_x: f64,
        cam_z: f64,
        config: DhConfig,
        ready_meshes: &HashSet<SectionPos>,
    ) -> Vec<SectionPos> {
        let max_dist = config.render_distance_blocks() as f64;
        let root_span = SectionPos::new(MAX_DETAIL, 0, 0).block_span() as f64;
        let radius_roots = (max_dist / root_span).ceil() as i32 + 1;
        let center_root =
            SectionPos::from_block(MAX_DETAIL, cam_x.floor() as i32, cam_z.floor() as i32);

        let mut selected = Vec::new();
        for rz in (center_root.z - radius_roots)..=(center_root.z + radius_roots) {
            for rx in (center_root.x - radius_roots)..=(center_root.x + radius_roots) {
                let root = SectionPos::new(MAX_DETAIL, rx, rz);
                select_node(
                    root,
                    cam_x,
                    cam_z,
                    max_dist,
                    config,
                    ready_meshes,
                    &mut selected,
                );
            }
        }
        // Sort front-to-back for early-Z rejection on the GPU.
        selected.sort_by(|a, b| {
            let da = a.min_horizontal_distance(cam_x, cam_z);
            let db = b.min_horizontal_distance(cam_x, cam_z);
            da.total_cmp(&db)
        });
        selected
    }

    /// Returns sections to populate in coarse-to-fine progressive order (detail `MAX_DETAIL` down
    /// to `0`, nearest-first within each detail level) so the full horizon fills in seconds and
    /// then sharpens smoothly without holes.
    pub fn desired_sections_coarse_to_fine(
        cam_x: f64,
        cam_z: f64,
        config: DhConfig,
    ) -> Vec<SectionPos> {
        let max_dist = config.render_distance_blocks() as f64;
        let mut by_detail: [Vec<SectionPos>; (MAX_DETAIL as usize) + 1] = Default::default();
        let root_span = SectionPos::new(MAX_DETAIL, 0, 0).block_span() as f64;
        let radius_roots = (max_dist / root_span).ceil() as i32 + 1;
        let center_root =
            SectionPos::from_block(MAX_DETAIL, cam_x.floor() as i32, cam_z.floor() as i32);

        for rz in (center_root.z - radius_roots)..=(center_root.z + radius_roots) {
            for rx in (center_root.x - radius_roots)..=(center_root.x + radius_roots) {
                collect_progressive_nodes(
                    SectionPos::new(MAX_DETAIL, rx, rz),
                    cam_x,
                    cam_z,
                    max_dist,
                    config,
                    &mut by_detail,
                );
            }
        }
        let mut ordered = Vec::new();
        for detail in (0..=MAX_DETAIL as usize).rev() {
            let bucket = &mut by_detail[detail];
            bucket.sort_by(|a, b| {
                a.min_horizontal_distance(cam_x, cam_z)
                    .total_cmp(&b.min_horizontal_distance(cam_x, cam_z))
            });
            ordered.extend(bucket.iter().copied());
        }
        ordered
    }
}

fn select_node(
    pos: SectionPos,
    cam_x: f64,
    cam_z: f64,
    max_dist: f64,
    config: DhConfig,
    ready_meshes: &HashSet<SectionPos>,
    out: &mut Vec<SectionPos>,
) {
    let dist = pos.min_horizontal_distance(cam_x, cam_z);
    if dist > max_dist {
        return;
    }
    if pos.detail > 0 && dist < config.detail_split_distance_blocks(pos.detail) as f64 {
        if let Some(children) = pos.children() {
            // Subdivide if at least one child is ready (or if parent itself isn't ready).
            let any_child_ready = children.iter().any(|c| ready_meshes.contains(c));
            let all_children_ready = children.iter().all(|c| ready_meshes.contains(c));
            if all_children_ready || (!ready_meshes.contains(&pos) && any_child_ready) {
                for child in children {
                    select_node(child, cam_x, cam_z, max_dist, config, ready_meshes, out);
                }
                return;
            }
            // Also check grandchildren when parent isn't ready so finer meshes always show.
            if !ready_meshes.contains(&pos) {
                for child in children {
                    select_node(child, cam_x, cam_z, max_dist, config, ready_meshes, out);
                }
                return;
            }
        }
    }
    if ready_meshes.contains(&pos) {
        out.push(pos);
    }
}

fn collect_progressive_nodes(
    pos: SectionPos,
    cam_x: f64,
    cam_z: f64,
    max_dist: f64,
    config: DhConfig,
    by_detail: &mut [Vec<SectionPos>; (MAX_DETAIL as usize) + 1],
) {
    let dist = pos.min_horizontal_distance(cam_x, cam_z);
    if dist > max_dist {
        return;
    }
    by_detail[pos.detail as usize].push(pos);
    if pos.detail > 0 && dist < config.detail_split_distance_blocks(pos.detail) as f64 {
        if let Some(children) = pos.children() {
            for child in children {
                collect_progressive_nodes(child, cam_x, cam_z, max_dist, config, by_detail);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_coordinates_and_downsampling_roundtrip() {
        let s0 = SectionPos::from_block(0, -1, 64);
        assert_eq!(s0, SectionPos::new(0, -1, 1));
        assert_eq!(s0.block_origin(), (-64, 64));
        assert_eq!(s0.local_cell(-1, 64), (63, 0));

        let parent = s0.parent().unwrap();
        assert_eq!(parent, SectionPos::new(1, -1, 0));
        assert!(parent.children().unwrap().contains(&s0));

        let mut store = DimensionLodStore::default();
        let col = LodColumn::from_solid_and_fluid(
            70,
            60,
            [120, 180, 90],
            Some(63),
            [63, 118, 228],
            15,
            0,
            SOURCE_LOADED_CHUNK,
        );
        {
            let sec = store.get_or_create_mut(s0);
            assert!(sec.set_cell(0, 0, col));
            // Lower priority source must not overwrite loaded chunk data.
            let weaker = LodColumn::from_solid_and_fluid(
                50,
                40,
                [10, 10, 10],
                None,
                [0, 0, 0],
                15,
                0,
                SOURCE_DISTANT_SURFACE,
            );
            assert!(!sec.set_cell(0, 0, weaker));
        }
        let propagated = store.propagate_up(s0);
        assert_eq!(propagated.len(), MAX_DETAIL as usize);
        let p = store.sections.get(&parent).unwrap();
        assert_eq!(p.get(32, 32).solid_top_y, 70);
    }

    #[test]
    fn quadtree_selection_falls_back_to_parent_when_children_unready() {
        let cfg = DhConfig::default();
        let root = SectionPos::new(MAX_DETAIL, 0, 0);
        let mut ready = HashSet::new();
        ready.insert(root);
        let vis = DimensionLodStore::select_visible_sections(10.0, 10.0, cfg, &ready);
        assert_eq!(vis, vec![root]);
    }
}
