//! Distant Sunrises multi-detail-level LOD section store and quadtree selector.

use crate::config::{DEFAULT_MAX_EXTRA_SPANS, DhConfig};
use std::collections::{HashMap, HashSet};

/// Width/depth in cells of one LOD section (`64 x 64`, matching DH `FullDataSourceV2.WIDTH`).
pub const SECTION_SIZE: usize = 64;
pub const SECTION_CELLS: usize = SECTION_SIZE * SECTION_SIZE;
/// Up to `d6` (`64 << 6 = 4096` blocks per section), covering render distances past 512 chunks.
pub const MAX_DETAIL: u8 = 6;

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
    /// Iris / Distant Sunrises 4-bit material ID (`DH_BLOCK_*`, `0..=15`) for solid terrain.
    pub material_id: u8,
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
        material_id: 0,
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

    #[allow(clippy::too_many_arguments)]
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
            material_id: 0,
        }
    }

    #[inline]
    pub const fn with_material(mut self, material_id: u8) -> Self {
        self.material_id = material_id & 0x0F;
        self
    }

    #[inline]
    pub const fn with_fluid_light(mut self, sky_light: u8, block_light: u8) -> Self {
        self.fluid_rgba[3] = Self::pack_light(sky_light, block_light);
        self
    }

    pub fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..2].copy_from_slice(&self.solid_top_y.to_le_bytes());
        out[2..4].copy_from_slice(&self.solid_bottom_y.to_le_bytes());
        out[4..8].copy_from_slice(&self.solid_rgba);
        out[8..10].copy_from_slice(&self.fluid_top_y.to_le_bytes());
        out[10..14].copy_from_slice(&self.fluid_rgba);
        out[14] = self.source;
        out[15] = self.material_id & 0x0F;
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
            material_id: b[15] & 0x0F,
        }
    }
}

/// One additional vertical terrain or fluid span beneath the primary surface of an LOD column
/// (overhangs, bridges, caves, Nether caverns beneath the bedrock ceiling, End floating islands).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtraSpan {
    /// Exclusive top Y of this span in blocks.
    pub top_y: i16,
    /// Inclusive bottom Y of this span in blocks (`bottom_y < top_y`).
    pub bottom_y: i16,
    /// RGB + packed light byte (`(sky_light << 4) | block_light`).
    pub rgba: [u8; 4],
    /// `true` for translucent fluids (water/lava surface layers), `false` for solid blocks.
    pub is_fluid: bool,
    /// Iris / Distant Sunrises 4-bit material ID (`DH_BLOCK_*`, `0..=15`).
    pub material_id: u8,
}

impl ExtraSpan {
    pub const SERIALIZED_LEN: usize = 9;

    pub fn new(
        top_y: i32,
        bottom_y: i32,
        rgb: [u8; 3],
        sky_light: u8,
        block_light: u8,
        is_fluid: bool,
    ) -> Self {
        let top = top_y.clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16;
        let bot = bottom_y.clamp(i16::MIN as i32 + 1, (top as i32) - 1) as i16;
        Self {
            top_y: top,
            bottom_y: bot,
            rgba: [
                rgb[0],
                rgb[1],
                rgb[2],
                LodColumn::pack_light(sky_light, block_light),
            ],
            is_fluid,
            material_id: if is_fluid {
                crate::colors::DH_BLOCK_WATER
            } else {
                crate::colors::DH_BLOCK_UNKNOWN
            },
        }
    }

    #[inline]
    pub const fn with_material(mut self, material_id: u8) -> Self {
        self.material_id = material_id & 0x0F;
        self
    }

    pub fn to_bytes(self) -> [u8; Self::SERIALIZED_LEN] {
        let mut out = [0u8; Self::SERIALIZED_LEN];
        out[0..2].copy_from_slice(&self.top_y.to_le_bytes());
        out[2..4].copy_from_slice(&self.bottom_y.to_le_bytes());
        out[4..8].copy_from_slice(&self.rgba);
        out[8] = u8::from(self.is_fluid) | ((self.material_id & 0x0F) << 4);
        out
    }

    pub fn from_bytes(b: [u8; Self::SERIALIZED_LEN]) -> Self {
        Self {
            top_y: i16::from_le_bytes([b[0], b[1]]),
            bottom_y: i16::from_le_bytes([b[2], b[3]]),
            rgba: [b[4], b[5], b[6], b[7]],
            is_fluid: (b[8] & 0x01) != 0,
            material_id: (b[8] >> 4) & 0x0F,
        }
    }
}

/// One vertical block sample used when extracting stacked spans from a chunk column.
#[derive(Clone, Copy, Debug)]
pub struct ColumnSample {
    /// Block Y coordinate (top of block is `y + 1`, bottom is `y`).
    pub y: i32,
    pub rgb: [u8; 3],
    pub sky_light: u8,
    pub block_light: u8,
    pub is_fluid: bool,
    pub material_id: u8,
}

/// Extract a primary [`LodColumn`] plus up to `max_extra_spans` lower [`ExtraSpan`] layers from
/// a top-down list of non-air block samples in one `(x, z)` column.
///
/// Handles:
/// - Overhangs, arches, bridges, floating islands (multiple solid spans separated by air gaps);
/// - Water/fluid surfaces above submerged floors;
/// - Cave culling below `cave_culling_height` in sunlit dimensions (`!has_ceiling`);
/// - Nether-style ceiling dimensions (`has_ceiling == true`), keeping cavern spans beneath the roof.
pub fn extract_column_spans(
    top_down_samples: &[ColumnSample],
    source: u8,
    max_extra_spans: usize,
    cave_culling_enabled: bool,
    cave_culling_height: i32,
    has_ceiling: bool,
) -> (LodColumn, Vec<ExtraSpan>) {
    if top_down_samples.is_empty() {
        return (LodColumn::EMPTY, Vec::new());
    }

    // Group contiguous vertical runs of solid or fluid blocks (top-down).
    let mut raw_spans: Vec<ExtraSpan> = Vec::new();
    let mut i = 0;
    while i < top_down_samples.len() {
        let head = top_down_samples[i];
        let top_y = head.y + 1;
        let mut bottom_y = head.y;
        let mut j = i + 1;
        while j < top_down_samples.len() {
            let next = top_down_samples[j];
            if next.is_fluid == head.is_fluid && next.y == bottom_y - 1 {
                bottom_y = next.y;
                j += 1;
            } else {
                break;
            }
        }
        raw_spans.push(
            ExtraSpan::new(
                top_y,
                bottom_y,
                head.rgb,
                head.sky_light,
                head.block_light,
                head.is_fluid,
            )
            .with_material(head.material_id),
        );
        i = j;
    }

    // Apply cave culling to lower spans in open-sky dimensions (never cull the topmost solid/fluid
    // surface, and never cull Nether cavern floors where `has_ceiling` is true).
    let mut primary_fluid: Option<ExtraSpan> = None;
    let mut primary_solid: Option<ExtraSpan> = None;
    let mut extra: Vec<ExtraSpan> = Vec::new();

    for span in raw_spans {
        if span.is_fluid {
            if primary_solid.is_none() && primary_fluid.is_none() {
                primary_fluid = Some(span);
                continue;
            }
        } else if primary_solid.is_none() {
            primary_solid = Some(span);
            continue;
        }
        if !has_ceiling && cave_culling_enabled && (span.top_y as i32) <= cave_culling_height {
            continue;
        }
        if extra.len() < max_extra_spans {
            extra.push(span);
        }
    }

    let col = match (primary_solid, primary_fluid) {
        (Some(solid), fluid) => {
            let (fy, frgb) = match fluid {
                Some(f) if f.top_y > solid.top_y => {
                    (Some(f.top_y as i32), [f.rgba[0], f.rgba[1], f.rgba[2]])
                }
                _ => (None, [0, 0, 0]),
            };
            LodColumn::from_solid_and_fluid(
                solid.top_y as i32,
                solid.bottom_y as i32,
                [solid.rgba[0], solid.rgba[1], solid.rgba[2]],
                fy,
                frgb,
                LodColumn::sky_light(solid.rgba),
                LodColumn::block_light(solid.rgba),
                source,
            )
            .with_material(solid.material_id)
        }
        (None, Some(fluid)) => LodColumn::from_solid_and_fluid(
            fluid.bottom_y as i32,
            fluid.bottom_y as i32 - 4,
            [fluid.rgba[0], fluid.rgba[1], fluid.rgba[2]],
            Some(fluid.top_y as i32),
            [fluid.rgba[0], fluid.rgba[1], fluid.rgba[2]],
            LodColumn::sky_light(fluid.rgba),
            LodColumn::block_light(fluid.rgba),
            source,
        )
        .with_material(fluid.material_id),
        (None, None) => LodColumn::EMPTY,
    };

    (col, extra)
}

/// One `64 x 64` cell LOD section in memory.
#[derive(Clone, Debug)]
pub struct LodSection {
    pub pos: SectionPos,
    pub columns: Vec<LodColumn>,
    /// Optional additional vertical spans per cell (`0..SECTION_CELLS`), kept empty for simple
    /// single-surface Overworld columns so memory stays flat and cache-friendly.
    pub extra_spans: HashMap<u16, Vec<ExtraSpan>>,
    pub populated: u16,
    pub revision: u64,
    pub dirty_save: bool,
}

impl LodSection {
    pub fn new(pos: SectionPos) -> Self {
        Self {
            pos,
            columns: vec![LodColumn::EMPTY; SECTION_CELLS],
            extra_spans: HashMap::new(),
            populated: 0,
            revision: 0,
            dirty_save: false,
        }
    }

    pub fn from_columns(pos: SectionPos, columns: Vec<LodColumn>) -> Self {
        Self::from_columns_and_spans(pos, columns, HashMap::new())
    }

    pub fn from_columns_and_spans(
        pos: SectionPos,
        columns: Vec<LodColumn>,
        extra_spans: HashMap<u16, Vec<ExtraSpan>>,
    ) -> Self {
        let populated = columns.iter().filter(|c| !c.is_empty()).count() as u16;
        Self {
            pos,
            columns,
            extra_spans,
            populated,
            revision: 1,
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

    #[inline]
    pub fn extra_spans_at(&self, cx: usize, cz: usize) -> &[ExtraSpan] {
        let idx = Self::index(cx, cz) as u16;
        self.extra_spans.get(&idx).map_or(&[], Vec::as_slice)
    }

    /// Sets cell `(cx, cz)` if `col.source >= existing.source`. Returns `true` if modified.
    pub fn set_cell(&mut self, cx: usize, cz: usize, col: LodColumn) -> bool {
        self.set_cell_with_spans(cx, cz, col, Vec::new())
    }

    /// Sets cell `(cx, cz)` and its lower vertical spans if `col.source >= existing.source`.
    pub fn set_cell_with_spans(
        &mut self,
        cx: usize,
        cz: usize,
        col: LodColumn,
        spans: Vec<ExtraSpan>,
    ) -> bool {
        if cx >= SECTION_SIZE || cz >= SECTION_SIZE || col.is_empty() {
            return false;
        }
        let idx = Self::index(cx, cz);
        let cur = self.columns[idx];
        if cur.source > col.source {
            return false;
        }
        let key = idx as u16;
        let same_spans =
            self.extra_spans.get(&key).map_or(&[][..], Vec::as_slice) == spans.as_slice();
        if cur == col && same_spans {
            return false;
        }
        if cur.is_empty() {
            self.populated = self.populated.saturating_add(1);
        }
        self.columns[idx] = col;
        if spans.is_empty() {
            self.extra_spans.remove(&key);
        } else {
            self.extra_spans.insert(key, spans);
        }
        self.revision = self.revision.wrapping_add(1);
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

    /// Downsamples both primary columns and extra vertical spans for a 2x2 child block.
    pub fn downsample_2x2_with_spans(
        &self,
        cx: usize,
        cz: usize,
        max_extra_spans: usize,
    ) -> (LodColumn, Vec<ExtraSpan>) {
        let col = self.downsample_2x2(cx, cz);
        if col.is_empty() || max_extra_spans == 0 || self.extra_spans.is_empty() {
            return (col, Vec::new());
        }
        let coords = [(cx, cz), (cx + 1, cz), (cx, cz + 1), (cx + 1, cz + 1)];
        let mut all_extra: Vec<ExtraSpan> = Vec::new();
        for (x, z) in coords {
            all_extra.extend_from_slice(self.extra_spans_at(x, z));
        }
        if all_extra.is_empty() {
            return (col, Vec::new());
        }
        // Sort descending by top_y and merge vertically overlapping spans of the same material kind.
        all_extra.sort_by_key(|s| std::cmp::Reverse(s.top_y));
        let mut merged: Vec<ExtraSpan> = Vec::new();
        for span in all_extra {
            if col.has_solid() && span.top_y >= col.solid_bottom_y && !span.is_fluid {
                continue;
            }
            if let Some(last) = merged.last_mut()
                && last.is_fluid == span.is_fluid
                && span.top_y >= last.bottom_y
            {
                last.bottom_y = last.bottom_y.min(span.bottom_y);
                continue;
            }
            if merged.len() < max_extra_spans {
                merged.push(span);
            }
        }
        (col, merged)
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
    let mut mat_counts = [0u8; 16];
    let mut mat_top_y = [EMPTY_Y; 16];

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
            let m = (col.material_id & 0x0F) as usize;
            mat_counts[m] += 1;
            mat_top_y[m] = mat_top_y[m].max(col.solid_top_y);
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

    let mut dominant_mat = 0u8;
    let mut best_count = 0u8;
    let mut best_y = EMPTY_Y;
    for m in 0..16 {
        let c = mat_counts[m];
        if c > best_count || (c == best_count && c > 0 && mat_top_y[m] > best_y) {
            best_count = c;
            best_y = mat_top_y[m];
            dominant_mat = m as u8;
        }
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
        material_id: dominant_mat,
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
        self.propagate_up_with_quality(start_pos, DEFAULT_MAX_EXTRA_SPANS)
    }

    /// Propagate dirty detail `d` updates up to parent detail `d + 1`..=`MAX_DETAIL`, preserving
    /// up to `max_extra_spans` stacked vertical spans per column.
    pub fn propagate_up_with_quality(
        &mut self,
        start_pos: SectionPos,
        max_extra_spans: usize,
    ) -> Vec<SectionPos> {
        let mut updated = Vec::new();
        let mut cur_pos = start_pos;
        while let Some(parent_pos) = cur_pos.parent() {
            let Some((child_columns, child_extra)) = self
                .sections
                .get(&cur_pos)
                .map(|s| (s.columns.clone(), s.extra_spans.clone()))
            else {
                break;
            };
            let child_tmp = LodSection {
                pos: cur_pos,
                columns: child_columns,
                extra_spans: child_extra,
                populated: 0,
                revision: 0,
                dirty_save: false,
            };
            let qx = (cur_pos.x & 1) as usize * (SECTION_SIZE / 2);
            let qz = (cur_pos.z & 1) as usize * (SECTION_SIZE / 2);
            let parent = self.get_or_create_mut(parent_pos);
            let mut changed = false;
            for pz in 0..(SECTION_SIZE / 2) {
                for px in 0..(SECTION_SIZE / 2) {
                    let (down, down_spans) =
                        child_tmp.downsample_2x2_with_spans(px * 2, pz * 2, max_extra_spans);
                    if !down.is_empty()
                        && parent.set_cell_with_spans(qx + px, qz + pz, down, down_spans)
                    {
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

    /// Returns the **leaf** sections to populate, mirroring DH's inside-out `LodQuadTree`: each
    /// distance ring is queued only at its target detail level (a node inside its split distance
    /// is replaced by its four children, never queued alongside them). The ring right outside
    /// vanilla's render distance therefore holds only `detail 0` sections; coarse parents are
    /// filled by downsampling from their children (`propagate_up`) or, in far rings, are
    /// themselves the target leaf.
    ///
    /// Order follows DH `WorldGenerationQueue.tryStartNextWorldGenTask` (lines 302-304):
    /// ascending `chebyshevDist(center, cam) / (detail + 1)`, so the sections nearest the camera
    /// come first and coarse leaves are favoured at equal distance. (DH divides by `detail`;
    /// `detail + 1` keeps detail 0 defined.)
    pub fn desired_sections_coarse_to_fine(
        cam_x: f64,
        cam_z: f64,
        config: DhConfig,
    ) -> Vec<SectionPos> {
        let max_dist = config.render_distance_blocks() as f64;
        let root_span = SectionPos::new(MAX_DETAIL, 0, 0).block_span() as f64;
        let radius_roots = (max_dist / root_span).ceil() as i32 + 1;
        let center_root =
            SectionPos::from_block(MAX_DETAIL, cam_x.floor() as i32, cam_z.floor() as i32);

        let mut leaves = Vec::new();
        for rz in (center_root.z - radius_roots)..=(center_root.z + radius_roots) {
            for rx in (center_root.x - radius_roots)..=(center_root.x + radius_roots) {
                collect_leaf_nodes(
                    SectionPos::new(MAX_DETAIL, rx, rz),
                    cam_x,
                    cam_z,
                    max_dist,
                    config,
                    &mut leaves,
                );
            }
        }
        let score = |p: SectionPos| {
            let (ox, oz) = p.block_origin();
            let half = (p.block_span() as f64) * 0.5;
            let chebyshev = ((ox as f64 + half) - cam_x)
                .abs()
                .max(((oz as f64 + half) - cam_z).abs());
            chebyshev / (p.detail as f64 + 1.0)
        };
        leaves.sort_by(|a, b| {
            score(*a)
                .total_cmp(&score(*b))
                .then_with(|| b.detail.cmp(&a.detail))
                .then_with(|| (a.x, a.z).cmp(&(b.x, b.z)))
        });
        leaves
    }
}

/// Leaf-only traversal: a node inside its split distance is replaced by its children; any other
/// in-range node is the target leaf for its ring and is emitted.
fn collect_leaf_nodes(
    pos: SectionPos,
    cam_x: f64,
    cam_z: f64,
    max_dist: f64,
    config: DhConfig,
    leaves: &mut Vec<SectionPos>,
) {
    let dist = pos.min_horizontal_distance(cam_x, cam_z);
    if dist > max_dist {
        return;
    }
    if pos.detail > 0
        && dist < config.detail_split_distance_blocks(pos.detail) as f64
        && let Some(children) = pos.children()
    {
        for child in children {
            collect_leaf_nodes(child, cam_x, cam_z, max_dist, config, leaves);
        }
        return;
    }
    leaves.push(pos);
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
    fn multi_span_extraction_handles_overhangs_cave_culling_and_nether_ceiling() {
        // Column with an overhang at Y=100..=104, air at Y=70..=99, ground at Y=64..=69,
        // and a deep cave floor at Y=20..=25.
        let mut samples = Vec::new();
        for y in (100..=104).rev() {
            samples.push(ColumnSample {
                y,
                rgb: [140, 140, 140],
                sky_light: 15,
                block_light: 0,
                is_fluid: false,
                material_id: crate::colors::DH_BLOCK_STONE,
            });
        }
        for y in (64..=69).rev() {
            samples.push(ColumnSample {
                y,
                rgb: [120, 180, 90],
                sky_light: 8,
                block_light: 0,
                is_fluid: false,
                material_id: crate::colors::DH_BLOCK_GRASS,
            });
        }
        for y in (20..=25).rev() {
            samples.push(ColumnSample {
                y,
                rgb: [80, 80, 80],
                sky_light: 0,
                block_light: 4,
                is_fluid: false,
                material_id: crate::colors::DH_BLOCK_DEEPSLATE,
            });
        }

        // Open-sky with cave culling at Y=60 keeps the overhang + ground (Y=64..70), culls Y=20..26.
        let (col, extra) = extract_column_spans(&samples, SOURCE_LOADED_CHUNK, 4, true, 60, false);
        assert_eq!(col.solid_top_y, 105);
        assert_eq!(col.solid_bottom_y, 100);
        assert_eq!(col.material_id, crate::colors::DH_BLOCK_STONE);
        assert_eq!(extra.len(), 1);
        assert_eq!((extra[0].top_y, extra[0].bottom_y), (70, 64));
        assert_eq!(extra[0].material_id, crate::colors::DH_BLOCK_GRASS);

        // Nether ceiling mode ignores cave culling so deep cavern floors stay visible.
        let (_, nether_extra) =
            extract_column_spans(&samples, SOURCE_LOADED_CHUNK, 4, true, 60, true);
        assert_eq!(nether_extra.len(), 2);
        assert_eq!((nether_extra[1].top_y, nether_extra[1].bottom_y), (26, 20));
        assert_eq!(
            nether_extra[1].material_id,
            crate::colors::DH_BLOCK_DEEPSLATE
        );
    }

    #[test]
    fn desired_sections_are_leaf_only_with_detail_zero_near_camera() {
        let cfg = DhConfig::default();
        let (cx, cz) = (1234.5, -987.25);
        let desired = DimensionLodStore::desired_sections_coarse_to_fine(cx, cz, cfg);
        assert!(!desired.is_empty());

        // The section(s) covering the camera, and everything inside detail 1's split distance,
        // are detail 0 leaves; no detail 6/5/4 ancestor covers the camera.
        let near = cfg.detail_split_distance_blocks(1) as f64;
        for &p in &desired {
            let d = p.min_horizontal_distance(cx, cz);
            if d < near {
                assert_eq!(p.detail, 0, "near section {p:?} at {d} must be detail 0");
            }
            if p.detail > 0 {
                assert!(
                    d >= cfg.detail_split_distance_blocks(p.detail) as f64,
                    "{p:?} is inside its split distance yet queued as a coarse leaf"
                );
            }
        }
        let cam_cover = |p: &SectionPos| p.min_horizontal_distance(cx, cz) == 0.0;
        assert!(desired.iter().any(|p| p.detail == 0 && cam_cover(p)));
        assert!(
            desired
                .iter()
                .filter(|p| cam_cover(p))
                .all(|p| p.detail == 0)
        );

        // Leaf-only: no queued section is an ancestor of another queued section.
        let set: HashSet<SectionPos> = desired.iter().copied().collect();
        assert_eq!(set.len(), desired.len());
        for &p in &desired {
            let mut cur = p;
            while let Some(parent) = cur.parent() {
                assert!(!set.contains(&parent), "{parent:?} queued above leaf {p:?}");
                cur = parent;
            }
        }
    }

    #[test]
    fn desired_sections_are_ordered_nearest_first() {
        let cfg = DhConfig::default();
        let desired = DimensionLodStore::desired_sections_coarse_to_fine(0.5, 0.5, cfg);
        assert_eq!(desired[0].detail, 0);
        assert_eq!(desired[0].min_horizontal_distance(0.5, 0.5), 0.0);
    }
}
