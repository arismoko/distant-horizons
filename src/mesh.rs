//! Greedy face-culled LOD mesher and background meshing worker thread.

use crate::lod::{EMPTY_Y, ExtraSpan, LodColumn, LodSection, SECTION_SIZE, SectionPos};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

/// Packed 20-byte LOD GPU vertex (`Float32x3`, `Uint32`, `Uint32`).
#[derive(Clone, Copy, Debug)]
pub struct LodVertex {
    pub pos: [f32; 3],
    pub color: u32,
    pub meta: u32,
}

impl LodVertex {
    pub const STRIDE: usize = 20;

    #[inline]
    pub fn pack_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
        (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
    }

    #[inline]
    pub fn pack_meta(
        normal_idx: u32,
        sky_light: u8,
        block_light: u8,
        is_fluid: bool,
        material_id: u8,
    ) -> u32 {
        Self::pack_meta_ext(
            normal_idx,
            sky_light,
            block_light,
            is_fluid,
            material_id,
            0,
            0,
            false,
            false,
            false,
        )
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn pack_meta_ext(
        normal_idx: u32,
        sky_light: u8,
        block_light: u8,
        is_fluid: bool,
        material_id: u8,
        detail: u8,
        source: u8,
        is_generic: bool,
        unshaded: bool,
        skip_clip_dist: bool,
    ) -> u32 {
        (normal_idx & 0x7)
            | (((sky_light as u32) & 0xF) << 4)
            | (((block_light as u32) & 0xF) << 8)
            | ((is_fluid as u32) << 12)
            | (((material_id as u32) & 0xF) << 13)
            | (((detail as u32) & 0x7) << 17)
            | (((source as u32) & 0x3) << 20)
            | ((is_generic as u32) << 22)
            | ((unshaded as u32) << 23)
            | ((skip_clip_dist as u32) << 24)
    }

    #[inline]
    pub fn append_bytes(self, dst: &mut Vec<u8>) {
        dst.extend_from_slice(&self.pos[0].to_le_bytes());
        dst.extend_from_slice(&self.pos[1].to_le_bytes());
        dst.extend_from_slice(&self.pos[2].to_le_bytes());
        dst.extend_from_slice(&self.color.to_le_bytes());
        dst.extend_from_slice(&self.meta.to_le_bytes());
    }
}

/// Completed CPU mesh buffers ready for GPU upload.
#[derive(Clone, Debug)]
pub struct SectionMeshData {
    pub dimension: String,
    pub pos: SectionPos,
    pub revision: u64,
    pub solid_vertex_bytes: Vec<u8>,
    pub solid_indices: Vec<u32>,
    pub fluid_vertex_bytes: Vec<u8>,
    pub fluid_indices: Vec<u32>,
    pub min_y: f32,
    pub max_y: f32,
    pub quad_count: u32,
}

impl SectionMeshData {
    pub fn is_empty(&self) -> bool {
        self.solid_indices.is_empty() && self.fluid_indices.is_empty()
    }
}

struct MeshBuilder {
    verts: Vec<u8>,
    indices: Vec<u32>,
    vert_count: u32,
    quad_count: u32,
}

impl MeshBuilder {
    fn with_capacity(quads: usize) -> Self {
        Self {
            verts: Vec::with_capacity(quads * 4 * LodVertex::STRIDE),
            indices: Vec::with_capacity(quads * 6),
            vert_count: 0,
            quad_count: 0,
        }
    }

    #[inline]
    fn push_quad(&mut self, corners: [[f32; 3]; 4], color: u32, meta: u32) {
        let base = self.vert_count;
        for pos in corners {
            LodVertex { pos, color, meta }.append_bytes(&mut self.verts);
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self.vert_count += 4;
        self.quad_count += 1;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct TopKey {
    y: i16,
    rgba: [u8; 4],
    material_id: u8,
    source: u8,
}

/// Build an optimized solid + translucent fluid mesh for `section`.
/// Positions are local to `section.pos.block_origin()` in X/Z and absolute in Y.
pub fn build_section_mesh(dimension: String, section: &LodSection) -> SectionMeshData {
    let cell = section.pos.cell_size() as f32;
    let detail = section.pos.detail;
    let mut solid = MeshBuilder::with_capacity(512);
    let mut fluid = MeshBuilder::with_capacity(128);
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    // 1. Greedy-merge solid top faces (+Y, normal index 1).
    let mut visited_solid = [false; SECTION_SIZE * SECTION_SIZE];
    for gz in 0..SECTION_SIZE {
        for gx in 0..SECTION_SIZE {
            let idx = LodSection::index(gx, gz);
            if visited_solid[idx] {
                continue;
            }
            let col = section.columns[idx];
            if !col.has_solid() {
                continue;
            }
            let key = TopKey {
                y: col.solid_top_y,
                rgba: col.solid_rgba,
                material_id: col.material_id,
                source: col.source,
            };
            let mut w = 1usize;
            while gx + w < SECTION_SIZE {
                let nidx = LodSection::index(gx + w, gz);
                if visited_solid[nidx] {
                    break;
                }
                let ncol = section.columns[nidx];
                if !ncol.has_solid()
                    || ncol.solid_top_y != key.y
                    || ncol.solid_rgba != key.rgba
                    || ncol.material_id != key.material_id
                    || ncol.source != key.source
                {
                    break;
                }
                w += 1;
            }
            let mut h = 1usize;
            'grow_z: while gz + h < SECTION_SIZE {
                for dx in 0..w {
                    let nidx = LodSection::index(gx + dx, gz + h);
                    if visited_solid[nidx] {
                        break 'grow_z;
                    }
                    let ncol = section.columns[nidx];
                    if !ncol.has_solid()
                        || ncol.solid_top_y != key.y
                        || ncol.solid_rgba != key.rgba
                        || ncol.material_id != key.material_id
                        || ncol.source != key.source
                    {
                        break 'grow_z;
                    }
                }
                h += 1;
            }
            for dz in 0..h {
                for dx in 0..w {
                    visited_solid[LodSection::index(gx + dx, gz + dz)] = true;
                }
            }

            let x0 = gx as f32 * cell;
            let x1 = (gx + w) as f32 * cell;
            let z0 = gz as f32 * cell;
            let z1 = (gz + h) as f32 * cell;
            let y = key.y as f32;
            min_y = min_y.min(y);
            max_y = max_y.max(y);

            let color = LodVertex::pack_rgba(key.rgba[0], key.rgba[1], key.rgba[2], 255);
            let meta = LodVertex::pack_meta_ext(
                1,
                LodColumn::sky_light(key.rgba),
                LodColumn::block_light(key.rgba),
                false,
                key.material_id,
                detail,
                key.source,
                false,
                false,
                false,
            );
            // CCW outward (+Y)
            solid.push_quad(
                [[x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0]],
                color,
                meta,
            );
        }
    }

    // 2. Greedy-merge translucent fluid top faces (+Y, normal index 1).
    let water_alpha = crate::resource_colors::water_still_rgba()[3];
    let mut visited_fluid = [false; SECTION_SIZE * SECTION_SIZE];
    for gz in 0..SECTION_SIZE {
        for gx in 0..SECTION_SIZE {
            let idx = LodSection::index(gx, gz);
            if visited_fluid[idx] {
                continue;
            }
            let col = section.columns[idx];
            if !col.has_fluid() {
                continue;
            }
            let key = TopKey {
                y: col.fluid_top_y,
                rgba: col.fluid_rgba,
                material_id: crate::colors::DH_BLOCK_WATER,
                source: col.source,
            };
            let mut w = 1usize;
            while gx + w < SECTION_SIZE {
                let nidx = LodSection::index(gx + w, gz);
                if visited_fluid[nidx] {
                    break;
                }
                let ncol = section.columns[nidx];
                if !ncol.has_fluid() || ncol.fluid_top_y != key.y || ncol.fluid_rgba != key.rgba {
                    break;
                }
                w += 1;
            }
            let mut h = 1usize;
            'grow_fz: while gz + h < SECTION_SIZE {
                for dx in 0..w {
                    let nidx = LodSection::index(gx + dx, gz + h);
                    if visited_fluid[nidx] {
                        break 'grow_fz;
                    }
                    let ncol = section.columns[nidx];
                    if !ncol.has_fluid() || ncol.fluid_top_y != key.y || ncol.fluid_rgba != key.rgba
                    {
                        break 'grow_fz;
                    }
                }
                h += 1;
            }
            for dz in 0..h {
                for dx in 0..w {
                    visited_fluid[LodSection::index(gx + dx, gz + dz)] = true;
                }
            }

            let x0 = gx as f32 * cell;
            let x1 = (gx + w) as f32 * cell;
            let z0 = gz as f32 * cell;
            let z1 = (gz + h) as f32 * cell;
            // Water surface sits slightly below full block top (vanilla 14/16 = 0.875)
            let y = key.y as f32 - 0.125;
            min_y = min_y.min(y);
            max_y = max_y.max(y);

            let color = LodVertex::pack_rgba(key.rgba[0], key.rgba[1], key.rgba[2], water_alpha);
            let meta = LodVertex::pack_meta_ext(
                1,
                LodColumn::sky_light(key.rgba),
                LodColumn::block_light(key.rgba),
                true,
                key.material_id,
                detail,
                key.source,
                false,
                false,
                false,
            );
            fluid.push_quad(
                [[x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0]],
                color,
                meta,
            );
        }
    }

    // 3. Emit vertical cliff/step side walls where a solid column stands taller than its neighbor
    //    (plus clean skirts along section boundaries so adjacent detail levels never show cracks).
    let edge_skirt = (4 * section.pos.cell_size()).min(32) as i16;
    for gz in 0..SECTION_SIZE {
        for gx in 0..SECTION_SIZE {
            let col = section.get(gx, gz);
            if !col.has_solid() {
                continue;
            }
            let top = col.solid_top_y;
            let extra = section.extra_spans_at(gx, gz);
            let has_overhang_gap = extra
                .first()
                .is_some_and(|below| col.solid_bottom_y > below.top_y);
            let bot = if has_overhang_gap {
                col.solid_bottom_y
            } else {
                col.solid_bottom_y.min(top.saturating_sub(edge_skirt))
            };
            let x0 = gx as f32 * cell;
            let x1 = (gx + 1) as f32 * cell;
            let z0 = gz as f32 * cell;
            let z1 = (gz + 1) as f32 * cell;
            let y1 = top as f32;
            let color =
                LodVertex::pack_rgba(col.solid_rgba[0], col.solid_rgba[1], col.solid_rgba[2], 255);
            let sky = LodColumn::sky_light(col.solid_rgba);
            let blk = LodColumn::block_light(col.solid_rgba);

            // North (-Z, normal 2)
            let north_y = if gz > 0 {
                neighbor_solid_y(section.get(gx, gz - 1), bot)
            } else {
                bot
            };
            if top > north_y {
                let y0 = north_y as f32;
                min_y = min_y.min(y0);
                let meta = LodVertex::pack_meta_ext(
                    2,
                    sky,
                    blk,
                    false,
                    col.material_id,
                    detail,
                    col.source,
                    false,
                    false,
                    false,
                );
                solid.push_quad(
                    [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
                    color,
                    meta,
                );
            }

            // South (+Z, normal 3)
            let south_y = if gz + 1 < SECTION_SIZE {
                neighbor_solid_y(section.get(gx, gz + 1), bot)
            } else {
                bot
            };
            if top > south_y {
                let y0 = south_y as f32;
                min_y = min_y.min(y0);
                let meta = LodVertex::pack_meta_ext(
                    3,
                    sky,
                    blk,
                    false,
                    col.material_id,
                    detail,
                    col.source,
                    false,
                    false,
                    false,
                );
                solid.push_quad(
                    [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                    color,
                    meta,
                );
            }

            // West (-X, normal 4)
            let west_y = if gx > 0 {
                neighbor_solid_y(section.get(gx - 1, gz), bot)
            } else {
                bot
            };
            if top > west_y {
                let y0 = west_y as f32;
                min_y = min_y.min(y0);
                let meta = LodVertex::pack_meta_ext(
                    4,
                    sky,
                    blk,
                    false,
                    col.material_id,
                    detail,
                    col.source,
                    false,
                    false,
                    false,
                );
                solid.push_quad(
                    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
                    color,
                    meta,
                );
            }

            // East (+X, normal 5)
            let east_y = if gx + 1 < SECTION_SIZE {
                neighbor_solid_y(section.get(gx + 1, gz), bot)
            } else {
                bot
            };
            if top > east_y {
                let y0 = east_y as f32;
                min_y = min_y.min(y0);
                let meta = LodVertex::pack_meta_ext(
                    5,
                    sky,
                    blk,
                    false,
                    col.material_id,
                    detail,
                    col.source,
                    false,
                    false,
                    false,
                );
                solid.push_quad(
                    [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
                    color,
                    meta,
                );
            }

            // Underside ceiling face (-Y, normal 0) when this primary span is an overhang or
            // Nether ceiling above a lower span separated by an air gap.
            if has_overhang_gap {
                let y_bot = col.solid_bottom_y as f32;
                min_y = min_y.min(y_bot);
                let meta = LodVertex::pack_meta_ext(
                    0,
                    sky,
                    blk,
                    false,
                    col.material_id,
                    detail,
                    col.source,
                    false,
                    false,
                    false,
                );
                solid.push_quad(
                    [
                        [x0, y_bot, z1],
                        [x0, y_bot, z0],
                        [x1, y_bot, z0],
                        [x1, y_bot, z1],
                    ],
                    color,
                    meta,
                );
            }
        }
    }

    // 4. Emit lower vertical spans (overhangs, caves, Nether caverns, End islands) with top,
    //    underside (when suspended above air), and neighbor-culled side walls.
    if !section.extra_spans.is_empty() {
        for (&cell_idx, spans) in &section.extra_spans {
            let idx = cell_idx as usize;
            let gx = idx % SECTION_SIZE;
            let gz = idx / SECTION_SIZE;
            let x0 = gx as f32 * cell;
            let x1 = (gx + 1) as f32 * cell;
            let z0 = gz as f32 * cell;
            let z1 = (gz + 1) as f32 * cell;

            for (si, span) in spans.iter().enumerate() {
                if span.top_y <= span.bottom_y {
                    continue;
                }
                let y0 = span.bottom_y as f32;
                let y1 = if span.is_fluid {
                    span.top_y as f32 - 0.125
                } else {
                    span.top_y as f32
                };
                min_y = min_y.min(y0);
                max_y = max_y.max(y1);
                let alpha = if span.is_fluid { water_alpha } else { 255 };
                let color = LodVertex::pack_rgba(span.rgba[0], span.rgba[1], span.rgba[2], alpha);
                let sky = LodColumn::sky_light(span.rgba);
                let blk = LodColumn::block_light(span.rgba);
                let target = if span.is_fluid {
                    &mut fluid
                } else {
                    &mut solid
                };

                // Top (+Y, normal 1)
                let col_src = section.get(gx, gz).source;
                let top_meta = LodVertex::pack_meta_ext(
                    1,
                    sky,
                    blk,
                    span.is_fluid,
                    span.material_id,
                    detail,
                    col_src,
                    false,
                    false,
                    false,
                );
                target.push_quad(
                    [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]],
                    color,
                    top_meta,
                );

                // Underside (-Y, normal 0) if there is an air gap to a lower span beneath this one.
                if !span.is_fluid
                    && let Some(next_below) = spans.get(si + 1)
                    && span.bottom_y > next_below.top_y
                {
                    let bot_meta = LodVertex::pack_meta_ext(
                        0,
                        sky,
                        blk,
                        false,
                        span.material_id,
                        detail,
                        col_src,
                        false,
                        false,
                        false,
                    );
                    target.push_quad(
                        [[x0, y0, z1], [x0, y0, z0], [x1, y0, z0], [x1, y0, z1]],
                        color,
                        bot_meta,
                    );
                }

                if !span.is_fluid {
                    // Emit side faces only where the neighboring cell does not cover this span.
                    type SideQuad = ((isize, isize), u32, [[f32; 3]; 4]);
                    let sides: [SideQuad; 4] = [
                        (
                            (0, -1),
                            2,
                            [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
                        ),
                        (
                            (0, 1),
                            3,
                            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                        ),
                        (
                            (-1, 0),
                            4,
                            [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
                        ),
                        (
                            (1, 0),
                            5,
                            [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
                        ),
                    ];
                    for ((dx, dz), normal, quad) in sides {
                        let nx = gx as isize + dx;
                        let nz = gz as isize + dz;
                        let covered = if nx >= 0
                            && nz >= 0
                            && (nx as usize) < SECTION_SIZE
                            && (nz as usize) < SECTION_SIZE
                        {
                            neighbor_covers_span(section, nx as usize, nz as usize, span)
                        } else {
                            false
                        };
                        if !covered {
                            let meta = LodVertex::pack_meta_ext(
                                normal,
                                sky,
                                blk,
                                false,
                                span.material_id,
                                detail,
                                col_src,
                                false,
                                false,
                                false,
                            );
                            target.push_quad(quad, color, meta);
                        }
                    }
                }
            }
        }
    }

    if !min_y.is_finite() {
        min_y = 0.0;
        max_y = 0.0;
    }

    SectionMeshData {
        dimension,
        pos: section.pos,
        revision: section.revision,
        solid_vertex_bytes: solid.verts,
        solid_indices: solid.indices,
        fluid_vertex_bytes: fluid.verts,
        fluid_indices: fluid.indices,
        min_y,
        max_y,
        quad_count: solid.quad_count + fluid.quad_count,
    }
}

/// True when any solid span in neighbor `(nx, nz)` completely covers `[span.bottom_y, span.top_y]`.
#[inline]
fn neighbor_covers_span(section: &LodSection, nx: usize, nz: usize, span: &ExtraSpan) -> bool {
    let col = section.get(nx, nz);
    if col.has_solid() && col.solid_bottom_y <= span.bottom_y && col.solid_top_y >= span.top_y {
        return true;
    }
    section.extra_spans_at(nx, nz).iter().any(|other| {
        !other.is_fluid && other.bottom_y <= span.bottom_y && other.top_y >= span.top_y
    })
}

/// Apply Distant Sunrises brightness and saturation multipliers to an sRGB color.
#[inline]
pub fn grade_rgb(rgb: [u8; 3], brightness: f32, saturation: f32) -> [u8; 3] {
    if (brightness - 1.0).abs() < 1e-4 && (saturation - 1.0).abs() < 1e-4 {
        return rgb;
    }
    let r = rgb[0] as f32 / 255.0;
    let g = rgb[1] as f32 / 255.0;
    let b = rgb[2] as f32 / 255.0;
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let sr = (luma + (r - luma) * saturation) * brightness;
    let sg = (luma + (g - luma) * saturation) * brightness;
    let sb = (luma + (b - luma) * saturation) * brightness;
    [
        (sr.clamp(0.0, 1.0) * 255.0).round() as u8,
        (sg.clamp(0.0, 1.0) * 255.0).round() as u8,
        (sb.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

#[inline]
fn neighbor_solid_y(neighbor: LodColumn, fallback: i16) -> i16 {
    if neighbor.solid_top_y == EMPTY_Y {
        fallback
    } else {
        neighbor.solid_top_y
    }
}

pub struct MeshJob {
    pub dimension: String,
    pub section: LodSection,
}

/// Background meshing worker thread producing GPU-ready vertex/index byte buffers.
pub struct MeshWorker {
    tx: Sender<MeshJob>,
    rx: Mutex<Receiver<SectionMeshData>>,
}

impl MeshWorker {
    pub fn spawn() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<MeshJob>();
        let (out_tx, out_rx) = mpsc::channel::<SectionMeshData>();
        let _ = thread::Builder::new()
            .name("dh-lod-mesh".into())
            .spawn(move || {
                while let Ok(job) = job_rx.recv() {
                    let mesh = build_section_mesh(job.dimension, &job.section);
                    let _ = out_tx.send(mesh);
                }
            });
        Self {
            tx: job_tx,
            rx: Mutex::new(out_rx),
        }
    }

    pub fn submit(&self, dimension: String, section: LodSection) {
        let _ = self.tx.send(MeshJob { dimension, section });
    }

    pub fn drain_finished(&self) -> Vec<SectionMeshData> {
        let mut out = Vec::new();
        if let Ok(rx) = self.rx.lock() {
            while let Ok(mesh) = rx.try_recv() {
                out.push(mesh);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Voxy-style 8-byte packed quads for 32³ octree sections
// ---------------------------------------------------------------------------------------------

/// One 8-byte quad in the exact bit layout of Voxy's `quad_format.glsl` (64-bit path) read by
/// `quads3.vert`:
///
/// | bits  | field |
/// |-------|-------|
/// | 0-2   | face (`Face::id`: 0 `-Y`, 1 `+Y`, 2 `-Z`, 3 `+Z`, 4 `-X`, 5 `+X`) |
/// | 3-6   | `w - 1` (first tangent extent, 1..=16) |
/// | 7-10  | `h - 1` (second tangent extent, 1..=16) |
/// | 11-15 | local z of the quad's minimum voxel |
/// | 16-20 | local y |
/// | 21-25 | local x |
/// | 26-45 | model id (`extractStateId`, 20 bits): index into the baked model table; carries the
///           `dhMaterialId` / `mc_Entity` of its [`crate::bakery::GpuModelEntry`] |
/// | 46-54 | biome id (`extractBiomeId`, 9 bits): biome tint row |
/// | 55-62 | light byte (`extractLightId`): `block_light << 4 | sky_light` |
///
/// Tangent axes (`RenderDataFactory.Mesher`): `±Y` faces `w` along X and `h` along Z; `±Z` faces
/// `w` along X and `h` along Y; `±X` faces `w` along Y and `h` along Z. The position is the
/// minimum voxel of the quad (not the face plane), so it always fits 5 bits; the face id selects
/// the side of the voxel the quad lies on. The extent limit of 16 is Voxy's
/// (`RenderDataFactory`: "length out of bounds" check, 4-bit fields).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedQuad(pub u64);

impl PackedQuad {
    /// Pack a quad; every field is masked to its width (`w`, `h` are clamped to `1..=16`).
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        x: u8,
        y: u8,
        z: u8,
        face: u8,
        w: u8,
        h: u8,
        model_id: u32,
        biome_id: u16,
        sky_light: u8,
        block_light: u8,
    ) -> Self {
        let w = if w == 0 {
            1
        } else if w > 16 {
            16
        } else {
            w
        };
        let h = if h == 0 {
            1
        } else if h > 16 {
            16
        } else {
            h
        };
        let light = ((block_light & 0xF) << 4) | (sky_light & 0xF);
        Self(
            (face as u64 & 0x7)
                | (((w - 1) as u64) << 3)
                | (((h - 1) as u64) << 7)
                | ((z as u64 & 0x1F) << 11)
                | ((y as u64 & 0x1F) << 16)
                | ((x as u64 & 0x1F) << 21)
                | ((model_id as u64 & 0xF_FFFF) << 26)
                | ((biome_id as u64 & 0x1FF) << 46)
                | ((light as u64) << 55),
        )
    }

    pub const fn face(self) -> u8 {
        (self.0 & 0x7) as u8
    }
    /// Width in voxels (`extractSize().x`).
    pub const fn width(self) -> u8 {
        ((self.0 >> 3) & 0xF) as u8 + 1
    }
    /// Height in voxels (`extractSize().y`).
    pub const fn height(self) -> u8 {
        ((self.0 >> 7) & 0xF) as u8 + 1
    }
    pub const fn z(self) -> u8 {
        ((self.0 >> 11) & 0x1F) as u8
    }
    pub const fn y(self) -> u8 {
        ((self.0 >> 16) & 0x1F) as u8
    }
    pub const fn x(self) -> u8 {
        ((self.0 >> 21) & 0x1F) as u8
    }
    pub const fn model_id(self) -> u32 {
        ((self.0 >> 26) & 0xF_FFFF) as u32
    }
    pub const fn biome_id(self) -> u16 {
        ((self.0 >> 46) & 0x1FF) as u16
    }
    /// Raw light byte, `block_light << 4 | sky_light`.
    pub const fn light(self) -> u8 {
        ((self.0 >> 55) & 0xFF) as u8
    }
    pub const fn sky_light(self) -> u8 {
        self.light() & 0xF
    }
    pub const fn block_light(self) -> u8 {
        self.light() >> 4
    }
    /// `quad == 0` is Voxy's empty-quad marker (`isQuadEmpty`).
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Meshed 32³ section: opaque and translucent quads in separate slices (separate draw ranges, as
/// Voxy keeps translucents in their own buffer), plus the section-local AABB and counts that feed
/// the indirect draw command (`DrawIndexedIndirect`: `index_count = quads * 6`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SectionQuads {
    pub pos: crate::octree::OctreeSectionPos,
    pub opaque: Vec<PackedQuad>,
    pub translucent: Vec<PackedQuad>,
    /// Section-local minimum corner in voxels (`0..=32`); `[0; 3]` when empty.
    pub aabb_min: [u8; 3],
    /// Section-local maximum corner in voxels (`0..=32`).
    pub aabb_max: [u8; 3],
    pub revision: u64,
}

impl SectionQuads {
    pub fn opaque_count(&self) -> u32 {
        self.opaque.len() as u32
    }
    pub fn translucent_count(&self) -> u32 {
        self.translucent.len() as u32
    }
    pub fn quad_count(&self) -> u32 {
        self.opaque_count() + self.translucent_count()
    }
    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.translucent.is_empty()
    }
    /// Indices a quad-index-buffer draw needs for the opaque / translucent ranges (6 per quad).
    pub fn index_counts(&self) -> (u32, u32) {
        (self.opaque_count() * 6, self.translucent_count() * 6)
    }
    /// World-space AABB in blocks (`pos.block_origin() + aabb * voxel_size`).
    pub fn world_aabb(&self) -> ([f64; 3], [f64; 3]) {
        let o = self.pos.block_origin();
        let s = f64::from(self.pos.voxel_size());
        let f = |i: usize, v: u8| f64::from(o[i]) + f64::from(v) * s;
        (
            [
                f(0, self.aabb_min[0]),
                f(1, self.aabb_min[1]),
                f(2, self.aabb_min[2]),
            ],
            [
                f(0, self.aabb_max[0]),
                f(1, self.aabb_max[1]),
                f(2, self.aabb_max[2]),
            ],
        )
    }
}

#[inline]
fn slice_voxel(face: crate::octree::Face, s: usize, a: usize, b: usize) -> [usize; 3] {
    match face.axis() {
        0 => [a, s, b],
        1 => [a, b, s],
        _ => [s, a, b],
    }
}

#[inline]
fn tangent_extent(face: crate::octree::Face, w: usize, h: usize) -> [usize; 3] {
    match face.axis() {
        0 => [w, 0, h],
        1 => [w, h, 0],
        _ => [0, w, h],
    }
}

/// Greedy-mesh one 32³ octree section into packed quads (`RenderDataFactory`): for each of the six
/// faces and each of the 32 slices, a 32×32 mask of visible faces is merged into rectangles of
/// identical (model, biome, light, layer), at most 16×16 (4-bit size fields).
///
/// A face is hidden when the voxel across it is in the same section (or in a loaded neighbour
/// slice of `borders`) and either its model occludes the opposite face
/// (`ModelQueries.faceOccludes`) or both are the same block and the model culls itself
/// (`cullsSame`: water, glass). A border that is not loaded reads as air, so faces on it are
/// emitted. The quad's light is the light of the voxel in front of it (the voxel's own light when
/// the neighbour is not loaded).
pub fn mesh_octree_section(
    section: &crate::octree::OctreeSection,
    borders: &crate::octree::NeighborBorders,
    models: &dyn crate::bakery::ModelLookup,
) -> SectionQuads {
    use crate::bakery::ModelMeta;
    use crate::octree::{Face, SECTION_EDGE, Voxel, voxel_index};
    use std::collections::HashMap;

    const N: usize = SECTION_EDGE;
    let dense = section.to_dense();
    let mut metas: HashMap<u32, ModelMeta> = HashMap::new();
    let mut meta_of = |v: Voxel| -> ModelMeta {
        if v.is_air() {
            return ModelMeta::EMPTY;
        }
        *metas
            .entry(v.block_id())
            .or_insert_with(|| models.meta(v.block_id()))
    };

    let mut out = SectionQuads {
        pos: section.pos,
        revision: section.revision,
        ..Default::default()
    };
    let mut bmin = [N as u8; 3];
    let mut bmax = [0u8; 3];
    let mut mask = vec![0u64; N * N];
    let mut used = vec![false; N * N];

    for face in Face::ALL {
        let (dx, dy, dz) = face.offset();
        for s in 0..N {
            mask.iter_mut().for_each(|m| *m = 0);
            used.iter_mut().for_each(|u| *u = false);
            let mut any = false;
            for b in 0..N {
                for a in 0..N {
                    let [x, y, z] = slice_voxel(face, s, a, b);
                    let v = dense[voxel_index(x, y, z)];
                    let meta = meta_of(v);
                    if meta.is_empty() || !meta.has_face(face) {
                        continue;
                    }
                    let (nx, ny, nz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                    let inside = (0..N as i32).contains(&nx)
                        && (0..N as i32).contains(&ny)
                        && (0..N as i32).contains(&nz);
                    let neighbor = if inside {
                        Some(dense[voxel_index(nx as usize, ny as usize, nz as usize)])
                    } else {
                        borders.get(face, x, y, z)
                    };
                    if let Some(nv) = neighbor
                        && !nv.is_air()
                    {
                        let nm = meta_of(nv);
                        if nm.face_occludes(face.opposite())
                            || (meta.culls_same() && nv.block_id() == v.block_id())
                        {
                            continue;
                        }
                    }
                    let light = neighbor.map_or(v.light(), |n| n.light());
                    let biome = if meta.biome_coloured() {
                        v.biome_id()
                    } else {
                        0
                    };
                    mask[b * N + a] = u64::from(meta.model_id)
                        | (u64::from(biome) << 20)
                        | (u64::from(light) << 29)
                        | (u64::from(meta.translucent()) << 37);
                    any = true;
                }
            }
            if !any {
                continue;
            }
            for b in 0..N {
                for a in 0..N {
                    let key = mask[b * N + a];
                    if key == 0 || used[b * N + a] {
                        continue;
                    }
                    let mut w = 1;
                    while a + w < N && w < 16 && mask[b * N + a + w] == key && !used[b * N + a + w]
                    {
                        w += 1;
                    }
                    let mut h = 1;
                    'grow: while b + h < N && h < 16 {
                        for aa in a..a + w {
                            if mask[(b + h) * N + aa] != key || used[(b + h) * N + aa] {
                                break 'grow;
                            }
                        }
                        h += 1;
                    }
                    for bb in b..b + h {
                        for aa in a..a + w {
                            used[bb * N + aa] = true;
                        }
                    }
                    let [x, y, z] = slice_voxel(face, s, a, b);
                    let q = PackedQuad::new(
                        x as u8,
                        y as u8,
                        z as u8,
                        face.id(),
                        w as u8,
                        h as u8,
                        (key & 0xF_FFFF) as u32,
                        ((key >> 20) & 0x1FF) as u16,
                        ((key >> 29) & 0xF) as u8,
                        ((key >> 33) & 0xF) as u8,
                    );
                    if (key >> 37) & 1 == 1 {
                        out.translucent.push(q);
                    } else {
                        out.opaque.push(q);
                    }
                    let ext = tangent_extent(face, w, h);
                    let lo = [x, y, z];
                    for i in 0..3 {
                        let hi = lo[i] + ext[i].max(usize::from(i == face.world_axis()));
                        bmin[i] = bmin[i].min(lo[i] as u8);
                        bmax[i] = bmax[i].max(hi as u8);
                    }
                }
            }
        }
    }
    if !out.is_empty() {
        out.aabb_min = bmin;
        out.aabb_max = bmax;
    }
    out
}

#[cfg(test)]
mod octree_mesh_tests {
    use super::*;
    use crate::bakery::{
        MODEL_BIOME_COLOURED, MODEL_CULLS_SAME, MODEL_FLUID, MODEL_FULLY_OPAQUE, MODEL_TRANSLUCENT,
        ModelMeta,
    };
    use crate::octree::{Face, NeighborBorders, OctreeSection, OctreeSectionPos, Voxel};

    const STONE: u32 = 1;
    const LOG: u32 = 2;
    const LEAVES: u32 = 3;
    const WATER: u32 = 4;

    fn models(id: u32) -> ModelMeta {
        match id {
            STONE | LOG => ModelMeta {
                model_id: id,
                flags: MODEL_FULLY_OPAQUE,
                face_exists: 0x3F,
                face_occludes: 0x3F,
                light_emission: 0,
            },
            // Leaves: every face exists but none occludes (cut-out texture), biome tinted.
            LEAVES => ModelMeta {
                model_id: 3,
                flags: MODEL_BIOME_COLOURED,
                face_exists: 0x3F,
                face_occludes: 0,
                light_emission: 0,
            },
            WATER => ModelMeta {
                model_id: 4,
                flags: MODEL_TRANSLUCENT | MODEL_FLUID | MODEL_CULLS_SAME | MODEL_BIOME_COLOURED,
                face_exists: 0x3F,
                face_occludes: 0,
                light_emission: 0,
            },
            _ => ModelMeta::EMPTY,
        }
    }

    fn pos() -> OctreeSectionPos {
        OctreeSectionPos::new(0, 0, 0, 0)
    }

    fn area(quads: &[PackedQuad], face: Face, plane: impl Fn(&PackedQuad) -> u8, at: u8) -> u32 {
        quads
            .iter()
            .filter(|q| q.face() == face.id() && plane(q) == at)
            .map(|q| u32::from(q.width()) * u32::from(q.height()))
            .sum()
    }

    #[test]
    fn packed_quad_is_eight_bytes_and_roundtrips_every_field() {
        assert_eq!(std::mem::size_of::<PackedQuad>(), 8);
        let q = PackedQuad::new(31, 17, 5, 5, 16, 3, 0xF_FFFF, 0x1FF, 15, 9);
        assert_eq!(
            (q.x(), q.y(), q.z(), q.face(), q.width(), q.height()),
            (31, 17, 5, 5, 16, 3)
        );
        assert_eq!(
            (q.model_id(), q.biome_id(), q.sky_light(), q.block_light()),
            (0xF_FFFF, 0x1FF, 15, 9)
        );
        assert_eq!(q.light(), 0x9F);
        // Bit positions are quads3.vert's: face 0-2, w 3-6, h 7-10, z 11, y 16, x 21, model 26.
        let q = PackedQuad::new(1, 2, 3, 4, 2, 3, 7, 0, 0, 0);
        assert_eq!(
            q.0,
            4 | (1 << 3) | (2 << 7) | (3 << 11) | (2 << 16) | (1 << 21) | (7 << 26)
        );
        let bytes = bytemuck::bytes_of(&q);
        assert_eq!(u64::from_le_bytes(bytes.try_into().unwrap()), q.0);
        // Out-of-range input is masked and sizes clamp, never panic.
        let q = PackedQuad::new(40, 0, 0, 9, 0, 99, 0, 0, 0, 0);
        assert_eq!((q.x(), q.face(), q.width(), q.height()), (8, 1, 1, 16));
        assert!(PackedQuad::default().is_empty());
    }

    /// Every visible voxel face of a section with canopies, overhangs, caves and water is covered
    /// by exactly one quad: no missing and no overlapping faces.
    #[test]
    fn greedy_quads_cover_each_visible_face_exactly_once() {
        use std::collections::HashMap;
        const N: usize = 32;
        let mut s = OctreeSection::new(pos());
        // Terrain with a cave and an overhang.
        for x in 0..N {
            for z in 0..N {
                let h = 8 + (x / 6 + z / 5) % 4;
                for y in 0..h {
                    let cave =
                        (10..14).contains(&x) && (12..20).contains(&z) && (4..7).contains(&y);
                    if !cave {
                        s.set(x, y, z, Voxel::new(STONE, 0, 0, 0));
                    }
                }
            }
        }
        for x in 20..28 {
            for z in 4..10 {
                s.set(x, 18, z, Voxel::new(STONE, 0, 15, 0)); // floating overhang slab
            }
        }
        // Tree: trunk plus leaf canopy with air gaps.
        for y in 10..16 {
            s.set(16, y, 16, Voxel::new(LOG, 0, 15, 0));
        }
        for x in 13..20usize {
            for z in 13..20usize {
                for y in 15..19usize {
                    if (x + y + z) % 5 != 0 {
                        s.set(x, y, z, Voxel::new(LEAVES, 3, 15, 0));
                    }
                }
            }
        }
        // A water pool.
        for x in 2..9 {
            for z in 22..30 {
                for y in 6..10 {
                    s.set(x, y, z, Voxel::new(WATER, 1, 15, 0));
                }
            }
        }

        let dense = s.to_dense();
        let at = |x: i32, y: i32, z: i32| {
            dense[crate::octree::voxel_index(x as usize, y as usize, z as usize)]
        };
        let mut expected: HashMap<([i32; 3], u8), u32> = HashMap::new();
        for x in 0..N as i32 {
            for y in 0..N as i32 {
                for z in 0..N as i32 {
                    let v = at(x, y, z);
                    let meta = models(v.block_id());
                    if v.is_air() || meta.is_empty() {
                        continue;
                    }
                    for face in Face::ALL {
                        if !meta.has_face(face) {
                            continue;
                        }
                        let (dx, dy, dz) = face.offset();
                        let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                        if (0..N as i32).contains(&nx)
                            && (0..N as i32).contains(&ny)
                            && (0..N as i32).contains(&nz)
                        {
                            let nv = at(nx, ny, nz);
                            if !nv.is_air() {
                                let nm = models(nv.block_id());
                                if nm.face_occludes(face.opposite())
                                    || (meta.culls_same() && nv.block_id() == v.block_id())
                                {
                                    continue;
                                }
                            }
                        }
                        expected.insert(([x, y, z], face.id()), 0);
                    }
                }
            }
        }
        assert!(expected.len() > 500);

        let m = mesh_octree_section(&s, &NeighborBorders::none(), &models);
        for q in m.opaque.iter().chain(&m.translucent) {
            let face = Face::ALL[q.face() as usize];
            let ext = tangent_extent(face, usize::from(q.width()), usize::from(q.height()));
            let lo = [q.x() as usize, q.y() as usize, q.z() as usize];
            for dx in 0..ext[0].max(usize::from(face.world_axis() == 0)) {
                for dy in 0..ext[1].max(usize::from(face.world_axis() == 1)) {
                    for dz in 0..ext[2].max(usize::from(face.world_axis() == 2)) {
                        let v = [
                            (lo[0] + dx) as i32,
                            (lo[1] + dy) as i32,
                            (lo[2] + dz) as i32,
                        ];
                        match expected.get_mut(&(v, q.face())) {
                            Some(n) => *n += 1,
                            None => panic!("quad covers a hidden face at {v:?} face {}", q.face()),
                        }
                    }
                }
            }
        }
        for (key, n) in &expected {
            assert_eq!(*n, 1, "face {key:?} covered {n} times");
        }
    }

    #[test]
    fn full_stone_section_is_24_greedy_quads_with_aabb() {
        let s = OctreeSection::filled(pos(), Voxel::new(STONE, 0, 15, 0));
        let m = mesh_octree_section(&s, &NeighborBorders::none(), &models);
        // 6 faces x (32/16)^2 quads of 16x16.
        assert_eq!(m.opaque_count(), 24);
        assert!(m.translucent.is_empty());
        assert_eq!(m.aabb_min, [0, 0, 0]);
        assert_eq!(m.aabb_max, [32, 32, 32]);
        assert!(m.opaque.iter().all(|q| q.width() == 16 && q.height() == 16));
        assert_eq!(m.index_counts(), (144, 0));
    }

    #[test]
    fn tree_canopy_air_gap_trunk_and_ground() {
        let mut s = OctreeSection::new(pos());
        for z in 0..32 {
            for x in 0..32 {
                s.fill_column(x, z, 0, 4, Voxel::new(STONE, 1, 15, 0));
            }
        }
        s.fill_column(16, 16, 4, 10, Voxel::new(LOG, 1, 15, 0));
        for z in 14..19 {
            for x in 14..19 {
                s.fill_column(x, z, 10, 12, Voxel::new(LEAVES, 1, 15, 0));
            }
        }
        let m = mesh_octree_section(&s, &NeighborBorders::none(), &models);
        // Ground top face at y = 3 covers everything except the trunk's footprint.
        assert_eq!(area(&m.opaque, Face::PosY, |q| q.y(), 3), 1023);
        // Trunk sides stand in the air gap: one 1x6 quad per side face.
        for face in [Face::PosX, Face::NegX, Face::PosZ, Face::NegZ] {
            let quads: Vec<_> = m
                .opaque
                .iter()
                .filter(|q| q.face() == face.id() && (q.x(), q.z()) == (16, 16) && q.y() == 4)
                .collect();
            assert_eq!(quads.len(), 1, "{face:?}");
            let q = quads[0];
            let vertical = if face.axis() == 2 {
                q.width()
            } else {
                q.height()
            };
            assert_eq!(vertical, 6, "{face:?}");
        }
        // Canopy underside: 25 leaves minus the one resting on the trunk (log occludes it).
        assert_eq!(area(&m.opaque, Face::NegY, |q| q.y(), 10), 24);
        // Leaves keep their own tint row; trunk top still shows under the non-occluding leaf? No:
        // the leaf above is non-occluding, so the trunk's top face is emitted.
        assert_eq!(area(&m.opaque, Face::PosY, |q| q.y(), 9), 1);
        assert!(
            m.opaque
                .iter()
                .filter(|q| q.model_id() == LEAVES)
                .all(|q| q.biome_id() == 1)
        );
        // No face is emitted in the air gap beside the trunk.
        assert!(
            m.opaque
                .iter()
                .all(|q| !(q.face() == Face::PosY.id() && q.y() > 3 && q.y() < 9))
        );
        assert_eq!((m.aabb_min[1], m.aabb_max[1]), (0, 12));
    }

    #[test]
    fn water_over_terrain_keeps_ground_and_goes_to_translucent_slice() {
        let mut s = OctreeSection::new(pos());
        for z in 0..32 {
            for x in 0..32 {
                s.fill_column(x, z, 0, 4, Voxel::new(STONE, 0, 15, 0));
                s.fill_column(x, z, 4, 6, Voxel::new(WATER, 2, 15, 0));
            }
        }
        let m = mesh_octree_section(&s, &NeighborBorders::none(), &models);
        // Underwater terrain top is visible through the (non-occluding) water.
        assert_eq!(area(&m.opaque, Face::PosY, |q| q.y(), 3), 1024);
        // Water top at y = 5; same-block faces inside the volume are culled (cullsSame).
        assert_eq!(area(&m.translucent, Face::PosY, |q| q.y(), 5), 1024);
        assert_eq!(area(&m.translucent, Face::PosY, |q| q.y(), 4), 0);
        assert_eq!(area(&m.translucent, Face::NegY, |q| q.y(), 4), 0);
        assert!(m.translucent.iter().all(|q| q.biome_id() == 2));
    }

    #[test]
    fn cross_section_borders_cull_shared_faces() {
        let mut a = OctreeSection::new(pos());
        for z in 0..32 {
            for y in 0..32 {
                a.fill_column(31, z, y, y + 1, Voxel::new(STONE, 0, 15, 0));
            }
        }
        let east_full = OctreeSection::filled(
            OctreeSectionPos::new(0, 1, 0, 0),
            Voxel::new(STONE, 0, 15, 0),
        );
        let east_air = OctreeSection::filled(
            OctreeSectionPos::new(0, 1, 0, 0),
            Voxel::air_with_light(15, 0),
        );

        let mut n: [Option<&OctreeSection>; 6] = [None; 6];
        let unloaded = mesh_octree_section(&a, &NeighborBorders::from_neighbors(n), &models);
        n[Face::PosX as usize] = Some(&east_full);
        let culled = mesh_octree_section(&a, &NeighborBorders::from_neighbors(n), &models);
        n[Face::PosX as usize] = Some(&east_air);
        let open = mesh_octree_section(&a, &NeighborBorders::from_neighbors(n), &models);

        let px = |m: &SectionQuads| area(&m.opaque, Face::PosX, |q| q.x(), 31);
        assert_eq!(px(&unloaded), 1024, "unloaded border reads as air");
        assert_eq!(px(&culled), 0, "stone neighbour hides the shared wall");
        assert_eq!(px(&open), 1024, "air neighbour keeps it");
        assert_eq!(unloaded.opaque, open.opaque);
        assert!(culled.opaque_count() < open.opaque_count());
        // The wall on the other side of the section is untouched by the east border.
        assert_eq!(area(&culled.opaque, Face::NegX, |q| q.x(), 31), 1024);
    }

    #[test]
    fn quad_light_comes_from_the_voxel_in_front() {
        let mut s = OctreeSection::new(pos());
        s.set(5, 5, 5, Voxel::new(STONE, 0, 0, 0));
        s.set(5, 6, 5, Voxel::air_with_light(12, 3));
        let m = mesh_octree_section(&s, &NeighborBorders::none(), &models);
        let top = m
            .opaque
            .iter()
            .find(|q| q.face() == Face::PosY.id())
            .unwrap();
        assert_eq!((top.sky_light(), top.block_light()), (12, 3));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod::SOURCE_LOADED_CHUNK;

    #[test]
    fn greedy_meshing_collapses_uniform_flat_section_into_one_top_quad() {
        let mut section = LodSection::new(SectionPos::new(0, 0, 0));
        let col = LodColumn::from_solid_and_fluid(
            64,
            60,
            [145, 189, 89],
            None,
            [0, 0, 0],
            15,
            0,
            SOURCE_LOADED_CHUNK,
        );
        for z in 0..64 {
            for x in 0..64 {
                section.set_cell(x, z, col);
            }
        }
        let mesh = build_section_mesh("minecraft:overworld".into(), &section);
        // 1 greedy top quad + 4*64 = 256 boundary skirt quads = 257 total quads!
        assert_eq!(mesh.quad_count, 257);
        assert_eq!(mesh.solid_indices.len(), 257 * 6);
        assert_eq!(mesh.solid_vertex_bytes.len(), 257 * 4 * LodVertex::STRIDE);
        assert!(mesh.fluid_indices.is_empty());
    }

    #[test]
    fn overhang_and_lower_cavern_emit_underside_and_cull_shared_neighbor_walls() {
        let mut section = LodSection::new(SectionPos::new(0, 0, 0));
        let roof = LodColumn::from_solid_and_fluid(
            128,
            120,
            [110, 45, 45],
            None,
            [0, 0, 0],
            0,
            0,
            SOURCE_LOADED_CHUNK,
        );
        let floor = ExtraSpan::new(40, 32, [90, 35, 35], 0, 8, false);
        section.set_cell_with_spans(10, 10, roof, vec![floor]);
        section.set_cell_with_spans(11, 10, roof, vec![floor]);

        let mesh = build_section_mesh("minecraft:the_nether".into(), &section);
        // Should emit overhang underside faces + lower cavern floor top & outer walls, while
        // culling the shared interior wall between (10,10) and (11,10).
        assert!(mesh.quad_count > 10);
        assert!(mesh.min_y <= 32.0);
        assert!(mesh.max_y >= 128.0);
    }
}
