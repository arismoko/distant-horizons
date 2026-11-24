//! Greedy face-culled LOD mesher and background meshing worker thread.

use crate::lod::{EMPTY_Y, LodColumn, LodSection, SECTION_SIZE, SectionPos};
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
    pub fn pack_meta(normal_idx: u32, sky_light: u8, block_light: u8, is_fluid: bool) -> u32 {
        (normal_idx & 0x7)
            | (((sky_light as u32) & 0xF) << 4)
            | (((block_light as u32) & 0xF) << 8)
            | ((is_fluid as u32) << 12)
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
}

/// Build an optimized solid + translucent fluid mesh for `section`.
/// Positions are local to `section.pos.block_origin()` in X/Z and absolute in Y.
pub fn build_section_mesh(dimension: String, section: &LodSection) -> SectionMeshData {
    let cell = section.pos.cell_size() as f32;
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
            };
            let mut w = 1usize;
            while gx + w < SECTION_SIZE {
                let nidx = LodSection::index(gx + w, gz);
                if visited_solid[nidx] {
                    break;
                }
                let ncol = section.columns[nidx];
                if !ncol.has_solid() || ncol.solid_top_y != key.y || ncol.solid_rgba != key.rgba {
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
                    if !ncol.has_solid() || ncol.solid_top_y != key.y || ncol.solid_rgba != key.rgba
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
            let meta = LodVertex::pack_meta(
                1,
                LodColumn::sky_light(key.rgba),
                LodColumn::block_light(key.rgba),
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

            let color = LodVertex::pack_rgba(key.rgba[0], key.rgba[1], key.rgba[2], 196);
            let meta = LodVertex::pack_meta(
                1,
                LodColumn::sky_light(key.rgba),
                LodColumn::block_light(key.rgba),
                true,
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
            let bot = col.solid_bottom_y.min(top.saturating_sub(edge_skirt));
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
                let meta = LodVertex::pack_meta(2, sky, blk, false);
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
                let meta = LodVertex::pack_meta(3, sky, blk, false);
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
                let meta = LodVertex::pack_meta(4, sky, blk, false);
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
                let meta = LodVertex::pack_meta(5, sky, blk, false);
                solid.push_quad(
                    [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
                    color,
                    meta,
                );
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
}
