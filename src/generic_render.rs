//! Distant Sunrises Generic Object Renderer (`IDhApiGenericObjectRenderer` /
//! `DhApiRenderableBoxGroup`), Distant Beacon Beams (`BeaconRenderHandler`), and procedural 3D
//! LOD Clouds (`CloudRenderHandler`).
//!
//! Matching upstream Distant Sunrises, beacons, clouds, and third-party mod distant box groups
//! all share one axis-aligned box group representation rendered inside DH's reversed-Z offscreen
//! LOD pass so they occlude cleanly against distant terrain and receive matching DH fog.

use crate::config::DhConfig;
use crate::lod::{MAX_DETAIL, SectionPos};
use crate::mesh::LodVertex;
use std::collections::{HashMap, HashSet};

/// Width of one LOD cloud cell in blocks (matching upstream `CloudRenderHandler.CLOUD_BOX_WIDTH`).
pub const CLOUD_BOX_WIDTH: i32 = 128;
/// Vertical thickness of one LOD cloud layer in blocks (`CloudRenderHandler.CLOUD_BOX_THICKNESS`).
pub const CLOUD_BOX_THICKNESS: f32 = 24.0;
/// Base altitude of vanilla 26.3 Overworld clouds.
pub const CLOUD_BASE_Y: f32 = 192.0;
/// Drift speed of LOD clouds along `-X` in blocks per second (`MOVE_SPEED_IN_BLOCKS_PER_SECOND`).
pub const CLOUD_DRIFT_BLOCKS_PER_SEC: f64 = 6.0;
/// Built-in group ID for distant beacon beams.
pub const GROUP_ID_BEACONS: &str = "distantsunrises:beacons";
/// Built-in group ID for LOD clouds.
pub const GROUP_ID_CLOUDS: &str = "distantsunrises:clouds";

/// Per-face directional shading multipliers for a [`RenderableBoxGroup`]
/// (matches upstream `DhApiRenderableBoxGroupShading`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxGroupShading {
    pub bottom: f32,
    pub top: f32,
    pub north: f32,
    pub south: f32,
    pub west: f32,
    pub east: f32,
}

impl BoxGroupShading {
    /// Vanilla directional face shading (`top = 1.0`, `north/south = 0.8`, `east/west = 0.65`, `bottom = 0.5`).
    pub const fn vanilla() -> Self {
        Self {
            bottom: 0.50,
            top: 1.00,
            north: 0.80,
            south: 0.80,
            west: 0.65,
            east: 0.65,
        }
    }

    /// Flat unshaded lighting (`1.0` on every face, used for glowing beacon beams).
    pub const fn unshaded() -> Self {
        Self {
            bottom: 1.0,
            top: 1.0,
            north: 1.0,
            south: 1.0,
            west: 1.0,
            east: 1.0,
        }
    }

    /// Soft cloud shading (gentle bottom shadow, bright top).
    pub const fn clouds() -> Self {
        Self {
            bottom: 0.78,
            top: 1.00,
            north: 0.92,
            south: 0.92,
            west: 0.86,
            east: 0.86,
        }
    }

    #[inline]
    pub const fn for_normal(self, normal_idx: u32) -> f32 {
        match normal_idx {
            0 => self.bottom,
            1 => self.top,
            2 => self.north,
            3 => self.south,
            4 => self.west,
            _ => self.east,
        }
    }
}

impl Default for BoxGroupShading {
    fn default() -> Self {
        Self::vanilla()
    }
}

/// One axis-aligned colored box relative to its owning [`RenderableBoxGroup::origin`]
/// (matches upstream `DhApiRenderableBox`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderableBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub color: [u8; 4],
    pub material_id: u8,
}

impl RenderableBox {
    pub const fn new(min: [f32; 3], max: [f32; 3], color: [u8; 4], material_id: u8) -> Self {
        Self {
            min,
            max,
            color,
            material_id,
        }
    }
}

/// Named group of axis-aligned boxes rendered in Distant Sunrises' reversed-Z world pass
/// (matches upstream `IDhApiRenderableBoxGroup`).
#[derive(Clone, Debug)]
pub struct RenderableBoxGroup {
    pub id: String,
    /// World-space anchor subtracted from camera position each frame to avoid float precision jitter.
    pub origin: [f64; 3],
    pub boxes: Vec<RenderableBox>,
    pub sky_light: u8,
    pub block_light: u8,
    pub shading: BoxGroupShading,
    pub unshaded: bool,
    pub skip_clip_dist: bool,
    pub active: bool,
    pub revision: u64,
}

impl RenderableBoxGroup {
    pub fn new(id: impl Into<String>, origin: [f64; 3]) -> Self {
        Self {
            id: id.into(),
            origin,
            boxes: Vec::new(),
            sky_light: 15,
            block_light: 0,
            shading: BoxGroupShading::vanilla(),
            unshaded: false,
            skip_clip_dist: true,
            active: true,
            revision: 1,
        }
    }
}

/// Compiled CPU vertex/index buffers for one [`RenderableBoxGroup`], ready for GPU upload.
#[derive(Clone, Debug)]
pub struct GenericGroupMesh {
    pub id: String,
    pub revision: u64,
    pub origin: [f64; 3],
    pub active: bool,
    pub solid_vertex_bytes: Vec<u8>,
    pub solid_indices: Vec<u32>,
    pub translucent_vertex_bytes: Vec<u8>,
    pub translucent_indices: Vec<u32>,
    pub quad_count: u32,
}

impl GenericGroupMesh {
    pub fn is_empty(&self) -> bool {
        self.solid_indices.is_empty() && self.translucent_indices.is_empty()
    }
}

/// Pack a [`RenderableBoxGroup`] into standard 20-byte [`LodVertex`] solid & translucent meshes.
pub fn build_box_group_mesh(group: &RenderableBoxGroup) -> GenericGroupMesh {
    let mut solid_v = Vec::new();
    let mut solid_i = Vec::new();
    let mut solid_count = 0u32;

    let mut trans_v = Vec::new();
    let mut trans_i = Vec::new();
    let mut trans_count = 0u32;

    let mut quad_count = 0u32;

    for b in &group.boxes {
        let [x0, y0, z0] = b.min;
        let [x1, y1, z1] = b.max;
        if x1 <= x0 || y1 <= y0 || z1 <= z0 || b.color[3] == 0 {
            continue;
        }
        let is_translucent = b.color[3] < 255;
        let (verts, indices, vcount) = if is_translucent {
            (&mut trans_v, &mut trans_i, &mut trans_count)
        } else {
            (&mut solid_v, &mut solid_i, &mut solid_count)
        };

        // Six outward-facing CCW quads matching `mesh.rs` winding.
        let faces: [(u32, [[f32; 3]; 4]); 6] = [
            // 0: -Y (bottom)
            (0, [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
            // 1: +Y (top)
            (1, [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]),
            // 2: -Z (north)
            (2, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]),
            // 3: +Z (south)
            (3, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
            // 4: -X (west)
            (4, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]),
            // 5: +X (east)
            (5, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]),
        ];

        for (normal_idx, corners) in faces {
            let shade = group.shading.for_normal(normal_idx);
            let r = ((b.color[0] as f32) * shade).round().clamp(0.0, 255.0) as u8;
            let g = ((b.color[1] as f32) * shade).round().clamp(0.0, 255.0) as u8;
            let bl = ((b.color[2] as f32) * shade).round().clamp(0.0, 255.0) as u8;
            let color = LodVertex::pack_rgba(r, g, bl, b.color[3]);
            let meta = LodVertex::pack_meta_ext(
                normal_idx,
                group.sky_light,
                group.block_light,
                is_translucent,
                b.material_id,
                0,
                3, // generic source
                true,
                group.unshaded,
                group.skip_clip_dist,
            );
            let base = *vcount;
            for pos in corners {
                LodVertex { pos, color, meta }.append_bytes(verts);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            *vcount += 4;
            quad_count += 1;
        }
    }

    GenericGroupMesh {
        id: group.id.clone(),
        revision: group.revision,
        origin: group.origin,
        active: group.active,
        solid_vertex_bytes: solid_v,
        solid_indices: solid_i,
        translucent_vertex_bytes: trans_v,
        translucent_indices: trans_i,
        quad_count,
    }
}

/// Detected beacon beam stored per dimension (`BeaconBeamDTO` counterpart).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeaconBeam {
    pub pos: [i32; 3],
    pub color: [u8; 3],
}

/// Per-dimension beacon tracker that updates when loaded chunks are scanned and builds the
/// `distantsunrises:beacons` [`RenderableBoxGroup`] for distant beams.
#[derive(Default)]
pub struct BeaconTracker {
    /// Active beacons indexed by block `(x, y, z)`.
    pub beams: HashMap<[i32; 3], [u8; 3]>,
    pub revision: u64,
    last_built_revision: u64,
    last_origin: [f64; 3],
    last_vanilla_far: i32,
}

impl BeaconTracker {
    /// Replace all recorded beacons inside chunk `(cx, cz)` with `found`.
    /// Returns `true` if the set of beacons changed.
    pub fn replace_chunk_beacons(&mut self, chunk: [i32; 2], found: &[BeaconBeam]) -> bool {
        let bx0 = chunk[0] << 4;
        let bz0 = chunk[1] << 4;
        let bx1 = bx0 + 16;
        let bz1 = bz0 + 16;

        let mut changed = false;
        let new_set: HashSet<[i32; 3]> = found.iter().map(|b| b.pos).collect();
        self.beams.retain(|pos, _| {
            let in_chunk = pos[0] >= bx0 && pos[0] < bx1 && pos[2] >= bz0 && pos[2] < bz1;
            if in_chunk && !new_set.contains(pos) {
                changed = true;
                false
            } else {
                true
            }
        });
        for b in found {
            if self.beams.insert(b.pos, b.color) != Some(b.color) {
                changed = true;
            }
        }
        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
        changed
    }

    /// Build or refresh the distant beacon [`RenderableBoxGroup`] when beacons, camera grid, or
    /// vanilla render distance changed. Returns `Some(group)` only when an update is needed.
    pub fn maybe_build_group(
        &mut self,
        camera_pos: [f64; 3],
        vanilla_far_blocks: f32,
        dh_far_blocks: f32,
        enabled: bool,
    ) -> Option<RenderableBoxGroup> {
        let snap_x = (camera_pos[0] / 1024.0).floor() * 1024.0;
        let snap_z = (camera_pos[2] / 1024.0).floor() * 1024.0;
        let origin = [snap_x, 0.0, snap_z];
        let v_far_i = (vanilla_far_blocks as i32) / 16;

        let needs_rebuild = self.last_built_revision != self.revision
            || self.last_origin != origin
            || self.last_vanilla_far != v_far_i;
        if !needs_rebuild && !self.beams.is_empty() {
            return None;
        }
        if self.beams.is_empty() && self.last_built_revision == self.revision {
            return None;
        }

        self.last_built_revision = self.revision;
        self.last_origin = origin;
        self.last_vanilla_far = v_far_i;

        let mut group = RenderableBoxGroup::new(GROUP_ID_BEACONS, origin);
        group.sky_light = 15;
        group.block_light = 15;
        group.shading = BoxGroupShading::unshaded();
        group.unshaded = true;
        group.skip_clip_dist = true;
        group.active = enabled && !self.beams.is_empty();
        group.revision = self.revision.wrapping_add(1);

        if !group.active {
            return Some(group);
        }

        // Only draw distant beacons past ~85% of vanilla render distance so they never z-fight
        // with vanilla's animated beacon beam renderer.
        let min_dist_sq = ((vanilla_far_blocks * 0.85) as f64).powi(2);
        let max_dist_sq = (dh_far_blocks as f64).powi(2);

        for (&pos, &rgb) in &self.beams {
            let dx = (pos[0] as f64 + 0.5) - camera_pos[0];
            let dz = (pos[2] as f64 + 0.5) - camera_pos[2];
            let dist_sq = dx * dx + dz * dz;
            if dist_sq < min_dist_sq || dist_sq > max_dist_sq {
                continue;
            }
            // Gently widen very distant beams so they stay sub-pixel readable at 200+ chunks.
            let dist = dist_sq.sqrt() as f32;
            let half_w = (0.25 + (dist / 1024.0) * 0.65).clamp(0.25, 1.8);
            let outer_w = half_w * 1.65;
            let lx = (pos[0] as f64 + 0.5 - origin[0]) as f32;
            let lz = (pos[2] as f64 + 0.5 - origin[2]) as f32;
            let y0 = (pos[1] + 1) as f32;
            let y1 = (pos[1] as f32 + 512.0).max(384.0);

            // Inner bright opaque core
            group.boxes.push(RenderableBox::new(
                [lx - half_w, y0, lz - half_w],
                [lx + half_w, y1, lz + half_w],
                [rgb[0], rgb[1], rgb[2], 255],
                crate::colors::DH_BLOCK_ILLUMINATED,
            ));
            // Outer translucent glow halo
            group.boxes.push(RenderableBox::new(
                [lx - outer_w, y0, lz - outer_w],
                [lx + outer_w, y1, lz + outer_w],
                [rgb[0], rgb[1], rgb[2], 96],
                crate::colors::DH_BLOCK_ILLUMINATED,
            ));
        }

        Some(group)
    }
}

/// Vanilla dye RGB colors for stained glass tinting above beacons (`DyeColor.java`).
pub fn stained_glass_tint(block_id: &str) -> Option<[u8; 3]> {
    let name = block_id.strip_prefix("minecraft:").unwrap_or(block_id);
    let prefix = name
        .strip_suffix("_stained_glass")
        .or_else(|| name.strip_suffix("_stained_glass_pane"))?;
    let rgb = match prefix {
        "white" => [249, 255, 254],
        "orange" => [249, 128, 29],
        "magenta" => [199, 78, 189],
        "light_blue" => [58, 179, 218],
        "yellow" => [254, 216, 61],
        "lime" => [128, 199, 31],
        "pink" => [243, 139, 170],
        "gray" => [71, 79, 82],
        "light_gray" => [157, 157, 151],
        "cyan" => [22, 156, 156],
        "purple" => [137, 50, 184],
        "blue" => [60, 68, 170],
        "brown" => [131, 84, 50],
        "green" => [94, 124, 22],
        "red" => [176, 46, 38],
        "black" => [29, 29, 33],
        _ => return None,
    };
    Some(rgb)
}

/// Blend a new stained glass color onto an existing beacon beam color matching vanilla
/// `BeaconBlockEntity.tick` (`(prev + glass) / 2`, or exact glass on first tint).
#[inline]
pub fn blend_beacon_tint(current: Option<[u8; 3]>, glass: [u8; 3]) -> [u8; 3] {
    match current {
        None => glass,
        Some(prev) => [
            ((prev[0] as u16 + glass[0] as u16) / 2) as u8,
            ((prev[1] as u16 + glass[1] as u16) / 2) as u8,
            ((prev[2] as u16 + glass[2] as u16) / 2) as u8,
        ],
    }
}

/// Procedural drifting 3D LOD cloud generator (matches upstream `CloudRenderHandler` concept).
#[derive(Default)]
pub struct CloudField {
    last_cell_origin: [i32; 2],
    last_radius_cells: i32,
    last_enabled: bool,
    last_dimension: String,
    last_rain_bucket: u8,
    revision: u64,
}

impl CloudField {
    /// Deterministic value-noise cloud density in `[0, 1]` at cloud cell `(cx, cz)`.
    #[inline]
    pub fn cloud_cell_filled(cx: i32, cz: i32) -> bool {
        // Combine coarse continent blobs with medium cloud puffs for natural Minecraft-like banks.
        let coarse = value_noise_2d(cx as f32 * 0.22, cz as f32 * 0.22);
        let detail = value_noise_2d(cx as f32 * 0.55 + 17.3, cz as f32 * 0.55 - 9.1);
        (coarse * 0.68 + detail * 0.32) > 0.56
    }

    /// Build or update the `distantsunrises:clouds` [`RenderableBoxGroup`] when the camera moves
    /// across a cloud cell, settings change, or weather/dimension changes.
    ///
    /// Returns `(Some(group), drift_x)` when geometry needs re-upload, or `(None, drift_x)` when
    /// only the group origin drifted smoothly along `-X`.
    pub fn update(
        &mut self,
        dimension: &str,
        camera_pos: [f64; 3],
        vanilla_far_blocks: f32,
        cfg: DhConfig,
        rain_level: f32,
        elapsed_secs: f64,
    ) -> (Option<RenderableBoxGroup>, [f64; 3]) {
        let active = cfg.enabled && cfg.clouds_enabled && dimension == "minecraft:overworld";
        let drift = (elapsed_secs * CLOUD_DRIFT_BLOCKS_PER_SEC).rem_euclid(CLOUD_BOX_WIDTH as f64);
        let cell_shift =
            (elapsed_secs * CLOUD_DRIFT_BLOCKS_PER_SEC / CLOUD_BOX_WIDTH as f64).floor() as i32;

        let center_cx = (camera_pos[0] / CLOUD_BOX_WIDTH as f64).floor() as i32;
        let center_cz = (camera_pos[2] / CLOUD_BOX_WIDTH as f64).floor() as i32;
        let radius_blocks = cfg.render_distance_blocks().clamp(256.0, 6144.0);
        let radius_cells = ((radius_blocks / CLOUD_BOX_WIDTH as f32).ceil() as i32).clamp(3, 36);
        let rain_bucket = (rain_level.clamp(0.0, 1.0) * 10.0).round() as u8;

        let origin = [
            (center_cx as f64) * (CLOUD_BOX_WIDTH as f64) - drift,
            CLOUD_BASE_Y as f64,
            (center_cz as f64) * (CLOUD_BOX_WIDTH as f64),
        ];

        let cell_origin = [center_cx + cell_shift, center_cz];
        let changed = self.last_cell_origin != cell_origin
            || self.last_radius_cells != radius_cells
            || self.last_enabled != active
            || self.last_dimension != dimension
            || self.last_rain_bucket != rain_bucket;

        if !changed {
            return (None, origin);
        }

        self.last_cell_origin = cell_origin;
        self.last_radius_cells = radius_cells;
        self.last_enabled = active;
        self.last_dimension = dimension.to_owned();
        self.last_rain_bucket = rain_bucket;
        self.revision = self.revision.wrapping_add(1);

        let mut group = RenderableBoxGroup::new(GROUP_ID_CLOUDS, origin);
        group.sky_light = 15;
        group.block_light = 0;
        group.shading = BoxGroupShading::clouds();
        group.unshaded = false;
        group.skip_clip_dist = true;
        group.active = active;
        group.revision = self.revision;

        if !active {
            return (Some(group), origin);
        }

        // Keep LOD clouds outside vanilla's cloud radius (~192 blocks or vanilla_far) so they
        // extend the sky horizon seamlessly without overlapping vanilla's closer cloud layer.
        let inner_r = vanilla_far_blocks.max(192.0);
        let inner_sq = inner_r * inner_r;
        let outer_sq = radius_blocks * radius_blocks;

        // Slightly darken cloud albedo during rain/storms matching vanilla sky tint.
        let rain = rain_level.clamp(0.0, 1.0);
        let base_c = (246.0 * (1.0 - 0.35 * rain)).round() as u8;
        let cloud_rgba = [base_c, base_c, (base_c as u16 + 4).min(255) as u8, 215];
        let w = CLOUD_BOX_WIDTH as f32;

        for dz in -radius_cells..=radius_cells {
            for dx in -radius_cells..=radius_cells {
                let wx = (dx as f32 + 0.5) * w;
                let wz = (dz as f32 + 0.5) * w;
                let d2 = wx * wx + wz * wz;
                if d2 < inner_sq || d2 > outer_sq {
                    continue;
                }
                let world_cx = center_cx + dx + cell_shift;
                let world_cz = center_cz + dz;
                if !Self::cloud_cell_filled(world_cx, world_cz) {
                    continue;
                }
                let x0 = (dx as f32) * w;
                let z0 = (dz as f32) * w;
                group.boxes.push(RenderableBox::new(
                    [x0, 0.0, z0],
                    [x0 + w, CLOUD_BOX_THICKNESS, z0 + w],
                    cloud_rgba,
                    crate::colors::DH_BLOCK_SNOW,
                ));
            }
        }

        (Some(group), origin)
    }
}

fn hash2i(x: i32, z: i32) -> f32 {
    let mut h = (x.wrapping_mul(374_761_393)) ^ (z.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    let u = (h ^ (h >> 16)) as u32;
    (u & 0x00FF_FFFF) as f32 / 16_777_215.0
}

fn value_noise_2d(x: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - ix as f32;
    let fz = z - iz as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sz = fz * fz * (3.0 - 2.0 * fz);
    let a = hash2i(ix, iz);
    let b = hash2i(ix + 1, iz);
    let c = hash2i(ix, iz + 1);
    let d = hash2i(ix + 1, iz + 1);
    let ab = a + (b - a) * sx;
    let cd = c + (d - c) * sx;
    ab + (cd - ab) * sz
}

/// Construct the exact DH reversed-Z `view_proj` matrix from Sunrise's `world.jittered` matrix
/// and its inverse, so `COMPOSITE_WGSL` can reconstruct exact camera-relative 3D world positions
/// from `(ndc_xy, dh_depth)` for SSAO and height/spherical fog.
///
/// In `TERRAIN_WGSL`:
/// `clip.x = row0 · p`, `clip.y = row1 · p`, `clip.w = row3 · p`, and
/// `clip.z = a * clip.w + b` where `a = -near / (far - near)` and `b = (near * far) / (far - near)`.
/// Replacing row 2 of `jittered` with `a * row3 + [0, 0, 0, b]` yields a true invertible 4×4 matrix
/// whose NDC `(x/w, y/w, z/w)` matches `TERRAIN_WGSL` identically.
pub fn dh_view_proj_and_inverse(
    jittered_cols: [f32; 16],
    near_dh: f32,
    far_dh: f32,
) -> ([f32; 16], [f32; 16]) {
    let denom = (far_dh - near_dh).max(1.0);
    let a = -near_dh / denom;
    let b = (near_dh * far_dh) / denom;

    // Column-major layout: element (row r, col c) is at index `c * 4 + r`.
    let mut m = jittered_cols;
    m[2] = a * m[3];
    m[6] = a * m[7];
    m[10] = a * m[11];
    m[14] = a * m[15] + b;

    let inv = invert_mat4_cols(m).unwrap_or([
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]);
    (m, inv)
}

/// Exact cofactor inverse of a column-major 4×4 matrix (uses `f64` internally for precision at
/// 10,000-block DH far planes).
pub fn invert_mat4_cols(m: [f32; 16]) -> Option<[f32; 16]> {
    let a: [f64; 16] = std::array::from_fn(|i| m[i] as f64);
    let mut inv = [0.0f64; 16];

    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15]
        + a[9] * a[7] * a[14]
        + a[13] * a[6] * a[11]
        - a[13] * a[7] * a[10];

    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15]
        - a[8] * a[7] * a[14]
        - a[12] * a[6] * a[11]
        + a[12] * a[7] * a[10];

    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15]
        + a[8] * a[7] * a[13]
        + a[12] * a[5] * a[11]
        - a[12] * a[7] * a[9];

    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14]
        - a[8] * a[6] * a[13]
        - a[12] * a[5] * a[10]
        + a[12] * a[6] * a[9];

    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15]
        - a[9] * a[3] * a[14]
        - a[13] * a[2] * a[11]
        + a[13] * a[3] * a[10];

    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15]
        + a[8] * a[3] * a[14]
        + a[12] * a[2] * a[11]
        - a[12] * a[3] * a[10];

    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15]
        - a[8] * a[3] * a[13]
        - a[12] * a[1] * a[11]
        + a[12] * a[3] * a[9];

    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14]
        + a[8] * a[2] * a[13]
        + a[12] * a[1] * a[10]
        - a[12] * a[2] * a[9];

    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15]
        + a[5] * a[3] * a[14]
        + a[13] * a[2] * a[7]
        - a[13] * a[3] * a[6];

    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15]
        - a[4] * a[3] * a[14]
        - a[12] * a[2] * a[7]
        + a[12] * a[3] * a[6];

    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15]
        + a[4] * a[3] * a[13]
        + a[12] * a[1] * a[7]
        - a[12] * a[3] * a[5];

    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14]
        - a[4] * a[2] * a[13]
        - a[12] * a[1] * a[6]
        + a[12] * a[2] * a[5];

    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11]
        - a[5] * a[3] * a[10]
        - a[9] * a[2] * a[7]
        + a[9] * a[3] * a[6];

    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11]
        + a[4] * a[3] * a[10]
        + a[8] * a[2] * a[7]
        - a[8] * a[3] * a[6];

    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11]
        - a[4] * a[3] * a[9]
        - a[8] * a[1] * a[7]
        + a[8] * a[3] * a[5];

    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10]
        + a[4] * a[2] * a[9]
        + a[8] * a[1] * a[6]
        - a[8] * a[2] * a[5];

    let det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if !det.is_finite() || det.abs() < 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    let out: [f32; 16] = std::array::from_fn(|i| (inv[i] * inv_det) as f32);
    Some(out)
}

/// Keep `SectionPos` / `MAX_DETAIL` referenced cleanly for future section-scoped generic culling.
#[allow(dead_code)]
fn _detail_bound_check(pos: SectionPos) -> bool {
    pos.detail <= MAX_DETAIL
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mul_mat4_vec4(m: [f32; 16], v: [f32; 4]) -> [f32; 4] {
        [
            m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
            m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
            m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
            m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
        ]
    }

    #[test]
    fn dh_view_proj_inverse_reconstructs_world_pos() {
        // Right-handed reversed-Z perspective (fov = 70 deg, aspect = 16/9, near = 0.05)
        let f = 1.0f32 / (70.0f32.to_radians() * 0.5).tan();
        let aspect = 16.0 / 9.0;
        let near_v = 0.05f32;
        let jittered = [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            near_v,
            0.0,
        ];
        let near_dh = 8.0;
        let far_dh = 4096.0;
        let (m_dh, inv_m_dh) = dh_view_proj_and_inverse(jittered, near_dh, far_dh);

        for rel in [
            [0.0, 0.0, -20.0],
            [120.5, -34.2, -500.0],
            [-800.0, 210.0, -3500.0],
        ] {
            let clip = mul_mat4_vec4(m_dh, [rel[0], rel[1], rel[2], 1.0]);
            let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1.0];
            let unproj = mul_mat4_vec4(inv_m_dh, ndc);
            let got = [
                unproj[0] / unproj[3],
                unproj[1] / unproj[3],
                unproj[2] / unproj[3],
            ];
            for i in 0..3 {
                assert!(
                    (got[i] - rel[i]).abs() < 0.05,
                    "axis {i}: got {}, expected {} (rel={:?})",
                    got[i],
                    rel[i],
                    rel
                );
            }
        }
    }

    #[test]
    fn box_group_mesh_splits_opaque_and_translucent() {
        let mut g = RenderableBoxGroup::new("test:boxes", [100.0, 0.0, 200.0]);
        g.boxes.push(RenderableBox::new(
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
            [255, 0, 0, 255],
            0,
        ));
        g.boxes.push(RenderableBox::new(
            [2.0, 0.0, 0.0],
            [3.0, 2.0, 1.0],
            [0, 0, 255, 128],
            12,
        ));
        let m = build_box_group_mesh(&g);
        assert_eq!(m.solid_indices.len(), 36);
        assert_eq!(m.translucent_indices.len(), 36);
        assert_eq!(m.quad_count, 12);
    }

    #[test]
    fn beacon_tracker_blends_stained_glass_and_culls_near_camera() {
        let red = stained_glass_tint("minecraft:red_stained_glass").unwrap();
        let blue = stained_glass_tint("minecraft:blue_stained_glass_pane").unwrap();
        let c1 = blend_beacon_tint(None, red);
        assert_eq!(c1, red);
        let c2 = blend_beacon_tint(Some(c1), blue);
        assert_eq!(
            c2,
            [
                ((176 + 60) / 2) as u8,
                ((46 + 68) / 2) as u8,
                ((38 + 170) / 2) as u8
            ]
        );

        let mut tracker = BeaconTracker::default();
        assert!(tracker.replace_chunk_beacons(
            [20, 0],
            &[BeaconBeam {
                pos: [325, 64, 5],
                color: c2
            }]
        ));
        // Near camera (dist ~5): culled
        let g_near = tracker
            .maybe_build_group([320.0, 70.0, 5.0], 192.0, 2048.0, true)
            .unwrap();
        assert!(g_near.boxes.is_empty());
        // Far camera (dist ~500 > 0.85 * 192): drawn with core + halo
        tracker.revision += 1;
        let g_far = tracker
            .maybe_build_group([-180.0, 70.0, 5.0], 192.0, 2048.0, true)
            .unwrap();
        assert_eq!(g_far.boxes.len(), 2);
    }

    #[test]
    fn cloud_field_generates_outside_vanilla_radius() {
        let mut field = CloudField::default();
        let cfg = DhConfig {
            render_distance_chunks: 128,
            ..DhConfig::default()
        };
        let (group, _origin) = field.update(
            "minecraft:overworld",
            [0.0, 70.0, 0.0],
            192.0,
            cfg,
            0.0,
            10.0,
        );
        let g = group.expect("initial cloud group");
        assert!(g.active);
        assert!(!g.boxes.is_empty());
        for b in &g.boxes {
            let cx = 0.5 * (b.min[0] + b.max[0]);
            let cz = 0.5 * (b.min[2] + b.max[2]);
            assert!(cx * cx + cz * cz >= 192.0 * 192.0);
        }
    }
}
