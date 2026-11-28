//! Persistent GPU storage-buffer arena for Voxy-style 8-byte packed quads.
//!
//! Layout and policy are taken from Voxy's `BasicSectionGeometryManager` /
//! `BasicSectionGeometryData` / `BufferArena` (read for behaviour; no code copied):
//! - a quad is 8 bytes (`new BufferArena(geometryCapacity, 8)`),
//! - a section's metadata record is 32 bytes (`SECTION_METADATA_SIZE = 32`, two `uvec4`s),
//! - a section's quads are split into 8 buckets (translucent, double-sided, then the six faces
//!   down/up/north/south/west/east) whose 16-bit counts live in the metadata record,
//! - section ids come from a bitset allocator over `maxSectionCount` (Voxy: `1 << 20`,
//!   `VoxyRenderSystem.java:119`), and freed ids get a zeroed metadata record,
//! - geometry is sub-allocated from one arena by a free list that returns the allocation size on
//!   free (`AllocationArena`).
//!
//! GPU objects are Sunrise `StorageBufferRef`s (sunrise#5589, ruling points 1, 2, 9, 10): they
//! survive resource reload and world close, are freed only by `release(host)`, and `write_range`
//! is queued and applied in order before the frame. Buffer sizes are capped by the device's
//! `max_buffer_size`, and by `max_storage_buffer_binding_size` because each buffer is bound whole
//! (ruling point 9); this module adds no other cap.
//!
//! Culled sections are drawn with `instance_count = 0` (ruling point 18): the command buffers use
//! one fixed slot per `(render queue index, bucket)`, rewritten every frame by `voxy_cmdgen.csh`.
//! Until count-driven draws are available, the draw count uses the id high-water mark.

use crate::octree::{MAX_OCTREE_DETAIL, OctreeSectionPos, child_index};
use crate::octree_ingest::is_out_of_build_height;
use mod_api::{
    BufferError, ClientHost, StorageBufferDescriptor, StorageBufferRef, StorageBufferSlice,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Bytes per packed quad (Voxy `BufferArena(geometryCapacity, 8)`).
pub const QUAD_BYTES: u64 = 8;
/// Bytes per section metadata record (Voxy `SECTION_METADATA_SIZE`).
pub const SECTION_METADATA_BYTES: u64 = 32;
/// Bytes per indexed indirect command: vanilla `RenderPass.INDIRECT_INDEXED_DRAW_SIZE = 20`.
pub const DRAW_INDEXED_INDIRECT_BYTES: u64 = 20;
/// Voxy's `maxSectionCount` (`VoxyRenderSystem.java:119`).
pub const VOXY_MAX_SECTIONS: u32 = 1 << 19;
/// Quad buckets per section: translucent, double-sided, down, up, north, south, west, east.
pub const BUCKETS: usize = 8;
/// Opaque command slots per section (every bucket but translucent).
pub const OPAQUE_SLOTS_PER_SECTION: u64 = 7;
/// Translucent command slots per section.
pub const TRANSLUCENT_SLOTS_PER_SECTION: u64 = 1;
/// Words per baked model record.
pub const MODEL_WORDS: u64 = 16;
/// State ids are 20 bits and biome ids 9 bits in the packed quad (Voxy `quad_format.glsl`).
pub const MODEL_COUNT: u64 = 1 << 20;
pub const BIOME_COUNT: u64 = 1 << 9;

/// Geometry arena size requested first. Each arena buffer is bound whole, so it must fit the
/// device's `max_storage_buffer_binding_size` (128 MiB by default); `create_storage_buffer` only
/// checks `max_buffer_size` and the game reports no limits to mods, so this stays at the default.
pub const VOXY_GEOMETRY_BYTES: u64 = 128 << 20;

/// Smallest geometry arena worth running with.
const MIN_GEOMETRY_BYTES: u64 = 64 << 20;

/// Requested arena capacities; halved at creation until the device accepts them.
#[derive(Clone, Copy, Debug)]
pub struct ArenaConfig {
    pub max_sections: u32,
    pub geometry_bytes: u64,
}

// ---------------------------------------------------------------------------------------------
// Packed quad (Voxy `quad_format.glsl`, 64-bit variant): one type, defined by the mesher
// ---------------------------------------------------------------------------------------------

pub use crate::mesh::PackedQuad;

// ---------------------------------------------------------------------------------------------
// Section metadata (Voxy `section.glsl` / `BasicSectionGeometryManager.SectionMeta`)
// ---------------------------------------------------------------------------------------------

/// Packs `(lod level, x, y, z)` into the two position words (Voxy `pos_util.glsl`):
/// word0 = level(4)@28 | y(8)@20 | z-high(20)@0, word1 = x(24)@4 | z-low(4)@28.
pub fn pack_section_pos(level: u8, x: i32, y: i32, z: i32) -> [u32; 2] {
    debug_assert!(level < 16);
    debug_assert!((-(1 << 23)..(1 << 23)).contains(&x) && (-(1 << 23)..(1 << 23)).contains(&z));
    debug_assert!((-128..128).contains(&y));
    let z24 = z as u32 & 0xFF_FFFF;
    let w0 = ((level as u32) << 28) | ((y as u32 & 0xFF) << 20) | (z24 >> 4);
    let w1 = ((z24 & 0xF) << 28) | ((x as u32 & 0xFF_FFFF) << 4);
    [w0, w1]
}

/// Inverse of [`pack_section_pos`].
pub fn unpack_section_pos(words: [u32; 2]) -> (u8, [i32; 3]) {
    let y = ((words[0] as i32) << 4) >> 24;
    let x = ((words[1] as i32) << 4) >> 8;
    let z = ((((words[0] & 0xF_FFFF) << 4) | (words[1] >> 28)) as i32) << 8 >> 8;
    ((words[0] >> 28) as u8, [x, y, z])
}

/// Section-local AABB in voxels: offset `0..32` and size `1..=32` per axis (5 bits each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SectionAabb {
    pub offset: [u8; 3],
    pub size: [u8; 3],
}

impl SectionAabb {
    pub const FULL: Self = Self {
        offset: [0; 3],
        size: [32; 3],
    };

    pub fn pack(self) -> u32 {
        let mut v = 0u32;
        for i in 0..3 {
            debug_assert!(self.offset[i] < 32 && (1..=32).contains(&self.size[i]));
            v |= (self.offset[i] as u32 & 31) << (5 * i);
            v |= ((self.size[i] - 1) as u32 & 31) << (15 + 5 * i);
        }
        v
    }
}

/// A section's quads, already sorted into the 8 buckets, ready for upload.
pub struct BuiltSection {
    pub level: u8,
    pub pos: [i32; 3],
    pub aabb: SectionAabb,
    /// Quads in bucket order; `counts[i]` quads belong to bucket `i`.
    pub quads: Vec<PackedQuad>,
    pub counts: [u16; BUCKETS],
    /// Whether the source covers this entire section, independent of its quad count.
    pub complete: bool,
}

impl BuiltSection {
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }
}

/// The 32-byte metadata record: `a = {pos0, pos1, aabb, quad_start}`,
/// `b = {translucent|double-sided, down|up, north|south, west|east}` (16-bit counts).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SectionMeta {
    pub a: [u32; 4],
    pub b: [u32; 4],
}

impl SectionMeta {
    pub fn new(section: &BuiltSection, quad_start: u32) -> Self {
        let p = pack_section_pos(
            section.level,
            section.pos[0],
            section.pos[1],
            section.pos[2],
        );
        let c = section.counts;
        let pair = |lo: usize, hi: usize| c[lo] as u32 | ((c[hi] as u32) << 16);
        Self {
            a: [p[0], p[1], section.aabb.pack(), quad_start],
            b: [pair(0, 1), pair(2, 3), pair(4, 5), pair(6, 7)],
        }
    }

    pub fn to_bytes(self) -> [u8; SECTION_METADATA_BYTES as usize] {
        let mut out = [0u8; SECTION_METADATA_BYTES as usize];
        for (i, w) in self.a.iter().chain(self.b.iter()).enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Indirect command (vanilla `RenderPass.INDIRECT_INDEXED_DRAW_SIZE = 20`)
// ---------------------------------------------------------------------------------------------

/// `DrawIndexedIndirect`: `indexCount, instanceCount, firstIndex, baseVertex, firstInstance`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

impl DrawIndexedIndirect {
    /// What `voxy_cmdgen.csh` writes for a visible bucket: six indices per quad, the bucket's
    /// first quad as `base_vertex = quad << 2`, and the section id as `first_instance`.
    pub fn for_bucket(quad_start: u32, quad_count: u32, section_id: u32) -> Self {
        Self {
            index_count: quad_count * 6,
            instance_count: 1,
            first_index: 0,
            base_vertex: (quad_start << 2) as i32,
            first_instance: section_id,
        }
    }

    /// A culled slot: `instance_count = 0`, so the draw is skipped on the GPU (ruling point 18).
    pub const CULLED: Self = Self {
        index_count: 0,
        instance_count: 0,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    };

    pub fn to_bytes(self) -> [u8; DRAW_INDEXED_INDIRECT_BYTES as usize] {
        let mut out = [0u8; DRAW_INDEXED_INDIRECT_BYTES as usize];
        out[0..4].copy_from_slice(&self.index_count.to_le_bytes());
        out[4..8].copy_from_slice(&self.instance_count.to_le_bytes());
        out[8..12].copy_from_slice(&self.first_index.to_le_bytes());
        out[12..16].copy_from_slice(&self.base_vertex.to_le_bytes());
        out[16..20].copy_from_slice(&self.first_instance.to_le_bytes());
        out
    }
}

// ---------------------------------------------------------------------------------------------
// CPU allocators
// ---------------------------------------------------------------------------------------------

/// Best-fit free list in element units, following Voxy's `AllocationArena.alloc` size ordering.
/// Address ordering coalesces adjacent runs without moving live allocations.
#[derive(Debug)]
pub struct FreeList {
    capacity: u64,
    free: BTreeMap<u64, u64>,
    by_size: BTreeSet<(u64, u64)>,
    allocated: HashMap<u64, u64>,
    used: u64,
}

impl FreeList {
    pub fn new(capacity: u64) -> Self {
        let mut list = Self {
            capacity,
            free: BTreeMap::new(),
            by_size: BTreeSet::new(),
            allocated: HashMap::new(),
            used: 0,
        };
        if capacity > 0 {
            list.insert_free(0, capacity);
        }
        list
    }

    fn insert_free(&mut self, start: u64, len: u64) {
        self.free.insert(start, len);
        self.by_size.insert((len, start));
    }

    fn remove_free(&mut self, start: u64, len: u64) {
        self.free.remove(&start);
        self.by_size.remove(&(len, start));
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    /// Smallest fitting run, with ties by address. `None` if no run is large enough.
    pub fn alloc(&mut self, len: u64) -> Option<u64> {
        if len == 0 {
            return None;
        }
        let (run, start) = self.by_size.range((len, 0)..).next().copied()?;
        self.remove_free(start, run);
        if run > len {
            self.insert_free(start + len, run - len);
        }
        self.allocated.insert(start, len);
        self.used += len;
        Some(start)
    }

    /// Frees the allocation that starts at `start`; returns its length, or `None` if unknown.
    pub fn free(&mut self, start: u64) -> Option<u64> {
        let len = self.allocated.remove(&start)?;
        self.used -= len;
        let mut new_start = start;
        let mut new_len = len;
        if let Some((&prev, &prev_len)) = self.free.range(..start).next_back()
            && prev + prev_len == start
        {
            self.remove_free(prev, prev_len);
            new_start = prev;
            new_len += prev_len;
        }
        if let Some(&next_len) = self.free.get(&(start + len)) {
            self.remove_free(start + len, next_len);
            new_len += next_len;
        }
        self.insert_free(new_start, new_len);
        Some(len)
    }
}

/// Section id allocator: lowest free id first, bounded by `max`.
#[derive(Debug)]
pub struct IdAllocator {
    max: u32,
    free: BTreeSet<u32>,
    next: u32,
}

impl IdAllocator {
    pub fn new(max: u32) -> Self {
        Self {
            max,
            free: BTreeSet::new(),
            next: 0,
        }
    }

    pub fn alloc(&mut self) -> Option<u32> {
        if let Some(id) = self.free.pop_first() {
            return Some(id);
        }
        (self.next < self.max).then(|| {
            self.next += 1;
            self.next - 1
        })
    }

    pub fn free(&mut self, id: u32) -> bool {
        if id >= self.next || !self.free.insert(id) {
            return false;
        }
        while self.next > 0 && self.free.remove(&(self.next - 1)) {
            self.next -= 1;
        }
        true
    }

    /// One past the highest live id; the number of slots cmdgen has to visit.
    pub fn high_water(&self) -> u32 {
        self.next
    }
}

// ---------------------------------------------------------------------------------------------
// The arena
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
pub enum ArenaError {
    /// The quad arena has no free run large enough (Voxy: "geometry buffer is full").
    GeometryFull,
    /// Every section id is in use.
    SectionsFull,
    /// The rebuilt tree exceeds the GPU node table.
    NodesFull,
    /// Nothing to upload (Voxy throws on an empty `BuiltSection`).
    EmptySection,
    UnknownSection(u32),
    Buffer(BufferError),
}

impl From<BufferError> for ArenaError {
    fn from(e: BufferError) -> Self {
        Self::Buffer(e)
    }
}

/// Size in bytes of `TransSortMeta` (`visible_count`, `pad0`, 256 counts, 256 offsets).
pub const TRANS_SORT_META_BYTES: u64 = (2 + 256 * 2) * 4;
/// Size in bytes of one `TransSortEntries` element (`uvec2(section_id, bucket)`).
pub const TRANS_SORT_ENTRY_BYTES: u64 = 8;
/// Size in bytes of one `GpuNode` record (`uvec4 a; uvec4 b;`).
pub const GPU_NODE_BYTES: u64 = 32;
/// Maximum nodes in the GPU hierarchical octree (`8 MiB` at 32 bytes/node).
pub const GPU_MAX_NODES: u64 = 1 << 18;
/// Maximum node indices in each ping-pong traversal queue (`512 KiB`).
pub const GPU_TRAV_QUEUE_CAP: u32 = 1 << 17;
/// Byte size shared by the flat root and ping-pong queues of node indices.
const GPU_TRAV_QUEUE_BYTES: u64 = GPU_TRAV_QUEUE_CAP as u64 * 4;
/// One traversal dispatch per octree level, from the detail-4 roots to detail-0 leaves.
pub const TRAVERSAL_ITERATIONS: usize = MAX_OCTREE_DETAIL as usize + 1;
/// Layout: `[wg_x, wg_y, wg_z, count]`, also usable as indirect dispatch arguments.
const QUEUE_META_BYTES: u64 = 16;
/// Count and three pad words before the render queue's section IDs.
pub const RENDER_QUEUE_HEADER_BYTES: usize = 16;

/// One 32-byte node in `NodeBuffer` for `voxy_traversal.csh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuNode {
    pub pos0: u32,
    pub pos1: u32,
    pub section_id: u32,
    pub child_base: u32,
    pub child_mask: u32,
    pub aabb: u32,
}

impl GpuNode {
    pub const NONE: u32 = u32::MAX;

    pub fn to_bytes(self) -> [u8; 32] {
        let words = [
            self.pos0,
            self.pos1,
            self.section_id,
            self.child_base,
            self.child_mask,
            self.aabb,
            0,
            0,
        ];
        let mut out = [0u8; 32];
        for (i, w) in words.into_iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

type NodeTree = (Vec<GpuNode>, Vec<u32>);
type LiveSectionMeta = (u8, [i32; 3], SectionAabb, bool);

fn out_of_build_height_child_mask(parent: OctreeSectionPos, y_bounds: Option<(i32, i32)>) -> u8 {
    let (Some(bounds), Some(children)) = (y_bounds, parent.children()) else {
        return 0;
    };
    // Each half contains the four horizontal octants with the same Y extent.
    let low = if is_out_of_build_height(children[0], bounds) {
        0x0f
    } else {
        0
    };
    let high = if is_out_of_build_height(children[4], bounds) {
        0xf0
    } else {
        0
    };
    low | high
}

/// Builds a compact top-down octree node table (`detail = MAX_OCTREE_DETAIL` roots down to
/// `detail = 0` leaves) with contiguous 8-child blocks for `voxy_traversal.csh`. Empty octants
/// count toward resolution but need no GPU node of their own.
fn build_node_tree(
    live_meta: &HashMap<u32, LiveSectionMeta>,
    known_empty: Option<&HashSet<OctreeSectionPos>>,
    y_bounds: Option<(i32, i32)>,
) -> NodeTree {
    let mut resolved_children = HashMap::<OctreeSectionPos, u8>::new();
    let uploaded = live_meta
        .values()
        .filter(|meta| meta.3)
        .map(|&(level, [x, y, z], _, _)| {
            OctreeSectionPos::new(level.min(MAX_OCTREE_DETAIL), x, y, z)
        });
    for mut pos in uploaded.chain(known_empty.into_iter().flatten().copied()) {
        while let Some(parent) = pos.parent() {
            let bit = 1
                << child_index(
                    (pos.x & 1) as usize,
                    (pos.y & 1) as usize,
                    (pos.z & 1) as usize,
                );
            let mask = resolved_children.entry(parent).or_default();
            if *mask & bit != 0 {
                break;
            }
            *mask |= bit | out_of_build_height_child_mask(parent, y_bounds);
            if *mask != u8::MAX {
                break;
            }
            // Complete coverage also resolves a meshless octant for its own parent.
            pos = parent;
        }
    }

    let mut nodes: Vec<GpuNode> = Vec::new();
    let mut root_index: HashMap<[i32; 3], usize> = HashMap::new();
    let mut roots: Vec<u32> = Vec::new();

    // Ensure a root node exists at `MAX_OCTREE_DETAIL` and return its index.
    let ensure_root = |pos: [i32; 3],
                       nodes: &mut Vec<GpuNode>,
                       root_index: &mut HashMap<[i32; 3], usize>,
                       roots: &mut Vec<u32>|
     -> usize {
        if let Some(&idx) = root_index.get(&pos) {
            return idx;
        }
        let idx = nodes.len();
        let [pos0, pos1] = pack_section_pos(MAX_OCTREE_DETAIL, pos[0], pos[1], pos[2]);
        nodes.push(GpuNode {
            pos0,
            pos1,
            section_id: GpuNode::NONE,
            child_base: GpuNode::NONE,
            child_mask: 0,
            aabb: SectionAabb::FULL.pack(),
        });
        root_index.insert(pos, idx);
        roots.push(idx as u32);
        idx
    };

    for (&section_id, &(level, pos, aabb, _)) in live_meta {
        let clamped_level = level.min(MAX_OCTREE_DETAIL);
        // Walk from root down to `clamped_level`, allocating 8-child blocks as needed.
        let shift = MAX_OCTREE_DETAIL - clamped_level;
        let root_pos = [pos[0] >> shift, pos[1] >> shift, pos[2] >> shift];
        let mut cur_idx = ensure_root(root_pos, &mut nodes, &mut root_index, &mut roots);
        let mut cur_pos = root_pos;
        let mut cur_level = MAX_OCTREE_DETAIL;
        while cur_level > clamped_level {
            let next_level = cur_level - 1;
            let next_shift = next_level - clamped_level;
            let next_pos = [
                pos[0] >> next_shift,
                pos[1] >> next_shift,
                pos[2] >> next_shift,
            ];
            let oct = child_index(
                (next_pos[0] & 1) as usize,
                (next_pos[1] & 1) as usize,
                (next_pos[2] & 1) as usize,
            );
            let child_base = if nodes[cur_idx].child_base != GpuNode::NONE {
                nodes[cur_idx].child_base as usize
            } else {
                let base = nodes.len();
                nodes[cur_idx].child_base = base as u32;
                for ci in 0..8usize {
                    let cx = cur_pos[0] * 2 + (ci & 1) as i32;
                    let cy = cur_pos[1] * 2 + ((ci >> 2) & 1) as i32;
                    let cz = cur_pos[2] * 2 + ((ci >> 1) & 1) as i32;
                    let [p0, p1] = pack_section_pos(next_level, cx, cy, cz);
                    nodes.push(GpuNode {
                        pos0: p0,
                        pos1: p1,
                        section_id: GpuNode::NONE,
                        child_base: GpuNode::NONE,
                        child_mask: 0,
                        aabb: SectionAabb::FULL.pack(),
                    });
                }
                base
            };
            nodes[cur_idx].child_mask |= 1u32 << oct;
            cur_idx = child_base + oct;
            cur_pos = next_pos;
            cur_level = next_level;
        }
        nodes[cur_idx].section_id = section_id;
        nodes[cur_idx].aabb = aabb.pack();
    }

    // Publish only after the whole residency snapshot has been linked. Rebuilding from
    // scratch also withdraws a child block when a retirement leaves an unresolved octant.
    for node in &mut nodes {
        if node.child_mask == 0 || node.section_id == GpuNode::NONE {
            continue;
        }
        let (level, [x, y, z]) = unpack_section_pos([node.pos0, node.pos1]);
        let pos = OctreeSectionPos::new(level, x, y, z);
        let resolved = resolved_children.get(&pos).copied().unwrap_or(0)
            | out_of_build_height_child_mask(pos, y_bounds);
        if resolved != u8::MAX {
            // A meshed parent remains the sole drawable for its volume until it can hand off
            // the entire volume. Keep the tight mesh bounds when that handoff is unavailable.
            node.child_base = GpuNode::NONE;
            node.child_mask = 0;
        } else {
            node.aabb = SectionAabb::FULL.pack();
        }
    }

    (nodes, roots)
}

/// CPU-only residency snapshot for pipeline tests; no storage buffers or renderer needed.
#[cfg(test)]
pub(crate) fn test_node_tree(
    sections: &[BuiltSection],
    y_bounds: Option<(i32, i32)>,
) -> Vec<GpuNode> {
    let mut live = HashMap::new();
    let mut empty = HashSet::new();
    for (id, section) in sections.iter().enumerate() {
        if !section.is_empty() {
            live.insert(
                id as u32,
                (section.level, section.pos, section.aabb, section.complete),
            );
        } else if section.complete {
            let [x, y, z] = section.pos;
            empty.insert(OctreeSectionPos::new(section.level, x, y, z));
        }
    }
    build_node_tree(&live, Some(&empty), y_bounds).0
}

/// Whole-buffer [`StorageBufferSlice`]s of the arena's bound buffers.
#[derive(Clone, Copy)]
pub struct ArenaSlices {
    pub quads: StorageBufferSlice,
    pub sections: StorageBufferSlice,
    pub opaque_commands: StorageBufferSlice,
    pub translucent_commands: StorageBufferSlice,
    pub trans_sort_meta: StorageBufferSlice,
    pub trans_sort_entries: StorageBufferSlice,
    pub nodes: StorageBufferSlice,
    pub roots: StorageBufferSlice,
    pub render_queue: StorageBufferSlice,
    pub queue_a: StorageBufferSlice,
    pub queue_b: StorageBufferSlice,
    pub queue_meta: [StorageBufferSlice; TRAVERSAL_ITERATIONS],
    pub models: StorageBufferSlice,
    pub biome_tints: StorageBufferSlice,
}

/// The persistent GPU buffers plus their CPU-side allocators.
pub struct GpuArena {
    pub quads: StorageBufferRef,
    pub sections: StorageBufferRef,
    /// `max_sections * 7` commands, `indirect: true`, written by `voxy_cmdgen.csh`.
    pub opaque_commands: StorageBufferRef,
    /// `max_sections` commands, `indirect: true`, sorted back-to-front by `voxy_trans_scatter.csh`.
    pub translucent_commands: StorageBufferRef,
    /// 256-bucket Manhattan-distance sort histogram & prefix-summed offsets (`voxy_trans_*.csh`).
    pub trans_sort_meta: StorageBufferRef,
    /// Visible translucent `(section_id, bucket)` pairs written by `voxy_cmdgen.csh`.
    pub trans_sort_entries: StorageBufferRef,
    /// Hierarchical octree node table (`voxy_traversal.csh`).
    pub nodes: StorageBufferRef,
    /// Top-level `detail == MAX_OCTREE_DETAIL` node indices, read directly in iteration 0.
    pub roots: StorageBufferRef,
    /// Count header followed by compact section IDs emitted by traversal.
    pub render_queue: StorageBufferRef,
    /// Flat ping-pong traversal queues of node indices (same layout as `roots`).
    pub queue_a: StorageBufferRef,
    pub queue_b: StorageBufferRef,
    /// Per-iteration metadata. Indirect reads must not alias writable storage in a dispatch.
    pub queue_meta: [StorageBufferRef; TRAVERSAL_ITERATIONS],
    /// Baked model table: `MODEL_WORDS` (16) `u32`s per 20-bit state id (see `voxy_quads.vsh`).
    pub models: StorageBufferRef,
    /// Biome tint LUT: grass, foliage, water `0x00BBGGRR` per 9-bit biome id.
    pub biome_tints: StorageBufferRef,
    /// Whole-buffer slices for set-1 bindings.
    pub slices: ArenaSlices,
    max_sections: u32,
    residency: ArenaResidency,
}

/// CPU-side section state, including changes not yet published to the GPU.
struct ArenaResidency {
    quad_alloc: FreeList,
    ids: IdAllocator,
    /// Quad-arena start (in quads) per live section id.
    live: HashMap<u32, u32>,
    /// `(level, pos, aabb, complete)` per live section id for building the GPU octree node table.
    live_meta: HashMap<u32, LiveSectionMeta>,
    /// Resolved empty meshes, scoped to the dimension whose geometry is resident.
    known_empty: HashMap<String, HashSet<OctreeSectionPos>>,
    last_flushed_y_bounds: Option<(i32, i32)>,
    nodes_dirty: bool,
    root_count: u32,
    dirty: BTreeMap<u32, SectionMeta>,
}

struct PendingFlush {
    nodes_bytes: Vec<u8>,
    roots_bytes: Vec<u8>,
    root_count: u32,
    section_runs: Vec<(u64, Vec<u8>)>,
}

impl GpuArena {
    pub fn new(host: &mut dyn ClientHost, config: ArenaConfig) -> Result<Self, ArenaError> {
        let initial_sections = config.max_sections.max(1);
        let min_sections = initial_sections.min(1 << 14);
        let mut sections = initial_sections;
        let mut geometry = config.geometry_bytes.max(MIN_GEOMETRY_BYTES);
        loop {
            match Self::try_new(host, sections, geometry) {
                Ok(arena) => return Ok(arena),
                // No renderer: smaller buffers cannot help.
                Err(e @ ArenaError::Buffer(BufferError::Unavailable)) => return Err(e),
                Err(ArenaError::Buffer(_))
                    if geometry > MIN_GEOMETRY_BYTES || sections > min_sections =>
                {
                    geometry = (geometry / 2).max(MIN_GEOMETRY_BYTES);
                    sections = (sections / 2).max(min_sections);
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn try_new(
        host: &mut dyn ClientHost,
        max_sections: u32,
        geometry_bytes: u64,
    ) -> Result<Self, ArenaError> {
        let geometry_bytes = geometry_bytes / QUAD_BYTES * QUAD_BYTES;
        let n = max_sections as u64;
        let mut made: Vec<StorageBufferRef> = Vec::new();
        let mut make = |host: &mut dyn ClientHost, desc: StorageBufferDescriptor| match host
            .create_storage_buffer(&desc)
        {
            Ok(b) => {
                made.push(b);
                Ok(b)
            }
            Err(e) => {
                for b in made.drain(..) {
                    let _ = b.release(&*host);
                }
                Err(ArenaError::from(e))
            }
        };
        let quads = make(host, StorageBufferDescriptor::new(geometry_bytes))?;
        let sections = make(
            host,
            StorageBufferDescriptor::new(n * SECTION_METADATA_BYTES),
        )?;
        let cmd_bytes = n * OPAQUE_SLOTS_PER_SECTION * DRAW_INDEXED_INDIRECT_BYTES;
        let tcmd_bytes = n * TRANSLUCENT_SLOTS_PER_SECTION * DRAW_INDEXED_INDIRECT_BYTES;
        let tsort_entry_bytes = n * TRANS_SORT_ENTRY_BYTES;
        let opaque_commands = make(host, StorageBufferDescriptor::new(cmd_bytes).indirect(true))?;
        let translucent_commands = make(
            host,
            StorageBufferDescriptor::new(tcmd_bytes).indirect(true),
        )?;
        let trans_sort_meta = make(host, StorageBufferDescriptor::new(TRANS_SORT_META_BYTES))?;
        let trans_sort_entries = make(host, StorageBufferDescriptor::new(tsort_entry_bytes))?;
        let node_bytes = GPU_MAX_NODES * GPU_NODE_BYTES;
        let render_queue_bytes = RENDER_QUEUE_HEADER_BYTES as u64 + n * 4;
        let nodes = make(host, StorageBufferDescriptor::new(node_bytes))?;
        let roots = make(host, StorageBufferDescriptor::new(GPU_TRAV_QUEUE_BYTES))?;
        let render_queue = make(host, StorageBufferDescriptor::new(render_queue_bytes))?;
        let queue_a = make(host, StorageBufferDescriptor::new(GPU_TRAV_QUEUE_BYTES))?;
        let queue_b = make(host, StorageBufferDescriptor::new(GPU_TRAV_QUEUE_BYTES))?;
        let queue_meta_desc = || StorageBufferDescriptor::new(QUEUE_META_BYTES).indirect(true);
        let mut queue_meta = [make(host, queue_meta_desc())?; TRAVERSAL_ITERATIONS];
        for meta in &mut queue_meta[1..] {
            *meta = make(host, queue_meta_desc())?;
        }
        let model_bytes = MODEL_COUNT * MODEL_WORDS * 4;
        let biome_bytes = BIOME_COUNT * 3 * 4;
        let models = make(host, StorageBufferDescriptor::new(model_bytes))?;
        let biome_tints = make(host, StorageBufferDescriptor::new(biome_bytes))?;
        let slices = ArenaSlices {
            quads: quads.slice(0, geometry_bytes),
            sections: sections.slice(0, n * SECTION_METADATA_BYTES),
            opaque_commands: opaque_commands.slice(0, cmd_bytes),
            translucent_commands: translucent_commands.slice(0, tcmd_bytes),
            trans_sort_meta: trans_sort_meta.slice(0, TRANS_SORT_META_BYTES),
            trans_sort_entries: trans_sort_entries.slice(0, tsort_entry_bytes),
            nodes: nodes.slice(0, node_bytes),
            roots: roots.slice(0, GPU_TRAV_QUEUE_BYTES),
            render_queue: render_queue.slice(0, render_queue_bytes),
            queue_a: queue_a.slice(0, GPU_TRAV_QUEUE_BYTES),
            queue_b: queue_b.slice(0, GPU_TRAV_QUEUE_BYTES),
            queue_meta: queue_meta.map(|meta| meta.slice(0, QUEUE_META_BYTES)),
            models: models.slice(0, model_bytes),
            biome_tints: biome_tints.slice(0, biome_bytes),
        };
        Ok(Self {
            slices,
            models,
            biome_tints,
            quads,
            sections,
            opaque_commands,
            translucent_commands,
            trans_sort_meta,
            trans_sort_entries,
            nodes,
            roots,
            render_queue,
            queue_a,
            queue_b,
            queue_meta,
            max_sections,
            residency: ArenaResidency::new(max_sections, geometry_bytes / QUAD_BYTES),
        })
    }

    pub fn max_sections(&self) -> u32 {
        self.max_sections
    }

    /// Number of command slots per bucket; bounds the number of resident sections.
    pub fn section_high_water(&self) -> u32 {
        self.residency.ids.high_water()
    }

    /// `max_count` for the opaque indirect draw.
    pub fn opaque_command_count(&self) -> u32 {
        self.residency.ids.high_water() * OPAQUE_SLOTS_PER_SECTION as u32
    }

    /// `max_count` for the translucent indirect draw.
    pub fn translucent_command_count(&self) -> u32 {
        self.residency.ids.high_water()
    }

    pub fn used_geometry_bytes(&self) -> u64 {
        self.residency.quad_alloc.used() * QUAD_BYTES
    }

    pub fn live_sections(&self) -> usize {
        self.residency.live.len()
    }

    pub fn root_count(&self) -> u32 {
        self.residency.root_count
    }

    /// Uploads `section` (replacing `old_id` when given, like `uploadReplaceSection`) and returns
    /// its new id. The quad bytes are written now; the metadata is queued for [`Self::flush`].
    pub fn upload_replace(
        &mut self,
        host: &mut dyn ClientHost,
        old_id: Option<u32>,
        section: &BuiltSection,
    ) -> Result<u32, ArenaError> {
        if section.is_empty() {
            return Err(ArenaError::EmptySection);
        }
        // The old section stays live until the new one is fully uploaded, so a failed replace
        // keeps showing the previous mesh instead of leaving a hole.
        if let Some(old) = old_id
            && !self.residency.live.contains_key(&old)
        {
            return Err(ArenaError::UnknownSection(old));
        }
        let start = self
            .residency
            .quad_alloc
            .alloc(section.quads.len() as u64)
            .ok_or(ArenaError::GeometryFull)?;
        let Some(id) = self.residency.ids.alloc() else {
            self.residency.quad_alloc.free(start);
            return Err(ArenaError::SectionsFull);
        };
        let mut bytes = Vec::with_capacity(section.quads.len() * QUAD_BYTES as usize);
        for q in &section.quads {
            bytes.extend_from_slice(&q.to_bytes());
        }
        if let Err(e) = self.quads.write_range(&*host, start * QUAD_BYTES, &bytes) {
            self.residency.ids.free(id);
            self.residency.quad_alloc.free(start);
            return Err(e.into());
        }
        self.residency.live.insert(id, start as u32);
        self.residency.live_meta.insert(
            id,
            (section.level, section.pos, section.aabb, section.complete),
        );
        self.residency.nodes_dirty = true;
        self.residency
            .dirty
            .insert(id, SectionMeta::new(section, start as u32));
        if let Some(old) = old_id {
            self.remove(old)?;
        }
        Ok(id)
    }

    /// Frees a section's quads and queues its metadata slot to be zeroed.
    pub fn remove(&mut self, id: u32) -> Result<(), ArenaError> {
        self.residency.remove(id)
    }

    /// Record a complete, empty mesh without allocating a section id.
    pub(crate) fn mark_empty(&mut self, dimension: String, pos: OctreeSectionPos) {
        self.residency.nodes_dirty |= self
            .residency
            .known_empty
            .entry(dimension)
            .or_default()
            .insert(pos);
    }

    /// Withdraw previous empty resolution when a result no longer proves it.
    pub(crate) fn forget_empty(&mut self, dimension: &str, pos: OctreeSectionPos) {
        if let Some(positions) = self.residency.known_empty.get_mut(dimension) {
            self.residency.nodes_dirty |= positions.remove(&pos);
        }
    }

    pub(crate) fn clear_dimension_coverage(&mut self, dimension: Option<&str>) {
        if let Some(dim) = dimension {
            self.residency.nodes_dirty |= self.residency.known_empty.remove(dim).is_some();
        } else {
            self.residency.nodes_dirty |= !self.residency.known_empty.is_empty();
            self.residency.known_empty.clear();
        }
    }

    /// Pending quad writes may already reuse retired ranges, and the buffer API queues each
    /// write separately. Only submit compute and draws after a whole flush succeeds.
    pub(crate) fn is_published(&self) -> bool {
        self.residency.is_published()
    }

    pub fn flush(
        &mut self,
        host: &mut dyn ClientHost,
        dimension: Option<&str>,
        y_bounds: Option<(i32, i32)>,
    ) -> Result<(), ArenaError> {
        let Some(pending) = self
            .residency
            .prepare_flush(dimension, y_bounds, GPU_MAX_NODES)?
        else {
            return Ok(());
        };
        // The buffer API queues each write separately. A failure leaves the whole batch pending
        // and traversal disabled until every write in a subsequent flush succeeds.
        if !pending.nodes_bytes.is_empty() {
            self.nodes.write_range(&*host, 0, &pending.nodes_bytes)?;
        }
        if !pending.roots_bytes.is_empty() {
            self.roots.write_range(&*host, 0, &pending.roots_bytes)?;
        }
        for (offset, bytes) in pending.section_runs {
            self.sections.write_range(&*host, offset, &bytes)?;
        }
        self.residency.commit_flush(pending.root_count);
        self.residency.last_flushed_y_bounds = y_bounds;
        Ok(())
    }

    /// Writes baked model records starting at `first_model` (`words.len()` a multiple of 16).
    pub fn write_models(
        &self,
        host: &mut dyn ClientHost,
        first_model: u32,
        words: &[u32],
    ) -> Result<(), ArenaError> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.models
            .write_range(&*host, first_model as u64 * MODEL_WORDS * 4, &bytes)?;
        Ok(())
    }

    /// Writes the biome tint LUT starting at `first_biome` (3 words per biome).
    pub fn write_biome_tints(
        &self,
        host: &mut dyn ClientHost,
        first_biome: u32,
        words: &[u32],
    ) -> Result<(), ArenaError> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.biome_tints
            .write_range(&*host, first_biome as u64 * 12, &bytes)?;
        Ok(())
    }

    /// Frees every GPU buffer (ruling point 1). Call on mod unload or re-init only; the arena
    /// otherwise survives resource reload and world close (ruling point 2). To empty it on a
    /// world change, `remove` each section instead. Failures are ignored, so one stale handle
    /// cannot leak the rest.
    pub fn release(self, host: &mut dyn ClientHost) {
        let _ = self.quads.release(&*host);
        let _ = self.sections.release(&*host);
        let _ = self.opaque_commands.release(&*host);
        let _ = self.translucent_commands.release(&*host);
        let _ = self.trans_sort_meta.release(&*host);
        let _ = self.trans_sort_entries.release(&*host);
        let _ = self.nodes.release(&*host);
        let _ = self.roots.release(&*host);
        let _ = self.render_queue.release(&*host);
        let _ = self.queue_a.release(&*host);
        let _ = self.queue_b.release(&*host);
        for meta in self.queue_meta {
            let _ = meta.release(&*host);
        }
        let _ = self.models.release(&*host);
        let _ = self.biome_tints.release(&*host);
    }
}

impl ArenaResidency {
    fn new(max_sections: u32, quad_capacity: u64) -> Self {
        Self {
            quad_alloc: FreeList::new(quad_capacity),
            ids: IdAllocator::new(max_sections),
            live: HashMap::new(),
            live_meta: HashMap::new(),
            known_empty: HashMap::new(),
            last_flushed_y_bounds: None,
            nodes_dirty: false,
            root_count: 0,
            dirty: BTreeMap::new(),
        }
    }

    fn is_published(&self) -> bool {
        !self.nodes_dirty && self.dirty.is_empty()
    }

    /// Frees a section's quads and zeroes its metadata so cmdgen emits `instance_count = 0`.
    fn remove(&mut self, id: u32) -> Result<(), ArenaError> {
        let start = self
            .live
            .remove(&id)
            .ok_or(ArenaError::UnknownSection(id))?;
        self.live_meta.remove(&id);
        self.nodes_dirty = true;
        self.quad_alloc.free(start as u64);
        self.ids.free(id);
        self.dirty.insert(id, SectionMeta::default());
        Ok(())
    }

    fn prepare_flush(
        &mut self,
        dimension: Option<&str>,
        y_bounds: Option<(i32, i32)>,
        max_nodes: u64,
    ) -> Result<Option<PendingFlush>, ArenaError> {
        self.nodes_dirty |= self.last_flushed_y_bounds != y_bounds;
        if self.is_published() {
            return Ok(None);
        }
        let mut pending = PendingFlush {
            nodes_bytes: Vec::new(),
            roots_bytes: Vec::new(),
            root_count: self.root_count,
            section_runs: Vec::new(),
        };
        if self.nodes_dirty {
            let known_empty = dimension.and_then(|dim| self.known_empty.get(dim));
            let (nodes, mut roots) = build_node_tree(&self.live_meta, known_empty, y_bounds);
            if nodes.len() as u64 > max_nodes {
                return Err(ArenaError::NodesFull);
            }
            roots.truncate(GPU_TRAV_QUEUE_CAP as usize);
            pending.root_count = roots.len() as u32;
            pending
                .nodes_bytes
                .reserve(nodes.len() * GPU_NODE_BYTES as usize);
            pending
                .nodes_bytes
                .extend(nodes.into_iter().flat_map(GpuNode::to_bytes));
            pending.roots_bytes.reserve(roots.len() * 4);
            pending
                .roots_bytes
                .extend(roots.into_iter().flat_map(u32::to_le_bytes));
        }

        // Coalesce consecutive metadata slots.
        let mut next = None;
        for (&id, meta) in &self.dirty {
            if next != Some(id) {
                pending
                    .section_runs
                    .push((u64::from(id) * SECTION_METADATA_BYTES, Vec::new()));
            }
            pending
                .section_runs
                .last_mut()
                .unwrap()
                .1
                .extend_from_slice(&meta.to_bytes());
            next = Some(id + 1);
        }
        Ok(Some(pending))
    }

    fn commit_flush(&mut self, root_count: u32) {
        self.root_count = root_count;
        self.nodes_dirty = false;
        self.dirty.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_list_best_fit_and_coalesce() {
        let mut f = FreeList::new(100);
        let a = f.alloc(10).unwrap();
        let b = f.alloc(20).unwrap();
        let c = f.alloc(30).unwrap();
        assert_eq!((a, b, c), (0, 10, 30));
        assert_eq!(f.used(), 60);
        assert_eq!(f.free(b), Some(20));
        // Hole of 20 at 10 is the smallest run that can fit 15.
        assert_eq!(f.alloc(15), Some(10));
        assert_eq!(f.free(10), Some(15));
        assert_eq!(f.free(a), Some(10));
        // 0..30 is one coalesced run again.
        assert_eq!(f.alloc(30), Some(0));
        assert_eq!(f.free(c), Some(30));
        assert_eq!(f.free(0), Some(30));
        assert_eq!(f.used(), 0);
        assert_eq!(f.alloc(100), Some(0));
    }

    #[test]
    fn free_list_best_fit_avoids_job5_fragmentation() {
        let mut f = FreeList::new(93);
        let large = f.alloc(60).unwrap();
        let small = std::array::from_fn::<_, 3, _>(|_| {
            f.alloc(1).unwrap(); // Live guard separating the free runs.
            f.alloc(10).unwrap()
        });
        assert_eq!(f.free(large), Some(60));
        for start in small {
            assert_eq!(f.free(start), Some(10));
        }
        let placements = std::array::from_fn::<_, 3, _>(|_| f.alloc(10).unwrap());
        // First-fit would take 0, 10, 20, leaving only 30 contiguous units at the low end.
        assert_eq!(f.alloc(60), Some(large));
        assert_eq!(placements, small);
        assert_eq!(f.used(), f.capacity());
    }

    #[test]
    fn section_position_round_trips_with_low_z_bits() {
        for level in [0u8, 3, 15] {
            for &x in &[0, 1, -1, 12345, -12345, (1 << 23) - 1, -(1 << 23)] {
                for &y in &[0, 7, -1, -128, 127] {
                    for &z in &[0, 5, -3, 0xF, -0x11, 99991, -99991, (1 << 23) - 1] {
                        let words = pack_section_pos(level, x, y, z);
                        assert_eq!(words[1] >> 28, z as u32 & 0xF);
                        assert_eq!(unpack_section_pos(words), (level, [x, y, z]));
                    }
                }
            }
        }
    }

    #[test]
    fn free_list_full_zero_and_unknown() {
        let mut f = FreeList::new(8);
        assert_eq!(f.alloc(0), None);
        assert_eq!(f.alloc(9), None);
        assert_eq!(f.alloc(8), Some(0));
        assert_eq!(f.alloc(1), None);
        assert_eq!(f.free(3), None);
        assert_eq!(f.free(0), Some(8));
        assert_eq!(f.free(0), None);
    }

    #[test]
    fn free_list_coalesces_both_neighbours() {
        let mut f = FreeList::new(30);
        let a = f.alloc(10).unwrap();
        let b = f.alloc(10).unwrap();
        let c = f.alloc(10).unwrap();
        f.free(a);
        f.free(c);
        f.free(b);
        assert_eq!(f.alloc(30), Some(0));
    }

    #[test]
    fn id_allocator_reuses_lowest_and_shrinks_high_water() {
        let mut ids = IdAllocator::new(3);
        assert_eq!(
            (ids.alloc(), ids.alloc(), ids.alloc()),
            (Some(0), Some(1), Some(2))
        );
        assert_eq!(ids.alloc(), None);
        assert!(ids.free(1));
        assert_eq!(ids.high_water(), 3);
        assert!(ids.free(2));
        assert_eq!(ids.high_water(), 1);
        assert!(!ids.free(2));
        assert_eq!(ids.alloc(), Some(1));
    }

    #[test]
    fn indirect_command_is_20_bytes_in_wgpu_field_order() {
        let cmd = DrawIndexedIndirect {
            index_count: 0x0102_0304,
            instance_count: 1,
            first_index: 7,
            base_vertex: -5,
            first_instance: 9,
        };
        let b = cmd.to_bytes();
        assert_eq!(b.len(), 20);
        assert_eq!(&b[0..4], &0x0102_0304u32.to_le_bytes());
        assert_eq!(&b[4..8], &1u32.to_le_bytes());
        assert_eq!(&b[8..12], &7u32.to_le_bytes());
        assert_eq!(&b[12..16], &(-5i32).to_le_bytes());
        assert_eq!(&b[16..20], &9u32.to_le_bytes());
    }

    #[test]
    fn bucket_command_and_culled_command() {
        let c = DrawIndexedIndirect::for_bucket(1000, 3, 42);
        assert_eq!(c.index_count, 18);
        assert_eq!(c.instance_count, 1);
        assert_eq!(c.base_vertex, 4000);
        assert_eq!(c.first_instance, 42);
        assert_eq!(DrawIndexedIndirect::CULLED.instance_count, 0);
    }

    #[test]
    fn packed_quad_is_stored_little_endian_in_the_arena() {
        let q = PackedQuad::new(31, 17, 3, 5, 16, 2, 0xF_FFFF, 0x1FF, 15, 15);
        assert_eq!(q.to_bytes(), q.0.to_le_bytes());
        assert_eq!(PackedQuad::default().to_bytes(), [0; 8]);
    }

    #[test]
    fn section_pos_round_trips_negative_coordinates() {
        for &(l, x, y, z) in &[
            (0u8, -1, -1, -1),
            (15, 8_388_607, 127, -8_388_608),
            (3, -77, -128, 12345),
        ] {
            assert_eq!(
                unpack_section_pos(pack_section_pos(l, x, y, z)),
                (l, [x, y, z])
            );
        }
    }

    #[test]
    fn metadata_record_is_32_bytes_with_paired_counts() {
        let s = BuiltSection {
            level: 2,
            pos: [1, 2, 3],
            aabb: SectionAabb::FULL,
            quads: vec![PackedQuad::default(); 36],
            counts: [1, 2, 3, 4, 5, 6, 7, 8],
            complete: false,
        };
        let m = SectionMeta::new(&s, 640);
        assert_eq!(m.a[3], 640);
        assert_eq!(m.b, [1 | 2 << 16, 3 | 4 << 16, 5 | 6 << 16, 7 | 8 << 16]);
        assert_eq!(m.to_bytes().len(), 32);
        assert_eq!(SectionAabb::FULL.pack() >> 15, 0b11111_11111_11111);
    }

    #[test]
    fn gpu_node_tree_links_roots_to_leaves() {
        let mut live = HashMap::new();
        live.insert(10, (0u8, [-3, 2, 5], SectionAabb::FULL, false));
        let (nodes, roots) = build_node_tree(&live, None, None);
        assert_eq!(roots.len(), 1);
        let root = nodes[roots[0] as usize];
        assert_eq!(
            unpack_section_pos([root.pos0, root.pos1]).0,
            MAX_OCTREE_DETAIL
        );
        assert_eq!(root.section_id, GpuNode::NONE);
        assert_ne!(root.child_mask, 0);
        // Walk down 4 levels from root to detail-0 leaf section 10.
        let mut cur = root;
        for _ in 1..TRAVERSAL_ITERATIONS {
            assert_ne!(cur.child_base, GpuNode::NONE);
            let ci = cur.child_mask.trailing_zeros() as usize;
            cur = nodes[cur.child_base as usize + ci];
        }
        assert_eq!(cur.section_id, 10);
        assert_eq!(unpack_section_pos([cur.pos0, cur.pos1]), (0, [-3, 2, 5]));
    }

    #[test]
    fn seven_of_eight_resolved_draws_parent() {
        let tight = SectionAabb {
            offset: [1, 2, 3],
            size: [4, 5, 6],
        };
        let level = MAX_OCTREE_DETAIL;
        let parent = OctreeSectionPos::new(level, 0, 0, 0);
        let kids = parent.children().unwrap();
        let live = HashMap::from([
            (1, (level, [0; 3], tight, true)),
            (2, (level - 1, [1, 0, 0], tight, true)),
            (3, (level - 2, [0; 3], tight, true)),
        ]);
        // Octant 0 has no mesh, but all eight of its own octants resolved.
        let grandkids = kids[0].children().unwrap();
        let mut empty: HashSet<_> = grandkids[1..].iter().chain(&kids[2..7]).copied().collect();
        let (nodes, roots) = build_node_tree(&live, Some(&empty), None);
        let root = nodes[roots[0] as usize];
        assert_eq!(root.section_id, 1);
        assert_eq!((root.child_base, root.child_mask), (GpuNode::NONE, 0));
        assert_eq!(root.aabb, tight.pack());

        empty.insert(kids[7]);
        let (nodes, roots) = build_node_tree(&live, Some(&empty), None);
        let root = nodes[roots[0] as usize];
        assert_eq!(root.section_id, 1);
        assert_ne!(root.child_base, GpuNode::NONE);
        assert_eq!(root.child_mask, 0b11);
        assert_eq!(root.aabb, SectionAabb::FULL.pack());
        assert_eq!(nodes[root.child_base as usize + 1].section_id, 2);
        let branch = nodes[root.child_base as usize];
        assert_eq!(branch.section_id, GpuNode::NONE);
        assert_eq!(branch.child_mask, 1);
        assert_eq!(nodes[branch.child_base as usize].section_id, 3);
    }

    #[test]
    fn incomplete_section_does_not_resolve_octant() {
        let level = MAX_OCTREE_DETAIL;
        let kids = OctreeSectionPos::new(level, 0, 0, 0).children().unwrap();
        let mut live = HashMap::from([
            (1, (level, [0; 3], SectionAabb::FULL, true)),
            (2, (level - 1, [0; 3], SectionAabb::FULL, false)),
        ]);
        let empty = kids[1..].iter().copied().collect();
        for complete in [false, true] {
            live.get_mut(&2).unwrap().3 = complete;
            let (nodes, roots) = build_node_tree(&live, Some(&empty), None);
            let root = nodes[roots[0] as usize];
            assert_eq!(root.section_id, 1);
            if complete {
                assert_eq!(root.child_mask, 1);
                assert_eq!(nodes[root.child_base as usize].section_id, 2);
            } else {
                assert_eq!((root.child_base, root.child_mask), (GpuNode::NONE, 0));
            }
        }
        // Without a meshed parent, the same incomplete section is reachable right away.
        live.remove(&1);
        live.get_mut(&2).unwrap().3 = false;
        let (nodes, roots) = build_node_tree(&live, Some(&empty), None);
        let root = nodes[roots[0] as usize];
        assert_eq!(root.child_mask, 1);
        assert_eq!(nodes[root.child_base as usize].section_id, 2);
    }

    #[test]
    fn build_height_resolves_meshless_octants_below_a_meshed_detail_four_root() {
        // Each detail-2 grandchild spans 128 blocks. Declare only the in-range rows.
        for (bounds, root_y, in_range_rows, child_mask) in [
            ((0, 256), 0, 0..2, 0x0f), // Nether and End
            ((-64, 320), 0, 0..3, 0xff),
            ((-64, 320), -1, -1..0, 0xf0),
        ] {
            let parent = OctreeSectionPos::new(4, 0, root_y, 0);
            let mut live = HashMap::from([(0, (4, [0, root_y, 0], SectionAabb::FULL, false))]);
            let mut empty = HashSet::new();
            let mut section_id = 1;
            for child in parent.children().unwrap() {
                let mut in_range = child
                    .children()
                    .unwrap()
                    .into_iter()
                    .filter(|pos| in_range_rows.contains(&pos.y));
                // Keep one terrain mesh per in-range branch; the remaining in-range siblings
                // completed as empty. No residency entry at all for out-of-build-height space.
                if let Some(pos) = in_range.next() {
                    live.insert(
                        section_id,
                        (pos.detail, [pos.x, pos.y, pos.z], SectionAabb::FULL, true),
                    );
                    section_id += 1;
                    empty.extend(in_range);
                }
            }
            let unknown = *empty.iter().next().unwrap();
            empty.remove(&unknown);
            let (nodes, roots) = build_node_tree(&live, Some(&empty), Some(bounds));
            let root = nodes[roots[0] as usize];
            assert_eq!(root.section_id, 0);
            assert_eq!((root.child_base, root.child_mask), (GpuNode::NONE, 0));

            empty.insert(unknown);
            let (nodes, roots) = build_node_tree(&live, Some(&empty), Some(bounds));
            let root = nodes[roots[0] as usize];
            assert_eq!(root.section_id, 0);
            assert_ne!(root.child_base, GpuNode::NONE);
            assert_eq!(root.child_mask, child_mask);
            for oct in 0..8 {
                if child_mask & (1 << oct) != 0 {
                    assert_ne!(
                        nodes[root.child_base as usize + oct].child_base,
                        GpuNode::NONE
                    );
                }
            }
        }
    }

    #[test]
    fn failed_flush_keeps_retirement_pending_until_parent_is_published() {
        let level = MAX_OCTREE_DETAIL;
        let kids = OctreeSectionPos::new(level, 0, 0, 0).children().unwrap();
        let dim = "test:overworld";
        let mut arena = ArenaResidency::new(3, 3);
        for (level, pos) in [(level, [0; 3]), (level - 1, [0; 3]), (level - 1, [1, 0, 0])] {
            let id = arena.ids.alloc().unwrap();
            let start = arena.quad_alloc.alloc(1).unwrap() as u32;
            arena.live.insert(id, start);
            arena
                .live_meta
                .insert(id, (level, pos, SectionAabb::FULL, true));
        }
        arena
            .known_empty
            .insert(dim.to_owned(), kids[2..].iter().copied().collect());
        arena.nodes_dirty = true;
        let initial = arena
            .prepare_flush(Some(dim), None, GPU_MAX_NODES)
            .unwrap()
            .unwrap();
        assert_eq!(&initial.nodes_bytes[16..20], &0b11u32.to_le_bytes());
        assert_eq!(initial.root_count, 1);
        arena.commit_flush(initial.root_count);
        assert!(arena.is_published());

        arena.remove(1).unwrap();
        assert!(!arena.is_published());
        assert!(matches!(
            arena.prepare_flush(Some(dim), None, 1),
            Err(ArenaError::NodesFull)
        ));
        assert!(arena.nodes_dirty);
        assert_eq!(arena.root_count, 1);
        assert_eq!(arena.dirty.get(&1), Some(&SectionMeta::default()));

        let failed_write = arena
            .prepare_flush(Some(dim), None, GPU_MAX_NODES)
            .unwrap()
            .unwrap();
        // The caller discards this batch if a buffer write fails, without committing.
        assert!(!arena.is_published());
        assert_eq!(arena.root_count, 1);
        let retry = arena
            .prepare_flush(Some(dim), None, GPU_MAX_NODES)
            .unwrap()
            .unwrap();
        assert_eq!(failed_write.nodes_bytes, retry.nodes_bytes);
        assert_eq!(failed_write.roots_bytes, retry.roots_bytes);
        assert_eq!(failed_write.section_runs, retry.section_runs);
        assert_eq!(&retry.nodes_bytes[8..12], &0u32.to_le_bytes());
        assert_eq!(&retry.nodes_bytes[12..16], &GpuNode::NONE.to_le_bytes());
        assert_eq!(&retry.nodes_bytes[16..20], &0u32.to_le_bytes());
        assert_eq!(
            retry.section_runs,
            [(
                SECTION_METADATA_BYTES,
                SectionMeta::default().to_bytes().to_vec()
            )]
        );
        arena.commit_flush(retry.root_count);
        assert!(arena.is_published());
        assert!(
            arena
                .prepare_flush(Some(dim), None, GPU_MAX_NODES)
                .unwrap()
                .is_none()
        );
        assert!(
            arena
                .prepare_flush(Some(dim), Some((0, 64)), GPU_MAX_NODES)
                .unwrap()
                .is_some()
        );
        // A bounds-only rebuild also invalidates the published snapshot until committed.
        assert!(!arena.is_published());
    }
}
