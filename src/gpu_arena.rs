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
//! one fixed slot per `(section id, bucket)`, rewritten every frame by `voxy_cmdgen.csh`, so the
//! draw count is simply the id high-water mark times the bucket count.

use mod_api::{
    BufferError, ClientHost, StorageBufferDescriptor, StorageBufferRef, StorageBufferSlice,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

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

/// Address-ordered free list with coalescing, in element units. `free` returns the allocation's
/// size, as Voxy's `AllocationArena.free` does.
#[derive(Debug)]
pub struct FreeList {
    capacity: u64,
    free: BTreeMap<u64, u64>,
    allocated: HashMap<u64, u64>,
    used: u64,
}

impl FreeList {
    pub fn new(capacity: u64) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self {
            capacity,
            free,
            allocated: HashMap::new(),
            used: 0,
        }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    /// First-fit by address. `None` when no free run is large enough.
    pub fn alloc(&mut self, len: u64) -> Option<u64> {
        if len == 0 {
            return None;
        }
        let (&start, &run) = self.free.iter().find(|&(_, &run)| run >= len)?;
        self.free.remove(&start);
        if run > len {
            self.free.insert(start + len, run - len);
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
            self.free.remove(&prev);
            new_start = prev;
            new_len += prev_len;
        }
        if let Some(&next_len) = self.free.get(&(start + len)) {
            self.free.remove(&(start + len));
            new_len += next_len;
        }
        self.free.insert(new_start, new_len);
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
/// Size in bytes of one ping-pong queue buffer (`4 * u32` header + `GPU_TRAV_QUEUE_CAP * u32`).
pub const GPU_TRAV_QUEUE_BYTES: u64 = (4 + GPU_TRAV_QUEUE_CAP as u64) * 4;
/// Size in bytes of one indirect compute dispatch buffer (`wg_x, wg_y, wg_z, pad`).
pub const GPU_DISPATCH_BYTES: u64 = 16;

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

/// Builds a compact top-down octree node table (`detail = MAX_OCTREE_DETAIL` roots down to
/// `detail = 0` leaves) with contiguous 8-child blocks for `voxy_traversal.csh`.
pub fn build_node_tree(live_meta: &HashMap<u32, (u8, [i32; 3], SectionAabb)>) -> NodeTree {
    use crate::octree::{MAX_OCTREE_DETAIL, child_index};

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

    for (&section_id, &(level, pos, aabb)) in live_meta {
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
            // Any internal node with children must keep FULL bounds regardless of HashMap
            // iteration order so frustum/Hi-Z culling never drops children outside a masked mesh.
            nodes[cur_idx].aabb = SectionAabb::FULL.pack();
            cur_idx = child_base + oct;
            cur_pos = next_pos;
            cur_level = next_level;
        }
        nodes[cur_idx].section_id = section_id;
        // Leaf nodes tighten to their exact geometry AABB; internal nodes keep FULL.
        if nodes[cur_idx].child_mask == 0 {
            nodes[cur_idx].aabb = aabb.pack();
        }
    }

    (nodes, roots)
}

fn rebuild_node_tree_if_dirty(
    nodes_dirty: &mut bool,
    live_meta: &HashMap<u32, (u8, [i32; 3], SectionAabb)>,
    max_nodes: u64,
) -> Result<Option<NodeTree>, ArenaError> {
    if !*nodes_dirty {
        return Ok(None);
    }
    let (nodes, roots) = build_node_tree(live_meta);
    if (nodes.len() as u64) > max_nodes {
        // This tree will not fit until section residency changes.
        *nodes_dirty = false;
        return Err(ArenaError::NodesFull);
    }
    Ok(Some((nodes, roots)))
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
    pub visibility: StorageBufferSlice,
    pub queue_a: StorageBufferSlice,
    pub queue_b: StorageBufferSlice,
    pub dispatch_a: StorageBufferSlice,
    pub dispatch_b: StorageBufferSlice,
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
    /// Top-level `detail == MAX_OCTREE_DETAIL` root node indices (`voxy_trav_init.csh`).
    pub roots: StorageBufferRef,
    /// Per-section visibility flags written by `voxy_traversal.csh` and read by `voxy_cmdgen.csh`.
    pub visibility: StorageBufferRef,
    /// Ping-pong traversal queue A (`count`, pad, `items[]`).
    pub queue_a: StorageBufferRef,
    /// Ping-pong traversal queue B (`count`, pad, `items[]`).
    pub queue_b: StorageBufferRef,
    /// Indirect compute dispatch arguments for queue A (`indirect: true`).
    pub dispatch_a: StorageBufferRef,
    /// Indirect compute dispatch arguments for queue B (`indirect: true`).
    pub dispatch_b: StorageBufferRef,
    /// Baked model table: `MODEL_WORDS` (16) `u32`s per 20-bit state id (see `voxy_quads.vsh`).
    pub models: StorageBufferRef,
    /// Biome tint LUT: grass, foliage, water `0x00BBGGRR` per 9-bit biome id.
    pub biome_tints: StorageBufferRef,
    /// Whole-buffer slices for set-1 bindings.
    pub slices: ArenaSlices,
    max_sections: u32,
    quad_alloc: FreeList,
    ids: IdAllocator,
    /// Quad-arena start (in quads) per live section id.
    live: HashMap<u32, u32>,
    /// `(level, pos, aabb)` per live section id for building the GPU octree node table.
    live_meta: HashMap<u32, (u8, [i32; 3], SectionAabb)>,
    nodes_dirty: bool,
    root_count: u32,
    dirty: BTreeMap<u32, SectionMeta>,
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
        let root_bytes = GPU_TRAV_QUEUE_CAP as u64 * 4;
        let vis_bytes = n * 4;
        let nodes = make(host, StorageBufferDescriptor::new(node_bytes))?;
        let roots = make(host, StorageBufferDescriptor::new(root_bytes))?;
        let visibility = make(host, StorageBufferDescriptor::new(vis_bytes))?;
        let queue_a = make(host, StorageBufferDescriptor::new(GPU_TRAV_QUEUE_BYTES))?;
        let queue_b = make(host, StorageBufferDescriptor::new(GPU_TRAV_QUEUE_BYTES))?;
        let dispatch_a = make(
            host,
            StorageBufferDescriptor::new(GPU_DISPATCH_BYTES).indirect(true),
        )?;
        let dispatch_b = make(
            host,
            StorageBufferDescriptor::new(GPU_DISPATCH_BYTES).indirect(true),
        )?;
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
            roots: roots.slice(0, root_bytes),
            visibility: visibility.slice(0, vis_bytes),
            queue_a: queue_a.slice(0, GPU_TRAV_QUEUE_BYTES),
            queue_b: queue_b.slice(0, GPU_TRAV_QUEUE_BYTES),
            dispatch_a: dispatch_a.slice(0, GPU_DISPATCH_BYTES),
            dispatch_b: dispatch_b.slice(0, GPU_DISPATCH_BYTES),
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
            visibility,
            queue_a,
            queue_b,
            dispatch_a,
            dispatch_b,
            max_sections,
            quad_alloc: FreeList::new(geometry_bytes / QUAD_BYTES),
            ids: IdAllocator::new(max_sections),
            live: HashMap::new(),
            live_meta: HashMap::new(),
            nodes_dirty: false,
            root_count: 0,
            dirty: BTreeMap::new(),
        })
    }

    pub fn max_sections(&self) -> u32 {
        self.max_sections
    }

    /// Sections cmdgen must visit (`indirect_lookup`-free: ids are dense up to this mark).
    pub fn section_high_water(&self) -> u32 {
        self.ids.high_water()
    }

    /// `max_count` for the opaque indirect draw.
    pub fn opaque_command_count(&self) -> u32 {
        self.ids.high_water() * OPAQUE_SLOTS_PER_SECTION as u32
    }

    /// `max_count` for the translucent indirect draw.
    pub fn translucent_command_count(&self) -> u32 {
        self.ids.high_water()
    }

    pub fn used_geometry_bytes(&self) -> u64 {
        self.quad_alloc.used() * QUAD_BYTES
    }

    pub fn live_sections(&self) -> usize {
        self.live.len()
    }

    pub fn root_count(&self) -> u32 {
        self.root_count
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
            && !self.live.contains_key(&old)
        {
            return Err(ArenaError::UnknownSection(old));
        }
        let start = self
            .quad_alloc
            .alloc(section.quads.len() as u64)
            .ok_or(ArenaError::GeometryFull)?;
        let Some(id) = self.ids.alloc() else {
            self.quad_alloc.free(start);
            return Err(ArenaError::SectionsFull);
        };
        let mut bytes = Vec::with_capacity(section.quads.len() * QUAD_BYTES as usize);
        for q in &section.quads {
            bytes.extend_from_slice(&q.to_bytes());
        }
        if let Err(e) = self.quads.write_range(&*host, start * QUAD_BYTES, &bytes) {
            self.ids.free(id);
            self.quad_alloc.free(start);
            return Err(e.into());
        }
        self.live.insert(id, start as u32);
        self.live_meta
            .insert(id, (section.level, section.pos, section.aabb));
        self.nodes_dirty = true;
        self.dirty
            .insert(id, SectionMeta::new(section, start as u32));
        if let Some(old) = old_id {
            self.remove(old)?;
        }
        Ok(id)
    }

    /// Frees a section's quads and zeroes its metadata so cmdgen emits `instance_count = 0`.
    pub fn remove(&mut self, id: u32) -> Result<(), ArenaError> {
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

    /// Writes queued metadata, coalescing consecutive ids into one `write_range`. The queue is
    /// cleared only once every run is written, so a failed flush is retried whole next call.
    pub fn flush(&mut self, host: &mut dyn ClientHost) -> Result<(), ArenaError> {
        let mut run_start = 0u32;
        let mut run: Vec<u8> = Vec::new();
        let mut next = 0u32;
        for (&id, meta) in &self.dirty {
            if !run.is_empty() && id != next {
                self.sections.write_range(
                    &*host,
                    run_start as u64 * SECTION_METADATA_BYTES,
                    &run,
                )?;
                run.clear();
            }
            if run.is_empty() {
                run_start = id;
            }
            run.extend_from_slice(&meta.to_bytes());
            next = id + 1;
        }
        if !run.is_empty() {
            self.sections
                .write_range(&*host, run_start as u64 * SECTION_METADATA_BYTES, &run)?;
        }
        self.dirty.clear();
        if let Some((node_list, root_list)) =
            rebuild_node_tree_if_dirty(&mut self.nodes_dirty, &self.live_meta, GPU_MAX_NODES)?
        {
            if !node_list.is_empty() {
                let mut nbytes = Vec::with_capacity(node_list.len() * GPU_NODE_BYTES as usize);
                for n in &node_list {
                    nbytes.extend_from_slice(&n.to_bytes());
                }
                self.nodes.write_range(&*host, 0, &nbytes)?;
            }
            let rcount = root_list.len().min(GPU_TRAV_QUEUE_CAP as usize);
            if rcount > 0 {
                let mut rbytes = Vec::with_capacity(rcount * 4);
                for &r in &root_list[..rcount] {
                    rbytes.extend_from_slice(&r.to_le_bytes());
                }
                self.roots.write_range(&*host, 0, &rbytes)?;
            }
            self.root_count = rcount as u32;
            self.nodes_dirty = false;
        }
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
    /// world change, `remove` each section instead.
    pub fn release(self, host: &mut dyn ClientHost) -> Result<(), ArenaError> {
        self.quads.release(&*host)?;
        self.sections.release(&*host)?;
        self.opaque_commands.release(&*host)?;
        self.translucent_commands.release(&*host)?;
        self.trans_sort_meta.release(&*host)?;
        self.trans_sort_entries.release(&*host)?;
        self.nodes.release(&*host)?;
        self.roots.release(&*host)?;
        self.visibility.release(&*host)?;
        self.queue_a.release(&*host)?;
        self.queue_b.release(&*host)?;
        self.dispatch_a.release(&*host)?;
        self.dispatch_b.release(&*host)?;
        self.models.release(&*host)?;
        self.biome_tints.release(&*host)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_list_first_fit_and_coalesce() {
        let mut f = FreeList::new(100);
        let a = f.alloc(10).unwrap();
        let b = f.alloc(20).unwrap();
        let c = f.alloc(30).unwrap();
        assert_eq!((a, b, c), (0, 10, 30));
        assert_eq!(f.used(), 60);
        assert_eq!(f.free(b), Some(20));
        // Hole of 20 at 10 is reused first-fit.
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
        live.insert(10, (0u8, [-3, 2, 5], SectionAabb::FULL));
        live.insert(20, (4u8, [-1, 0, 0], SectionAabb::FULL));
        let (nodes, roots) = build_node_tree(&live);
        assert_eq!(roots.len(), 1);
        let root = nodes[roots[0] as usize];
        assert_eq!(root.section_id, 20);
        assert_ne!(root.child_mask, 0);
        // Walk down 4 levels from root to detail-0 leaf section 10.
        let mut cur = root;
        for _ in 0..4 {
            assert_ne!(cur.child_base, GpuNode::NONE);
            let ci = cur.child_mask.trailing_zeros() as usize;
            cur = nodes[cur.child_base as usize + ci];
        }
        assert_eq!(cur.section_id, 10);
        assert_eq!(unpack_section_pos([cur.pos0, cur.pos1]), (0, [-3, 2, 5]));
    }

    #[test]
    fn nodes_full_clears_dirty_until_residency_changes() {
        let live = HashMap::from([(
            1,
            (crate::octree::MAX_OCTREE_DETAIL, [0; 3], SectionAabb::FULL),
        )]);
        let mut nodes_dirty = true;
        assert!(matches!(
            rebuild_node_tree_if_dirty(&mut nodes_dirty, &live, 0),
            Err(ArenaError::NodesFull)
        ));
        assert!(!nodes_dirty);
        assert!(matches!(
            rebuild_node_tree_if_dirty(&mut nodes_dirty, &live, 0),
            Ok(None)
        ));

        nodes_dirty = true;
        assert!(matches!(
            rebuild_node_tree_if_dirty(&mut nodes_dirty, &live, 1),
            Ok(Some(_))
        ));
        assert!(nodes_dirty);
    }

    #[test]
    fn gpu_node_tree_keeps_parent_aabb_full_regardless_of_insert_order() {
        let tight = SectionAabb {
            offset: [1, 2, 3],
            size: [4, 5, 6],
        };
        let mut live = HashMap::new();
        live.insert(1, (4u8, [0, 0, 0], tight));
        live.insert(2, (0u8, [0, 0, 0], tight));
        let (nodes, roots) = build_node_tree(&live);
        assert_eq!(nodes[roots[0] as usize].aabb, SectionAabb::FULL.pack());
    }
}
