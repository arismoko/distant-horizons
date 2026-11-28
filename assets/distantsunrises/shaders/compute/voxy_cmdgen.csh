#version 450
// Generate indirect commands from the compact traversal output (Voxy lod/gl46/cmdgen.comp).
// Until count-driven draws are available (#5931), clear the unused tail up to the ID high-water.
// Opaque slots: queue index * 7 + bucket (double-sided, down, up, north, south, west, east).
// first_instance is the section ID, not the queue index.
layout(local_size_x = 128) in;

struct SectionMeta {
    uvec4 a; // pos0, pos1, aabb, quad_start
    uvec4 b; // translucent|double-sided, down|up, north|south, west|east
};

struct DrawCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int base_vertex;
    uint first_instance;
};

layout(std140) uniform CmdgenParams {
    ivec4 base_section; // xyz = camera block >> 5, w = section ID high-water
    uvec4 flags;        // x = sort_translucent (1 = classic/pack radix sort, 0 = direct OIT emit)
};

layout(std430) readonly buffer SectionBuffer { SectionMeta sections[]; };
layout(std430) readonly buffer RenderQueue {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
    uint section_ids[];
} render_queue;
layout(std430) buffer OpaqueCommands { DrawCommand opaque_cmds[]; };
layout(std430) buffer TranslucentCommands { DrawCommand translucent_cmds[]; };
layout(std430) buffer TransSortMeta {
    uint visible_count;
    uint pad0;
    uint bucket_counts[256];
    uint bucket_offsets[256];
};
layout(std430) buffer TransSortEntries { uvec2 trans_entries[]; };

ivec3 section_pos(uvec2 p) {
    int y = (int(p.x) << 4) >> 24;
    int x = (int(p.y) << 4) >> 8;
    int z = int(((p.x & 0xFFFFFu) << 4) | (p.y >> 28));
    z = (z << 8) >> 8;
    return ivec3(x, y, z);
}

// Matches MAX_QUADS_PER_COMMAND in render_bridge.rs: the size of the shared quad index buffer.
const uint MAX_QUADS_PER_COMMAND = 65536u;

DrawCommand make(uint start, uint count, uint id, bool visible) {
    count = min(count, MAX_QUADS_PER_COMMAND);
    DrawCommand c;
    c.index_count = visible ? count * 6u : 0u;
    c.instance_count = visible ? 1u : 0u;
    c.first_index = 0u;
    c.base_vertex = visible ? int(start << 2) : 0;
    c.first_instance = visible ? id : 0u;
    return c;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint sections_count = uint(base_section.w);
    if (gid >= sections_count) return;

    uint o = gid * 7u;
    // Scatter later fills the sorted prefix. Clear every slot first, including the unused tail.
    DrawCommand empty = make(0u, 0u, 0u, false);
    translucent_cmds[gid] = empty;
    if (gid >= min(render_queue.count, sections_count)) {
        for (uint i = 0u; i < 7u; i++) opaque_cmds[o + i] = empty;
        return;
    }

    uint sid = render_queue.section_ids[gid];
    SectionMeta meta = sections[sid];
    if ((meta.b.x | meta.b.y | meta.b.z | meta.b.w) == 0u) {
        for (uint i = 0u; i < 7u; i++) opaque_cmds[o + i] = empty;
        return;
    }
    uint detail = meta.a.x >> 28;
    ivec3 ipos = section_pos(meta.a.xy);
    // Face bucket visibility from the section's position relative to the camera section
    // (a face pointing away from the camera cannot be seen), per Voxy cmdgen's `msk`.
    ivec3 rel = ipos - (base_section.xyz >> int(detail));

    uint ptr = meta.a.w;
    uint c_trans = meta.b.x & 0xFFFFu;
    uint c_dbl = meta.b.x >> 16;
    uint c_down = meta.b.y & 0xFFFFu;
    uint c_up = meta.b.y >> 16;
    uint c_north = meta.b.z & 0xFFFFu;
    uint c_south = meta.b.z >> 16;
    uint c_west = meta.b.w & 0xFFFFu;
    uint c_east = meta.b.w >> 16;

    uint s_trans = ptr;
    uint s_dbl = s_trans + c_trans;
    uint s_down = s_dbl + c_dbl;
    uint s_up = s_down + c_down;
    uint s_north = s_up + c_up;
    uint s_south = s_north + c_north;
    uint s_west = s_south + c_south;
    uint s_east = s_west + c_west;

    if (c_trans != 0u) {
        if (flags.x != 0u) {
            uint dist = min((uint(abs(rel.x) + abs(rel.y) + abs(rel.z)) << detail), 255u);
            uint bucket = 255u - dist;
            uint slot = atomicAdd(visible_count, 1u);
            atomicAdd(bucket_counts[bucket], 1u);
            trans_entries[slot] = uvec2(sid, bucket);
        } else {
            translucent_cmds[gid] = make(s_trans, c_trans, sid, true);
        }
    }
    opaque_cmds[o + 0u] = make(s_dbl, c_dbl, sid, c_dbl != 0u);
    opaque_cmds[o + 1u] = make(s_down, c_down, sid, c_down != 0u && rel.y > -1);
    opaque_cmds[o + 2u] = make(s_up, c_up, sid, c_up != 0u && rel.y < 1);
    opaque_cmds[o + 3u] = make(s_north, c_north, sid, c_north != 0u && rel.z > -1);
    opaque_cmds[o + 4u] = make(s_south, c_south, sid, c_south != 0u && rel.z < 1);
    opaque_cmds[o + 5u] = make(s_west, c_west, sid, c_west != 0u && rel.x > -1);
    opaque_cmds[o + 6u] = make(s_east, c_east, sid, c_east != 0u && rel.x < 1);
}
