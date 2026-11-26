#version 450
// Scatters visible translucent sections into TranslucentCommands in back-to-front Manhattan
// distance order using the prefix-summed bucket offsets (behaviour: Voxy lod/gl46/buildtranslucents.comp).
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

layout(std430) readonly buffer SectionBuffer { SectionMeta sections[]; };
layout(std430) buffer TranslucentCommands { DrawCommand translucent_cmds[]; };
layout(std430) buffer TransSortMeta {
    uint visible_count;
    uint opaque_count;
    uint bucket_counts[256];
    uint bucket_offsets[256];
};
layout(std430) readonly buffer TransSortEntries { uvec2 trans_entries[]; };

const uint MAX_QUADS_PER_COMMAND = 65536u;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= visible_count) return;

    uvec2 entry = trans_entries[idx];
    uint sid = entry.x;
    uint bucket = entry.y;
    uint dst = atomicAdd(bucket_offsets[bucket], 1u);

    SectionMeta meta = sections[sid];
    uint count = min(meta.b.x & 0xFFFFu, MAX_QUADS_PER_COMMAND);
    DrawCommand c;
    c.index_count = count * 6u;
    c.instance_count = 1u;
    c.first_index = 0u;
    c.base_vertex = int(meta.a.w << 2);
    c.first_instance = sid;
    translucent_cmds[dst] = c;
}
