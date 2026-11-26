#version 450
// Per-section frustum + Hi-Z occlusion culling and indirect command generation
// (behaviour: Voxy lod/gl46/cmdgen.comp + hierarchical/screenspace.glsl).
//
// One invocation per section id. Every (section, bucket) slot is rewritten every frame:
// a culled / empty / back-facing bucket gets instance_count = 0 (ruling point 18), a visible one
// gets {index_count = quads*6, 1, 0, base_vertex = first_quad*4, first_instance = section id}
// (Voxy writeCmd). Slots: opaque = id*7 + k (k 0 = double-sided, 1..6 = down,up,north,south,
// west,east), translucent = id.
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
    mat4 view_proj;        // camera-relative, same matrix the quad pass uses
    ivec4 base_section;    // xyz = camera block position >> 5, w = section count (id high-water)
    vec4 camera_sub_pos;   // xyz = camera - base_section*32, w = unused
    ivec4 hiz;             // xy = mip 0 size, z = mip count, w = hiz enabled (0/1)
};

layout(std430) readonly buffer SectionBuffer { SectionMeta sections[]; };
layout(std430) readonly buffer VisibilityBuffer { uint visible_flags[]; };
layout(std430) buffer OpaqueCommands { DrawCommand opaque_cmds[]; };
layout(std430) buffer TranslucentCommands { DrawCommand translucent_cmds[]; };
layout(std430) buffer TransSortMeta {
    uint visible_count;
    uint pad0;
    uint bucket_counts[256];
    uint bucket_offsets[256];
};
layout(std430) buffer TransSortEntries { uvec2 trans_entries[]; };
uniform sampler2D HizPyramid;

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

// Left/right/bottom/top planes of the clip volume (Gribb-Hartmann); near/far are not tested:
// the far plane is deliberately beyond vanilla's (Voxy also skips far).
bool outside_frustum(vec3 mn, vec3 mx) {
    mat4 m = transpose(view_proj);
    vec4 planes[4] = vec4[4](m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1]);
    for (int i = 0; i < 4; i++) {
        vec3 p = vec3(planes[i].x >= 0.0 ? mx.x : mn.x,
                      planes[i].y >= 0.0 ? mx.y : mn.y,
                      planes[i].z >= 0.0 ? mx.z : mn.z);
        if (dot(planes[i].xyz, p) + planes[i].w < 0.0) return true;
    }
    return false;
}

// Screen-space AABB of the box against the pyramid; true if fully hidden by vanilla depth.
bool occluded_by_hiz(vec3 mn, vec3 mx) {
    if (hiz.w == 0) return false;
    vec3 smin = vec3(1e30);
    vec3 smax = vec3(-1e30);
    for (int i = 0; i < 8; i++) {
        vec3 c = vec3((i & 1) != 0 ? mx.x : mn.x, (i & 2) != 0 ? mx.y : mn.y, (i & 4) != 0 ? mx.z : mn.z);
        vec4 clip = view_proj * vec4(c, 1.0);
        if (clip.w <= 0.0) return false; // straddles the camera plane: keep (conservative)
        vec3 ndc = clip.xyz / clip.w;
        // wgpu texture space: y = 0 is the top row.
        vec3 s = vec3(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
        smin = min(smin, s);
        smax = max(smax, s);
    }
    smin.xy = clamp(smin.xy, vec2(0.0), vec2(1.0));
    smax.xy = clamp(smax.xy, vec2(0.0), vec2(1.0));
    vec2 px = (smax.xy - smin.xy) * vec2(hiz.xy);
    // Pick the mip where the box covers about 2 texels per axis (Voxy: floor(log2(max))-1).
    int level = int(floor(log2(max(max(px.x, px.y), 1.0)))) - 1;
    level = clamp(level, 0, hiz.z - 1);
    ivec2 size = max(ivec2(1), hiz.xy >> level);
    ivec2 lo = ivec2(floor(smin.xy * vec2(size)));
    ivec2 hi = min(ivec2(ceil(smax.xy * vec2(size))), size - 1);
    lo = min(lo, size - 1);
    // Farthest occluder over the footprint (reversed-Z: smaller = farther; reduction = min).
    float far_occluder = 1.0;
    for (int y = lo.y; y <= hi.y; y++) {
        for (int x = lo.x; x <= hi.x; x++) {
            far_occluder = min(far_occluder, texelFetch(HizPyramid, ivec2(x, y), level).r);
        }
    }
    // Reversed-Z: the box's nearest point has the LARGEST depth. Sky texels are 0, so a footprint
    // containing sky never occludes (far_occluder = 0).
    float nearest = max(smax.z, 0.0);
    return far_occluder > 0.0 && far_occluder >= nearest;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(base_section.w)) return;

    SectionMeta meta = sections[id];
    uint total = meta.b.x | meta.b.y | meta.b.z | meta.b.w;
    bool live = total != 0u;

    uint detail = meta.a.x >> 28;
    ivec3 ipos = section_pos(meta.a.xy);
    float scale = float(1u << detail);

    ivec3 aabb_off = (ivec3(int(meta.a.z)) >> ivec3(0, 5, 10)) & 31;
    ivec3 aabb_size = ((ivec3(int(meta.a.z)) >> ivec3(15, 20, 25)) & 31) + 1;
    vec3 corner = vec3(((ipos << int(detail)) - base_section.xyz) << 5) - camera_sub_pos.xyz;
    vec3 mn = corner + vec3(aabb_off) * scale;
    vec3 mx = mn + vec3(aabb_size) * scale;

    bool visible = live && visible_flags[id] != 0u && !outside_frustum(mn, mx) && !occluded_by_hiz(mn, mx);

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

    // Clear this translucent slot; visible translucent sections are compacted and sorted
    // back-to-front by Manhattan distance into translucent_cmds[0..visible_count] by
    // voxy_trans_prefix.csh + voxy_trans_scatter.csh.
    translucent_cmds[id] = make(0u, 0u, 0u, false);
    if (visible && c_trans != 0u) {
        uint dist = min(uint(abs(rel.x) + abs(rel.y) + abs(rel.z)), 255u);
        uint bucket = 255u - dist;
        uint slot = atomicAdd(visible_count, 1u);
        atomicAdd(bucket_counts[bucket], 1u);
        trans_entries[slot] = uvec2(id, bucket);
    }
    uint o = id * 7u;
    opaque_cmds[o + 0u] = make(s_dbl, c_dbl, id, visible && c_dbl != 0u);
    opaque_cmds[o + 1u] = make(s_down, c_down, id, visible && c_down != 0u && rel.y > -1);
    opaque_cmds[o + 2u] = make(s_up, c_up, id, visible && c_up != 0u && rel.y < 1);
    opaque_cmds[o + 3u] = make(s_north, c_north, id, visible && c_north != 0u && rel.z > -1);
    opaque_cmds[o + 4u] = make(s_south, c_south, id, visible && c_south != 0u && rel.z < 1);
    opaque_cmds[o + 5u] = make(s_west, c_west, id, visible && c_west != 0u && rel.x > -1);
    opaque_cmds[o + 6u] = make(s_east, c_east, id, visible && c_east != 0u && rel.x < 1);
}
