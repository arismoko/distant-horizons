#version 450
// Top-down GPU hierarchical octree traversal (behaviour: Voxy HierarchicalOcclusionTraverser /
// traversal_dev.comp). Dispatched via dispatch_indirect across 5 iterations (detail 4 -> 0):
// frustum-culls and Hi-Z-culls subtrees in O(1), evaluates screen-space projected voxel error,
// subdivides into existing children via ping-pong queues, and marks visible leaf/LOD sections.
// A visible node that wants finer detail but lacks child octants records the missing children in
// RequestBuffer; the CPU reads it back without blocking and loads those sections first.
layout(local_size_x = 64) in;

struct GpuNode {
    uvec4 a; // x = pos0, y = pos1, z = section_id (0xFFFFFFFF if none), w = child_base (0xFFFFFFFF if leaf)
    uvec4 b; // x = child_mask (bits 0..7), y = packed aabb, z = 0, w = 0
};

layout(std140) uniform TraversalParams {
    mat4 view_proj;        // camera-relative view-projection
    ivec4 base_section;    // xyz = camera block >> 5, w = max_queue
    vec4 camera_sub_pos;   // xyz = camera - base_section*32, w = focal_px
    ivec4 hiz;             // xy = mip 0 size, z = mip count, w = hiz enabled (0/1)
    vec4 lod_cfg;          // x = pixel error threshold, y = max requests (RequestBuffer capacity), zw = unused
};

layout(std430) readonly buffer NodeBuffer { GpuNode nodes[]; };
layout(std430) readonly buffer QueueIn {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
    uint items[];
} queue_in;
layout(std430) buffer QueueOut {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
    uint items[];
} queue_out;
layout(std430) buffer DispatchOut {
    uint wg_x;
    uint wg_y;
    uint wg_z;
    uint pad;
} dispatch_out;
layout(std430) buffer VisibilityBuffer { uint visible_flags[]; };
layout(std430) buffer RequestBuffer {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
    uvec2 items[];
} req_out;
uniform sampler2D HizPyramid;

ivec3 section_pos(uvec2 p) {
    int y = (int(p.x) << 4) >> 24;
    int x = (int(p.y) << 4) >> 8;
    int z = int(((p.x & 0xFFFFFu) << 4) | (p.y >> 28));
    z = (z << 8) >> 8;
    return ivec3(x, y, z);
}

// Inverse of `section_pos` (`gpu_arena::pack_section_pos`).
uvec2 pack_section_pos(uint detail, ivec3 p) {
    uint z24 = uint(p.z) & 0xFFFFFFu;
    uint w0 = (detail << 28) | ((uint(p.y) & 0xFFu) << 20) | (z24 >> 4);
    uint w1 = ((z24 & 0xFu) << 28) | ((uint(p.x) & 0xFFFFFFu) << 4);
    return uvec2(w0, w1);
}

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

bool occluded_by_hiz(vec3 mn, vec3 mx) {
    if (hiz.w == 0) return false;
    vec3 smin = vec3(1e30);
    vec3 smax = vec3(-1e30);
    for (int i = 0; i < 8; i++) {
        vec3 c = vec3((i & 1) != 0 ? mx.x : mn.x, (i & 2) != 0 ? mx.y : mn.y, (i & 4) != 0 ? mx.z : mn.z);
        vec4 clip = view_proj * vec4(c, 1.0);
        if (clip.w <= 0.0) return false;
        vec3 ndc = clip.xyz / clip.w;
        vec3 s = vec3(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
        smin = min(smin, s);
        smax = max(smax, s);
    }
    smin.xy = clamp(smin.xy, vec2(0.0), vec2(1.0));
    smax.xy = clamp(smax.xy, vec2(0.0), vec2(1.0));
    vec2 px = (smax.xy - smin.xy) * vec2(hiz.xy);
    int level = int(floor(log2(max(max(px.x, px.y), 1.0)))) - 1;
    level = clamp(level, 0, hiz.z - 1);
    ivec2 size = max(ivec2(1), hiz.xy >> level);
    ivec2 lo = ivec2(floor(smin.xy * vec2(size)));
    ivec2 hi = min(ivec2(ceil(smax.xy * vec2(size))), size - 1);
    lo = min(lo, size - 1);
    float far_occluder = 1.0;
    for (int y = lo.y; y <= hi.y; y++) {
        for (int x = lo.x; x <= hi.x; x++) {
            far_occluder = min(far_occluder, texelFetch(HizPyramid, ivec2(x, y), level).r);
        }
    }
    float nearest = max(smax.z, 0.0);
    return far_occluder > 0.0 && far_occluder >= nearest;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= queue_in.count) return;

    uint node_idx = queue_in.items[gid];
    GpuNode n = nodes[node_idx];

    uint detail = n.a.x >> 28;
    ivec3 ipos = section_pos(n.a.xy);
    float scale = float(1u << detail);

    ivec3 aabb_off = (ivec3(int(n.b.y)) >> ivec3(0, 5, 10)) & 31;
    ivec3 aabb_size = ((ivec3(int(n.b.y)) >> ivec3(15, 20, 25)) & 31) + 1;
    vec3 corner = vec3(((ipos << int(detail)) - base_section.xyz) << 5) - camera_sub_pos.xyz;
    vec3 mn = corner + vec3(aabb_off) * scale;
    vec3 mx = mn + vec3(aabb_size) * scale;

    if (outside_frustum(mn, mx) || occluded_by_hiz(mn, mx)) {
        return;
    }

    uint section_id = n.a.z;
    uint child_base = n.a.w;
    uint child_mask = n.b.x & 0xFFu;
    bool has_geom = section_id != 0xFFFFFFFFu;
    bool has_children = child_mask != 0u && child_base != 0xFFFFFFFFu;

    // Screen-space projected voxel error: closest point of node AABB to the camera.
    vec3 closest = clamp(vec3(0.0), mn, mx);
    float dist = max(length(closest), 1.0);
    float voxel_px = (scale * camera_sub_pos.w) / dist;

    // A visible node that wants finer detail asks for every child octant it does not have.
    if (detail > 0u && voxel_px > lod_cfg.x) {
        uint max_req = uint(lod_cfg.y);
        for (uint ci = 0u; ci < 8u; ci++) {
            if (!has_children || (child_mask & (1u << ci)) == 0u) {
                uint slot = atomicAdd(req_out.count, 1u);
                if (slot < max_req) {
                    ivec3 child = ipos * 2 + ivec3(int(ci & 1u), int((ci >> 2) & 1u), int((ci >> 1) & 1u));
                    req_out.items[slot] = pack_section_pos(detail - 1u, child);
                }
            }
        }
    }

    bool subdivide = detail > 0u && has_children && (voxel_px > lod_cfg.x || !has_geom);
    if (subdivide) {
        uint max_q = uint(base_section.w);
        for (uint ci = 0u; ci < 8u; ci++) {
            if ((child_mask & (1u << ci)) != 0u) {
                uint slot = atomicAdd(queue_out.count, 1u);
                if (slot < max_q) {
                    queue_out.items[slot] = child_base + ci;
                    atomicMax(dispatch_out.wg_x, (slot + 64u) >> 6);
                }
            }
        }
        // If this parent still owns uncovered octants (partial child coverage), draw it too.
        if (has_geom && child_mask != 0xFFu) {
            visible_flags[section_id] = 1u;
        }
    } else if (has_geom) {
        visible_flags[section_id] = 1u;
    }
}
