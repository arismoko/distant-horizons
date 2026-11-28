#version 450
// Top-down GPU octree traversal (Voxy HierarchicalOcclusionTraverser / traversal_dev.comp).
// Roots are always at MAX_OCTREE_DETAIL (4), so iteration 4 sees only detail 0.
// The last-iteration guard also prefers a node's own mesh to enqueueing work no pass will read.
layout(local_size_x = 64) in;

struct GpuNode {
    uvec4 a; // pos0, pos1, section_id, child_base (0xFFFFFFFF = absent id/base)
    uvec4 b; // child_mask (bits 0..7), packed aabb, padding
};

layout(std140) uniform TraversalParams {
    mat4 view_proj;
    ivec4 base_section;    // xyz = camera block >> 5, w = max_queue
    vec4 camera_sub_pos;   // xyz = camera - base_section*32
    float minSSS;
    float render_distance; // blocks (horizontal radius)
    uint is_last_iter;
    uint max_sections;
};

layout(std430) readonly buffer NodeBuffer { GpuNode nodes[]; };
layout(std430) readonly buffer QueueIn { uint items[]; } queue_in;
layout(std430) buffer QueueOut { uint items[]; } queue_out;
layout(std430) readonly buffer QueueMetaIn { uvec4 args; } queue_meta_in;
// Four scalar lanes of a uvec4 so wg_x and count can be atomic independently.
layout(std430) buffer QueueMetaOut {
    uint wg_x;
    uint wg_y;
    uint wg_z;
    uint count;
} queue_meta_out;
layout(std430) buffer RenderQueue {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
    uint section_ids[];
} render_queue;

ivec3 section_pos(uvec2 p) {
    int y = (int(p.x) << 4) >> 24;
    int x = (int(p.y) << 4) >> 8;
    int z = int(((p.x & 0xFFFFFu) << 4) | (p.y >> 28));
    z = (z << 8) >> 8;
    return ivec3(x, y, z);
}

bool outside_frustum(vec3 mn, vec3 mx) {
    mat4 m = transpose(view_proj);
    // CameraMatrices::view_projection uses Sunrise's reversed-Z [0,1] clip space.
    // The near plane is w - z >= 0 (not w + z); deliberately skip the far plane.
    vec4 planes[5] = vec4[5](m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] - m[2]);
    for (int i = 0; i < 5; i++) {
        vec4 plane = planes[i];
        vec3 p = vec3(plane.x < 0.0 ? mn.x : mx.x,
                      plane.y < 0.0 ? mn.y : mx.y,
                      plane.z < 0.0 ? mn.z : mx.z);
        if (dot(plane.xyz, p) + plane.w < 0.0) return true;
    }
    return false;
}

float three_face_area(vec2 x, vec2 y, vec2 z) {
    // Magnitudes of the 2D cross products of each incident edge pair.
    return dot(abs(vec3(x.x, x.x, y.x) * vec3(y.y, z.y, z.y)
                 - vec3(x.y, x.y, y.y) * vec3(y.x, z.x, z.x)), vec3(1.0));
}

float screen_size(vec3 origin, float edge) {
    // Voxy screenspace.glsl: project eight corners via P000 + projected axes; estimate
    // silhouette area by averaging the three faces incident to each opposite corner.
    vec4 p000 = view_proj * vec4(origin, 1.0);
    vec4 dx = view_proj[0] * edge;
    vec4 dy = view_proj[1] * edge;
    vec4 dz = view_proj[2] * edge;
    vec2 p[8];
    for (int i = 0; i < 8; i++) {
        vec4 clip = p000;
        if ((i & 1) != 0) clip += dx;
        if ((i & 2) != 0) clip += dy;
        if ((i & 4) != 0) clip += dz;
        // If any corner is on or behind the camera plane, skip projection and
        // return an area above minSSS so non-leaves take the subdivision path.
        if (clip.w <= 0.0) return 1e30;
        p[i] = (0.5 / clip.w) * clip.xy; // viewport units; translation cancels in edge differences
    }
    return 0.5 * (three_face_area(p[1] - p[0], p[2] - p[0], p[4] - p[0])
                + three_face_area(p[6] - p[7], p[5] - p[7], p[3] - p[7]));
}

void enqueue_children(uint child_base, uint child_mask, uint max_q) {
    // Voxy queue.glsl uses 32 threads and atomicAdd at workgroup thresholds. With 64 threads,
    // publish the largest group needed by an in-bounds child, regardless of insertion order.
    for (uint ci = 0u; ci < 8u; ci++) {
        if ((child_mask & (1u << ci)) == 0u) continue;
        uint slot = atomicAdd(queue_meta_out.count, 1u);
        if (slot < max_q) {
            queue_out.items[slot] = child_base + ci;
            atomicMax(queue_meta_out.wg_x, (slot >> 6) + 1u);
        }
    }
}

void emit_render(uint section_id) {
    uint r = atomicAdd(render_queue.count, 1u);
    if (r < max_sections) {
        render_queue.section_ids[r] = section_id;
    }
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint max_q = uint(base_section.w);
    if (gid >= min(queue_meta_in.args.w, max_q)) return;

    GpuNode n = nodes[queue_in.items[gid]];
    uint detail = n.a.x >> 28;
    ivec3 ipos = section_pos(n.a.xy);
    float edge = float(32u << detail);
    vec3 mn = vec3(((ipos << int(detail)) - base_section.xyz) << 5) - camera_sub_pos.xyz;
    vec3 mx = mn + vec3(edge);
    if (outside_frustum(mn, mx)) return;

    // Match Voxy's closest-point horizontal range gate before evaluating subdivision.
    vec2 closest = clamp(vec2(0.0), mn.xz, mx.xz);
    float radius_sq = render_distance * render_distance;
    if (dot(closest, closest) > radius_sq) return;

    uint section_id = n.a.z;
    uint child_base = n.a.w;
    uint child_mask = n.b.x & 0xFFu;
    bool has_geom = section_id != 0xFFFFFFFFu;
    bool has_children = child_mask != 0u && child_base != 0xFFFFFFFFu;
    bool can_enqueue = has_children && is_last_iter == 0u;
    bool should_descend = detail > 0u && screen_size(mn, edge) > minSSS;

    if (should_descend) {
        if (can_enqueue) {
            enqueue_children(child_base, child_mask, max_q);
        } else if (has_geom) {
            // When descent is wanted but unavailable, match Voxy's furthestPointToCamera
            // and add its 16*16*16 buffer to the squared horizontal distance.
            vec2 far_xz = vec2(mx.x < 0.0 ? mn.x : mx.x, mx.z < 0.0 ? mn.z : mx.z);
            if (dot(far_xz, far_xz) + 4096.0 <= radius_sq) {
                emit_render(section_id);
            }
        }
    } else if (has_geom) {
        emit_render(section_id);
    } else if (detail > 0u && can_enqueue) {
        enqueue_children(child_base, child_mask, max_q);
    }
}
