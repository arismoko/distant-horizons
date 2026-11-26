#version 450
// Initialize GPU hierarchical traversal for the frame:
// clears per-section visibility flags and seeds QueueA + DispatchA from RootBuffer.
layout(local_size_x = 64) in;

layout(std140) uniform TravInitParams {
    uvec4 counts; // x = section_count, y = root_count, z = max_queue, w = unused
};

layout(std430) readonly buffer RootBuffer { uint roots[]; };
layout(std430) buffer VisibilityBuffer { uint visible_flags[]; };
layout(std430) buffer QueueA {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
    uint items[];
} queue_a;
layout(std430) buffer DispatchA {
    uint wg_x;
    uint wg_y;
    uint wg_z;
    uint pad;
} dispatch_a;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id == 0u) {
        uint rc = min(counts.y, counts.z);
        queue_a.count = rc;
        dispatch_a.wg_x = (rc + 63u) >> 6;
        dispatch_a.wg_y = 1u;
        dispatch_a.wg_z = 1u;
        dispatch_a.pad = 0u;
    }
    if (id < counts.x) {
        visible_flags[id] = 0u;
    }
    if (id < counts.y && id < counts.z) {
        queue_a.items[id] = roots[id];
    }
}
