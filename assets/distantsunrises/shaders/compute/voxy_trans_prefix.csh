#version 450
// Exclusive prefix sum over the 256 Manhattan-distance translucent buckets so farthest sections
// (bucket 0) draw before nearest sections (bucket 255) (behaviour: Voxy util/prefixsum).
layout(local_size_x = 256) in;

layout(std430) buffer TransSortMeta {
    uint visible_count;
    uint pad0;
    uint bucket_counts[256];
    uint bucket_offsets[256];
};

shared uint s[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    s[tid] = bucket_counts[tid];
    barrier();
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        uint v = (tid >= stride) ? s[tid - stride] : 0u;
        barrier();
        s[tid] += v;
        barrier();
    }
    bucket_offsets[tid] = (tid == 0u) ? 0u : s[tid - 1u];
}
