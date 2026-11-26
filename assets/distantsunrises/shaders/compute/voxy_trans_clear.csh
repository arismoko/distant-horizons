#version 450
// Clears the 256-bucket Manhattan-distance translucent sort histogram before voxy_cmdgen.csh
// (behaviour: Voxy MDICSectionRenderer translucent sort reset).
layout(local_size_x = 256) in;

layout(std430) buffer TransSortMeta {
    uint visible_count;
    uint opaque_count;
    uint bucket_counts[256];
    uint bucket_offsets[256];
};

void main() {
    uint tid = gl_LocalInvocationID.x;
    if (tid == 0u) {
        visible_count = 0u;
        opaque_count = 0u;
    }
    bucket_counts[tid] = 0u;
    bucket_offsets[tid] = 0u;
}
