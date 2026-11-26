#version 450
// Reset the output ping-pong queue and indirect dispatch arguments before each traversal iteration.
layout(local_size_x = 1) in;

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

#ifdef HIZ_PROOF
#include "hiz_proof.glsl"
#endif

void main() {
    queue_out.count = 0u;
    dispatch_out.wg_x = 0u;
    dispatch_out.wg_y = 1u;
    dispatch_out.wg_z = 1u;
    dispatch_out.pad = 0u;
#ifdef HIZ_PROOF
    atomicAdd(hiz_proof.trav_clears, 1u);
#endif
}
