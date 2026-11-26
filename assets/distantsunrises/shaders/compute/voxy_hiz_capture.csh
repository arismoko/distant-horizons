#version 450
// Copies the existing R32Float mip-0 view, never WorldDepth.
#include "hiz_proof.glsl"
layout(local_size_x = 8, local_size_y = 8) in;

layout(std140) uniform HizProofParams {
    uvec4 size_serial; // width, height, submission serial low/high
};
uniform sampler2D HizMip0;

void main() {
    uvec2 xy = gl_GlobalInvocationID.xy;
    uvec2 size = size_serial.xy;
    if (any(notEqual(uvec2(textureSize(HizMip0, 0)), size)) || any(greaterThanEqual(xy, size))) return;

    hiz_proof.payload[PROOF_MIP0_OFFSET + xy.y * size.x + xy.x] = floatBitsToUint(texelFetch(HizMip0, ivec2(xy), 0).r);
    if (gl_LocalInvocationIndex == 0u) atomicAdd(hiz_proof.copy_groups, 1u);
    if (all(equal(xy, uvec2(0u)))) {
        hiz_proof.serial_lo = size_serial.z;
        hiz_proof.serial_hi = size_serial.w;
    }
}
