#version 450
// Hi-Z depth pyramid builder (behaviour: Voxy hiz/hiz.comp + HiZBuffer2, reduction = min for
// reversed-Z so a texel keeps the FARTHEST occluder depth; 0.0 = far/sky).
//
// Two pipelines share this file:
//   HIZ_FROM_WORLD_DEPTH defined : mip 0 <- vanilla `WorldDepth` (TextureSource::WorldDepth)
//   undefined                    : mip N <- mip N-1 (RenderTargetColorMip, mip_levels = 1)
// One dispatch per destination mip; reading mip N-1 while writing mip N are distinct
// subresources (sunrise#5590 hazard rule).
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std140) uniform HizParams {
    ivec2 src_size;   // size of the sampled source level
    ivec2 dst_size;   // size of the written level
};

uniform sampler2D HizSource;
layout(r32f) uniform writeonly image2D HizDest;

void main() {
    ivec2 dst = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(dst, dst_size))) {
        return;
    }
#ifdef HIZ_FROM_WORLD_DEPTH
    float d = texelFetch(HizSource, min(dst, src_size - 1), 0).r;
#else
    // 2x2 footprint; an odd source dimension adds one extra row/column to the last texel so no
    // source texel is dropped (conservative, standard mip-reduction for non-power-of-two sizes).
    ivec2 base = dst * 2;
    ivec2 extent = ivec2(2);
    if ((src_size.x & 1) == 1 && dst.x == dst_size.x - 1) extent.x = 3;
    if ((src_size.y & 1) == 1 && dst.y == dst_size.y - 1) extent.y = 3;
    float d = 1.0;
    for (int y = 0; y < extent.y; y++) {
        for (int x = 0; x < extent.x; x++) {
            d = min(d, texelFetch(HizSource, min(base + ivec2(x, y), src_size - 1), 0).r);
        }
    }
#endif
    imageStore(HizDest, dst, vec4(d));
}
