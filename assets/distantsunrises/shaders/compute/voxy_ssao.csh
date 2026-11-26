#version 450
// Screen-space ambient occlusion on distant LOD terrain (behaviour: Voxy post/ssao.comp).
// Reads reversed-Z infinite-far WorldDepth after opaque LODs, reconstructs view-space positions
// analytically (zero matrix-inversion precision loss at 2048+ blocks), and writes a scalar AO
// factor (0.55..1.0; 1.0 for sky and vanilla foreground chunks) into an R32Float storage image.
layout(local_size_x = 8, local_size_y = 8) in;

layout(std140) uniform SsaoParams {
    vec4 proj_info;  // x = proj[0][0], y = proj[1][1], z = near, w = clip_dist
    ivec4 viewport;  // xy = width, height; zw = unused
    uvec4 stamp;     // xy = this submission's publication serial (never 0); zw = unused
};

uniform sampler2D WorldDepthTex;
// One invocation publishes `stamp.xy` here; `voxy_ssao_apply.fsh` draws only where it reads the
// serial it expects, so an output this dispatch never wrote is never applied.
layout(std430) buffer SsaoStamp { uvec2 published; };
layout(r32f) uniform writeonly image2D SsaoOut;

// 8 evenly spaced screen-space directions.
const vec2 DIRS[8] = vec2[8](
    vec2( 1.0,  0.0),
    vec2(-1.0,  0.0),
    vec2( 0.0,  1.0),
    vec2( 0.0, -1.0),
    vec2( 0.70710678,  0.70710678),
    vec2(-0.70710678,  0.70710678),
    vec2( 0.70710678, -0.70710678),
    vec2(-0.70710678, -0.70710678)
);

vec3 view_pos_at(ivec2 px, out float view_z) {
    ivec2 c = clamp(px, ivec2(0), viewport.xy - ivec2(1));
    float d = texelFetch(WorldDepthTex, c, 0).r;
    if (d <= 1e-6) {
        view_z = 0.0;
        return vec3(0.0);
    }
    view_z = proj_info.z / d;
    vec2 uv = (vec2(c) + 0.5) / vec2(viewport.xy);
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3(ndc.x * view_z / proj_info.x, ndc.y * view_z / proj_info.y, -view_z);
}

void main() {
    // Before every early return below. Invocation (0, 0) always lies inside the viewport, and the
    // dispatch's writes become visible to the later draw pass as a whole.
    if (gl_GlobalInvocationID.x == 0u && gl_GlobalInvocationID.y == 0u) {
        published = stamp.xy;
    }
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= viewport.x || px.y >= viewport.y) return;

    float z0;
    vec3 p0 = view_pos_at(px, z0);
    // Leave sky and vanilla foreground chunks untouched (ao = 1.0).
    if (z0 <= 0.0 || z0 < proj_info.w + 4.0) {
        imageStore(SsaoOut, px, vec4(1.0, 0.0, 0.0, 1.0));
        return;
    }

    // Reconstruct view-space normal using the closer neighbor on each axis to avoid edge halos.
    float zr, zl, zd, zu;
    vec3 pr = view_pos_at(px + ivec2(1, 0), zr);
    vec3 pl = view_pos_at(px - ivec2(1, 0), zl);
    vec3 pd = view_pos_at(px + ivec2(0, 1), zd);
    vec3 pu = view_pos_at(px - ivec2(0, 1), zu);

    vec3 dx = (zr > 0.0 && (zl <= 0.0 || abs(zr - z0) < abs(zl - z0))) ? (pr - p0) : (p0 - pl);
    vec3 dy = (zd > 0.0 && (zu <= 0.0 || abs(zd - z0) < abs(zu - z0))) ? (pd - p0) : (p0 - pu);
    vec3 n = cross(dy, dx);
    float nlen = length(n);
    if (nlen <= 1e-5) {
        imageStore(SsaoOut, px, vec4(1.0, 0.0, 0.0, 1.0));
        return;
    }
    n /= nlen;

    // World-space occlusion radius scales gently with distance so coarse LOD hills get soft
    // valley shading without blackening silhouettes.
    float radius_world = clamp(z0 * 0.045, 2.5, 28.0);
    float occ = 0.0;
    for (int i = 0; i < 8; i++) {
        for (int step = 1; step <= 2; step++) {
            ivec2 spx = px + ivec2(round(DIRS[i] * float(step * 4)));
            float zs;
            vec3 ps = view_pos_at(spx, zs);
            if (zs <= 0.0) continue;
            vec3 v = ps - p0;
            float dist = length(v);
            if (dist <= 0.05 || dist >= radius_world) continue;
            float ndot = dot(v / dist, n);
            if (ndot > 0.12) {
                float falloff = 1.0 - (dist / radius_world);
                occ += (ndot - 0.12) * falloff * falloff;
            }
        }
    }
    float ao = clamp(1.0 - (occ / 16.0) * 1.4, 0.58, 1.0);
    imageStore(SsaoOut, px, vec4(ao, 0.0, 0.0, 1.0));
}
