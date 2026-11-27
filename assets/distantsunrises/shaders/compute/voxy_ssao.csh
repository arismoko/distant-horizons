#version 450
// Screen-space ambient occlusion on distant LOD terrain (behaviour: Voxy post/ssao.comp).
// Reads reversed-Z infinite-far WorldDepth after opaque LODs, reconstructs view-space positions
// analytically, and writes a half-res R32Float image. Each texel stores the 2x2 LOD validity
// mask in the integer part and AO * 0.5 in the fractional part; zero marks an empty cell.
layout(local_size_x = 8, local_size_y = 8) in;

layout(std140) uniform SsaoParams {
    vec4 proj_info;  // x = proj[0][0], y = proj[1][1], z = near, w = clip_dist
    ivec4 viewport;  // xy = half-res SSAO size, zw = full-res WorldDepth size
};

uniform sampler2D WorldDepthTex;
layout(r32f) uniform writeonly image2D SsaoOut;

bool is_lod_pixel(ivec2 fpx, out float view_z) {
    view_z = 0.0;
    if (fpx.x >= viewport.z || fpx.y >= viewport.w) return false;
    float d = texelFetch(WorldDepthTex, fpx, 0).r;
    if (d <= 1e-6) return false;
    view_z = proj_info.z / d;
    return view_z >= proj_info.w + 4.0;
}

vec3 view_pos(ivec2 fpx, float view_z) {
    vec2 uv = (vec2(fpx) + 0.5) / vec2(viewport.zw);
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3(ndc.x * view_z / proj_info.x, ndc.y * view_z / proj_info.y, -view_z);
}

vec3 view_pos_at(ivec2 fpx, out float view_z) {
    ivec2 c = clamp(fpx, ivec2(0), viewport.zw - ivec2(1));
    float d = texelFetch(WorldDepthTex, c, 0).r;
    if (d <= 1e-6) {
        view_z = 0.0;
        return vec3(0.0);
    }
    view_z = proj_info.z / d;
    return view_pos(c, view_z);
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= viewport.x || px.y >= viewport.y) return;

    ivec2 base_px = px * 2;
    float z00, z10, z01, z11;
    bool ok00 = is_lod_pixel(base_px + ivec2(0, 0), z00);
    bool ok10 = is_lod_pixel(base_px + ivec2(1, 0), z10);
    bool ok01 = is_lod_pixel(base_px + ivec2(0, 1), z01);
    bool ok11 = is_lod_pixel(base_px + ivec2(1, 1), z11);
    uint mask = (ok00 ? 1u : 0u) | (ok10 ? 2u : 0u) | (ok01 ? 4u : 0u) | (ok11 ? 8u : 0u);
    if (mask == 0u) {
        imageStore(SsaoOut, px, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Use the first valid pixel, even if the top-left pixel is sky or foreground.
    ivec2 full_px = ok00 ? base_px
                  : ok10 ? (base_px + ivec2(1, 0))
                  : ok01 ? (base_px + ivec2(0, 1))
                         : (base_px + ivec2(1, 1));
    float z0 = ok00 ? z00 : ok10 ? z10 : ok01 ? z01 : z11;
    vec3 p0 = view_pos(full_px, z0);

    // Reconstruct view-space normal using the closer neighbor on each axis to avoid edge halos.
    float zr, zl, zd, zu;
    vec3 pr = view_pos_at(full_px + ivec2(1, 0), zr);
    vec3 pl = view_pos_at(full_px - ivec2(1, 0), zl);
    vec3 pd = view_pos_at(full_px + ivec2(0, 1), zd);
    vec3 pu = view_pos_at(full_px - ivec2(0, 1), zu);

    vec3 dx = (zr > 0.0 && (zl <= 0.0 || abs(zr - z0) < abs(zl - z0))) ? (pr - p0) : (p0 - pl);
    vec3 dy = (zd > 0.0 && (zu <= 0.0 || abs(zd - z0) < abs(zu - z0))) ? (pd - p0) : (p0 - pu);
    vec3 n = cross(dy, dx);
    float nlen = length(n);
    if (nlen <= 1e-5) {
        imageStore(SsaoOut, px, vec4(float(mask) + 0.5, 0.0, 0.0, 1.0));
        return;
    }
    n /= nlen;

    // World-space occlusion radius scales gently with distance so coarse LOD hills get soft
    // valley shading without blackening silhouettes.
    float radius_world = clamp(z0 * 0.045, 2.5, 28.0);
    // Rotate the four orthogonal directions across each 2x2 half-res neighborhood.
    uint rot = uint((px.x & 1) | ((px.y & 1) << 1));
    float base_ang = float(rot) * 0.39269908; // pi / 8
    float occ = 0.0;
    for (int i = 0; i < 4; i++) {
        float ang = base_ang + float(i) * 1.5707963; // pi / 2
        vec2 dir = vec2(cos(ang), sin(ang));
        for (int step = 1; step <= 2; step++) {
            ivec2 spx = full_px + ivec2(round(dir * float(step * 4)));
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
    float ao = clamp(1.0 - (occ / 8.0) * 1.4, 0.58, 1.0);
    imageStore(SsaoOut, px, vec4(float(mask) + ao * 0.5, 0.0, 0.0, 1.0));
}
