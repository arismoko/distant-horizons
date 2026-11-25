#version 450
// Sky-masked composite of the offscreen LOD target over vanilla, with distance / height /
// environmental fog and spiral SSAO (port of the previous composite pass).
// Reversed-Z everywhere: depth 0.0 = sky / empty.

layout(std140) uniform CompositeParams {
    mat4 inv_view_proj;
    vec4 fog_color;
    vec4 fog_params;      // near_dh, far_dh, fog_start, fog_end
    vec4 height_and_env;  // camera_y, height_fog_base_y, env_fog_start, env_fog_end
    vec4 ssao_params;     // radius, strength, bias, fade_dist
    uvec4 flags0;         // fog_enabled, is_hdr, fog_falloff, fog_spherical
    uvec4 flags1;         // height_fog_enabled, height_fog_mix_mode, ssao_samples, fog_type
    vec4 sharpen;         // x = cas amount
};

uniform sampler2D LodColor;
uniform sampler2D LodDepth;
uniform sampler2D WorldDepth;

layout(location = 0) in vec2 uv_in;
layout(location = 0) out vec4 out_color;

vec3 unproject(vec2 uv, float depth) {
    vec4 ndc = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    vec4 w = inv_view_proj * ndc;
    return w.xyz / max(abs(w.w), 1e-6) * (w.w < 0.0 ? -1.0 : 1.0);
}

float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float lod_depth_at(ivec2 p) { return texelFetch(LodDepth, p, 0).r; }

float ssao(ivec2 px, ivec2 dims, vec3 rel, float view_dist) {
    uint n = flags1.z;
    float fade = ssao_params.w;
    if (n == 0u || view_dist >= fade) return 0.0;
    vec2 fd = max(vec2(dims), vec2(1.0));
    ivec2 hi = dims - 1;
    ivec2 pr = clamp(px + ivec2(1, 0), ivec2(0), hi);
    ivec2 pl = clamp(px - ivec2(1, 0), ivec2(0), hi);
    ivec2 pd = clamp(px + ivec2(0, 1), ivec2(0), hi);
    ivec2 pu = clamp(px - ivec2(0, 1), ivec2(0), hi);
    float dr = lod_depth_at(pr), dl = lod_depth_at(pl), dd = lod_depth_at(pd), du = lod_depth_at(pu);
    vec3 xr = unproject((vec2(pr) + 0.5) / fd, max(dr, 1e-6));
    vec3 xl = unproject((vec2(pl) + 0.5) / fd, max(dl, 1e-6));
    vec3 xd = unproject((vec2(pd) + 0.5) / fd, max(dd, 1e-6));
    vec3 xu = unproject((vec2(pu) + 0.5) / fd, max(du, 1e-6));
    vec3 ddx_r = xr - rel, ddx_l = rel - xl;
    vec3 dpdx = (dr > 0.0 && (dl <= 0.0 || dot(ddx_r, ddx_r) < dot(ddx_l, ddx_l))) ? ddx_r : ddx_l;
    vec3 ddy_d = xd - rel, ddy_u = rel - xu;
    vec3 dpdy = (dd > 0.0 && (du <= 0.0 || dot(ddy_d, ddy_d) < dot(ddy_u, ddy_u))) ? ddy_d : ddy_u;
    vec3 nrm = cross(dpdy, dpdx);
    float nl = length(nrm);
    if (nl < 1e-5) return 0.0;
    nrm /= nl;
    if (dot(nrm, -rel) < 0.0) nrm = -nrm;

    float radius = ssao_params.x;
    float px_radius = clamp(radius * fd.y / max(view_dist * 1.4, 1.0), 2.5, 42.0);
    float step_r = px_radius / float(n);
    float angle = ign(vec2(px)) * 6.2831853;
    float r = step_r;
    float ao = 0.0, hits = 0.0;
    for (uint i = 0u; i < n; i++) {
        vec2 off = vec2(cos(angle), sin(angle)) * r;
        r += step_r;
        angle += 2.39996323;
        ivec2 sp = clamp(px + ivec2(round(off)), ivec2(0), hi);
        float sd = lod_depth_at(sp);
        if (sd <= 0.0) continue;
        vec3 diff = unproject((vec2(sp) + 0.5) / fd, sd) - rel;
        float dist = length(diff);
        if (dist > 1e-3 && dist < radius * 2.5) {
            ao += max(dot(nrm, diff / dist) - ssao_params.z, 0.0) * (1.0 - clamp(dist / (radius * 2.5), 0.0, 1.0));
            hits += 1.0;
        }
    }
    float raw = ao / max(hits, 1.0);
    float dist_fade = 1.0 - clamp(view_dist / max(fade, 1.0), 0.0, 1.0);
    return clamp(smoothstep(0.0, 0.55, raw) * ssao_params.y * dist_fade, 0.0, 0.65);
}

float fog_curve(float t_raw, uint mode) {
    float t = clamp(t_raw, 0.0, 1.0);
    if (mode == 0u) return t;
    if (mode == 1u) return clamp(1.0 - exp(-2.5 * t), 0.0, 1.0);
    return clamp(1.0 - exp(-2.8 * t * t), 0.0, 1.0);
}

float mix_height_fog(float far_f, float h, uint mode) {
    if (mode == 0u) return far_f;
    if (mode == 1u) return max(far_f, h);
    if (mode == 2u) return clamp(far_f + h, 0.0, 1.0);
    if (mode == 3u) return far_f * h;
    return 1.0 - (1.0 - far_f) * (1.0 - h);
}

vec3 fetch_masked(ivec2 p, ivec2 dims, vec3 fallback) {
    ivec2 c = clamp(p, ivec2(0), dims - 1);
    return lod_depth_at(c) <= 0.0 ? fallback : texelFetch(LodColor, c, 0).rgb;
}

vec3 cas(ivec2 px, ivec2 dims, vec3 e, float amount) {
    if (amount <= 0.001) return e;
    vec3 a = fetch_masked(px + ivec2(-1, -1), dims, e), b = fetch_masked(px + ivec2(0, -1), dims, e);
    vec3 c = fetch_masked(px + ivec2(1, -1), dims, e), d = fetch_masked(px + ivec2(-1, 0), dims, e);
    vec3 f = fetch_masked(px + ivec2(1, 0), dims, e), g = fetch_masked(px + ivec2(-1, 1), dims, e);
    vec3 h = fetch_masked(px + ivec2(0, 1), dims, e), i = fetch_masked(px + ivec2(1, 1), dims, e);
    vec3 mn = min(min(min(d, e), min(f, b)), h) + min(min(min(a, g), c), i);
    vec3 mx = max(max(max(d, e), max(f, b)), h) + max(max(max(a, g), c), i);
    vec3 rcp_mx = vec3(1.0) / max(mx, vec3(1e-4));
    vec3 amp = inversesqrt(max(clamp(min(mn, vec3(2.0) - mx) * rcp_mx, vec3(0.0), vec3(1.0)), vec3(1e-4)));
    float peak = 8.0 - 3.0 * clamp(amount, 0.0, 1.0);
    vec3 w = vec3(-1.0) / (amp * peak);
    vec3 rcp_w = vec3(1.0) / max(vec3(1.0) + 4.0 * w, vec3(1e-4));
    return clamp(((b + d + f + h) * w + e) * rcp_w, vec3(0.0), vec3(1.0));
}

void main() {
    ivec2 dh_dims = textureSize(LodColor, 0);
    ivec2 mc_dims = textureSize(WorldDepth, 0);
    ivec2 px = clamp(ivec2(gl_FragCoord.xy), ivec2(0), dh_dims - 1);
    ivec2 mc_px = clamp(ivec2((vec2(px) + 0.5) * vec2(mc_dims) / max(vec2(dh_dims), vec2(1.0))), ivec2(0), mc_dims - 1);

    if (texelFetch(WorldDepth, mc_px, 0).r > 0.0) discard; // vanilla geometry in front
    float dh_depth = lod_depth_at(px);
    if (dh_depth <= 0.0) discard;

    vec3 col = cas(px, dh_dims, texelFetch(LodColor, px, 0).rgb, sharpen.x);

    float near_dh = fog_params.x, far_dh = fog_params.y;
    float view_dist = (near_dh * far_dh) / max(dh_depth * (far_dh - near_dh) + near_dh, 1e-5);
    vec2 uv = (vec2(px) + 0.5) / max(vec2(dh_dims), vec2(1.0));
    vec3 rel = unproject(uv, dh_depth);

    col *= 1.0 - ssao(px, dh_dims, rel, view_dist);

    uint fog_type = flags1.w;
    float fog_factor = 0.0;
    if (fog_type != 0u) {
        float s = height_and_env.z;
        float e = max(height_and_env.w, s + 1.0);
        fog_factor = clamp((length(rel) - s) / (e - s), 0.0, 1.0);
    } else if (flags0.x != 0u) {
        float fs = fog_params.z;
        float fe = max(fog_params.w, fs + 1.0);
        float d = flags0.w != 0u ? length(rel) : length(rel.xz);
        float far_fog = fog_curve((d - fs) / (fe - fs), flags0.z);
        float height_fog = 0.0;
        if (flags1.x != 0u) {
            float world_y = rel.y + height_and_env.x;
            float below = max(height_and_env.y - world_y, 0.0);
            float h_t = clamp(below / 96.0, 0.0, 1.0) * clamp((d - fs * 0.45) / max(fe * 0.55, 1.0), 0.0, 1.0);
            height_fog = fog_curve(h_t, flags0.z) * 0.65;
        }
        fog_factor = mix_height_fog(far_fog, height_fog, flags1.y);
    }
    vec3 fc = flags0.y != 0u ? pow(max(fog_color.rgb, vec3(0.0)), vec3(2.2)) : fog_color.rgb;
    col = mix(col, fc, fog_factor);
    out_color = vec4(max(col, vec3(0.0)), 1.0);
}
