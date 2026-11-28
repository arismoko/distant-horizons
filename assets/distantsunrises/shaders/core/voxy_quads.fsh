#version 450
// Baked-texture shading for GPU-pulled quads (behaviour: Voxy lod/gl46/quads.frag).
// Lighting and fog are applied in the same gamma space as vanilla terrain.
// Define TRANSLUCENT for the translucent pass (alpha kept, only alpha == 0 discarded);
// the opaque pass forces alpha = 1 and discards cutout texels with alpha <= 0.1 (Voxy).

layout(std140) uniform QuadParams {
    mat4 view_proj;
    ivec4 base_section;
    vec4 camera_sub_pos;
    vec4 depth_and_light;
    vec4 fog;
    vec4 fog_color;
    uvec4 flags;
};

layout(std430) readonly buffer ModelBuffer { uint models[]; };
uniform sampler2D BlockAtlas;

layout(location = 0) in flat uvec4 inter;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 rel_pos;
layout(location = 4) in flat vec3 lit_shade;

layout(location = 0) out vec4 out_color;

vec4 unpack_rgba(uint c) {
    return vec4(float(c & 0xFFu), float((c >> 8) & 0xFFu), float((c >> 16) & 0xFFu), float(c >> 24)) / 255.0;
}

void main() {
    // Hoist screen-space derivatives to uniform control flow before any discard/branch.
    vec2 duv_x = dFdx(uv);
    vec2 duv_y = dFdy(uv);

    uint face = inter.x & 0xFFu;
    uint model_id = inter.x >> 8;
    uint mbase = model_id * 16u;
    uint mflags = models[mbase + 12u];
    bool translucent = (mflags & 1u) != 0u;
    bool cutout = (mflags & 4u) != 0u;
    uint tint_kind = (mflags >> 4) & 15u;

    // Clip region: vanilla's loaded chunk area is circular around the camera's 16x16 chunk
    // (until #5911 provides exact per-chunk AABB depth bounds); `clip_dist = 0` disables.
    float clip_dist = depth_and_light.z;
    if (clip_dist > 0.0) {
        vec2 cam_in_chunk = fract(camera_sub_pos.xz / 16.0) * 16.0;
        vec2 rel_chunk_center = rel_pos.xz + (cam_in_chunk - vec2(8.0));
        if (length(rel_chunk_center) < clip_dist) {
            discard;
        }
    }

    uint tile_origin = models[mbase + face];
    uint tile_size_packed = models[mbase + 14u];
    vec2 tile_px = vec2(float(tile_size_packed & 0xFFFFu), float(tile_size_packed >> 16));
    uint face_tint_state = (mflags >> (16u + face * 2u)) & 3u;
    vec3 tint = vec3(1.0);
    bool has_tint = inter.y != 0xFFFFFFFFu && inter.y != 0u;
    if (has_tint) {
        tint = vec3(float(inter.y & 0xFFu), float((inter.y >> 8) & 0xFFu), float((inter.y >> 16) & 0xFFu)) / 255.0;
    }

    vec4 avg_albedo = unpack_rgba(models[mbase + 6u + face]);
    vec4 albedo;
    float tex_alpha;
    if (tile_px.x > 0.0) {
        vec2 atlas_px = vec2(textureSize(BlockAtlas, 0));
        vec2 tile = floor(uv);
        vec2 f = uv - tile; // repeat inside the tile across merged quads
        vec2 origin = vec2(float(tile_origin & 0xFFFFu), float(tile_origin >> 16));
        vec2 inset = vec2(0.5) / tile_px; // keep bilinear taps inside the tile
        vec2 p = (origin + clamp(f, inset, vec2(1.0) - inset) * tile_px) / atlas_px;
        vec4 c0 = textureLod(BlockAtlas, p, 0.0);
        tex_alpha = c0.a;

        // 3-level software mip chain (4x4 -> 2x2 -> 1x1 avg) driven by screen-space UV footprint.
        // For partial-tint faces (e.g. grass_block_side: greyscale grass fringe on top row + brown
        // dirt below), test greyscale on the UNBLENDED nearest texel (`texelFetch`) so bilinear
        // filtering between grass and dirt never strips the green grass lip, and tint greyscale
        // texels BEFORE averaging for mip 1 / mip 2.
        float foot = max(length(duv_x), length(duv_y));
        float lod = clamp(log2(max(foot * tile_px.x, 1.0)), 0.0, 2.0);

        ivec2 iorigin = ivec2(origin);
        ivec2 itile = ivec2(tile_px);
        if (has_tint) {
            if (tint_kind == 4u || face_tint_state == 2u) {
                c0.rgb *= tint;
                avg_albedo.rgb *= tint;
            } else if (face_tint_state == 1u) {
                ivec2 nearest_px = iorigin + clamp(ivec2(floor(f * tile_px)), ivec2(0), itile - ivec2(1));
                vec4 raw0 = texelFetch(BlockAtlas, nearest_px, 0);
                if (abs(raw0.r - raw0.g) < 0.06 && abs(raw0.g - raw0.b) < 0.06) {
                    c0.rgb = raw0.rgb * tint;
                }
            }
        }

        if (lod <= 0.01) {
            albedo = c0;
        } else if (has_tint && face_tint_state == 1u) {
            // Partial-tint mip 1/2: sample unblended texels across the 4x4 tile, tint each
            // greyscale grass-fringe texel first, then average so distant slopes keep their green
            // grass lip instead of turning solid brown dirt.
            vec4 acc = vec4(0.0);
            vec4 q0 = texelFetch(BlockAtlas, iorigin + ivec2(0, 0), 0);
            vec4 q1 = texelFetch(BlockAtlas, iorigin + ivec2(itile.x - 1, 0), 0);
            vec4 q2 = texelFetch(BlockAtlas, iorigin + ivec2(0, itile.y - 1), 0);
            vec4 q3 = texelFetch(BlockAtlas, iorigin + ivec2(itile.x - 1, itile.y - 1), 0);
            if (abs(q0.r - q0.g) < 0.06 && abs(q0.g - q0.b) < 0.06) q0.rgb *= tint;
            if (abs(q1.r - q1.g) < 0.06 && abs(q1.g - q1.b) < 0.06) q1.rgb *= tint;
            if (abs(q2.r - q2.g) < 0.06 && abs(q2.g - q2.b) < 0.06) q2.rgb *= tint;
            if (abs(q3.r - q3.g) < 0.06 && abs(q3.g - q3.b) < 0.06) q3.rgb *= tint;
            acc = 0.25 * (q0 + q1 + q2 + q3);
            albedo = mix(c0, acc, clamp(lod * 0.5, 0.0, 1.0));
        } else {
            // Exact 2x2 mip (mip 1) from the four quadrant centers of the 4x4 tile.
            vec2 q00 = (origin + vec2(0.25, 0.25) * tile_px) / atlas_px;
            vec2 q10 = (origin + vec2(0.75, 0.25) * tile_px) / atlas_px;
            vec2 q01 = (origin + vec2(0.25, 0.75) * tile_px) / atlas_px;
            vec2 q11 = (origin + vec2(0.75, 0.75) * tile_px) / atlas_px;
            vec4 s00 = textureLod(BlockAtlas, q00, 0.0);
            vec4 s10 = textureLod(BlockAtlas, q10, 0.0);
            vec4 s01 = textureLod(BlockAtlas, q01, 0.0);
            vec4 s11 = textureLod(BlockAtlas, q11, 0.0);
            if (has_tint && (tint_kind == 4u || face_tint_state == 2u)) {
                s00.rgb *= tint; s10.rgb *= tint; s01.rgb *= tint; s11.rgb *= tint;
            }
            vec4 c1 = 0.25 * (s00 + s10 + s01 + s11);
            if (lod <= 1.0) {
                albedo = mix(c0, c1, lod);
            } else {
                albedo = mix(c1, avg_albedo, lod - 1.0);
            }
        }
    } else {
        albedo = avg_albedo;
        tex_alpha = albedo.a;
        if (has_tint && (tint_kind == 4u || face_tint_state == 2u)) {
            albedo.rgb *= tint;
        }
    }

#ifdef TRANSLUCENT
    if (tex_alpha == 0.0) discard;
#else
    if (cutout && tex_alpha <= 0.1) discard;
    albedo.a = 1.0;
#endif

    vec3 rgb = albedo.rgb * lit_shade;

    // Atmospheric aerial-perspective haze + smooth far-horizon fade (Voxy
    // `blit_texture_depth_cutout.frag`: keeps foreground/midground crisp, adds soft pale-blue
    // haze across distant peaks and ridges, then blends smoothly into sky over the last 15%).
    float d = fog.w != 0.0 ? length(rel_pos) : length(rel_pos.xz);
    float aerial_start = min(fog.x * 0.30, 360.0);
    float t_aerial = clamp((d - aerial_start) / max(fog.y - aerial_start, 1.0), 0.0, 1.0);
    float haze = smoothstep(0.0, 1.0, t_aerial) * 0.58;
    float fade_start = fog.y * 0.85;
    float horizon_fade = smoothstep(fade_start, fog.y, d);
    float f = clamp(max(haze, horizon_fade), 0.0, 1.0);
    rgb = mix(rgb, fog_color.rgb, f);

    out_color = vec4(max(rgb, vec3(0.0)), albedo.a);
}
