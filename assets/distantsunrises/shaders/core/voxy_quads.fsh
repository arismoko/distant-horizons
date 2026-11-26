#version 450
// Baked-texture shading for GPU-pulled quads (behaviour: Voxy lod/gl46/quads.frag).
// Face texels come from the mipmapped 16x16 BlockAtlas via textureGrad.
// Outputs linear HDR: sRGB albedo is decoded before lighting when is_hdr is set.
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
// 16x16 face tiles with a hardware mip chain (`pipeline::ATLAS_MIP_LEVELS` = 5, tile sizes
// 16, 8, 4, 2, 1) and a trilinear sampler (`render_bridge::atlas_sampler`, `max_lod` = 4).
uniform sampler2D BlockAtlas;
const float ATLAS_MAX_LOD = 4.0;
// Depth of the furthest meshed-column box per pixel (`render_bridge::setup_bounds`).
uniform sampler2D BoundDepth;

layout(location = 0) in flat uvec4 inter;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 rel_pos;
layout(location = 3) in flat float face_shade;

layout(location = 0) out vec4 out_color;

vec3 srgb_to_linear(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

// Greyscale texel of a partially tinted face (the part the biome tint multiplies).
bool is_grey(vec3 c) {
    return abs(c.r - c.g) < 0.06 && abs(c.g - c.b) < 0.06;
}

vec4 unpack_rgba(uint c) {
    return vec4(float(c & 0xFFu), float((c >> 8) & 0xFFu), float((c >> 16) & 0xFFu), float(c >> 24)) / 255.0;
}

void main() {
    // Derivatives in uniform control flow, before any discard or branch.
    vec2 duv_x = dFdx(uv);
    vec2 duv_y = dFdy(uv);

    uint face = inter.x & 0xFFu;
    uint model_id = inter.x >> 8;
    uint mbase = model_id * 16u;
    uint mflags = models[mbase + 12u];
    bool translucent = (mflags & 1u) != 0u;
    bool cutout = (mflags & 4u) != 0u;
    uint tint_kind = (mflags >> 4) & 15u;

    // Footprint bound (behaviour: Voxy lod/gl46/quads.frag's bounding-buffer test): where the
    // meshed vanilla chunk columns cover this pixel, the furthest depth of their boxes
    // (`voxy_bounds.vsh` stores 1 - depth; 0 = no column covers the pixel) is the bound, and an
    // LOD fragment nearer than it is vanilla's terrain to draw, not ours. Off (`flags.z = 0`)
    // when the footprint is empty or unavailable: nothing is clipped then.
    if (flags.z != 0u) {
        float bound = 1.0 - texelFetch(BoundDepth, ivec2(gl_FragCoord.xy), 0).r;
        if (gl_FragCoord.z > bound) {
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
        vec2 f = uv - floor(uv); // repeat inside the tile across merged quads
        vec2 origin = vec2(float(tile_origin & 0xFFFFu), float(tile_origin >> 16));

        // Screen-space footprint in level-0 tile texels. The hardware picks the mip level from the
        // explicit atlas-UV gradients (`textureGrad`); the footprint is capped at the coarsest
        // level that still has one texel per tile, so trilinear taps never leave the tile.
        vec2 grad_x = duv_x * tile_px;
        vec2 grad_y = duv_y * tile_px;
        float rho = max(max(length(grad_x), length(grad_y)), 1e-6);
        float max_lod = min(ATLAS_MAX_LOD, log2(min(tile_px.x, tile_px.y)));
        float lod = clamp(log2(rho), 0.0, max_lod);
        float grad_scale = min(1.0, exp2(max_lod) / rho);
        vec2 ddx_atlas = grad_x * grad_scale / atlas_px;
        vec2 ddy_atlas = grad_y * grad_scale / atlas_px;

        // Half a texel of the coarser mip level the sampler may blend in: keeps every bilinear
        // tap inside this face's tile at all levels (at the last level the tile is one texel and
        // the sample sits on its centre).
        vec2 inset = vec2(0.5 * exp2(ceil(lod))) / tile_px;
        vec2 fc = clamp(f, inset, vec2(1.0) - inset);
        vec2 p = (origin + fc * tile_px) / atlas_px;
        vec4 c = textureGrad(BlockAtlas, p, ddx_atlas, ddy_atlas);
        tex_alpha = c.a;

        if (has_tint) {
            if (tint_kind == 4u || face_tint_state == 2u) {
                // Tinting commutes with the mip average.
                c.rgb *= tint;
            } else if (face_tint_state == 1u) {
                // Partial tint (e.g. grass_block_side: greyscale grass fringe on top + brown dirt
                // below). Test greyscale on the UNBLENDED level-0 texel so filtering between grass
                // and dirt never strips the green lip ...
                ivec2 iorigin = ivec2(origin);
                ivec2 itile = ivec2(tile_px);
                ivec2 nearest_px = iorigin + clamp(ivec2(floor(f * tile_px)), ivec2(0), itile - ivec2(1));
                vec3 raw0 = texelFetch(BlockAtlas, nearest_px, 0).rgb;
                if (is_grey(raw0)) c.rgb = raw0 * tint;
                // ... and at coarser levels average unblended level-0 texels over the footprint
                // (4x4 taps), tinting each greyscale texel first, so distant grass-side slopes stay
                // green instead of turning solid dirt brown.
                if (lod > 0.0) {
                    vec2 centre = fc * tile_px;
                    float span = exp2(lod);
                    vec3 acc = vec3(0.0);
                    for (int ty = 0; ty < 4; ty++) {
                        for (int tx = 0; tx < 4; tx++) {
                            vec2 o = (vec2(float(tx), float(ty)) + 0.5) / 4.0 - 0.5;
                            ivec2 tp = iorigin + clamp(ivec2(floor(centre + o * span)), ivec2(0), itile - ivec2(1));
                            vec3 t = texelFetch(BlockAtlas, tp, 0).rgb;
                            acc += is_grey(t) ? t * tint : t;
                        }
                    }
                    c.rgb = mix(c.rgb, acc / 16.0, min(lod, 1.0));
                }
            }
        }
        albedo = c;
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

    bool hdr = flags.x != 0u;
    vec3 rgb = hdr ? srgb_to_linear(albedo.rgb) : albedo.rgb;

    // Vanilla-style lightmap curve (block<<4 | sky) with non-linear sky/block response and
    // gamma-appropriate ambient floor so shadowed faces read naturally instead of jet black.
    float block = float((inter.z >> 4) & 15u) / 15.0;
    float sky = float(inter.z & 15u) / 15.0;
    float sky_curve = pow(sky * depth_and_light.w, 1.35);
    float block_curve = pow(block, 1.35);
    float min_light = hdr ? 0.08 : 0.34;
    float light = clamp(max(sky_curve, block_curve) * (1.0 - min_light) + min_light, min_light, 1.0);
    float shade = flags.y != 0u ? face_shade : 1.0;
    rgb *= light * shade;

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
    vec3 fc = hdr ? srgb_to_linear(fog_color.rgb) : fog_color.rgb;
    rgb = mix(rgb, fc, f);

    out_color = vec4(max(rgb, vec3(0.0)), albedo.a);
}
