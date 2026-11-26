#version 450
// Baked-texture shading for GPU-pulled quads (behaviour: Voxy lod/gl46/quads.frag).
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
uniform sampler2D BlockAtlas;

layout(location = 0) in flat uvec4 inter;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 rel_pos;
layout(location = 3) in flat float face_shade;

layout(location = 0) out vec4 out_color;

vec3 srgb_to_linear(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

vec4 unpack_rgba(uint c) {
    return vec4(float(c & 0xFFu), float((c >> 8) & 0xFFu), float((c >> 16) & 0xFFu), float(c >> 24)) / 255.0;
}

float fog_curve(float t_raw, uint mode) {
    float t = clamp(t_raw, 0.0, 1.0);
    if (mode == 0u) return t;
    if (mode == 1u) return clamp(1.0 - exp(-2.5 * t), 0.0, 1.0);
    return clamp(1.0 - exp(-2.8 * t * t), 0.0, 1.0);
}

void main() {
    uint face = inter.x & 0xFFu;
    uint model_id = inter.x >> 8;
    uint mbase = model_id * 16u;
    uint mflags = models[mbase + 12u];
    bool translucent = (mflags & 1u) != 0u;
    bool cutout = (mflags & 4u) != 0u;
    uint tint_kind = (mflags >> 4) & 15u;

    // Clip radius: the vanilla chunk area is not redrawn (clip_dist = 0 disables).
    float clip_dist = depth_and_light.z;
    if (clip_dist > 0.0 && length(rel_pos.xz) < clip_dist) {
        discard;
    }

    uint tile_origin = models[mbase + face];
    uint tile_size_packed = models[mbase + 14u];
    vec2 tile_px = vec2(float(tile_size_packed & 0xFFFFu), float(tile_size_packed >> 16));

    vec4 albedo;
    float tex_alpha;
    if (tile_px.x > 0.0) {
        vec2 atlas_px = vec2(textureSize(BlockAtlas, 0));
        vec2 tile = floor(uv);
        vec2 f = uv - tile; // repeat inside the tile across merged quads
        vec2 origin = vec2(float(tile_origin & 0xFFFFu), float(tile_origin >> 16));
        vec2 inset = vec2(0.5) / tile_px; // keep bilinear taps inside the tile
        vec2 p = (origin + clamp(f, inset, vec2(1.0) - inset) * tile_px) / atlas_px;
        vec2 dx = dFdx(uv) * tile_px / atlas_px;
        vec2 dy = dFdy(uv) * tile_px / atlas_px;
        albedo = textureGrad(BlockAtlas, p, dx, dy);
        tex_alpha = textureLod(BlockAtlas, p, 0.0).a;
    } else {
        albedo = unpack_rgba(models[mbase + 6u + face]);
        tex_alpha = albedo.a;
    }

#ifdef TRANSLUCENT
    if (tex_alpha == 0.0) discard;
#else
    if (cutout && tex_alpha <= 0.1) discard;
    albedo.a = 1.0;
#endif

    // Biome / constant tint. Voxy's "partial" tint (grass-like grey texels only) is the
    // conditional case: tint kinds 1-3 apply only to near-grey texels, kind 4 always.
    if (inter.y != 0xFFFFFFFFu) {
        vec3 tint = vec3(float(inter.y & 0xFFu), float((inter.y >> 8) & 0xFFu), float((inter.y >> 16) & 0xFFu)) / 255.0;
        bool grey = abs(albedo.r - albedo.g) < 0.02 && abs(albedo.g - albedo.b) < 0.02;
        if (tint_kind == 4u || grey) {
            albedo.rgb *= tint;
        }
    }

    // Inside the radius vanilla's own chunks cover, vanilla draws the world.
    if (length(rel_pos.xz) < depth_and_light.z) discard;

    bool hdr = flags.x != 0u;
    vec3 rgb = hdr ? srgb_to_linear(albedo.rgb) : albedo.rgb;

    // Vanilla-style lightmap: light id = block<<4 | sky (what mesh.rs packs).
    float block = float((inter.z >> 4) & 15u) / 15.0;
    float sky = float(inter.z & 15u) / 15.0;
    float light = clamp(max(sky * depth_and_light.w, block) * 0.90 + 0.10, 0.08, 1.0);
    float shade = flags.y != 0u ? face_shade : 1.0;
    rgb *= light * shade;

    // Fog (distance from the camera, spherical or cylindrical).
    float d = fog.w != 0.0 ? length(rel_pos) : length(rel_pos.xz);
    float f = fog_curve((d - fog.x) / max(fog.y - fog.x, 1.0), uint(fog.z));
    vec3 fc = hdr ? srgb_to_linear(fog_color.rgb) : fog_color.rgb;
    rgb = mix(rgb, fc, f);

    out_color = vec4(max(rgb, vec3(0.0)), albedo.a);
}
