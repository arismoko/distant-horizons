#version 450
// GPU-pulled 8-byte quads (behaviour: Voxy lod/gl46/quads3.vert + quad_util.glsl +
// quad_format.glsl). No vertex attributes: vertex = gl_VertexIndex (already includes the indirect
// command's base_vertex = first_quad*4), section = gl_InstanceIndex (= first_instance = id).
// Four vertices per quad; the shared Quads index pattern expands them to two triangles.

struct SectionMeta {
    uvec4 a; // pos0, pos1, aabb, quad_start
    uvec4 b;
};

layout(std140) uniform QuadParams {
    mat4 view_proj;        // camera-relative, jittered (world.jittered)
    ivec3 base_section;    // camera block >> 5
    int shading_enabled;
    vec3 camera_sub_pos;   // camera - base_section*32
    float near_plane;      // depth = near_plane / w
    vec4 fog_and_clip;     // x = start, y = end, z = clip distance (0 = off), w = spherical (0/1)
    vec3 fog_color;        // sRGB
};

layout(std430) readonly buffer QuadBuffer { uvec2 quads[]; };
layout(std430) readonly buffer SectionBuffer { SectionMeta sections[]; };
layout(std430) readonly buffer ModelBuffer { uint models[]; };      // 16 words per model
layout(std430) readonly buffer BiomeTintBuffer { uint biome_tints[]; }; // grass, foliage, water
uniform sampler2D Lightmap;

layout(location = 0) out flat uvec4 inter;   // x: face | model<<8 ; y: tint rgb ; z: unused ; w: lod
layout(location = 1) out vec2 uv;            // tile units across the merged quad
layout(location = 2) out vec3 rel_pos;       // camera-relative world position
layout(location = 4) out flat vec3 lit_shade;

// Vanilla CardinalLighting.DEFAULT: down 0.5, up 1.0, north/south 0.8, west/east 0.6.
const float FACE_SHADE[6] = float[6](0.5, 1.0, 0.8, 0.8, 0.6, 0.6);

vec3 swizzle_axis(uint axis, vec3 d) {
    if (axis == 0u) return d.xzy;  // y-normal faces
    if (axis == 2u) return d.zxy;  // x-normal faces
    return d;                      // z-normal faces
}

uint field(uvec2 q, uint bits, uint shift) {
    uint v = shift >= 32u ? (q.y >> (shift - 32u)) : (q.x >> shift);
    if (shift < 32u && shift + bits > 32u) {
        v |= q.y << (32u - shift);
    }
    return v & ((1u << bits) - 1u);
}

void main() {
    uvec2 q = quads[uint(gl_VertexIndex) >> 2];
    uint corner = uint(gl_VertexIndex) & 3u;
    SectionMeta meta = sections[uint(gl_InstanceIndex)];

    uint face = field(q, 3u, 0u);
    uvec2 size = uvec2(field(q, 4u, 3u), field(q, 4u, 7u)) + 1u;
    vec3 qpos = vec3(field(q, 5u, 21u), field(q, 5u, 16u), field(q, 5u, 11u));
    uint model_id = field(q, 20u, 26u);
    uint biome = field(q, 9u, 46u);
    uint light = field(q, 8u, 55u);

    uint detail = meta.a.x >> 28;
    float lod_scale = float(1u << detail);
    int y = (int(meta.a.x) << 4) >> 24;
    int x = (int(meta.a.y) << 4) >> 8;
    int z = (int(((meta.a.x & 0xFFFFFu) << 4) | (meta.a.y >> 28)) << 8) >> 8;
    ivec3 section = (ivec3(x, y, z) << int(detail)) - base_section;

    uint axis = face >> 1;
    vec3 base = qpos + swizzle_axis(axis, vec3(0.0, 0.0, float(face & 1u)));
    vec2 corner_mask = vec2(float((corner >> 1) & 1u), float(corner & 1u));
    vec3 local = base + swizzle_axis(axis, vec3(vec2(size) * corner_mask, 0.0));
    vec3 world = local * lod_scale + vec3(section << 5) - camera_sub_pos;

    // Model entry: tint kind selects biome LUT column or a constant.
    uint mbase = model_id * 16u;
    uint mflags = models[mbase + 12u];
    uint tint_kind = (mflags >> 4) & 15u;
    // Coarse LODs (detail >= 1) quantize sea-level water (y=62) up to top_y = 64.0, 1 block above
    // detail-0 water (63.0), which creates a dark step line at the detail-0/detail-1 ocean border.
    // Snap coarse sea-level water tops back to 63.0 so all LOD rings stay coplanar across oceans.
    bool is_water = (mflags & 1u) != 0u && tint_kind == 3u;
    if (detail >= 1u && face == 1u && is_water) {
        int top_block_y = ((y << int(detail)) << 5) + int(local.y) * int(1u << detail);
        if (top_block_y == 64) {
            world.y -= 1.0;
        }
    }

    vec4 clip = view_proj * vec4(world, 1.0);
    // The world's reversed depth over an infinite far plane (depth = near / w), so LOD and vanilla
    // geometry order correctly where they overlap and nothing clips at vanilla's far plane.
    clip.z = near_plane;
    gl_Position = clip;

    bool double_sided = (mflags & 8u) != 0u;
    // Exact per-quad backface culling for single-sided faces (keeps `.cull(false)` safe for
    // double-sided foliage while never drawing undersides or back walls from the wrong side).
    if (!double_sided) {
        bool back = (face == 0u && world.y < 0.0)
            || (face == 1u && world.y > 0.0)
            || (face == 2u && world.z < 0.0)
            || (face == 3u && world.z > 0.0)
            || (face == 4u && world.x < 0.0)
            || (face == 5u && world.x > 0.0);
        if (back) {
            gl_Position = vec4(0.0);
            inter = uvec4(0u);
            uv = vec2(0.0);
            rel_pos = vec3(0.0);
            lit_shade = vec3(1.0);
            return;
        }
    }
    uint tint = 0xFFFFFFFFu;
    if (tint_kind == 4u) {
        tint = models[mbase + 13u];
    } else if (tint_kind >= 1u && tint_kind <= 3u) {
        tint = biome_tints[biome * 3u + (tint_kind - 1u)];
    }

    inter = uvec4(face | (model_id << 8), tint, 0u, detail);
    uv = vec2(size) * corner_mask;
    rel_pos = world;

    // Match terrain.wgsl: sample the center of the (block, sky) lightmap texel.
    vec2 lm_uv = (vec2(float((light >> 4) & 15u), float(light & 15u)) + 0.5) / 16.0;
    vec3 lm = textureLod(Lightmap, lm_uv, 0.0).rgb;
    float shade = (shading_enabled != 0 && (mflags & 2u) != 0u) ? FACE_SHADE[face] : 1.0;
    lit_shade = lm * shade;
}
