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
    ivec4 base_section;    // xyz = camera block >> 5
    vec4 camera_sub_pos;   // xyz = camera - base_section*32
    vec4 depth_and_light;  // x = near_dh, y = far_dh, z = clip_dist (blocks, 0 = off), w = sun brightness
    vec4 fog;              // x = start, y = end, z = falloff mode, w = spherical (0/1)
    vec4 fog_color;        // sRGB
    uvec4 flags;           // x = is_hdr, y = shading, z = debug mode, w = unused
};

layout(std430) readonly buffer QuadBuffer { uvec2 quads[]; };
layout(std430) readonly buffer SectionBuffer { SectionMeta sections[]; };
layout(std430) readonly buffer ModelBuffer { uint models[]; };      // 16 words per model
layout(std430) readonly buffer BiomeTintBuffer { uint biome_tints[]; }; // grass, foliage, water

layout(location = 0) out flat uvec4 inter;   // x: face | model<<8 ; y: tint rgb ; z: light ; w: lod
layout(location = 1) out vec2 uv;            // tile units across the merged quad
layout(location = 2) out vec3 rel_pos;       // camera-relative world position
layout(location = 3) out flat float face_shade;

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
    ivec3 section = (ivec3(x, y, z) << int(detail)) - base_section.xyz;

    uint axis = face >> 1;
    vec3 base = qpos + swizzle_axis(axis, vec3(0.0, 0.0, float(face & 1u)));
    vec2 corner_mask = vec2(float((corner >> 1) & 1u), float(corner & 1u));
    vec3 local = base + swizzle_axis(axis, vec3(vec2(size) * corner_mask, 0.0));
    vec3 world = local * lod_scale + vec3(section << 5) - camera_sub_pos.xyz;

    vec4 clip = view_proj * vec4(world, 1.0);
    // Reversed-Z on [near_dh, far_dh]: w = near -> 1, w = far -> 0 (keeps LODs from fighting
    // vanilla's far plane; matches the previous renderer's depth replacement).
    float near_dh = depth_and_light.x;
    float far_dh = depth_and_light.y;
    clip.z = (near_dh * far_dh - near_dh * clip.w) / max(far_dh - near_dh, 1.0);
    gl_Position = clip;

    // Model entry: tint kind selects biome LUT column or a constant.
    uint mbase = model_id * 16u;
    uint mflags = models[mbase + 12u];
    uint tint_kind = (mflags >> 4) & 15u;
    uint tint = 0xFFFFFFFFu;
    if (tint_kind == 4u) {
        tint = models[mbase + 13u];
    } else if (tint_kind >= 1u && tint_kind <= 3u) {
        tint = biome_tints[biome * 3u + (tint_kind - 1u)];
    }

    inter = uvec4(face | (model_id << 8), tint, light, detail);
    uv = vec2(size) * corner_mask;
    rel_pos = world;
    face_shade = ((mflags & 2u) != 0u) ? FACE_SHADE[face] : 1.0;
}
