#version 450
// Translucent LOD terrain / fluid vertex entrypoint (`dh_water`, falls back to `dh_terrain`) for `PackProgramMode::ModVertexPackFragment` (#5700 part C).
//
// Unpacks Voxy's 8-byte `PackedQuad` (voxy `lod/quad_format.glsl`, `lod/quad_util.glsl`
// `setupQuad` / `getQuadCornerPos`) from the storage-buffer arena by vertex index and emits the
// pack fragment interface that Iris gives `dh_terrain.fsh` (`DHTerrainTransformer`: `dhMaterialId`,
// `iris_vBlockPos`, lightmap coord, normal, colour) plus the vanilla gbuffers varyings
// (`mc_Entity`, world / view / clip position). The translated pack fragment shader is linked
// directly against these outputs and writes the pack's `colortex*` draw buffers.
//
// Mod bindings use the mod's own set (`set = 1` in source; the engine places it after the pack's
// sets, #5700 ruling 42). Draw contract: indexed-indirect, 6 indices per quad producing vertex ids
// `4 * quad + corner`, `first_vertex` = first quad of the section * 4, `first_instance` = index into
// `SectionBuffer` (`SectionMeta`, the same 32-byte rows as `voxy_quads.vsh`).

layout(set = 1, binding = 0, std140) uniform SceneUniform {
    mat4 modelView;      // camera-relative world -> view
    mat4 projection;     // view -> clip (w = -view.z)
    ivec3 baseSectionPos; // block origin the section offsets are relative to
    uint frameId;
    vec3 cameraSubPos;   // camera position - baseSectionPos
    float _pad0;
    vec2 depthRemap;     // clip.z = x + y * clip.w, maps LOD depth into the pack's main depth
    vec2 _pad1;
};

// One `PackedQuad` per element: x = bits 0..31, y = bits 32..63 (little endian).
layout(set = 1, binding = 1, std430) readonly buffer QuadArena { uvec2 quadData[]; };
// Voxy packed section position (`pos_util.glsl`): x = lod[31:28] | y[27:20] | z_hi[19:0], y = z_lo[31:28] | x[27:0].
struct SectionMeta {
    uvec4 a; // pos0, pos1, aabb, quad_start
    uvec4 b;
};
layout(set = 1, binding = 2, std430) readonly buffer SectionBuffer { SectionMeta sections[]; };
// 16 uints per model (same as voxy_quads): tile_origin[6] (x | y << 16 in atlas px), avg_rgba[6],
// mflags@12 (bit0 translucent, bit1 face shading, bit2 cutout, bits4..8 tint kind, bits8..12
// dhMaterialId), const_tint@13, tile_size@14 (w | h << 16, 0 = flat colour), mc_Entity@15.
layout(set = 1, binding = 3, std430) readonly buffer ModelBuffer { uint modelData[]; };
// Biome tint LUT: grass, foliage, water per biome id, 0x00BBGGRR.
layout(set = 1, binding = 4, std430) readonly buffer BiomeTintBuffer { uint biomeTints[]; };
layout(set = 1, binding = 5) uniform sampler2D BlockAtlas;

// Pack fragment interface.
layout(location = 0) out vec4 vColor;               // sRGB vertex colour (tint * directional shade)
layout(location = 1) out vec4 vColorLinear;         // same colour, linear
layout(location = 2) out vec2 vLmCoord;             // vanilla lightmap uv (x = block, y = sky)
layout(location = 3) out vec3 vNormal;              // view-space normal
layout(location = 4) flat out int dhMaterialId;     // DH_BLOCK_* (model mflags bits 8..11)
layout(location = 5) flat out ivec4 mc_Entity;      // (block.properties id, 0, 0, 1)
layout(location = 6) out vec3 vWorldPos;            // camera-relative world position
layout(location = 7) out vec3 vViewPos;
layout(location = 8) out vec4 vClipPos;
layout(location = 9) out vec2 vTexCoord;            // face-local uv in blocks

const uint MODEL_STRIDE = 16u;

const vec3 FACE_NORMALS[6] = vec3[](
    vec3(0, -1, 0), vec3(0, 1, 0), vec3(0, 0, -1), vec3(0, 0, 1), vec3(-1, 0, 0), vec3(1, 0, 0));

vec3 swizzleAxis(uint axis, vec3 d) {
    if (axis == 1u) return d;
    if (axis == 0u) return d.xzy;
    return d.zxy;
}

vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

float directionalShade(uint face) {
    if (face == 0u) return 0.5;
    if (face == 1u) return 1.0;
    if ((face >> 1) == 1u) return 0.8;
    return 0.6;
}

void main() {
    uint quadIndex = uint(gl_VertexIndex) >> 2;
    uint corner = uint(gl_VertexIndex) & 3u;
    uvec2 q = quadData[quadIndex];

    uint face = q.x & 7u;
    ivec2 size = ivec2((q.x >> 3) & 15u, (q.x >> 7) & 15u) + ivec2(1);
    vec3 pos = vec3((q.x >> 21) & 31u, (q.x >> 16) & 31u, (q.x >> 11) & 31u);
    uint modelId = ((q.x >> 26) & 63u) | ((q.y & 0x3FFFu) << 6);
    uint biome = (q.y >> 14) & 0x1FFu;
    uint light = (q.y >> 23) & 0xFFu;

    // Section placement (voxy `pos_util.glsl`).
    uvec2 sPos = sections[uint(gl_InstanceIndex)].a.xy;
    uint lodLevel = sPos.x >> 28;
    int sy = (int(sPos.x) << 4) >> 24;
    int sx = (int(sPos.y) << 4) >> 8;
    int sz = int((sPos.x & ((1u << 20) - 1u)) << 4) | int(sPos.y >> 28);
    sz = (sz << 8) >> 8;
    float lodScale = float(1u << lodLevel);
    ivec3 baseSection = (ivec3(sx, sy, sz) << int(lodLevel)) - (baseSectionPos >> 5);

    uint m = modelId * MODEL_STRIDE;
    uint mflags = modelData[m + 12u];
    uint mcEntity = modelData[m + 15u];

    float depthOffset = 0.0;

    uint axis = face >> 1;
    vec3 quadStart = pos + swizzleAxis(axis, vec3(0.0, 0.0, mix(depthOffset, 1.0 - depthOffset, float(face & 1u))));
    vec3 basePoint = quadStart * lodScale + vec3(baseSection << 5);
    vec2 sizeAddin = vec2(size);

    vec2 mask = vec2(float((corner >> 1) & 1u), float(corner & 1u)) * lodScale;
    vec3 point = basePoint + swizzleAxis(axis, vec3(sizeAddin * mask, 0.0));
    vec3 worldPos = point - cameraSubPos;
    vec4 viewPos = modelView * vec4(worldPos, 1.0);
    vec4 clip = projection * viewPos;
    clip.z = depthRemap.x + depthRemap.y * clip.w;

    // Colour: baked tile (or the face average) * biome / constant tint * directional shade.
    uint tileOrigin = modelData[m + face];
    uint tileSize = modelData[m + 14u];
    vec2 tilePx = vec2(float(tileSize & 0xFFFFu), float(tileSize >> 16));
    vec2 faceUV = vec2(0.0);
    vec4 base;
    if (tilePx.x > 0.0) {
        vec2 originPx = vec2(float(tileOrigin & 0xFFFFu), float(tileOrigin >> 16));
        faceUV = (originPx + tilePx * 0.5) / vec2(textureSize(BlockAtlas, 0));
        base = textureLod(BlockAtlas, faceUV, 0.0);
    } else {
        uint avg = modelData[m + 6u + face];
        base = vec4(float(avg & 0xFFu), float((avg >> 8) & 0xFFu), float((avg >> 16) & 0xFFu), float(avg >> 24)) / 255.0;
    }
    base.a = max(base.a, 1.0 / 255.0); // translucent pass keeps the atlas alpha (dh_water blend)
    uint tintKind = (mflags >> 4) & 15u;
    uint tintColour = 0xFFFFFFFFu;
    if (tintKind == 4u) tintColour = modelData[m + 13u];
    else if (tintKind >= 1u && tintKind <= 3u) tintColour = biomeTints[biome * 3u + (tintKind - 1u)];
    if (tintColour != 0xFFFFFFFFu) {
        base.rgb *= vec3(float(tintColour & 0xFFu), float((tintColour >> 8) & 0xFFu), float((tintColour >> 16) & 0xFFu)) / 255.0;
    }
    float shade = ((mflags & 2u) != 0u) ? directionalShade(face) : 1.0;
    base.rgb *= shade;

    vColor = base;
    vColorLinear = vec4(srgbToLinear(base.rgb), base.a);
    vLmCoord = clamp(vec2(float((light >> 4) & 0xFu), float(light & 0xFu)) / 15.0 * (15.0 / 16.0) + 0.5 / 16.0,
                     vec2(8.0 / 256.0), vec2(248.0 / 256.0));
    vNormal = mat3(modelView) * FACE_NORMALS[face];
    dhMaterialId = int((mflags >> 8) & 0xFu);
    mc_Entity = ivec4(int(mcEntity), 0, 0, 1);
    vWorldPos = worldPos;
    vViewPos = viewPos.xyz;
    vClipPos = clip;
    vTexCoord = faceUV;
    gl_Position = clip;
}
