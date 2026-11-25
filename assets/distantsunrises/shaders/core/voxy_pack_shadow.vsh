#version 450
// Shadow-pass LOD vertex entrypoint (`dh_shadow`, PackPassSlot::Shadow) for `PackProgramMode::ModVertexPackFragment` (#5700 part C).
//
// Unpacks Voxy's 8-byte `PackedQuad` (voxy `lod/quad_format.glsl`, `lod/quad_util.glsl`
// `setupQuad` / `getQuadCornerPos`) from the storage-buffer arena by vertex index and emits the
// pack fragment interface that Iris gives `dh_terrain.fsh` (`DHTerrainTransformer`: `dhMaterialId`,
// `iris_vBlockPos`, lightmap coord, normal, colour) plus the vanilla gbuffers varyings
// (`mc_Entity`, world / view / clip position). The translated pack fragment shader is linked
// directly against these outputs and writes the pack's `colortex*` draw buffers.
//
// `SceneUniform.modelView` / `projection` carry the pack's `shadowModelView` / `shadowProjection`
// (`ShaderPackShadowView`, camera-relative, current frame) and `depthRemap` is unused so
// `clip.z` is the orthographic shadow depth untouched.
//
// Mod bindings use the mod's own set (`set = 1` in source; the engine places it after the pack's
// sets, #5700 ruling 42). Draw contract: indexed-indirect, 6 indices per quad producing vertex ids
// `4 * quad + corner`, `first_vertex` = first quad of the section * 4, `first_instance` = index into
// `SectionPositions` (Voxy `positionBuffer[gl_BaseInstance]`).

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
layout(set = 1, binding = 2, std430) readonly buffer SectionPositions { uvec2 positionBuffer[]; };
// 16 uints per model: faceData[6], flagsA, colourTint, customId (= mc_Entity id), pad[7].
layout(set = 1, binding = 3, std430) readonly buffer ModelBuffer { uint modelData[]; };
layout(set = 1, binding = 4, std430) readonly buffer ModelColours { uint colourData[]; };
layout(set = 1, binding = 5) uniform sampler2D blockModelAtlas;

// Pack fragment interface.
layout(location = 0) out vec4 vColor;               // sRGB vertex colour (tint * directional shade)
layout(location = 1) out vec4 vColorLinear;         // same colour, linear
layout(location = 2) out vec2 vLmCoord;             // vanilla lightmap uv (x = block, y = sky)
layout(location = 3) out vec3 vNormal;              // view-space normal
layout(location = 4) flat out int dhMaterialId;     // DH_BLOCK_* (model flagsA bits 8..11)
layout(location = 5) flat out ivec4 mc_Entity;      // (block.properties id, 0, 0, 1)
layout(location = 6) out vec3 vWorldPos;            // camera-relative world position
layout(location = 7) out vec3 vViewPos;
layout(location = 8) out vec4 vClipPos;
layout(location = 9) out vec2 vTexCoord;            // face-local uv in blocks

const uint MODEL_STRIDE = 16u;
const float TILE_MIP = 4.0; // 16 px face tile -> 1 px; the pack fragment has no atlas sampler

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
    uvec2 sPos = positionBuffer[gl_InstanceIndex];
    uint lodLevel = sPos.x >> 28;
    int sy = (int(sPos.x) << 4) >> 24;
    int sx = (int(sPos.y) << 4) >> 8;
    int sz = int((sPos.x & ((1u << 20) - 1u)) << 4) | int(sPos.y >> 28);
    sz = (sz << 8) >> 8;
    float lodScale = float(1u << lodLevel);
    ivec3 baseSection = (ivec3(sx, sy, sz) << int(lodLevel)) - (baseSectionPos >> 5);

    uint m = modelId * MODEL_STRIDE;
    uint faceData = modelData[m + face];
    uint flagsA = modelData[m + 6u];
    uint tintBase = modelData[m + 7u];
    uint customId = modelData[m + 8u];

    // Face rectangle inside the block (voxy `getFaceSize`).
    vec4 fs = vec4(faceData & 0xFu, (faceData >> 4) & 0xFu, (faceData >> 8) & 0xFu, (faceData >> 12) & 0xFu) / 16.0
            + vec4(0.0, 1.0 / 16.0, 0.0, 1.0 / 16.0);
    fs.xz -= vec2(0.00005);
    fs.yw -= fs.xz;
    uint indentEnc = (faceData >> 16) & 63u;
    indentEnc += uint(indentEnc == 63u);
    float depthOffset = float(indentEnc) / 64.0;

    uint axis = face >> 1;
    vec3 quadStart = pos + swizzleAxis(axis, vec3(fs.xz, mix(depthOffset, 1.0 - depthOffset, float(face & 1u))));
    vec3 basePoint = quadStart * lodScale + vec3(baseSection << 5);
    vec2 sizeAddin = fs.yw + vec2(size) - 1.0;

    vec2 mask = vec2(float((corner >> 1) & 1u), float(corner & 1u)) * lodScale;
    vec3 point = basePoint + swizzleAxis(axis, vec3(sizeAddin * mask, 0.0));
    vec3 worldPos = point - cameraSubPos;
    vec4 viewPos = modelView * vec4(worldPos, 1.0);
    vec4 clip = projection * viewPos;
    // Orthographic shadow projection: keep clip.z from the matrix (depthRemap is not applied).

    // Colour: atlas tile average * conditional biome tint * directional shade.
    vec2 modelUV = vec2(float(modelId & 0xFFu), float((modelId >> 8) & 0xFFu)) * (1.0 / 256.0);
    vec2 tile = vec2(1.0 / (3.0 * 256.0), 1.0 / (2.0 * 256.0));
    vec2 faceUV = modelUV + vec2(float(face >> 1), float(face & 1u)) * tile;
    vec4 base = textureLod(blockModelAtlas, faceUV + tile * 0.5, TILE_MIP);
    base.a = 1.0;

    uint tintState = (faceData >> 24) & 3u;
    uint tintColour = tintBase;
    if ((flagsA & 2u) != 0u) tintColour = colourData[tintColour + biome];
    bool doTint = tintColour != 0xFFFFFFFFu && (tintState == 2u
        || (tintState == 1u && abs(base.r - base.g) < 0.02 && abs(base.g - base.b) < 0.02));
    if (doTint) {
        base.rgb *= vec3((uvec3(tintColour) >> uvec3(16, 8, 0)) & uvec3(255u)) / 255.0;
    }
    float shade = ((flagsA & 8u) != 0u) ? directionalShade(face) : 1.0;
    base.rgb *= shade;

    vColor = base;
    vColorLinear = vec4(srgbToLinear(base.rgb), base.a);
    vLmCoord = clamp(vec2(float((light >> 4) & 0xFu), float(light & 0xFu)) / 15.0 * (15.0 / 16.0) + 0.5 / 16.0,
                     vec2(8.0 / 256.0), vec2(248.0 / 256.0));
    vNormal = mat3(modelView) * FACE_NORMALS[face];
    dhMaterialId = int((flagsA >> 8) & 0xFu);
    mc_Entity = ivec4(int(customId), 0, 0, 1);
    vWorldPos = worldPos;
    vViewPos = viewPos.xyz;
    vClipPos = clip;
    vTexCoord = faceUV;
    gl_Position = clip;
}
