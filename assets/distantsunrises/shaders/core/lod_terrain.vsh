#version 330

// Distant Sunrises LOD terrain vertex shader (vanilla Sunrise core-shader GLSL style).
// Targets Sunrise's `sunrise-glsl` / `#2005` mod render pipeline contract (`#4758`
// `WorldRenderView` + `Camera` matrices) with an independent reversed-Z projection over
// `[NearDh, FarDh]`, debug visualization modes, and generic box-group support.

in vec3 Position;
in vec4 Color;
in uint PackedMeta;
in vec4 SectionOffset; // xyz = origin - camera_pos, w = cell_size

layout(std140) uniform DhFrameUniforms {
    mat4 ViewProjMat;
    vec4 ClipAndLight;   // x = near_dh, y = far_dh, z = clip_dist, w = sun_brightness
    vec4 NoiseAndFade;   // x = noise_intensity, y = fade_band, z = brightness_mul, w = saturation_mul
    uvec4 Flags;         // x = shading_enabled, y = noise_enabled, z = fade_enabled, w = debug_mode
};

out vec4 vertexColor;
out vec3 relPos;
out vec3 blockPos;
flat out uint packedMeta;
flat out int dhMaterialId;
flat out float cellSize;

float dirShade(uint normalIdx) {
    if (normalIdx == 0u) return 0.50;
    if (normalIdx == 1u) return 1.00;
    if (normalIdx == 2u || normalIdx == 3u) return 0.80;
    return 0.65;
}

vec3 detailDebugColor(uint detail) {
    if (detail == 0u) return vec3(0.20, 0.90, 0.30);
    if (detail == 1u) return vec3(0.15, 0.85, 0.85);
    if (detail == 2u) return vec3(0.25, 0.55, 0.98);
    if (detail == 3u) return vec3(0.55, 0.30, 0.95);
    if (detail == 4u) return vec3(0.85, 0.25, 0.85);
    if (detail == 5u) return vec3(0.95, 0.25, 0.25);
    return vec3(0.98, 0.60, 0.15);
}

vec3 sourceDebugColor(uint src) {
    if (src == 1u) return vec3(0.95, 0.75, 0.20);
    if (src == 2u) return vec3(0.20, 0.85, 0.35);
    if (src == 3u) return vec3(0.30, 0.70, 0.98);
    return vec3(0.60, 0.60, 0.60);
}

void main() {
    vec3 rel = Position + SectionOffset.xyz;
    vec4 clip = ViewProjMat * vec4(rel, 1.0);

    float nearDh = ClipAndLight.x;
    float farDh = ClipAndLight.y;
    clip.z = (nearDh * farDh - nearDh * clip.w) / max(farDh - nearDh, 1.0);

    uint normalIdx = PackedMeta & 7u;
    float skyLight = float((PackedMeta >> 4u) & 15u) / 15.0;
    float blockLight = float((PackedMeta >> 8u) & 15u) / 15.0;
    uint detail = (PackedMeta >> 17u) & 7u;
    uint source = (PackedMeta >> 20u) & 3u;
    uint isGeneric = (PackedMeta >> 22u) & 1u;
    uint unshaded = (PackedMeta >> 23u) & 1u;
    uint debugMode = Flags.w;

    vec3 rgb;
    if (debugMode == 1u || debugMode == 4u) {
        rgb = detailDebugColor(detail) * dirShade(normalIdx);
    } else if (debugMode == 2u) {
        rgb = sourceDebugColor(source) * dirShade(normalIdx);
    } else if (debugMode == 3u) {
        rgb = vec3(blockLight, 0.5 * (skyLight + blockLight), skyLight);
    } else if (unshaded != 0u) {
        rgb = Color.rgb;
    } else {
        float sun = ClipAndLight.w;
        float light = clamp(max(skyLight * sun, blockLight) * 0.90 + 0.10, 0.08, 1.0);

        float shade = 1.0;
        if (Flags.x != 0u && isGeneric == 0u) {
            shade = dirShade(normalIdx);
            float heightHint = clamp((Position.y - 62.0) / 180.0, -0.06, 0.08);
            shade = clamp(shade + heightHint, 0.45, 1.06);
        }

        rgb = Color.rgb * light * shade * NoiseAndFade.z;
        float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
        rgb = clamp(mix(vec3(luma), rgb, NoiseAndFade.w), 0.0, 1.0);
    }

    gl_Position = clip;
    vertexColor = vec4(rgb, Color.a);
    relPos = rel;
    blockPos = Position;
    packedMeta = PackedMeta;
    dhMaterialId = int((PackedMeta >> 13u) & 15u);
    cellSize = max(SectionOffset.w, 1.0);
}
