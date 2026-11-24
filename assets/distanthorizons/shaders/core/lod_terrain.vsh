#version 330

// Distant Horizons LOD terrain vertex shader (vanilla Sunrise core-shader GLSL style).
// Targets Sunrise's upcoming `sunrise-glsl` / `#2005` mod render pipeline contract (`#4758`
// `WorldRenderView` + `Camera` matrices) with an independent reversed-Z projection over
// `[NearDh, FarDh]`.

in vec3 Position;
in vec4 Color;
in uint PackedMeta;
in vec4 SectionOffset; // xyz = section_origin - camera_pos, w = cell_size

layout(std140) uniform DhFrameUniforms {
    mat4 ViewProjMat;
    vec4 ClipAndLight;   // x = near_dh, y = far_dh, z = clip_dist, w = sun_brightness
    vec4 NoiseAndFade;   // x = noise_intensity, y = fade_band, z = brightness_mul, w = saturation_mul
    uvec4 Flags;         // x = shading_enabled, y = noise_enabled, z = fade_enabled, w = is_hdr
};

out vec4 vertexColor;
out vec3 relPos;
out vec3 blockPos;
flat out uint packedMeta;

float dirShade(uint normalIdx) {
    if (normalIdx == 0u) return 0.50;
    if (normalIdx == 1u) return 1.00;
    if (normalIdx == 2u || normalIdx == 3u) return 0.80;
    return 0.65;
}

void main() {
    vec3 rel = Position + SectionOffset.xyz;
    vec4 clip = ViewProjMat * vec4(rel, 1.0);

    float nearDh = ClipAndLight.x;
    float farDh = ClipAndLight.y;
    // Exact reversed-Z depth remap over [nearDh, farDh] without fighting vanilla's far plane.
    clip.z = (nearDh * farDh - nearDh * clip.w) / max(farDh - nearDh, 1.0);

    uint normalIdx = PackedMeta & 7u;
    float skyLight = float((PackedMeta >> 4u) & 15u) / 15.0;
    float blockLight = float((PackedMeta >> 8u) & 15u) / 15.0;
    float sun = ClipAndLight.w;
    float light = clamp(max(skyLight * sun, blockLight) * 0.90 + 0.10, 0.08, 1.0);

    float shade = 1.0;
    if (Flags.x != 0u) {
        shade = dirShade(normalIdx);
        float heightHint = clamp((Position.y - 62.0) / 180.0, -0.06, 0.08);
        shade = clamp(shade + heightHint, 0.45, 1.06);
    }

    vec3 rgb = Color.rgb * light * shade * NoiseAndFade.z;
    float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    rgb = clamp(mix(vec3(luma), rgb, NoiseAndFade.w), 0.0, 1.0);

    gl_Position = clip;
    vertexColor = vec4(rgb, Color.a);
    relPos = rel;
    blockPos = Position;
    packedMeta = PackedMeta;
}
