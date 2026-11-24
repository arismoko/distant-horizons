#version 330

// Distant Horizons LOD terrain fragment shader (vanilla Sunrise core-shader GLSL style).
// Applies near-edge dither fade, debug cell wireframes, and procedural surface noise.

layout(std140) uniform DhFrameUniforms {
    mat4 ViewProjMat;
    vec4 ClipAndLight;   // x = near_dh, y = far_dh, z = clip_dist, w = sun_brightness
    vec4 NoiseAndFade;   // x = noise_intensity, y = fade_band, z = brightness_mul, w = saturation_mul
    uvec4 Flags;         // x = shading_enabled, y = noise_enabled, z = fade_enabled, w = debug_mode
};

in vec4 vertexColor;
in vec3 relPos;
in vec3 blockPos;
flat in uint packedMeta;
flat in float cellSize;

out vec4 fragColor;

float hash3(vec3 p) {
    float h = dot(p, vec3(12.9898, 78.233, 37.719));
    return fract(sin(h) * 43758.5453);
}

float bayer4x4(vec2 p) {
    int x = int(p.x) & 3;
    int y = int(p.y) & 3;
    int idx = y * 4 + x;
    const float m[16] = float[16](
        0.0,  8.0,  2.0, 10.0,
        12.0, 4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
        15.0, 7.0, 13.0,  5.0
    );
    return (m[idx] + 0.5) / 16.0;
}

void main() {
    uint skipClip = (packedMeta >> 24u) & 1u;
    float dist = length(relPos.xz);
    float clipDist = ClipAndLight.z;
    float fadeBand = NoiseAndFade.y;
    if (skipClip == 0u && clipDist > 0.0) {
        if (Flags.z != 0u && fadeBand > 0.5) {
            float t = smoothstep(clipDist, clipDist + fadeBand, dist);
            if (t <= bayer4x4(gl_FragCoord.xy)) {
                discard;
            }
        } else if (dist < clipDist) {
            discard;
        }
    }

    vec3 rgb = vertexColor.rgb;
    uint debugMode = Flags.w;
    uint isFluid = (packedMeta >> 12u) & 1u;
    uint isGeneric = (packedMeta >> 22u) & 1u;

    if (debugMode == 4u) {
        vec2 f = fract(blockPos.xz / cellSize);
        float edge = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
        if (edge < 0.05) {
            rgb = vec3(0.05, 0.05, 0.05);
        }
    } else if (debugMode == 0u && Flags.y != 0u && isFluid == 0u && isGeneric == 0u) {
        float intensity = NoiseAndFade.x;
        float lum = (rgb.r + rgb.g + rgb.b) / 3.0;
        float amp = (1.0 - pow(lum * 2.0 - 1.0, 2.0)) * intensity;
        vec3 qpos = floor(blockPos);
        float rnd = hash3(qpos) * 2.0 * amp - amp;
        float dropoff = clamp(dist / 512.0, 0.0, 1.0);
        vec3 noisy = clamp(rgb + (vec3(1.0) - rgb) * rnd, 0.0, 1.0);
        rgb = mix(noisy, rgb, dropoff);
    }

    fragColor = vec4(rgb, vertexColor.a);
}
