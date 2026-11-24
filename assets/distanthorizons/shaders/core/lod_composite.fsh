#version 330

// Fullscreen sky-masked atmospheric horizon fog composite for Distant Horizons LOD terrain.
// Preserves Sunrise's linear HDR light pipeline (docs/hdr-passes.md).

layout(std140) uniform DhCompositeUniforms {
    vec4 FogColor;
    vec4 Params;  // x = near_dh, y = far_dh, z = fog_start, w = fog_end
    uvec4 Flags;  // x = fog_enabled, y = is_hdr, z = 0, w = 0
};

uniform sampler2D DhColorSampler;
uniform sampler2D DhDepthSampler;
uniform sampler2D WorldDepthSampler;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    ivec2 dhDims = textureSize(DhColorSampler, 0);
    ivec2 mcDims = textureSize(WorldDepthSampler, 0);
    ivec2 px = clamp(ivec2(gl_FragCoord.xy), ivec2(0), dhDims - ivec2(1));
    ivec2 mcPx = clamp(
        ivec2((vec2(px) + vec2(0.5)) * vec2(mcDims) / max(vec2(dhDims), vec2(1.0))),
        ivec2(0),
        mcDims - ivec2(1)
    );

    float mcDepth = texelFetch(WorldDepthSampler, mcPx, 0).r;
    if (mcDepth > 0.0) {
        discard;
    }

    float dhDepth = texelFetch(DhDepthSampler, px, 0).r;
    if (dhDepth <= 0.0) {
        discard;
    }

    vec3 col = texelFetch(DhColorSampler, px, 0).rgb;

    if (Flags.x != 0u) {
        float nearDh = Params.x;
        float farDh = Params.y;
        float fogStart = Params.z;
        float fogEnd = max(Params.w, fogStart + 1.0);
        float viewDist = (nearDh * farDh) / max(dhDepth * (farDh - nearDh) + nearDh, 1e-5);
        float t = clamp((viewDist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
        float fogFactor = clamp(1.0 - exp(-2.8 * t * t), 0.0, 1.0);
        if (Flags.y != 0u) {
            vec3 linCol = pow(max(col, vec3(0.0)), vec3(2.2));
            vec3 linFog = pow(max(FogColor.rgb, vec3(0.0)), vec3(2.2));
            col = mix(linCol, linFog, fogFactor);
        } else {
            col = mix(col, FogColor.rgb, fogFactor);
        }
    } else if (Flags.y != 0u) {
        col = pow(max(col, vec3(0.0)), vec3(2.2));
    }

    fragColor = vec4(max(col, vec3(0.0)), 1.0);
}
