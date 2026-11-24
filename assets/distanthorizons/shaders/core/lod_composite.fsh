#version 330

// Fullscreen sky-masked composite + spiral SSAO + far/height/environmental fog for Distant Horizons.
// Preserves Sunrise's linear HDR light pipeline (docs/hdr-passes.md).

layout(std140) uniform DhCompositeUniforms {
    mat4 InvViewProj;
    vec4 FogColor;
    vec4 FogParams;    // x = near_dh, y = far_dh, z = fog_start, w = fog_end
    vec4 HeightAndEnv; // x = camera_y, y = height_fog_base_y, z = env_fog_start, w = env_fog_end
    vec4 SsaoParams;   // x = ssao_radius, y = ssao_strength, z = ssao_bias, w = ssao_fade_dist
    uvec4 Flags0;      // x = fog_enabled, y = is_hdr, z = fog_falloff, w = fog_spherical
    uvec4 Flags1;      // x = height_fog_enabled, y = height_fog_mix_mode, z = ssao_samples, w = fog_type
};

uniform sampler2D DhColorSampler;
uniform sampler2D DhDepthSampler;
uniform sampler2D WorldDepthSampler;

in vec2 texCoord;
out vec4 fragColor;

vec3 unprojectDh(vec2 uv, float depth) {
    vec4 ndc = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    vec4 wpos = InvViewProj * ndc;
    float s = wpos.w < 0.0 ? -1.0 : 1.0;
    return wpos.xyz / max(abs(wpos.w), 1e-6) * s;
}

float interleavedGradientNoise(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

float computeSsao(ivec2 px, ivec2 dims, vec3 relPos, float viewDist) {
    uint sampleCount = Flags1.z;
    if (sampleCount == 0u || viewDist >= SsaoParams.w) {
        return 0.0;
    }

    vec2 fDims = max(vec2(dims), vec2(1.0));
    ivec2 pxR = clamp(px + ivec2(1, 0), ivec2(0), dims - ivec2(1));
    ivec2 pxL = clamp(px - ivec2(1, 0), ivec2(0), dims - ivec2(1));
    ivec2 pxD = clamp(px + ivec2(0, 1), ivec2(0), dims - ivec2(1));
    ivec2 pxU = clamp(px - ivec2(0, 1), ivec2(0), dims - ivec2(1));

    float dr = texelFetch(DhDepthSampler, pxR, 0).r;
    float dl = texelFetch(DhDepthSampler, pxL, 0).r;
    float dd = texelFetch(DhDepthSampler, pxD, 0).r;
    float du = texelFetch(DhDepthSampler, pxU, 0).r;

    vec3 dxR = unprojectDh((vec2(pxR) + 0.5) / fDims, max(dr, 1e-6)) - relPos;
    vec3 dxL = relPos - unprojectDh((vec2(pxL) + 0.5) / fDims, max(dl, 1e-6));
    vec3 dpdx = (dr > 0.0 && (dl <= 0.0 || dot(dxR, dxR) < dot(dxL, dxL))) ? dxR : dxL;

    vec3 dyD = unprojectDh((vec2(pxD) + 0.5) / fDims, max(dd, 1e-6)) - relPos;
    vec3 dyU = relPos - unprojectDh((vec2(pxU) + 0.5) / fDims, max(du, 1e-6));
    vec3 dpdy = (dd > 0.0 && (du <= 0.0 || dot(dyD, dyD) < dot(dyU, dyU))) ? dyD : dyU;

    vec3 n = cross(dpdy, dpdx);
    float nLen = length(n);
    if (nLen < 1e-5) {
        return 0.0;
    }
    n /= nLen;
    if (dot(n, -relPos) < 0.0) {
        n = -n;
    }

    float radiusWorld = SsaoParams.x;
    float strength = SsaoParams.y;
    float bias = SsaoParams.z;
    float pxRadius = clamp(radiusWorld * fDims.y / max(viewDist * 1.4, 1.0), 2.5, 42.0);
    float rStep = pxRadius / float(sampleCount);

    float angle = interleavedGradientNoise(vec2(px)) * 6.2831853;
    float r = rStep;
    float ao = 0.0;
    float hits = 0.0;

    for (uint i = 0u; i < sampleCount; i++) {
        vec2 off = vec2(cos(angle), sin(angle)) * r;
        r += rStep;
        angle += 2.39996323;

        ivec2 sp = clamp(px + ivec2(round(off)), ivec2(0), dims - ivec2(1));
        float sd = texelFetch(DhDepthSampler, sp, 0).r;
        if (sd <= 0.0) {
            continue;
        }
        vec3 sRel = unprojectDh((vec2(sp) + 0.5) / fDims, sd);
        vec3 diff = sRel - relPos;
        float sDist = length(diff);
        if (sDist > 1e-3 && sDist < radiusWorld * 2.5) {
            float ndotl = max(dot(n, diff / sDist) - bias, 0.0);
            float falloff = 1.0 - clamp(sDist / (radiusWorld * 2.5), 0.0, 1.0);
            ao += ndotl * falloff;
            hits += 1.0;
        }
    }

    float rawAo = ao / max(hits, 1.0);
    float distFade = 1.0 - clamp(viewDist / max(SsaoParams.w, 1.0), 0.0, 1.0);
    return clamp(smoothstep(0.0, 0.55, rawAo) * strength * distFade, 0.0, 0.65);
}

float fogCurve(float tRaw, uint falloff) {
    float t = clamp(tRaw, 0.0, 1.0);
    if (falloff == 0u) return t;
    if (falloff == 1u) return clamp(1.0 - exp(-2.5 * t), 0.0, 1.0);
    return clamp(1.0 - exp(-2.8 * t * t), 0.0, 1.0);
}

float mixHeightFog(float farF, float heightF, uint mode) {
    if (mode == 0u) return farF;
    if (mode == 1u) return max(farF, heightF);
    if (mode == 2u) return clamp(farF + heightF, 0.0, 1.0);
    if (mode == 3u) return farF * heightF;
    return 1.0 - (1.0 - farF) * (1.0 - heightF);
}

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
    float nearDh = FogParams.x;
    float farDh = FogParams.y;
    float viewDist = (nearDh * farDh) / max(dhDepth * (farDh - nearDh) + nearDh, 1e-5);
    vec2 uv = (vec2(px) + vec2(0.5)) / max(vec2(dhDims), vec2(1.0));
    vec3 relPos = unprojectDh(uv, dhDepth);

    float ao = computeSsao(px, dhDims, relPos, viewDist);
    col *= (1.0 - ao);

    uint fogType = Flags1.w;
    float fogFactor = 0.0;
    if (fogType != 0u) {
        float envStart = HeightAndEnv.z;
        float envEnd = max(HeightAndEnv.w, envStart + 1.0);
        fogFactor = clamp((length(relPos) - envStart) / (envEnd - envStart), 0.0, 1.0);
    } else if (Flags0.x != 0u) {
        float fogStart = FogParams.z;
        float fogEnd = max(FogParams.w, fogStart + 1.0);
        float activeDist = Flags0.w != 0u ? length(relPos) : length(relPos.xz);
        float farFog = fogCurve((activeDist - fogStart) / (fogEnd - fogStart), Flags0.z);

        float heightFog = 0.0;
        if (Flags1.x != 0u) {
            float worldY = relPos.y + HeightAndEnv.x;
            float below = max(HeightAndEnv.y - worldY, 0.0);
            float hT = clamp(below / 96.0, 0.0, 1.0)
                * clamp((activeDist - fogStart * 0.45) / max(fogEnd * 0.55, 1.0), 0.0, 1.0);
            heightFog = fogCurve(hT, Flags0.z) * 0.65;
        }
        fogFactor = mixHeightFog(farFog, heightFog, Flags1.y);
    }

    if (Flags0.y != 0u) {
        vec3 linCol = pow(max(col, vec3(0.0)), vec3(2.2));
        vec3 linFog = pow(max(FogColor.rgb, vec3(0.0)), vec3(2.2));
        col = mix(linCol, linFog, fogFactor);
    } else {
        col = mix(col, FogColor.rgb, fogFactor);
    }

    fragColor = vec4(max(col, vec3(0.0)), 1.0);
}
