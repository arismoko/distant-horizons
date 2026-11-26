#version 450
// Applies the R32Float SSAO factor multiplicatively onto WorldColor via Translucent alpha
// (src = 0, alpha = 1 - ao => dst * ao, preserving linear HDR light and skipping vanilla/sky).
// `expected.xy` is the serial of the submission whose `voxy_ssao` output this draw samples;
// `published` is what that dispatch wrote. They differ when the dispatch never ran for this
// target and size (graph dropped, dispatch refused, node not run before this stage, target
// recreated), and then the texture is not read.
layout(std140) uniform SsaoApplyParams {
    uvec4 expected;
};
layout(std430) readonly buffer SsaoStamp { uvec2 published; };
uniform sampler2D SsaoTex;

layout(location = 0) out vec4 out_color;

void main() {
    if (any(notEqual(published, expected.xy))) {
        discard;
    }
    float ao = texelFetch(SsaoTex, ivec2(gl_FragCoord.xy), 0).r;
    if (ao >= 0.995) {
        discard;
    }
    out_color = vec4(0.0, 0.0, 0.0, clamp(1.0 - ao, 0.0, 0.5));
}
