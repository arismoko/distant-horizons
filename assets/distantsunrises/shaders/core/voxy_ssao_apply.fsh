#version 450
// Applies the R32Float SSAO factor multiplicatively onto WorldColor via Translucent alpha
// (src = 0, alpha = 1 - ao => dst * ao, preserving linear HDR light and skipping vanilla/sky).
uniform sampler2D SsaoTex;

layout(location = 0) out vec4 out_color;

void main() {
    float ao = texelFetch(SsaoTex, ivec2(gl_FragCoord.xy), 0).r;
    if (ao >= 0.995) {
        discard;
    }
    out_color = vec4(0.0, 0.0, 0.0, clamp(1.0 - ao, 0.0, 0.5));
}
