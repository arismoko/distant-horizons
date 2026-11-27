#version 450
// Applies the R32Float SSAO factor multiplicatively onto WorldColor via Translucent alpha
// (src = 0, alpha = 1 - ao => dst * ao, preserving linear HDR light and skipping vanilla/sky).
uniform sampler2D SsaoTex;

layout(location = 0) out vec4 out_color;

void main() {
    ivec2 half_size = textureSize(SsaoTex, 0);
    ivec2 fp = ivec2(gl_FragCoord.xy);
    // The owning cell's integer part carries one validity bit per full-res pixel.
    ivec2 cell = clamp(fp >> 1, ivec2(0), half_size - ivec2(1));
    uint sub = uint((fp.x & 1) | ((fp.y & 1) << 1));
    float raw_c = texelFetch(SsaoTex, cell, 0).r;
    uint mask_c = uint(floor(raw_c));
    if ((mask_c & (1u << sub)) == 0u) {
        discard;
    }

    // Centered 2x2 tent (9/16 owning cell, 3/16 axial neighbors, 1/16 diagonal),
    // reusing raw_c so surviving pixels perform 4 fetches total and w_sum >= 9/16 always.
    int nx = (fp.x & 1) == 0 ? max(cell.x - 1, 0) : min(cell.x + 1, half_size.x - 1);
    int ny = (fp.y & 1) == 0 ? max(cell.y - 1, 0) : min(cell.y + 1, half_size.y - 1);
    float rx = texelFetch(SsaoTex, ivec2(nx, cell.y), 0).r;
    float ry = texelFetch(SsaoTex, ivec2(cell.x, ny), 0).r;
    float rxy = texelFetch(SsaoTex, ivec2(nx, ny), 0).r;

    float ao_sum = (fract(raw_c) * 2.0) * 0.5625;
    float w_sum = 0.5625;
    if (floor(rx) >= 1.0) { ao_sum += (fract(rx) * 2.0) * 0.1875; w_sum += 0.1875; }
    if (floor(ry) >= 1.0) { ao_sum += (fract(ry) * 2.0) * 0.1875; w_sum += 0.1875; }
    if (floor(rxy) >= 1.0) { ao_sum += (fract(rxy) * 2.0) * 0.0625; w_sum += 0.0625; }
    float ao = ao_sum / w_sum;
    if (ao >= 0.995) {
        discard;
    }
    out_color = vec4(0.0, 0.0, 0.0, clamp(1.0 - ao, 0.0, 0.5));
}
