#version 450
// Fullscreen triangle for multiplicative HDR-safe SSAO composite onto opaque LODs.
void main() {
    uint idx = uint(gl_VertexIndex);
    vec2 pos = vec2(float((idx & 1u) << 2) - 1.0, float((idx & 2u) << 1) - 1.0);
    gl_Position = vec4(pos, 0.0, 1.0);
}
