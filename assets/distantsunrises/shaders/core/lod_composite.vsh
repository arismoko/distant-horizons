#version 330

// Fullscreen triangle vertex shader for sky-masked LOD horizon compositing.

out vec2 texCoord;

void main() {
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 p = positions[gl_VertexID];
    gl_Position = vec4(p, 0.0, 1.0);
    texCoord = p * 0.5 + 0.5;
}
