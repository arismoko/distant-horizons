#version 450
// Fallback fragment stage of voxy_pack_water.vsh, paired by name. With a shader pack active its
// `dh_water` program replaces this stage; it only has to link to the vertex stage's outputs so the
// pipeline validates, and draws the vertex colour alone otherwise.
layout(location = 0) in vec4 vColor;
layout(location = 1) in vec4 vColorLinear;
layout(location = 2) in vec2 vLmCoord;
layout(location = 3) in vec3 vNormal;
layout(location = 4) in flat int dhMaterialId;
layout(location = 5) in flat ivec4 mc_Entity;
layout(location = 6) in vec3 vWorldPos;
layout(location = 7) in vec3 vViewPos;
layout(location = 8) in vec4 vClipPos;
layout(location = 9) in vec2 vTexCoord;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vColorLinear;
}
