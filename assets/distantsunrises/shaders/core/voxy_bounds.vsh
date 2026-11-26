#version 450
// Footprint bounds: one box per chunk column the level renderer has meshed, over the level's whole
// build height, rasterized into a depth-only target (behaviour: Voxy's bounding-box pass, whose
// back faces keep the furthest depth per pixel; no code shared). No vertex attributes: the corner
// is the index value (0..7: x | y << 1 | z << 2), the column is gl_InstanceIndex.
//
// The depth written is the world's reversed-Z near / w, exactly what voxy_quads.vsh writes for an
// LOD vertex, from the same view_proj and the same camera split. The target is cleared to 1.0 and
// the pipeline's depth test is Less, so the smallest value (the furthest face of the boxes, which
// for a convex box seen from outside is its back face) is kept per pixel.

layout(std140) uniform BoundsParams {
    mat4 view_proj;       // camera-relative, same matrix as QuadParams.view_proj
    ivec4 base_section;   // xyz = camera block >> 5, as QuadParams.base_section
    vec4 camera_sub_pos;  // xyz = camera - base_section*32, w = camera near
    ivec4 range;          // x = level min block y, y = min y + height
};

layout(std430) readonly buffer Columns { ivec2 columns[]; };

void main() {
    uint corner = uint(gl_VertexIndex) & 7u;
    ivec2 column = columns[uint(gl_InstanceIndex)];
    ivec3 block = ivec3(
        column.x * 16 + int(corner & 1u) * 16,
        ((corner >> 1) & 1u) == 0u ? range.x : range.y,
        column.y * 16 + int((corner >> 2) & 1u) * 16
    );
    // Whole blocks are subtracted before the fractional camera, like the quads' section math.
    vec3 rel = vec3(block - (base_section.xyz << 5)) - camera_sub_pos.xyz;
    vec4 clip = view_proj * vec4(rel, 1.0);
    // z / w = near / w; z >= 0 and z <= w clip at the near plane exactly like the quads' depth.
    clip.z = camera_sub_pos.w;
    gl_Position = clip;
}
