#version 450
// Footprint bounds: one box per chunk column the level renderer has meshed, over the level's whole
// build height, rasterized into a depth-only target (behaviour: Voxy's bounding-box pass, whose
// back faces keep the furthest depth per pixel; no code shared). No vertex attributes: the corner
// is the index value (0..7: x | y << 1 | z << 2), the column is gl_InstanceIndex.
//
// The LOD quads store the reversed-Z depth near / w (voxy_quads.vsh). Here the stored depth is
// 1 - near / w, so the "nearer wins" test the pipeline keeps (greater wins) keeps the largest w:
// the furthest face of the boxes, which for a convex box seen from outside is its back face. A
// target cleared to 0 therefore holds no bound (near / w = 1 never cuts a fragment).

layout(std140) uniform BoundsParams {
    mat4 view_proj;       // camera-relative, same matrix as QuadParams.view_proj
    ivec4 camera_block;   // xyz = floor(camera)
    vec4 camera_sub_pos;  // xyz = camera - camera_block, w = camera near
    vec4 range;           // x = level min block y, y = min y + height, z = corner bias in blocks
};

layout(std430) readonly buffer Columns { ivec2 columns[]; };

void main() {
    uint corner = uint(gl_VertexIndex) & 7u;
    ivec2 column = columns[uint(gl_InstanceIndex)];
    float x = float(column.x * 16 + int(corner & 1u) * 16 - camera_block.x) - camera_sub_pos.x;
    float z = float(column.y * 16 + int((corner >> 2) & 1u) * 16 - camera_block.z) - camera_sub_pos.z;
    float y = mix(range.x, range.y, float((corner >> 1) & 1u))
        - (float(camera_block.y) + camera_sub_pos.y);
    vec3 rel = vec3(x, y, z);
    // Pull the corner toward the camera so an LOD fragment on the footprint's far wall survives.
    float len = max(length(rel), 1e-3);
    rel *= max(len - range.z, 0.0) / len;
    vec4 clip = view_proj * vec4(rel, 1.0);
    // z / w = 1 - near / w; z >= 0 clips at the near plane exactly like the quads' depth.
    clip.z = clip.w - camera_sub_pos.w;
    gl_Position = clip;
}
