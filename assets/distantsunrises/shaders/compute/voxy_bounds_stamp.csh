#version 450
// Publishes this submission's serial after the footprint-bounds pass (render_bridge::setup_bounds).
// voxy_quads.fsh clips against the bound texture only where it reads the serial it was drawn
// with, so a bounds pass that never ran in this submission (the graph was dropped after the draws
// were prepared) leaves the stamp behind and nothing is clipped.
layout(local_size_x = 1) in;

layout(std140) uniform BoundsStampParams {
    uvec4 serial;  // xy = serial (never 0); zw = unused
};

layout(std430) buffer BoundsStamp { uvec2 published; };

void main() {
    published = serial.xy;
}
