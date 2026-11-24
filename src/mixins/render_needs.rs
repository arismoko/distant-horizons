fn render_needs(needs: &mut RenderNeeds) {
    if distanthorizons::render_bridge::needs_world_depth() {
        needs.world_depth = true;
    }
}
