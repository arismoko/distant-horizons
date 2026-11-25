fn render_needs(needs: &mut RenderNeeds) {
    if distantsunrises::render_bridge::needs_world_depth() {
        needs.world_depth = true;
    }
}
