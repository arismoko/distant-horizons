# Distant Horizons for Sunrise

A work-in-progress native Sunrise port of
[Distant Horizons](https://gitlab.com/distant-horizons-team/distant-horizons/)
(originally by James Seibel and contributors), targeting Sunrise's general mod API with no
Distant Horizons–specific engine seams and no `[native.*]` mixins once the underlying mod API
primitives land.

## Status

| Stage | Scope | Status |
|---|---|---|
| **Stage 1** | End-to-end LOD prototype on Sunrise: multi-layer LOD store (`DHLOD002`), background meshing, per-world disk storage, reversed-Z offscreen rendering & sky-masked HDR fog composite, settings UI, and F3 debug telemetry (using two temporary `[[native.mixin]]` render hooks) | **Done** (pinned to Sunrise Dev `ebd2d226`) |
| **Stage 2** | Full data & worldgen parity: zero-copy chunk snapshots, non-blocking chunk loading, saved Anvil region import, off-thread detached worldgen, multiplayer LOD streaming, and public mod-to-mod API for external generators / LOD providers | **Not done** (blocked on Sunrise API PRs) |
| **Stage 3** | Pure general mod-API renderer in GLSL: delete both temporary `[[native.mixin]]` hooks (`render_needs`, `world_drawn`) so `sunrise-mod.toml` has zero `[native.*]` entries | **Not done** (blocked on `#1161` and `#2005`) |
| **Stage 4** | Native drawing under translated Iris/OptiFine shader packs (`dh_terrain`, `dh_water`, `dh_shadow`, auxiliary depth planes, and pack uniforms) | **Not done** (shader-pack binding design pending) |

## What works today

- **LOD data & storage:** Multi-layer vertical column spans (overhangs, arches, caves, Nether ceilings) with cave culling, detail levels `0`–`6` (up to 512 chunks), background disk persistence (`DHLOD002` format with backward-compatible `DHLOD001` reads), and automatic quadtree parent mipmapping.
- **Meshing & rendering:** Background section mesher with neighbor face culling, side-plane frustum culling, independent reversed-Z projection and depth buffer, sky-masked compositing into Sunrise's linear HDR world target, configurable dithering fade, noise texture, and atmospheric distance fog.
- **Data capture (prototype):** Single-player/integrated-server capture of loaded chunks via `read_columns` and surface preview sampling via `terrain_surface`.
- **UI & telemetry:** Mod settings screen (render distance, quality presets, vertical quality, cave culling, fog/fade/noise/shading/brightness/saturation, distant generation mode) and F3 debug overlay entries.

## What does not work yet

- **Full-height background chunk snapshots (client & server):** Loaded chunk ingestion still uses synchronous tick-thread `read_columns` rather than zero-copy off-thread snapshots (so multiplayer client-side LOD recording from received chunks and full-light/3D-biome snapshots are not wired yet).
- **Saved region import:** Previously explored Anvil `.mca` regions outside the loaded radius are not imported in the background yet.
- **Full off-thread distant world generation:** Distant terrain outside loaded chunks currently uses basic surface column sampling; off-thread carvers, caves, biomes, trees/features, Nether, End, and custom dimension generators are not wired yet.
- **Multiplayer server-sent LODs & public mod-to-mod API:** Remote DH server LOD streaming and the public mod-to-mod API (letting other Sunrise mods override distant generation or push external/server-sent LOD data into Distant Horizons) are not implemented yet.
- **Zero-mixin rendering:** Rendering still relies on two temporary `[[native.mixin]]` hooks (`sunrise_mod_render::render::render_needs` and `world_drawn`) with WGSL shaders until Sunrise's general mod render pipeline and world stage APIs land.
- **Translated Iris/OptiFine shader packs:** When an Iris/OptiFine shader pack is active, Distant Horizons currently draws in its own standalone world-composite pass after vanilla world rendering rather than participating in the shader pack's gbuffer, shadow, deferred, or composite passes. Shader-pack shadows, water reflections, and pack fog do not apply to distant LOD terrain yet.

## Known Sunrise mod-API gaps

Progress on Stages 2–4 tracks these open issues on `sunrise-org/sunrise`:

- **Immutable column snapshots (`ChunkSnapshot`, client & server):** [sunrise-org/sunrise#5233](https://github.com/sunrise-org/sunrise/issues/5233)
- **Non-blocking server chunk requests (`LevelRef::request_chunk`):** [sunrise-org/sunrise#5234](https://github.com/sunrise-org/sunrise/issues/5234)
- **Read-only saved Anvil chunk inspection (`SavedChunkReader`):** [sunrise-org/sunrise#5235](https://github.com/sunrise-org/sunrise/issues/5235)
- **Off-thread detached world generation (`DetachedChunkGenerator`):** [sunrise-org/sunrise#5236](https://github.com/sunrise-org/sunrise/issues/5236)
- **World render stages & camera matrices (`RenderLevelStage` / `WorldRenderView`):** [sunrise-org/sunrise#1161](https://github.com/sunrise-org/sunrise/issues/1161)
- **General mod render pipelines, retained meshes, uniforms & render targets:** [sunrise-org/sunrise#2005](https://github.com/sunrise-org/sunrise/issues/2005)
- **Typed mod-to-mod events & library dependency identity:** [sunrise-org/sunrise#5137](https://github.com/sunrise-org/sunrise/issues/5137), [sunrise-org/sunrise#5136](https://github.com/sunrise-org/sunrise/issues/5136), and [sunrise-org/sunrise#3885](https://github.com/sunrise-org/sunrise/issues/3885)
- **Shader-pack geometry binding for mod pipelines (Stage 4):** General geometry-layer binding proposal pending alignment with Sunrise's shader-pack runtime refactor.

## Building

Requires `cargo-sunrise-mod` matching the pinned Sunrise Dev/SDK commit in `sunrise-mod.toml`:

```sh
cargo sunrise-mod check
cargo sunrise-mod build
```

## License

Licensed under LGPL-3.0-or-later (`LICENSE.LESSER.txt` and `LICENSE.txt`), matching upstream
Distant Horizons. Logos remain under CC BY-SA 4.0 (`LICENSE-LOGOS.txt`).
