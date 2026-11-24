# Distant Horizons for Sunrise

A work-in-progress native Sunrise port of
[Distant Horizons](https://gitlab.com/distant-horizons-team/distant-horizons/)
(originally by James Seibel and contributors), targeting Sunrise's general mod API with no
Distant Horizons–specific engine seams and no `[native.*]` mixins once the underlying mod API
primitives land.

## Status

| Stage | Scope | Status |
|---|---|---|
| **Stage 1** | End-to-end LOD prototype on Sunrise: multi-layer LOD store, background meshing, per-world crash-safe region storage (`DHREG002`), reversed-Z offscreen rendering & sky-masked HDR composite, settings UI, and F3 debug telemetry (using two temporary `[[native.mixin]]` render hooks) | **Done** (pinned to Sunrise Dev `3d34bbe6`) |
| **Stage 2a** | Public mod-to-mod LOD API (`src/api.rs`), `FullDataSourceV2DTO` wire decoder with LZ4/Zstd/LZMA2 (`src/dh_v2.rs`), optional droppable multiplayer LOD sync (`distanthorizons:lod_sync`, `SessionConfig`, active `PartialColumnUpdate` broadcast, per-server disk cache, `src/network.rs`), client `/dh` commands, Controls keybinds, spiral SSAO, CAS sharpening, full DH far/height/environmental fog, debug render modes, generic box renderer (`GenericObjectRenderer`), distant beacon beams, procedural 3D LOD clouds, active resource-pack block/colormap colors (`src/resource_colors.rs`), and GLSL shaders (`assets/distanthorizons/shaders/core/`) | **Done** (on `sunrise`, Dev `3d34bbe6`) |
| **Stage 2b** | Full data & worldgen parity: zero-copy `ChunkSnapshot` (client & server), non-blocking `LevelRef::request_chunk` queue, saved Anvil `.mca` `SavedChunkReader` import, and off-thread `DetachedChunkGenerator` worldgen | **Staged on branch `stage2-data-gaps`** (`#5339`–`#5341`, `#5347` merged on `sunrise-org/sunrise` `main`; waiting on Dev SDK cut) |
| **Stage 3** | Pure general mod-API renderer in GLSL: delete both temporary `[[native.mixin]]` hooks (`render_needs`, `world_drawn`) so `sunrise-mod.toml` has zero `[native.*]` entries | **In progress** (`#4758`, `#5185`, `#5252` merged; `#5426`–`#5430` / `#2005` pipeline stack next) |
| **Stage 4** | Native drawing under translated Iris/OptiFine shader packs (`dh_terrain`, `dh_water`, `dh_shadow`, auxiliary depth planes, and pack uniforms) | **In progress** (`#5386` & `#5387` merged on `main`; translator/runtime `#5490` open) |

## Differences from the original

- **LOD storage format (`DHREG002`):** Stores LOD data in region-packed Zstd-compressed binary files (`16×16` sections per `.dhreg` file, double-buffered CRC32-checksummed headers, copy-on-write append with streaming atomic compaction) rather than Java Distant Horizons' `DistantHorizons.sqlite` database. Existing Java DH SQLite databases are not shared or read directly.

## What works today

- **LOD data & storage:** Multi-layer vertical column spans (overhangs, arches, caves, Nether ceilings) with cave culling, detail levels `0`–`6` (up to 512 chunks), crash-safe region-packed Zstd disk persistence (`DHREG002` with double-buffered CRC32 headers, streaming atomic compaction, and legacy `.dhlod` fallback), and automatic quadtree parent mipmapping.
- **Meshing & rendering:** Background section mesher with neighbor face culling, side-plane frustum culling, active resource-pack block/grass/foliage color extraction, independent reversed-Z projection and depth buffer, sky-masked HDR-preserving compositing, Contrast-Adaptive Sharpening (CAS), spiral depth-buffer SSAO, configurable dithering fade, noise texture, brightness/saturation grading, weather- and time-aware lighting, full far/height/environmental fog, debug render modes, generic box-group rendering (`GenericObjectRenderer`), distant tinted beacon beams, and drifting 3D LOD clouds.
- **Public mod-to-mod API & multiplayer sync:** Public Rust API (`distanthorizons::api`) for custom `LodWorldGenerator` overrides, `RemoteLodProvider` clients, `GenericObjectRenderer` custom geometry, column/chunk submission, and `FullDataSourceV2DTO` wire decoding (LZ4/Zstd/LZMA2), plus an optional droppable `distanthorizons:lod_sync` play payload channel with `SessionConfig` negotiation, rate-limited section streaming, active `PartialColumnUpdate` delta broadcasts, dedicated/LAN per-player distant generation, and per-server local disk caching.
- **Data capture (on Dev `f8aa80e4`):** Single-player/integrated-server capture of loaded chunks via `read_columns` and surface preview sampling via `terrain_surface`. Full-height zero-copy snapshots (`#5339`), async chunk requests (`#5340`), saved Anvil region import (`#5341`), and detached off-thread worldgen (`#5347`) are wired on branch `stage2-data-gaps` ready to merge as soon as those PRs hit a Dev SDK.
- **UI, controls, commands & telemetry:** Mod settings screen (render distance, horizontal/vertical quality, cave culling, fog/fade/noise/shading/brightness/saturation, distant generation mode), `/dh` and `/distanthorizons` client commands, Controls keybinds (toggle rendering, cycle quality, clear cache), and F3 debug overlay entries.

## What does not work yet

- **Full-height background chunk snapshots, saved region import & detached worldgen on `sunrise` branch:** Wired on branch `stage2-data-gaps`; merges to `sunrise` once `#5339`–`#5341` and `#5347` land in a Sunrise Dev SDK.
- **Typed mod-to-mod event bus:** Public trait/function API (`src/api.rs`) is in place; typed `#[derive(mod_api::Event)]` events wait on `#5137`/`#5136`.
- **Zero-mixin rendering:** Rendering still relies on two temporary `[[native.mixin]]` hooks (`sunrise_mod_render::render::render_needs` and `world_drawn`) with WGSL shaders until Sunrise's `#4758` + `#1161` + `#2005` GLSL mod render pipeline APIs land (GLSL shaders are already staged in `assets/distanthorizons/shaders/core/`).
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
