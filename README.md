# Distant Sunrises for Sunrise

A native Sunrise port of [Distant Sunrises](https://gitlab.com/distant-sunrises-team/distant-sunrises/)
(originally by James Seibel and contributors) with a Voxy-style renderer. It runs entirely on
Sunrise's general mod API: `sunrise-mod.toml` has no `[[native.mixin]]` rows and no other
`[native.*]` table, and there are no WGSL shaders. All shaders are GLSL.

## Architecture

- **32³ octree sections.** Terrain is stored and meshed as 32×32×32 voxel sections (`src/octree.rs`,
  `src/octree_ingest.rs`). Loaded chunks and distant ungenerated terrain (sampled off-thread with
  `DetachedChunkGenerator`) are ingested into per-dimension octree stores by `src/pipeline.rs`.
- **8-byte packed quads.** A greedy mesher (`src/mesh.rs`) merges faces of the same model, biome and
  light into quads of at most 16×16 voxels, packed into one `u64` in Voxy's `quad_format` layout.
  `PackedQuad` is defined once, in `mesh.rs`, and re-exported by `gpu_arena`.
- **Mipmapped baked atlas.** `src/bakery.rs` bakes block models into six face grids of 16×16 texels.
  Face tiles are packed as space allows into one 2048×2048 dynamic texture with five mip levels.
  A 16-word record per distinct baked model and a biome tint table go into storage buffers.
- **GPU arena.** `src/gpu_arena.rs` keeps every section's quads in one persistent storage-buffer
  arena (free-list sub-allocation, 32-byte section metadata, 8 quad buckets per section: translucent,
  double-sided and the six faces). A section that cannot be stored keeps its previous mesh.
- **GLSL Hi-Z compute and GPU indirect draws.** Each frame a compute node (`voxy_hiz.csh`) builds a
  Hi-Z depth pyramid from the world depth, `voxy_cmdgen.csh` frustum, Hi-Z and back-face culls every
  section and rewrites its indirect draw commands, and two `draw_indirect` calls (opaque after the
  opaque features, translucent after the translucent blocks) pull quads from the arena in
  `voxy_quads.vsh/fsh`. Settings that tune the look (fog, shading, overdraw prevention) feed its
  uniform block.
- **Shader packs.** When the active Iris/OptiFine shader pack compiles `dh_terrain`, `dh_water` or
  `dh_shadow`, the mod's vertex stages `voxy_pack_*.vsh` link against the pack's fragment shader and
  the pack-linked pipelines draw instead (`src/shader_pack.rs`).
- **Storage.** Octree sections are stored as `DHLOD003` records (DH-exact palette and run layout with a
  CRC32) in crash-safe `.dhoct` region files under `<world_dir>/distant_sunrises/<dimension>/`
  (`src/storage.rs`), written and loaded on a background IO worker. `src/dh_v2.rs` decodes Distant
  Horizons' `FullDataSourceV2` payloads (LZ4, Zstd, LZMA2) for the public API.
- **Multiplayer: `octree_sync`.** `src/network.rs` (`octree_sync`) streams octree sections from a
  published server to remote clients over one raw, optional play channel
  (`distantsunrises:octree_sync`) in Distant Horizons' wire layout: session-config negotiation,
  per-player rate and concurrency limiters, timestamp-based "up to date" replies, chunked payloads
  (1,048,000-byte splits). A full section reply holds its request slot until the engine accepts its
  final frame for the local connection queue or the transfer is cancelled. Queue acceptance is
  not confirmation of remote delivery. Once the final frame is accepted, later revisions can be
  sent without another request. These partial updates require negotiated real-time updates and a
  player in the same dimension and within the negotiated range; they share the reply bandwidth
  limit. The integrated server's local window reads the shared terrain store instead. Receive
  buffers are size- and count-capped. Clients without the mod, and servers without it, are never
  blocked from joining.
- **Public API.** `src/api.rs` offers `LodWorldGenerator` overrides, `RemoteLodProvider` clients,
  column and section submission, `FullDataSourceV2` decoding and cache queries.
- **Client UI.** Mod settings (`Options > Mods`), `/ds` and `/distantsunrises` client commands,
  controls keybinds, and F3 debug entries.

## Differences from the original

- Sections are 32³ voxel octree nodes, not 64×64 column quadtree cells, and the renderer is a
  Voxy-style GPU-driven one rather than Distant Horizons' OpenGL one.
- Existing Java Distant Horizons `.sqlite` databases are not read; the mod keeps its own `.dhoct`
  region files.
- The multiplayer protocol follows Distant Horizons' message layout but is a different channel and
  is not compatible with Java DH servers or clients.

## Known limits

- The GPU arena is 128 MiB of quads and has no distance-based eviction yet: once it is full, new
  sections are not stored (the failure is logged).
- `octree_sync` serves sections by dimension. A published server drains all dimensions' outgoing
  terrain traffic after successful Overworld ticks when the mod is enabled and the server has time.
- Shader-pack shadows draw with the camera-culled command list, so casters outside the view are
  missing from shadow maps.

## Building

Requires `cargo-sunrise-mod` matching the pinned Sunrise Dev/SDK commit in `sunrise-mod.toml`:

```sh
cargo sunrise-mod check
cargo sunrise-mod build
```

## Measuring far-terrain cost

On Linux with Python 3.11+, use a trusted `target/sunrise-mod/run/app` staged by the SDK tool for
this checkout, `sunrise-ctl` from the matching game release, and that release's verified game and
SDK metadata. The script does not build, download, or stage source packages. For example, on a
machine with a dedicated Xvfb display and a working graphics driver:

```sh
xvfb-run -a -s '-screen 0 1280x720x24' python3 tools/bench_far_terrain.py \
  --app "$STAGED_APP" --ctl "$SUNRISE_CTL" \
  --release "$SUNRISE_RELEASE_JSON" --sdk "$SDK_JSON" \
  --resources "$PREPARED_RESOURCES" --out "$NEW_PRIVATE_RESULT_DIR" \
  --seed 1234 --camera -525 105 64 0 18 --distance 4 --settle 15 --seconds 20
```

Both rows use the same game executable, resources, seed, fixed spectator camera, clear noon and
near render distance. Each starts a fresh scratch save (no pre-existing chunks or LOD cache), with
its own game, options, offline player, and XDG state. The baseline omits the native package and
libraries; the mod row must map the selected native library and answer `/ds` in the running game.
The supplied stage and resources are not edited. `--runs N` repeats the pair with fresh state and
alternates the order. It does not reuse a shader or terrain cache from an earlier run.

`report.json` records frame times from `sunrise-ctl bench`, integrated server-thread CPU over the
same measurement window, `.dhoct` apparent and allocated bytes after graceful shutdown, and
near-terrain counts and mod store counts at both boundaries. `inputs.json` records file SHA-256s;
each row preserves the raw control results, library mapping, logs, and a same-camera screenshot.
Inspect the screenshots side by side to confirm visible distant terrain. An active section count
is not a count of visible geometry, and the vanilla row does not render the mod's far coverage.
These are fixed-duration costs, not an equal-coverage speedup or a comparison with Sunrise's
former built-in far terrain.

## License

Licensed under LGPL-3.0-or-later (`LICENSE.LESSER.txt` and `LICENSE.txt`), matching upstream
Distant Sunrises. Logos remain under CC BY-SA 4.0 (`LICENSE-LOGOS.txt`).
