# Distant Horizons for Sunrise

A native Sunrise port of [Distant Horizons](https://gitlab.com/distant-horizons-team/distant-horizons/)
(originally by James Seibel and contributors), built on Sunrise's general mod API.

## What it does

Distant Horizons generates, caches, and renders multi-level-of-detail (LOD) terrain beyond
the vanilla render distance using an independent reversed-Z projection and depth buffer,
sky-masked compositing with distance fog, background meshing, and per-world disk storage.

## Shader Pack Support

When no Iris/OptiFine shader pack is active, Distant Horizons draws distant terrain into
Sunrise's linear HDR world color target with sky-masked atmospheric fog compositing.

When an Iris/OptiFine shader pack is active, native shader-pack geometry integration
(`dh_terrain`, `dh_water`, `dh_shadow`, auxiliary depth planes, and pack uniforms via
`RenderPipeline::pack_binding`) is not yet available in Sunrise. Until that pack-binding
support lands, Distant Horizons draws with its own standalone world-composite pass after
vanilla world rendering rather than participating in the shader pack's gbuffer, shadow,
deferred, or composite passes (so shader-pack shadows, water reflections, and pack fog
do not yet apply to distant LOD terrain).

## Building

Requires `cargo-sunrise-mod` matching the pinned Sunrise Dev/SDK commit in `sunrise-mod.toml`:

```sh
cargo sunrise-mod check
cargo sunrise-mod build
```

## License

Licensed under LGPL-3.0-or-later (`LICENSE.LESSER.txt` and `LICENSE.txt`), matching upstream
Distant Horizons. Logos remain under CC BY-SA 4.0 (`LICENSE-LOGOS.txt`).
