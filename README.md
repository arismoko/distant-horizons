# Distant Horizons for Sunrise

A native Sunrise port of [Distant Horizons](https://gitlab.com/distant-horizons-team/distant-horizons/)
(originally by James Seibel and contributors), built on Sunrise's general mod API.

## What it does

Distant Horizons generates, caches, and renders multi-level-of-detail (LOD) terrain beyond
the vanilla render distance using an independent reversed-Z projection and depth buffer,
sky-masked compositing with distance fog, background meshing, and per-world disk storage.

## Building

Requires `cargo-sunrise-mod` matching the pinned Sunrise Dev/SDK commit in `sunrise-mod.toml`:

```sh
cargo sunrise-mod check
cargo sunrise-mod build
```

## License

Licensed under LGPL-3.0-or-later (`LICENSE.LESSER.txt` and `LICENSE.txt`), matching upstream
Distant Horizons. Logos remain under CC BY-SA 4.0 (`LICENSE-LOGOS.txt`).
