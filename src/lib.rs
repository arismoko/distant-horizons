//! Distant Horizons native mod for Sunrise.
//!
//! Stage 1 architecture:
//! - Server side samples loaded chunks (`read_columns`) and distant ungenerated terrain
//!   (`terrain_surface`) during `level_tick_post` under `host.server_has_time()` budgeting,
//!   downsamples into the multi-detail-level quadtree, and persists RLE-compressed sections to
//!   `<world_dir>/distant_horizons/<dimension>/` on a background IO worker.
//! - Background meshing worker generates greedy-merged, face-culled solid and fluid meshes.
//! - Client side tracks camera, day-time lighting, settings (`Options > Mods`), fog colors
//!   (`render_fog`), and F3 debug telemetry (`register_debug_entries`).
//! - `sunrise-mod-render` mixins (`render_needs`, `world_drawn`) render LOD geometry with an
//!   independent reversed-Z depth buffer and sky-masked atmospheric fog compositing.

pub mod colors;
pub mod config;
pub mod lod;
pub mod mesh;
pub mod render_bridge;
pub mod storage;

use crate::colors::{biome_water_color, block_color, is_skippable_for_lod, is_water_block};
use crate::config::{DhConfig, MOD_ID};
use crate::lod::{
    DimensionLodStore, LodColumn, SECTION_SIZE, SOURCE_DISTANT_SURFACE, SOURCE_LOADED_CHUNK,
    SectionPos,
};
use crate::render_bridge::{runtime, stats, sun_brightness_from_day_time};
use mod_api::client::fog::FogType;
use mod_api::{
    self, BlockChangedEvent, ChunkLoadEvent, ClientHost, ClientLevelLoadEvent,
    ClientLevelUnloadEvent, ClientTickPreEvent, DebugEntry, DebugGroup, Heightmap, Hooks, HostSide,
    LevelDimension, LevelHost, LevelLoadEvent, LevelTickPostEvent, LevelUnloadEvent,
    MAX_TERRAIN_COLUMNS, RegisterDebugEntriesEvent, Registrar, RegistrationContext, RenderFogEvent,
    ServerHost, ServerStartedEvent, ServerStoppingEvent, SettingChangedEvent,
};
use std::collections::HashSet;
use std::sync::atomic::Ordering;

/// Depth below the `WorldSurface` heightmap scanned per loaded chunk column to see through
/// foliage/water to the solid ocean/ground floor.
const LOADED_SCAN_BELOW: u32 = 24;
const LOADED_SCAN_ABOVE: u32 = 2;

/// Number of loaded chunks sampled per server tick when time remains.
const CHUNKS_PER_TICK: usize = 4;

/// Number of distant surface columns sampled per server tick (capped well under 4096 so ticks
/// stay responsive even on software setups).
const DISTANT_COLS_PER_TICK: usize = 1024;

#[derive(Default)]
struct DistantHorizonsMod;

#[mod_api::sunrise_mod]
impl Hooks for DistantHorizonsMod {
    fn register(&self, registrar: &mut dyn Registrar, _cx: &RegistrationContext) {
        DhConfig::register_settings(registrar);
    }

    fn register_debug_entries(
        &self,
        _cx: &RegistrationContext,
        event: &mut RegisterDebugEntriesEvent<'_>,
    ) {
        event.entries.push(
            DebugEntry::text(
                format!("{MOD_ID}:status"),
                DebugGroup::ChunkGeneration,
                |_host, lines| {
                    let s = stats();
                    lines.push(format!(
                        "DH: sec={} ready={} drawn={} quads={} gen={} saved={}",
                        s.active_sections,
                        s.ready_meshes,
                        s.rendered_sections,
                        s.rendered_quads,
                        s.generated_columns,
                        s.saved_sections
                    ));
                },
            )
            .with_reduced_debug_info(true),
        );
    }

    fn setting_changed(&self, host: &mut dyn ClientHost, _event: &mut SettingChangedEvent<'_>) {
        sync_client_config(host);
    }

    fn client_level_load(&self, host: &mut dyn ClientHost, event: &mut ClientLevelLoadEvent<'_>) {
        sync_client_config(host);
        let rt = runtime();
        let mut st = rt.inner.lock().unwrap();
        st.active_dimension = event.dimension.to_owned();
        st.gpu_reset_epoch = st.gpu_reset_epoch.wrapping_add(1);
        // Re-submit any already-known sections in this dimension to the mesher so the GPU cache
        // immediately populates after switching dimensions or rejoining a world.
        if let Some(dim) = st.dimensions.get(event.dimension) {
            for sec in dim.sections.values() {
                if sec.populated > 0 {
                    rt.mesher.submit(event.dimension.to_owned(), sec.clone());
                }
            }
        }
    }

    fn client_level_unload(
        &self,
        _host: &mut dyn ClientHost,
        event: &mut ClientLevelUnloadEvent<'_>,
    ) {
        let rt = runtime();
        let mut st = rt.inner.lock().unwrap();
        if st.active_dimension == event.dimension {
            st.gpu_reset_epoch = st.gpu_reset_epoch.wrapping_add(1);
        }
    }

    fn client_tick_pre(&self, host: &mut dyn ClientHost, _event: &mut ClientTickPreEvent) {
        let rt = runtime();
        let cfg = DhConfig::read_from_client(host);
        rt.enabled_flag.store(cfg.enabled, Ordering::Relaxed);

        let cam = host.camera();
        let clock = host.clock();
        let mut st = rt.inner.lock().unwrap();
        st.config = cfg;
        if let Some(camera) = cam {
            st.camera_pos = camera.position;
            st.has_camera = true;
            if st.vanilla_far_blocks <= 16.0 && camera.far.is_finite() && camera.far > 64.0 {
                st.vanilla_far_blocks = camera.far * 0.25;
            }
        }
        if let Some(clk) = clock {
            st.sun_brightness = sun_brightness_from_day_time(clk.day_time, &st.active_dimension);
        }
    }

    fn render_fog(&self, host: &dyn ClientHost, event: &mut RenderFogEvent<'_>) {
        if !host.on(MOD_ID) {
            return;
        }
        let rt = runtime();
        let mut st = rt.inner.lock().unwrap();
        st.fog_color = event.fog.color;
        if event.fog.render_distance_end.is_finite() && event.fog.render_distance_end > 16.0 {
            st.vanilla_far_blocks = event.fog.render_distance_end;
        }
        // Push vanilla's atmospheric render-distance fog out past vanilla's clip so the horizon
        // reveals Distant Horizons LOD terrain (DH applies its own atmospheric horizon fog in the
        // composite pass). Underwater/lava/powder-snow fog stays vanilla-exact.
        if st.config.enabled && event.fog_type == FogType::Atmospheric {
            let dh_far = st.config.render_distance_blocks().max(512.0);
            event.fog.render_distance_start = event.fog.render_distance_start.max(dh_far * 0.75);
            event.fog.render_distance_end = event.fog.render_distance_end.max(dh_far);
        }
    }

    fn server_started(&self, host: &mut dyn ServerHost, _event: &mut ServerStartedEvent) {
        let world_dir = host.world_dir();
        let rt = runtime();
        let mut st = rt.inner.lock().unwrap();
        st.world_dir = world_dir;
    }

    fn server_stopping(&self, host: &mut dyn ServerHost, _event: &mut ServerStoppingEvent) {
        let world_dir = host.world_dir();
        flush_dirty_sections(world_dir);
    }

    fn level_load(&self, host: &mut dyn ServerHost, event: &mut LevelLoadEvent<'_>) {
        let world_dir = host.world_dir();
        let rt = runtime();
        let mut st = rt.inner.lock().unwrap();
        if st.world_dir.is_none() {
            st.world_dir = world_dir;
        }
        st.dimensions.entry(event.dimension.to_owned()).or_default();
    }

    fn level_unload(&self, host: &mut dyn ServerHost, event: &mut LevelUnloadEvent<'_>) {
        let world_dir = host.world_dir();
        flush_dimension_dirty(world_dir, event.dimension);
    }

    fn chunk_load(&self, host: &mut dyn ServerHost, event: &mut ChunkLoadEvent<'_>) {
        if !host.on(MOD_ID) {
            return;
        }
        enqueue_chunk(event.dimension, event.chunk);
    }

    fn block_changed(&self, _level: &dyn LevelHost, event: &mut BlockChangedEvent) {
        let chunk = [event.pos.x >> 4, event.pos.z >> 4];
        let dim_id = match &event.dimension {
            LevelDimension::Overworld => "minecraft:overworld",
            LevelDimension::Nether => "minecraft:the_nether",
            LevelDimension::End => "minecraft:the_end",
            LevelDimension::Custom(key) => key.id(),
        };
        enqueue_chunk(dim_id, chunk);
    }

    fn level_tick_post(&self, host: HostSide<'_>, event: &mut LevelTickPostEvent) {
        let HostSide::Server(host) = host else {
            return;
        };
        if !host.on(MOD_ID) || !host.server_has_time() {
            return;
        }
        let dim = event.level.dimension_id();
        tick_server_dimension(host, dim);
    }
}

fn sync_client_config(host: &dyn ClientHost) {
    let cfg = DhConfig::read_from_client(host);
    let rt = runtime();
    rt.enabled_flag.store(cfg.enabled, Ordering::Relaxed);
    rt.inner.lock().unwrap().config = cfg;
}

fn enqueue_chunk(dimension: &str, chunk: [i32; 2]) {
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let set = st
        .pending_chunk_set
        .entry(dimension.to_owned())
        .or_default();
    if set.insert(chunk) {
        st.pending_chunks
            .entry(dimension.to_owned())
            .or_default()
            .push_back(chunk);
    }
}

fn tick_server_dimension(host: &mut dyn ServerHost, dim: &str) {
    let rt = runtime();
    let world_dir = host.world_dir();

    // 1. Sample a small batch of newly loaded / modified chunks via `read_columns`.
    let chunks_to_sample: Vec<[i32; 2]> = {
        let mut st = rt.inner.lock().unwrap();
        if st.world_dir.is_none() && world_dir.is_some() {
            st.world_dir = world_dir.clone();
        }
        let mut batch = Vec::with_capacity(CHUNKS_PER_TICK);
        let st_ref = &mut *st;
        if let Some(queue) = st_ref.pending_chunks.get_mut(dim) {
            let mut set = st_ref.pending_chunk_set.get_mut(dim);
            for _ in 0..CHUNKS_PER_TICK {
                let Some(c) = queue.pop_front() else {
                    break;
                };
                if let Some(ref mut s) = set {
                    s.remove(&c);
                }
                batch.push(c);
            }
        }
        batch
    };

    let mut dirty_sections: HashSet<SectionPos> = HashSet::new();
    if !chunks_to_sample.is_empty() {
        sample_loaded_chunks(host, dim, &chunks_to_sample, &mut dirty_sections);
    }

    // 2. If time remains and distant generation is on, sample distant terrain around the camera
    //    using `terrain_surface` and schedule disk loads for any previously saved sections.
    let (distant_on, cam, cfg, active_dim) = {
        let st = rt.inner.lock().unwrap();
        (
            st.config.enabled && st.config.distant_generation,
            if st.has_camera {
                Some(st.camera_pos)
            } else {
                None
            },
            st.config,
            st.active_dimension.clone(),
        )
    };

    if distant_on && host.server_has_time() && dim == active_dim {
        let center = cam.unwrap_or([0.0, 64.0, 0.0]);
        sample_distant_sections(host, dim, center, cfg, &mut dirty_sections);
    }

    // 3. Propagate dirty sections up the quadtree, queue meshing, and persist dirty sections.
    if !dirty_sections.is_empty() {
        let mut st = rt.inner.lock().unwrap();
        let save_root = st.world_dir.clone().or(world_dir);
        let store = st.dimensions.entry(dim.to_owned()).or_default();
        let mut to_mesh: HashSet<SectionPos> = dirty_sections.clone();
        for &pos in &dirty_sections {
            for parent_pos in store.propagate_up(pos) {
                to_mesh.insert(parent_pos);
            }
        }
        for pos in to_mesh {
            if let Some(sec) = store.sections.get_mut(&pos) {
                if sec.dirty_mesh && sec.populated > 0 {
                    sec.dirty_mesh = false;
                    rt.mesher.submit(dim.to_owned(), sec.clone());
                }
                if sec.dirty_save && sec.populated > 0 {
                    if let Some(ref dir) = save_root {
                        sec.dirty_save = false;
                        rt.io.request_save(dir.clone(), dim.to_owned(), sec.clone());
                    }
                }
            }
        }
    }
}

fn sample_loaded_chunks(
    host: &dyn ServerHost,
    dim: &str,
    chunks: &[[i32; 2]],
    dirty_sections: &mut HashSet<SectionPos>,
) {
    let mut columns = Vec::with_capacity(chunks.len() * 256);
    for &[cx, cz] in chunks {
        let bx0 = cx << 4;
        let bz0 = cz << 4;
        for lz in 0..16 {
            for lx in 0..16 {
                columns.push((bx0 + lx, bz0 + lz));
            }
        }
    }
    let Ok(read) = host.read_columns(
        dim,
        &columns,
        Heightmap::WorldSurface,
        (LOADED_SCAN_BELOW, LOADED_SCAN_ABOVE),
    ) else {
        return;
    };

    let span = (LOADED_SCAN_BELOW + LOADED_SCAN_ABOVE) as usize;
    if span == 0 || read.bases.len() != columns.len() {
        return;
    }

    // Resolve palette block IDs once per batch.
    let palette_ids: Vec<&str> = read
        .palette
        .iter()
        .map(|state| state.block().id(host).unwrap_or("minecraft:stone"))
        .collect();

    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let store = st.dimensions.entry(dim.to_owned()).or_default();

    for (col_idx, &(bx, bz)) in columns.iter().enumerate() {
        let Some(base_y) = read.bases[col_idx] else {
            continue;
        };
        let slice_start = col_idx * span;
        if slice_start + span > read.blocks.len() {
            break;
        }
        let col_blocks = &read.blocks[slice_start..slice_start + span];
        let water_hint = read.water_colors.get(col_idx).copied().unwrap_or(0);

        let mut fluid_top: Option<i32> = None;
        let mut solid_top: Option<(i32, &str)> = None;

        // Scan top-down through the sampled band.
        for offset in (0..span).rev() {
            let pal_idx = col_blocks[offset] as usize;
            let block_id = palette_ids.get(pal_idx).copied().unwrap_or("minecraft:air");
            let y = base_y - (LOADED_SCAN_BELOW as i32) + (offset as i32) + 1;
            if is_water_block(block_id) {
                if fluid_top.is_none() {
                    fluid_top = Some(y);
                }
                continue;
            }
            if is_skippable_for_lod(block_id) {
                continue;
            }
            solid_top = Some((y, block_id));
            break;
        }

        let (top_y, block_id) = match solid_top {
            Some(v) => v,
            None => match fluid_top {
                Some(fy) => (fy - 4, "minecraft:sand"),
                None => continue,
            },
        };

        let solid_rgb = block_color(block_id, "minecraft:plains", water_hint);
        let fluid_rgb = biome_water_color("minecraft:plains", water_hint);
        let lod_col = LodColumn::from_solid_and_fluid(
            top_y,
            top_y - 8,
            solid_rgb,
            fluid_top,
            fluid_rgb,
            15,
            0,
            SOURCE_LOADED_CHUNK,
        );

        let sec_pos = SectionPos::from_block(0, bx, bz);
        let (lx, lz) = sec_pos.local_cell(bx, bz);
        if store.get_or_create_mut(sec_pos).set_cell(lx, lz, lod_col) {
            dirty_sections.insert(sec_pos);
        }
    }
}

fn sample_distant_sections(
    host: &dyn ServerHost,
    dim: &str,
    cam: [f64; 3],
    cfg: DhConfig,
    dirty_sections: &mut HashSet<SectionPos>,
) {
    let rt = runtime();
    let desired = DimensionLodStore::desired_sections_coarse_to_fine(cam[0], cam[2], cfg);

    // Pick the nearest desired section that still has unpopulated cells, and issue async disk
    // load checks for unseen sections first.
    let target_section: Option<(SectionPos, Vec<(usize, usize)>)> = {
        let mut st = rt.inner.lock().unwrap();
        let save_root = st.world_dir.clone();
        let store = st.dimensions.entry(dim.to_owned()).or_default();

        // Trigger background disk loads for up to 8 unseen sections per tick.
        if let Some(ref dir) = save_root {
            let mut loads = 0;
            for &pos in &desired {
                if store.disk_checked.insert(pos) {
                    rt.io.request_load(dir.clone(), dim.to_owned(), pos);
                    loads += 1;
                    if loads >= 8 {
                        break;
                    }
                }
            }
        }

        let mut picked = None;
        for pos in desired {
            if store.sampled_distant_sections.contains(&pos) {
                continue;
            }
            let sec = store.sections.get(&pos);
            let mut missing = Vec::with_capacity(DISTANT_COLS_PER_TICK);
            for cz in 0..SECTION_SIZE {
                for cx in 0..SECTION_SIZE {
                    let is_empty = sec.map(|s| s.get(cx, cz).is_empty()).unwrap_or(true);
                    if is_empty {
                        missing.push((cx, cz));
                        if missing.len() >= DISTANT_COLS_PER_TICK.min(MAX_TERRAIN_COLUMNS) {
                            break;
                        }
                    }
                }
                if missing.len() >= DISTANT_COLS_PER_TICK.min(MAX_TERRAIN_COLUMNS) {
                    break;
                }
            }
            if missing.is_empty() {
                store.sampled_distant_sections.insert(pos);
            } else {
                picked = Some((pos, missing));
                break;
            }
        }
        picked
    };

    let Some((pos, cells)) = target_section else {
        return;
    };

    let (ox, oz) = pos.block_origin();
    let step = pos.cell_size();
    let half = step >> 1;
    let query_cols: Vec<(i32, i32)> = cells
        .iter()
        .map(|&(cx, cz)| {
            (
                ox + (cx as i32) * step + half,
                oz + (cz as i32) * step + half,
            )
        })
        .collect();

    let Ok(samples) = host.terrain_surface(dim, &query_cols) else {
        return;
    };

    let mut st = rt.inner.lock().unwrap();
    let store = st.dimensions.entry(dim.to_owned()).or_default();
    let mut updated_any = false;
    let mut non_empty_count = 0u64;

    {
        let sec = store.get_or_create_mut(pos);
        for (&(cx, cz), sample) in cells.iter().zip(samples.iter()) {
            let Some(col) = sample else {
                continue;
            };
            let block_id = col.block.id(host).unwrap_or("minecraft:stone");
            let solid_rgb = block_color(block_id, &col.biome, 0);
            let fluid_rgb = biome_water_color(&col.biome, 0);
            let skirt = (step * 4).clamp(8, 32);
            let lod_col = LodColumn::from_solid_and_fluid(
                col.height,
                col.height - skirt,
                solid_rgb,
                col.water,
                fluid_rgb,
                15,
                0,
                SOURCE_DISTANT_SURFACE,
            );
            if sec.set_cell(cx, cz, lod_col) {
                updated_any = true;
            }
            non_empty_count += 1;
        }
    }

    store.generated_columns = store.generated_columns.saturating_add(non_empty_count);
    if non_empty_count == 0 {
        // Void / generatorless dimension: don't loop forever on the same section.
        store.sampled_distant_sections.insert(pos);
    }
    if updated_any {
        dirty_sections.insert(pos);
    }
}

fn flush_dimension_dirty(world_dir: Option<std::path::PathBuf>, dim: &str) {
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let Some(dir) = world_dir.or_else(|| st.world_dir.clone()) else {
        return;
    };
    if let Some(store) = st.dimensions.get_mut(dim) {
        for sec in store.sections.values_mut() {
            if sec.dirty_save && sec.populated > 0 {
                sec.dirty_save = false;
                rt.io.request_save(dir.clone(), dim.to_owned(), sec.clone());
            }
        }
    }
}

fn flush_dirty_sections(world_dir: Option<std::path::PathBuf>) {
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let Some(dir) = world_dir.or_else(|| st.world_dir.clone()) else {
        return;
    };
    for (dim, store) in st.dimensions.iter_mut() {
        for sec in store.sections.values_mut() {
            if sec.dirty_save && sec.populated > 0 {
                sec.dirty_save = false;
                rt.io.request_save(dir.clone(), dim.clone(), sec.clone());
            }
        }
    }
}
