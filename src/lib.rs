//! Distant Sunrises native mod for Sunrise.
//!
//! Stage 1 architecture:
//! - Server side samples loaded chunks (`read_columns`) and distant ungenerated terrain
//!   (`terrain_surface`) during `level_tick_post` under `host.server_has_time()` budgeting,
//!   downsamples into the multi-detail-level quadtree, and persists RLE-compressed sections to
//!   `<world_dir>/distant_sunrises/<dimension>/` on a background IO worker.
//! - Background meshing worker generates greedy-merged, face-culled solid and fluid meshes.
//! - Client side tracks camera, day-time lighting, settings (`Options > Mods`), fog colors
//!   (`render_fog`), and F3 debug telemetry (`register_debug_entries`).
//! - `sunrise-mod-render` mixins (`render_needs`, `world_drawn`) render LOD geometry with an
//!   independent reversed-Z depth buffer and sky-masked atmospheric fog compositing.

pub mod api;
pub mod colors;
pub mod commands;
pub mod config;
pub mod dh_v2;
pub mod generic_render;
pub mod lod;
pub mod mesh;
pub mod network;
pub mod render_bridge;
pub mod resource_colors;
pub mod shader_pack;
pub mod stage3_camera;
pub mod storage;

use crate::api::{LodGenRequest, LodGenStep, LodSectionPos, ingest_gen_output};
use crate::colors::{
    biome_water_color, block_color, classify_dh_material, is_skippable_for_lod, is_water_block,
};
use crate::commands::DhKeybinds;
use crate::config::{DhConfig, MOD_ID};
use crate::generic_render::{
    BeaconBeam, GROUP_ID_BEACONS, GROUP_ID_CLOUDS, blend_beacon_tint, build_box_group_mesh,
    stained_glass_tint,
};
use crate::lod::{
    DimensionLodStore, LodColumn, SECTION_SIZE, SOURCE_DISTANT_SURFACE, SOURCE_LOADED_CHUNK,
    SectionPos,
};
use crate::render_bridge::{runtime, stats, sun_brightness_from_day_time_and_weather};
use mod_api::client::fog::FogType;
use mod_api::{
    self, BlockChangedEvent, ChunkLoadEvent, ClientBlockChangedEvent, ClientChunkLoadEvent,
    ClientChunkUnloadEvent, ClientHost, ClientLevelLoadEvent, ClientLevelUnloadEvent,
    ClientReloadHost, ClientSetupContext, ClientSetupEvent, ClientTickPreEvent, CommandEvent,
    DebugEntry, DebugGroup, EntityRef, Flow, Heightmap, Hooks, HostSide, LevelDimension, LevelHost,
    LevelLoadEvent, LevelTickPostEvent, LevelUnloadEvent, MAX_TERRAIN_COLUMNS,
    RegisterClientCommandsEvent, RegisterDebugEntriesEvent, Registrar, RegistrationContext,
    RenderFogEvent, ServerHost, ServerStartedEvent, ServerStoppingEvent, SettingChangedEvent,
};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::Ordering;

/// Depth below the `WorldSurface` heightmap scanned per loaded chunk column to see through
/// foliage/water to the solid ocean/ground floor.
const LOADED_SCAN_BELOW: u32 = 24;
const LOADED_SCAN_ABOVE: u32 = 2;

/// Number of loaded chunks sampled per server tick when time remains.
const CHUNKS_PER_TICK: usize = 4;
/// Per-tick wall-time budget for synchronous server-thread `terrain_surface` fallback batches
/// (8% of a 50 ms server tick). On fast CPUs multiple 64-column batches run per tick within this
/// budget; if a single batch exceeds it on a slower CPU, a post-completion cooldown of
/// `elapsed * DISTANT_SAMPLE_COOLDOWN_MULT` (capped at `DISTANT_SAMPLE_MAX_COOLDOWN`) keeps the
/// server thread under ~10% duty cycle until off-thread detached generation lands.
const DISTANT_SAMPLE_BUDGET: std::time::Duration = std::time::Duration::from_millis(4);
const DISTANT_SAMPLE_MAX_COOLDOWN: std::time::Duration = std::time::Duration::from_millis(250);
const DISTANT_SAMPLE_COOLDOWN_MULT: u32 = 9;

/// Number of empty cells sampled per synchronous server-thread `terrain_surface` batch (taken in
/// row-major order from the nearest unsampled section).
const DISTANT_COLS_PER_TICK: usize = 64;

#[derive(Default)]
struct DistantHorizonsMod {
    keybinds: Mutex<DhKeybinds>,
}

#[mod_api::sunrise_mod]
impl Hooks for DistantHorizonsMod {
    fn register(&self, registrar: &mut dyn Registrar, _cx: &RegistrationContext) {
        DhConfig::register_settings(registrar);
        commands::register_key_mappings(registrar);
        network::register_payloads(registrar);
    }

    fn client_setup(&self, host: &mut dyn ClientSetupContext, _event: &mut ClientSetupEvent) {
        *self.keybinds.lock().unwrap() = DhKeybinds::resolve(host);
        let _ = resource_colors::reload_from_resources(host);
    }

    fn client_resources_reloaded(
        &self,
        host: &dyn ClientReloadHost,
        _event: &mut mod_api::ClientResourcesReloadedEvent,
    ) {
        let _ = resource_colors::reload_from_resources(host);
        let _ = commands::remesh_active_dimension();
    }

    fn register_client_commands(
        &self,
        _host: &mut dyn ClientHost,
        event: &mut RegisterClientCommandsEvent<'_>,
    ) {
        commands::register_client_commands(event.commands);
    }

    fn command(&self, host: &mut dyn ClientHost, event: &mut CommandEvent<'_>) -> Flow {
        commands::handle_client_command(host, event.command)
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
        let remote_cache = host
            .client_session()
            .client_server()
            .filter(|s| !s.integrated_local)
            .map(|s| {
                let key = s.address.or(s.name).unwrap_or("remote");
                network::remote_cache_root(&host.instance_dir(), key)
            });
        let rt = runtime();
        let mut st = rt.inner.lock().unwrap();
        st.active_dimension = event.dimension.to_owned();
        st.remote_cache_dir = remote_cache;
        st.client_loaded_chunks.clear();
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
        let remote_cache = {
            let mut st = runtime().inner.lock().unwrap();
            st.remote_requested.remove(event.dimension);
            st.remote_session_config = None;
            st.remote_config_requested = false;
            st.client_loaded_chunks.clear();
            if st.active_dimension == event.dimension {
                st.gpu_reset_epoch = st.gpu_reset_epoch.wrapping_add(1);
            }
            st.remote_cache_dir.take()
        };
        if let Some(dir) = remote_cache {
            flush_dimension_dirty(Some(dir), event.dimension);
        }
    }

    fn client_chunk_load(&self, host: &mut dyn ClientHost, event: &mut ClientChunkLoadEvent<'_>) {
        if !host.on(MOD_ID) {
            return;
        }
        let rt = runtime();
        rt.inner
            .lock()
            .unwrap()
            .client_loaded_chunks
            .insert(event.chunk);
    }

    fn client_chunk_unload(
        &self,
        _host: &mut dyn ClientHost,
        event: &mut ClientChunkUnloadEvent<'_>,
    ) {
        let rt = runtime();
        rt.inner
            .lock()
            .unwrap()
            .client_loaded_chunks
            .remove(&event.chunk);
    }

    fn client_block_changed(&self, host: &dyn ClientHost, event: &mut ClientBlockChangedEvent<'_>) {
        if !host.on(MOD_ID) || event.predicted {
            return;
        }
        // Record chunk coordinate for future client-side ChunkSnapshot refresh (G1/#5233).
        let _ = (event.dimension, [event.pos.x >> 4, event.pos.z >> 4]);
    }

    fn client_tick_pre(&self, host: &mut dyn ClientHost, _event: &mut ClientTickPreEvent) {
        let rt = runtime();
        let cfg = DhConfig::read_from_client(host);
        rt.enabled_flag.store(cfg.enabled, Ordering::Relaxed);

        if host.playing() {
            let kb = self.keybinds.lock().unwrap();
            kb.poll(host);
        }

        let cam = host.camera();
        let clock = host.clock();
        let gfx_opts = host.graphics_options();
        let active_dim_now = rt.inner.lock().unwrap().active_dimension.clone();
        let (rain_level, thunder_level) = host
            .level(&active_dim_now)
            .and_then(|lvl| lvl.weather(host))
            .unwrap_or((0.0, 0.0));

        let remote_batch = {
            let mut st = rt.inner.lock().unwrap();
            st.config = DhConfig::read_from_client(host);
            st.vanilla_render_distance_chunks = gfx_opts.render_distance.max(2);
            st.rain_level = rain_level.clamp(0.0, 1.0);
            st.thunder_level = thunder_level.clamp(0.0, 1.0);
            if let Some(camera) = cam {
                st.camera = Some(stage3_camera::CameraMatrices::from_camera(&camera));
                st.camera_pos = camera.position;
                st.has_camera = true;
                st.vanilla_far_blocks = (st.vanilla_render_distance_chunks as f32) * 16.0;
            }
            if let Some(clk) = clock {
                st.sun_brightness = sun_brightness_from_day_time_and_weather(
                    clk.day_time,
                    &st.active_dimension,
                    st.rain_level,
                    st.thunder_level,
                );
            }

            let active_dim = st.active_dimension.clone();
            if st.has_camera {
                let cam_pos = st.camera_pos;
                let v_far = st.vanilla_far_blocks;
                let dh_far = cfg.render_distance_blocks().max(256.0);
                let elapsed = st.start_instant.elapsed().as_secs_f64();
                let rain = st.rain_level;

                // Update procedural 3D LOD clouds (drifts smoothly every tick; only re-meshes on cell/config change).
                let (cloud_group, cloud_origin) =
                    st.cloud_field
                        .update(&active_dim, cam_pos, v_far, cfg, rain, elapsed);
                if let Some(group) = cloud_group {
                    let mesh = build_box_group_mesh(&group);
                    st.generic_groups
                        .entry(active_dim.clone())
                        .or_default()
                        .insert(GROUP_ID_CLOUDS.to_owned(), mesh);
                    st.generic_groups_revision = st.generic_groups_revision.wrapping_add(1);
                } else if let Some(existing) = st
                    .generic_groups
                    .get_mut(&active_dim)
                    .and_then(|m| m.get_mut(GROUP_ID_CLOUDS))
                {
                    existing.origin = cloud_origin;
                }

                // Update distant beacon beams (only rebuilds when beacons, origin grid, or render distance change).
                let beacon_group = st
                    .beacons
                    .entry(active_dim.clone())
                    .or_default()
                    .maybe_build_group(
                        cam_pos,
                        v_far,
                        dh_far,
                        cfg.enabled && cfg.beacon_beams_enabled,
                    );
                if let Some(group) = beacon_group {
                    let mesh = build_box_group_mesh(&group);
                    st.generic_groups
                        .entry(active_dim.clone())
                        .or_default()
                        .insert(GROUP_ID_BEACONS.to_owned(), mesh);
                    st.generic_groups_revision = st.generic_groups_revision.wrapping_add(1);
                }
            }

            if cfg.enabled && st.has_camera {
                if let Some(provider) = st.remote_providers.get(&active_dim).cloned() {
                    let desired = DimensionLodStore::desired_sections_coarse_to_fine(
                        st.camera_pos[0],
                        st.camera_pos[2],
                        cfg,
                    );
                    let mut to_request = Vec::with_capacity(4);
                    for pos in desired {
                        let populated = st
                            .dimensions
                            .get(&active_dim)
                            .and_then(|d| d.sections.get(&pos))
                            .map(|s| s.populated > 0)
                            .unwrap_or(false);
                        if populated {
                            continue;
                        }
                        if st
                            .remote_requested
                            .entry(active_dim.clone())
                            .or_default()
                            .insert(pos)
                        {
                            to_request.push(pos);
                            if to_request.len() >= 4 {
                                break;
                            }
                        }
                    }
                    Some((provider, active_dim, to_request))
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some((provider, dim, positions)) = remote_batch {
            for pos in positions {
                provider.request_section(&dim, LodSectionPos::from(pos), 0);
            }
        } else {
            network::poll_remote_sections(host);
        }
    }

    fn render_fog(&self, host: &dyn ClientHost, event: &mut RenderFogEvent<'_>) {
        if !host.on(MOD_ID) {
            return;
        }
        let rt = runtime();
        let mut st = rt.inner.lock().unwrap();
        st.fog_color = event.fog.color;
        st.fog_type = match event.fog_type {
            FogType::Atmospheric => 0,
            FogType::Water => 1,
            FogType::Lava => 2,
            FogType::PowderSnow => 3,
        };
        st.env_fog_start = event.fog.environmental_start;
        st.env_fog_end = event.fog.environmental_end;
        if event.fog.render_distance_end.is_finite() && event.fog.render_distance_end > 16.0 {
            st.vanilla_far_blocks = event.fog.render_distance_end;
        }
        // Push vanilla's atmospheric render-distance fog out past vanilla's clip so the horizon
        // reveals Distant Sunrises LOD terrain (DH applies its own atmospheric/height fog in the
        // composite pass). Underwater/lava/powder-snow fog stays vanilla-exact.
        if st.config.enabled
            && st.config.disable_vanilla_fog
            && event.fog_type == FogType::Atmospheric
        {
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
    let collect_deltas = host.server_is_published();
    let mut partial_deltas: HashMap<SectionPos, network::SectionColumnDelta> = HashMap::new();
    if !chunks_to_sample.is_empty() {
        partial_deltas = sample_loaded_chunks(
            host,
            dim,
            &chunks_to_sample,
            &mut dirty_sections,
            collect_deltas,
        );
    }

    // 2. If time remains and distant generation is on, sample distant terrain around connected
    //    players in `dim` (round-robin, supporting dedicated & LAN servers) or the local camera.
    let (distant_on, cam, cfg, active_dim, rr_cursor) = {
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
            st.server_player_round_robin.get(dim).copied().unwrap_or(0),
        )
    };

    if distant_on && host.server_has_time() {
        let dim_id = mod_api::Identifier::parse(dim).ok();
        let player_centers: Vec<(u128, [f64; 3])> = host
            .players()
            .server_players()
            .into_iter()
            .filter(|&p| dim_id.is_some() && p.dimension(host) == dim_id)
            .filter_map(|p| EntityRef::from(p).position(host).map(|pos| (p.uuid(), pos)))
            .collect();
        if !player_centers.is_empty() {
            // Evict cache entries for players no longer in this dimension.
            {
                let mut st = rt.inner.lock().unwrap();
                st.desired_cache.retain(|(d, key), _| {
                    d != dim || *key == 0 || player_centers.iter().any(|(u, _)| u == key)
                });
            }
            let n = player_centers.len();
            for offset in 0..n {
                let idx = (rr_cursor + offset) % n;
                let (player_key, center) = player_centers[idx];
                match sample_distant_sections(
                    host,
                    dim,
                    player_key,
                    center,
                    cfg,
                    &mut dirty_sections,
                ) {
                    SampleOutcome::Cooldown => break,
                    SampleOutcome::Sampled => {
                        let mut st = rt.inner.lock().unwrap();
                        st.server_player_round_robin
                            .insert(dim.to_owned(), (idx + 1) % n);
                        break;
                    }
                    SampleOutcome::Exhausted => {
                        let mut st = rt.inner.lock().unwrap();
                        st.server_player_round_robin
                            .insert(dim.to_owned(), (idx + 1) % n);
                    }
                }
            }
        } else if dim == active_dim
            && let Some(center) = cam
        {
            let _ = sample_distant_sections(host, dim, 0, center, cfg, &mut dirty_sections);
        }
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
                if sec.dirty_save
                    && sec.populated > 0
                    && let Some(ref dir) = save_root
                {
                    sec.dirty_save = false;
                    rt.io.request_save(dir.clone(), dim.to_owned(), sec.clone());
                }
            }
        }
    }

    // 4. On published LAN/dedicated servers, stream incremental column deltas to connected remote
    //    DH clients in this dimension.
    if !partial_deltas.is_empty() {
        let _ = network::broadcast_partial_updates(host, dim, &partial_deltas);
    }
}

fn sample_loaded_chunks(
    host: &dyn ServerHost,
    dim: &str,
    chunks: &[[i32; 2]],
    dirty_sections: &mut HashSet<SectionPos>,
    collect_deltas: bool,
) -> HashMap<SectionPos, network::SectionColumnDelta> {
    let mut deltas: HashMap<SectionPos, network::SectionColumnDelta> = HashMap::new();
    let level_ref = host.level(dim);
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
        return deltas;
    };

    let span = (LOADED_SCAN_BELOW + LOADED_SCAN_ABOVE) as usize;
    if span == 0 || read.bases.len() != columns.len() {
        return deltas;
    }

    // Resolve palette block IDs once per batch.
    let palette_ids: Vec<&str> = read
        .palette
        .iter()
        .map(|state| state.block().id(host).unwrap_or("minecraft:stone"))
        .collect();

    // Sample biomes on the vanilla 4x4 quart grid per chunk (16 calls/chunk vs 256) so grass,
    // foliage, and water tints on loaded chunks match surrounding distant terrain seamlessly.
    let mut chunk_biomes: Vec<[String; 16]> = Vec::with_capacity(chunks.len());
    for (ci, &[cx, cz]) in chunks.iter().enumerate() {
        let bx0 = cx << 4;
        let bz0 = cz << 4;
        let mut grid: [String; 16] = std::array::from_fn(|_| "minecraft:plains".to_owned());
        if let Some(level) = level_ref {
            for qz in 0..4 {
                for qx in 0..4 {
                    let lx = qx * 4 + 2;
                    let lz = qz * 4 + 2;
                    let col_idx = ci * 256 + lz * 16 + lx;
                    let y = read.bases.get(col_idx).and_then(|b| *b).unwrap_or(64);
                    if let Some(b) = level.biome_at(host, [bx0 + lx as i32, y, bz0 + lz as i32]) {
                        grid[qz * 4 + qx] = b;
                    }
                }
            }
        }
        chunk_biomes.push(grid);
    }

    let mut chunk_beacons: Vec<Vec<BeaconBeam>> = vec![Vec::new(); chunks.len()];

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

        let ci = col_idx / 256;
        let local_idx = col_idx % 256;
        let qx = (local_idx % 16) / 4;
        let qz = (local_idx / 16) / 4;
        let biome_id = chunk_biomes
            .get(ci)
            .map(|g| g[qz * 4 + qx].as_str())
            .unwrap_or("minecraft:plains");

        let mut fluid_top: Option<i32> = None;
        let mut solid_top: Option<(i32, &str)> = None;
        let mut beacon_y: Option<i32> = None;
        let mut beacon_tint: Option<[u8; 3]> = None;

        // Scan top-down through the sampled band for surface + any sky-visible beacon & stained glass tint.
        for offset in (0..span).rev() {
            let pal_idx = col_blocks[offset] as usize;
            let block_id = palette_ids.get(pal_idx).copied().unwrap_or("minecraft:air");
            let y = base_y - (LOADED_SCAN_BELOW as i32) + (offset as i32) + 1;
            if block_id == "minecraft:beacon" {
                beacon_y = Some(y);
            }
            if is_water_block(block_id) {
                if fluid_top.is_none() {
                    fluid_top = Some(y);
                }
                continue;
            }
            if is_skippable_for_lod(block_id) {
                continue;
            }
            if solid_top.is_none() {
                solid_top = Some((y, block_id));
                if beacon_y.is_some() {
                    break;
                }
            }
        }

        if let Some(by) = beacon_y {
            // Walk bottom-up above the beacon to apply vanilla stained-glass tint averaging.
            for (offset, &pal_u16) in col_blocks.iter().enumerate() {
                let y = base_y - (LOADED_SCAN_BELOW as i32) + (offset as i32) + 1;
                if y <= by {
                    continue;
                }
                let pal_idx = pal_u16 as usize;
                let block_id = palette_ids.get(pal_idx).copied().unwrap_or("minecraft:air");
                if let Some(glass_rgb) = stained_glass_tint(block_id) {
                    beacon_tint = Some(blend_beacon_tint(beacon_tint, glass_rgb));
                }
            }
            if let Some(list) = chunk_beacons.get_mut(ci) {
                list.push(BeaconBeam {
                    pos: [bx, by, bz],
                    color: beacon_tint.unwrap_or([255, 255, 255]),
                });
            }
        }

        let (top_y, block_id) = match solid_top {
            Some(v) => v,
            None => match fluid_top {
                Some(fy) => (fy - 4, "minecraft:sand"),
                None => continue,
            },
        };

        let solid_rgb = block_color(block_id, biome_id, water_hint);
        let fluid_rgb = biome_water_color(biome_id, water_hint);
        let lod_col = LodColumn::from_solid_and_fluid(
            top_y,
            top_y - 8,
            solid_rgb,
            fluid_top,
            fluid_rgb,
            15,
            0,
            SOURCE_LOADED_CHUNK,
        )
        .with_material(classify_dh_material(block_id));

        let sec_pos = SectionPos::from_block(0, bx, bz);
        let (lx, lz) = sec_pos.local_cell(bx, bz);
        let sec = store.get_or_create_mut(sec_pos);
        if sec.set_cell(lx, lz, lod_col) {
            dirty_sections.insert(sec_pos);
            if collect_deltas {
                let entry = deltas.entry(sec_pos).or_default();
                entry.revision = sec.revision;
                entry.columns.push((lx as u8, lz as u8, lod_col));
            }
        }
    }

    let tracker = st.beacons.entry(dim.to_owned()).or_default();
    for (ci, &chunk) in chunks.iter().enumerate() {
        if let Some(found) = chunk_beacons.get(ci) {
            tracker.replace_chunk_beacons(chunk, found);
        }
    }
    deltas
}

enum SampleOutcome {
    /// Dimension cooldown has not elapsed yet; do not advance round-robin.
    Cooldown,
    /// Player's desired-section list has no unsampled sections left; try the next player.
    Exhausted,
    /// Ran at least one sampling batch (or hit a rate-limited terminal path); advance round-robin.
    Sampled,
}

/// Advances unthrottled async disk-load checks and, when this dimension's cooldown has elapsed,
/// runs distant-sampling batches up to `DISTANT_SAMPLE_BUDGET` (4 ms/tick).
fn sample_distant_sections(
    host: &dyn ServerHost,
    dim: &str,
    player_key: u128,
    cam: [f64; 3],
    cfg: DhConfig,
    dirty_sections: &mut HashSet<SectionPos>,
) -> SampleOutcome {
    let rt = runtime();
    let cam_block = [cam[0].floor() as i32, cam[2].floor() as i32];
    let cache_key = (dim.to_owned(), player_key);

    // Refresh the cached desired-section list for `(dim, player_key)` when config changes or the
    // camera moves at least `SECTION_SIZE` (64) blocks Manhattan distance (matching DH's
    // `QuadTree.setCenterBlockPos` using `FullDataSourceV2.WIDTH` = 64 for
    // `blockDistanceForNodeClearing`). Within that 64-block window, ordering and detail splits
    // may lag by up to 63 blocks. Async disk loads always run unthrottled.
    {
        let mut st = rt.inner.lock().unwrap();
        let needs_rebuild = match st.desired_cache.get(&cache_key) {
            Some(c) => {
                c.config != cfg
                    || (c.center[0] - cam_block[0]).abs() + (c.center[1] - cam_block[1]).abs()
                        >= SECTION_SIZE as i32
            }
            None => true,
        };
        if needs_rebuild {
            let list = DimensionLodStore::desired_sections_coarse_to_fine(cam[0], cam[2], cfg);
            st.desired_cache.insert(
                cache_key.clone(),
                render_bridge::DesiredCache {
                    center: cam_block,
                    config: cfg,
                    list,
                    disk_cursor: 0,
                    sample_cursor: 0,
                },
            );
        }

        let save_root = st.world_dir.clone();
        let now = std::time::Instant::now();
        let sample_due = st
            .next_distant_sample
            .get(dim)
            .is_none_or(|&next| now >= next);

        let st_ref = &mut *st;
        let cache = st_ref.desired_cache.get_mut(&cache_key).unwrap();
        let store = st_ref.dimensions.entry(dim.to_owned()).or_default();

        // Trigger background disk loads for up to 8 unseen sections per tick (unthrottled).
        if let Some(ref dir) = save_root {
            let mut loads = 0;
            while cache.disk_cursor < cache.list.len() && loads < 8 {
                let pos = cache.list[cache.disk_cursor];
                cache.disk_cursor += 1;
                if store.disk_checked.insert(pos) {
                    rt.io.request_load(dir.clone(), dim.to_owned(), pos);
                    loads += 1;
                }
            }
        }

        if !sample_due {
            return SampleOutcome::Cooldown;
        }
    }

    let budget_start = std::time::Instant::now();
    let mut ran_any = false;

    loop {
        let (target_section, custom_gen, builtin_disabled) = {
            let mut st = rt.inner.lock().unwrap();
            let custom_gen = st.world_generators.get(dim).cloned();
            let builtin_disabled = st.disabled_builtin_gen.contains(dim);
            let st_ref = &mut *st;
            let Some(cache) = st_ref.desired_cache.get_mut(&cache_key) else {
                break;
            };
            let store = st_ref.dimensions.entry(dim.to_owned()).or_default();

            let mut picked = None;
            while cache.sample_cursor < cache.list.len() {
                let pos = cache.list[cache.sample_cursor];
                if store.sampled_distant_sections.contains(&pos) {
                    cache.sample_cursor += 1;
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
                    cache.sample_cursor += 1;
                } else {
                    picked = Some((pos, missing));
                    break;
                }
            }
            (picked, custom_gen, builtin_disabled)
        };

        let Some((pos, cells)) = target_section else {
            break;
        };

        // If a mod registered a custom `LodWorldGenerator` covering this detail level, delegate
        // generation to it instead of calling built-in `terrain_surface`.
        if let Some(generator) = custom_gen
            && generator.detail_range().contains(&pos.detail)
        {
            generator.pre_task_start();
            let req = LodGenRequest {
                dimension: dim.to_owned(),
                pos: LodSectionPos::from(pos),
                target_step: LodGenStep::Surface,
            };
            let mut produced = 0usize;
            generator.generate(&req, &mut |out| {
                if let Ok(n) = ingest_gen_output(dim, out) {
                    produced = produced.saturating_add(n);
                }
            });
            let mut st = rt.inner.lock().unwrap();
            let store = st.dimensions.entry(dim.to_owned()).or_default();
            store.sampled_distant_sections.insert(pos);
            store.generated_columns = store.generated_columns.saturating_add(produced as u64);
            ran_any = true;
            if budget_start.elapsed() >= DISTANT_SAMPLE_BUDGET {
                break;
            }
            continue;
        }

        if builtin_disabled {
            let mut st = rt.inner.lock().unwrap();
            st.next_distant_sample.insert(
                dim.to_owned(),
                std::time::Instant::now() + DISTANT_SAMPLE_MAX_COOLDOWN,
            );
            return if ran_any {
                SampleOutcome::Sampled
            } else {
                SampleOutcome::Cooldown
            };
        }

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
            let cooldown = (budget_start.elapsed() * DISTANT_SAMPLE_COOLDOWN_MULT)
                .min(DISTANT_SAMPLE_MAX_COOLDOWN);
            rt.inner
                .lock()
                .unwrap()
                .next_distant_sample
                .insert(dim.to_owned(), std::time::Instant::now() + cooldown);
            return if ran_any {
                SampleOutcome::Sampled
            } else {
                SampleOutcome::Cooldown
            };
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
                )
                .with_material(classify_dh_material(block_id));
                if sec.set_cell(cx, cz, lod_col) {
                    updated_any = true;
                }
                non_empty_count += 1;
            }
        }

        store.generated_columns = store.generated_columns.saturating_add(non_empty_count);
        if !updated_any {
            // Void / partial-None section with no further fillable cells: mark sampled so
            // `sample_cursor` advances past it instead of re-picking it forever.
            store.sampled_distant_sections.insert(pos);
        }
        if updated_any {
            dirty_sections.insert(pos);
        }
        ran_any = true;

        if budget_start.elapsed() >= DISTANT_SAMPLE_BUDGET {
            break;
        }
    }

    if ran_any {
        let elapsed = budget_start.elapsed();
        if elapsed > DISTANT_SAMPLE_BUDGET {
            let cooldown =
                (elapsed * DISTANT_SAMPLE_COOLDOWN_MULT).min(DISTANT_SAMPLE_MAX_COOLDOWN);
            rt.inner
                .lock()
                .unwrap()
                .next_distant_sample
                .insert(dim.to_owned(), std::time::Instant::now() + cooldown);
        }
        SampleOutcome::Sampled
    } else {
        SampleOutcome::Exhausted
    }
}

fn flush_dimension_dirty(world_dir: Option<std::path::PathBuf>, dim: &str) {
    let rt = runtime();
    {
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
    rt.io.flush();
}

fn flush_dirty_sections(world_dir: Option<std::path::PathBuf>) {
    let rt = runtime();
    {
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
    rt.io.flush();
}
