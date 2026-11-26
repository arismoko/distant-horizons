//! Distant Sunrises native mod for Sunrise.
//!
//! Architecture (Voxy-style 32³ octree renderer on Sunrise's general mod API, zero `[native.*]`):
//! - The server ingests loaded chunks (`ChunkSnapshot`) and distant ungenerated terrain
//!   (`DetachedChunkGenerator::sample_surface_columns`) during `level_tick_post` under
//!   `host.server_has_time()` budgeting into 32³ octree sections (`octree`, `pipeline`), and
//!   persists them (`DHLOD003` / `.dhoct`, `storage`) on a background IO worker.
//! - `pipeline` greedy-meshes each section into 8-byte packed quads (`mesh`) using the baked
//!   per-block models and 16x16 mipmapped face atlas (`bakery`) and hands them to the GPU arena.
//! - `network::octree_sync` streams sections to remote clients over one raw play channel.
//! - The client tracks camera, day-time lighting, settings (`Options > Mods`), fog colors
//!   (`render_fog`) and F3 telemetry. `gpu_arena` + GLSL compute (Hi-Z, command generation) and
//!   GPU indirect draws render the LODs; shader packs draw them through pack-linked pipelines.

pub mod api;
pub mod bakery;
pub mod colors;
pub mod commands;
pub mod config;
pub mod dh_v2;
pub mod gen_worker;
pub mod gpu_arena;
pub mod lod;
pub mod mesh;
pub mod network;
pub mod octree;
pub mod octree_ingest;
pub mod pipeline;
pub mod render_bridge;
pub mod resource_colors;
pub mod shader_pack;
pub mod stage3_camera;
pub mod storage;

use crate::api::{LodGenRequest, LodGenStep, LodSectionPos, ingest_gen_output};
use crate::colors::{
    block_color, classify_dh_material, is_skippable_for_lod, is_water_block, water_surface_color,
};
use crate::commands::DhKeybinds;
use crate::config::{DhConfig, MOD_ID};
use crate::lod::{
    DimensionLodStore, LodColumn, SECTION_SIZE, SOURCE_DISTANT_SURFACE, SOURCE_LOADED_CHUNK,
    SectionPos,
};
use crate::render_bridge::{runtime, stats, sun_brightness_from_day_time_and_weather};
use mod_api::client::fog::FogType;
use mod_api::{
    self, BlockChangedEvent, ChunkLoadEvent, ClientBlockChangedEvent, ClientChunkLoadEvent,
    ClientChunkUnloadEvent, ClientHost, ClientLevelLoadEvent, ClientLevelUnloadEvent,
    ClientReloadHost, ClientSetupContext, ClientSetupEvent, ClientTickPreEvent, DebugEntry,
    DebugGroup, EntityRef, Heightmap, Hooks, Host, HostSide, LevelDimension, LevelHost,
    LevelLoadEvent, LevelTickPostEvent, LevelUnloadEvent, RegisterClientCommandsEvent,
    RegisterDebugEntriesEvent, Registrar, RegistrationContext, RenderFogEvent, ServerHost,
    ServerStartedEvent, ServerStoppingEvent, SettingChangedEvent,
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
/// Per-tick wall-time budget for custom `LodWorldGenerator` callbacks on the server thread.
const DISTANT_SAMPLE_BUDGET: std::time::Duration = std::time::Duration::from_millis(4);
/// Cooldown applied when built-in generation is disabled or no detached generator is available.
const DISTANT_SAMPLE_MAX_COOLDOWN: std::time::Duration = std::time::Duration::from_millis(250);

#[derive(Default)]
struct DistantHorizonsMod {
    keybinds: Mutex<DhKeybinds>,
}

#[mod_api::sunrise_mod]
impl Hooks for DistantHorizonsMod {
    fn register(&self, registrar: &mut dyn Registrar, _cx: &RegistrationContext) {
        DhConfig::register_settings(registrar);
        commands::register_key_mappings(registrar);
        network::octree_sync::register_payloads(registrar);
    }

    fn register_render_pipelines(
        &self,
        _context: &RegistrationContext,
        event: &mut mod_api::RegisterRenderPipelinesEvent<'_>,
    ) {
        render_bridge::register_pipelines(event);
    }

    fn prepare_render_buffers(
        &self,
        host: &dyn ClientHost,
        event: &mut mod_api::PrepareRenderBuffersEvent<'_>,
    ) {
        if !host.on(MOD_ID) {
            return;
        }
        let camera = stage3_camera::CameraMatrices::from_camera(event.view.camera);
        render_bridge::begin_frame(camera.position);
        render_bridge::prepare_draws(host, &camera, event.context);
    }

    fn frame_graph_setup(
        &self,
        host: &dyn ClientHost,
        event: &mut mod_api::FrameGraphSetupEvent<'_>,
    ) {
        if !host.on(MOD_ID) {
            return;
        }
        let camera = stage3_camera::CameraMatrices::from_camera(event.view.camera);
        render_bridge::setup_compute(host, &camera, event.graph);
    }

    fn render_level_stage(
        &self,
        host: &dyn ClientHost,
        event: &mut mod_api::RenderLevelStageEvent<'_>,
    ) {
        if host.on(MOD_ID) {
            render_bridge::draw_stage(event.stage, event.context);
        }
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
        pipeline::invalidate_models();
    }

    fn register_client_commands(
        &self,
        _host: &mut dyn ClientHost,
        event: &mut RegisterClientCommandsEvent<'_>,
    ) {
        commands::register_client_commands(event.commands);
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
                        "DH: sec={} drawn={} quads={} gen={} saved={}",
                        s.active_sections,
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

    fn setting_changed(&self, host: &dyn Host, _event: &mut SettingChangedEvent<'_>) {
        sync_client_config(host);
    }

    fn client_level_load(&self, host: &mut dyn ClientHost, event: &mut ClientLevelLoadEvent<'_>) {
        sync_client_config(host);
        let mut st = runtime().inner.lock().unwrap();
        st.active_dimension = event.dimension.to_owned();
        st.client_loaded_chunks.clear();
    }

    fn client_level_unload(
        &self,
        _host: &mut dyn ClientHost,
        event: &mut ClientLevelUnloadEvent<'_>,
    ) {
        let _ = render_bridge::clear_dimension(event.dimension);
        network::octree_sync::client_reset();
        let mut st = runtime().inner.lock().unwrap();
        st.remote_requested.remove(event.dimension);
        st.client_loaded_chunks.clear();
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
        if host.on(MOD_ID) {
            let dim = runtime().inner.lock().unwrap().active_dimension.clone();
            pipeline::client_tick(host, &dim);
            render_bridge::upload_pending(
                host,
                gpu_arena::ArenaConfig {
                    max_sections: gpu_arena::VOXY_MAX_SECTIONS,
                    geometry_bytes: gpu_arena::VOXY_GEOMETRY_BYTES,
                },
            );
        }
        let rt = runtime();
        let cfg = DhConfig::read_from_host(host);
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
            st.config = DhConfig::read_from_host(host);
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
            poll_remote_octree(host);
        }
    }

    fn render_fog(&self, host: &dyn ClientHost, event: &mut RenderFogEvent<'_>) {
        if !host.on(MOD_ID) {
            return;
        }
        let rt = runtime();
        let mut st = rt.inner.lock().unwrap();
        st.fog_color = event.fog.color;
        // Vanilla render distance in blocks (matches DH `MinecraftClientWrapper.getRenderDistance()` * 16).
        // Do NOT read `event.fog.render_distance_end`, which becomes `2e9` when Sunrise fog is disabled
        // and would push `clip_dist` to ~1e9 (discarding all DS chunks).
        let rd_chunks = host.graphics_options().render_distance.max(2);
        st.vanilla_far_blocks = (rd_chunks as f32) * 16.0;
        // Disable vanilla atmospheric environmental + render-distance fog so foreground chunks
        // are not hazed out at the vanilla border before Distant Sunrises draws far fog
        // (matches DH `MixinFogRenderer.java:144-148` and Sunrise `fog.rs:447-448`).
        // Underwater/lava/powder-snow fog stays vanilla-exact.
        if st.config.enabled
            && st.config.disable_vanilla_fog
            && event.fog_type == FogType::Atmospheric
        {
            event.fog.environmental_start = 1e9;
            event.fog.environmental_end = 2e9;
            event.fog.render_distance_start = 1e9;
            event.fog.render_distance_end = 2e9;
        }
    }

    fn server_started(&self, host: &mut dyn ServerHost, _event: &mut ServerStartedEvent) {
        let world_dir = host.world_dir();
        let rt = runtime();
        rt.gen_pool.shutdown_and_join();
        pipeline::reset_server_intake();
        rt.gen_pool.ensure_started();
        let mut st = rt.inner.lock().unwrap();
        st.world_dir = world_dir;
        pipeline::install_sync_source();
    }

    fn server_stopping(&self, host: &mut dyn ServerHost, _event: &mut ServerStoppingEvent) {
        let rt = runtime();
        let world_dir = host
            .world_dir()
            .or_else(|| rt.inner.lock().unwrap().world_dir.clone());
        rt.gen_pool.shutdown_and_join();
        pipeline::finish_server_io(host, &rt.io, world_dir.as_deref(), None);
    }

    fn level_load(&self, host: &mut dyn ServerHost, event: &mut LevelLoadEvent<'_>) {
        let world_dir = host.world_dir();
        let rt = runtime();
        rt.gen_pool.allow_dimension(event.dimension);
        let mut st = rt.inner.lock().unwrap();
        if st.world_dir.is_none() {
            st.world_dir = world_dir;
        }
        st.dimensions.entry(event.dimension.to_owned()).or_default();
    }

    fn level_unload(&self, host: &mut dyn ServerHost, event: &mut LevelUnloadEvent<'_>) {
        let rt = runtime();
        let world_dir = host
            .world_dir()
            .or_else(|| rt.inner.lock().unwrap().world_dir.clone());
        rt.gen_pool.reset_dimension(event.dimension);
        pipeline::finish_server_io(host, &rt.io, world_dir.as_deref(), Some(event.dimension));
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
        if runtime().gen_pool.dimension_allowed(dim) {
            tick_server_dimension(host, dim);
        }
    }
}

/// Client side of `octree_sync` on a remote server: send our config once, then ask for the
/// sections around the camera and adopt what comes back.
fn poll_remote_octree(host: &mut dyn ClientHost) {
    let is_remote = host
        .client_session()
        .client_server()
        .is_some_and(|s| !s.integrated_local);
    if !is_remote || !host.has_channel(&network::octree_sync::octree_sync_channel_id()) {
        return;
    }
    let (enabled, dim, camera) = {
        let st = runtime().inner.lock().unwrap();
        (
            st.config.enabled && st.has_camera,
            st.active_dimension.clone(),
            st.camera_pos,
        )
    };
    if !enabled {
        return;
    }
    network::octree_sync::client_send_config_once(host);
    pipeline::client_sync(
        host,
        &dim,
        camera,
        STORED_RADIUS_SECTIONS,
        SYNC_REQUESTS_PER_TICK,
    );
}

fn sync_client_config(host: &dyn Host) {
    let cfg = DhConfig::read_from_host(host);
    let rt = runtime();
    rt.enabled_flag.store(cfg.enabled, Ordering::Relaxed);
    rt.inner.lock().unwrap().config = cfg;
    rt.gen_pool.reconfigure(cfg.thread_preset);
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

/// Level-0 sections around a player asked of disk per tick budget / radius.
const STORED_RADIUS_SECTIONS: i32 = 8;
const STORED_REQUESTS_PER_TICK: usize = 64;
const SYNC_REQUESTS_PER_TICK: usize = 16;
const SAVES_PER_TICK: usize = 32;

fn tick_server_dimension(host: &mut dyn ServerHost, dim: &str) {
    let rt = runtime();
    let world_dir = host
        .world_dir()
        .or_else(|| rt.inner.lock().unwrap().world_dir.clone());
    pipeline::advance_intake(host, &rt.io, world_dir.as_deref(), dim);
    let intake_ready = !pipeline::has_pending_intake(dim);

    // 1. Sample a small batch of newly loaded / modified chunks via `ChunkSnapshot`.
    let chunks_to_sample: Vec<[i32; 2]> = {
        let mut st = rt.inner.lock().unwrap();
        if st.world_dir.is_none() && world_dir.is_some() {
            st.world_dir = world_dir.clone();
        }
        let mut batch = Vec::with_capacity(CHUNKS_PER_TICK);
        let st_ref = &mut *st;
        if intake_ready && let Some(queue) = st_ref.pending_chunks.get_mut(dim) {
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

    // 1b. Leave completed generation in the existing bounded pool while intake awaits disk.
    if intake_ready {
        drain_gen_results(host, dim, &mut dirty_sections);
    }
    if pipeline::has_pending_intake(dim) {
        pipeline::advance_intake(host, &rt.io, world_dir.as_deref(), dim);
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
                match sample_distant_sections(host, dim, player_key, center, cfg) {
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
            let _ = sample_distant_sections(host, dim, 0, center, cfg);
        }
    }

    // 3. Propagate dirty sections up the quadtree of the column store.
    if !dirty_sections.is_empty() {
        let mut st = rt.inner.lock().unwrap();
        let store = st.dimensions.entry(dim.to_owned()).or_default();
        for &pos in &dirty_sections {
            store.propagate_up(pos);
        }
    }

    // 4. Octree persistence and sync: adopt stored sections near the players, save changed ones,
    //    and serve connected clients.
    let players: Vec<_> = host
        .players()
        .server_players()
        .into_iter()
        .filter_map(|p| EntityRef::from(p).position(host).map(|pos| (p, pos)))
        .collect();
    pipeline::set_player_positions(
        players
            .iter()
            .map(|&(p, pos)| (p, (pos[0] as i32, pos[2] as i32)))
            .collect(),
    );
    if let Some(dir) = world_dir {
        let center = players.first().map(|&(_, pos)| pos).or(cam);
        if let Some(center) = center {
            pipeline::request_stored_sections(
                &rt.io,
                &dir,
                dim,
                center,
                STORED_RADIUS_SECTIONS,
                STORED_REQUESTS_PER_TICK,
            );
        }
        pipeline::flush_saves(&rt.io, &dir, SAVES_PER_TICK);
    }
    if dim == "minecraft:overworld" && host.server_is_published() {
        network::octree_sync::server_tick(host);
    }
}

fn sample_loaded_chunks(
    host: &dyn ServerHost,
    dim: &str,
    chunks: &[[i32; 2]],
    dirty_sections: &mut HashSet<SectionPos>,
) {
    let Some(level) = host.level(dim) else {
        return;
    };

    let snapshots: Vec<([i32; 2], Option<mod_api::ChunkSnapshot>)> = chunks
        .iter()
        .map(|&pos| {
            (
                pos,
                level.chunk_ref(host, pos).and_then(|c| c.snapshot(host)),
            )
        })
        .collect();
    for (chunk, snapshot) in &snapshots {
        if let Some(snapshot) = snapshot {
            pipeline::ingest_chunk(
                host,
                dim,
                *chunk,
                snapshot,
                octree_ingest::IngestSource::Loaded,
            );
        }
    }
    ingest_chunk_snapshots(host, dim, &snapshots, dirty_sections);
}

/// Converts chunk snapshots (live G1 or saved G3) into LOD columns and spans.
fn ingest_chunk_snapshots(
    host: &dyn ServerHost,
    dim: &str,
    snapshots: &[([i32; 2], Option<mod_api::ChunkSnapshot>)],
    dirty_sections: &mut HashSet<SectionPos>,
) {
    let mut block_id_cache: HashMap<mod_api::BlockState, &str> = HashMap::new();

    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let max_spans = st.config.vertical_quality.clamp(1, 8) as usize;
    let store = st.dimensions.entry(dim.to_owned()).or_default();

    for ([cx, cz], snap_opt) in snapshots {
        let Some(snap) = snap_opt else {
            continue;
        };
        let bx0 = cx << 4;
        let bz0 = cz << 4;

        for lz in 0..16u8 {
            for lx in 0..16u8 {
                let Some(base_y) = snap.heightmap(Heightmap::WorldSurface, lx, lz) else {
                    continue;
                };
                let bx = bx0 + i32::from(lx);
                let bz = bz0 + i32::from(lz);
                let ocean_y = snap
                    .heightmap(Heightmap::OceanFloor, lx, lz)
                    .unwrap_or(base_y);
                let y_min = (base_y - (LOADED_SCAN_BELOW as i32) + 1)
                    .min(ocean_y - 2)
                    .max(snap.min_y());
                let y_max = base_y + (LOADED_SCAN_ABOVE as i32);

                let mut fluid_top: Option<i32> = None;
                // Contiguous vertical solid spans `(top_y, lowest_solid_y, top_block_id)` for
                // tree canopies, overhangs, and ground underneath (DH `EDhApiVerticalQuality`).
                let mut solid_spans: Vec<(i32, i32, &str)> = Vec::with_capacity(max_spans);
                let mut in_solid_span = false;

                // Scan top-down through the sampled band for surface + vertical spans.
                for y in (y_min..=y_max).rev() {
                    let block_id = match snap.block_state(lx, y, lz) {
                        Some(state) => *block_id_cache
                            .entry(state)
                            .or_insert_with(|| state.block().id(host).unwrap_or("minecraft:stone")),
                        None => "minecraft:air",
                    };
                    if is_water_block(block_id) {
                        in_solid_span = false;
                        if fluid_top.is_none() {
                            fluid_top = Some(y);
                        }
                        continue;
                    }
                    if is_skippable_for_lod(block_id) {
                        in_solid_span = false;
                        continue;
                    }
                    if in_solid_span {
                        if let Some(last) = solid_spans.last_mut() {
                            last.1 = y;
                        }
                    } else if solid_spans.len() < max_spans {
                        solid_spans.push((y, y, block_id));
                        in_solid_span = true;
                    }
                }

                let (top_y, primary_bot_y, block_id) = match solid_spans.first().copied() {
                    Some((ty, by, id)) => {
                        let bot = if solid_spans.len() > 1 {
                            by - 1
                        } else {
                            (by - 1).min(ty - 8)
                        };
                        (ty, bot, id)
                    }
                    None => match fluid_top {
                        Some(fy) => (fy - 4, fy - 12, "minecraft:sand"),
                        None => continue,
                    },
                };

                let biome_id = snap
                    .biome(lx, top_y, lz)
                    .map(|k| k.identifier().as_str())
                    .unwrap_or("minecraft:plains");
                let fluid_biome_id = fluid_top
                    .and_then(|fy| snap.biome(lx, fy, lz))
                    .map(|k| k.identifier().as_str())
                    .unwrap_or(biome_id);
                let solid_rgb = block_color(block_id, biome_id, 0);
                let fluid_rgb = water_surface_color(fluid_biome_id, 0);
                let (solid_sky, solid_blk) = if snap.has_light() {
                    snap.light(lx, top_y + 1, lz).unwrap_or((15, 0))
                } else {
                    let sky = match fluid_top {
                        Some(fy) if fy > top_y => 15u8.saturating_sub((fy - top_y).min(15) as u8),
                        _ => 15,
                    };
                    (sky, 0)
                };
                let (fluid_sky, fluid_blk) = match fluid_top {
                    Some(fy) if snap.has_light() => snap.light(lx, fy + 1, lz).unwrap_or((15, 0)),
                    _ => (15, 0),
                };
                let lod_col = LodColumn::from_solid_and_fluid(
                    top_y,
                    primary_bot_y,
                    solid_rgb,
                    fluid_top,
                    fluid_rgb,
                    solid_sky,
                    solid_blk,
                    SOURCE_LOADED_CHUNK,
                )
                .with_fluid_light(fluid_sky, fluid_blk)
                .with_material(classify_dh_material(block_id));

                let mut extra_spans = Vec::new();
                for &(s_top, s_bot, s_block) in solid_spans.iter().skip(1) {
                    let s_biome = snap
                        .biome(lx, s_top, lz)
                        .map(|k| k.identifier().as_str())
                        .unwrap_or(biome_id);
                    let s_rgb = block_color(s_block, s_biome, 0);
                    let (s_sky, s_blk) = if snap.has_light() {
                        snap.light(lx, s_top + 1, lz).unwrap_or((12, 0))
                    } else {
                        (12, 0)
                    };
                    let s_bot_clamped = (s_bot - 1).clamp(i16::MIN as i32 + 1, s_top - 1) as i16;
                    extra_spans.push(crate::lod::ExtraSpan {
                        top_y: s_top.clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16,
                        bottom_y: s_bot_clamped,
                        rgba: [
                            s_rgb[0],
                            s_rgb[1],
                            s_rgb[2],
                            LodColumn::pack_light(s_sky, s_blk),
                        ],
                        is_fluid: false,
                        material_id: classify_dh_material(s_block),
                    });
                }

                let sec_pos = SectionPos::from_block(0, bx, bz);
                let (cell_x, cell_z) = sec_pos.local_cell(bx, bz);
                let sec = store.get_or_create_mut(sec_pos);
                let changed = if extra_spans.is_empty() {
                    sec.set_cell(cell_x, cell_z, lod_col)
                } else {
                    sec.set_cell_with_spans(cell_x, cell_z, lod_col, extra_spans)
                };
                if changed {
                    dirty_sections.insert(sec_pos);
                }
            }
        }
    }
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
    let mut detached_gen: Option<Option<mod_api::DetachedChunkGenerator>> = None;

    loop {
        if !rt.gen_pool.has_capacity() {
            break;
        }

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
            let mut scan = cache.sample_cursor;
            while scan < cache.list.len() {
                let pos = cache.list[scan];
                if store.sampled_distant_sections.contains(&pos) {
                    if scan == cache.sample_cursor {
                        cache.sample_cursor += 1;
                    }
                    scan += 1;
                    continue;
                }
                if rt.gen_pool.is_in_flight(dim, pos) {
                    scan += 1;
                    continue;
                }
                let sec = store.sections.get(&pos);
                let mut missing = Vec::new();
                for cz in 0..SECTION_SIZE {
                    for cx in 0..SECTION_SIZE {
                        let is_empty = sec.map(|s| s.get(cx, cz).is_empty()).unwrap_or(true);
                        if is_empty {
                            missing.push((cx, cz));
                        }
                    }
                }
                if missing.is_empty() {
                    store.sampled_distant_sections.insert(pos);
                    if scan == cache.sample_cursor {
                        cache.sample_cursor += 1;
                    }
                    scan += 1;
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
        // generation to it instead of calling built-in `DetachedChunkGenerator`.
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

        let generator = detached_gen
            .get_or_insert_with(|| host.level(dim).and_then(|l| l.detached_generator(host)));
        let Some(gen_handle) = generator.clone() else {
            rt.inner.lock().unwrap().next_distant_sample.insert(
                dim.to_owned(),
                std::time::Instant::now() + DISTANT_SAMPLE_MAX_COOLDOWN,
            );
            return if ran_any {
                SampleOutcome::Sampled
            } else {
                SampleOutcome::Cooldown
            };
        };

        let saved_reader = host.level(dim).and_then(|l| l.saved_chunk_reader(host));
        let feature_stage = match cfg.distant_generation_mode {
            2 => Some(mod_api::GenerationStage::Features),
            3 => Some(mod_api::GenerationStage::Light),
            _ => None,
        };
        if rt.gen_pool.enqueue(
            dim.to_owned(),
            pos,
            gen_handle,
            saved_reader,
            feature_stage,
            cells,
        ) {
            ran_any = true;
        } else {
            break;
        }
    }

    if ran_any {
        SampleOutcome::Sampled
    } else {
        SampleOutcome::Exhausted
    }
}

/// Ingest completed off-thread `DetachedChunkGenerator` section results on the server thread.
fn drain_gen_results(host: &dyn ServerHost, dim: &str, dirty_sections: &mut HashSet<SectionPos>) {
    let rt = runtime();
    let mut results = rt.gen_pool.drain_for_dim(dim);
    if results.is_empty() {
        return;
    }
    // G3: chunks read from saved Anvil storage go through the same snapshot ingest as live chunks.
    let saved: Vec<([i32; 2], Option<mod_api::ChunkSnapshot>)> = results
        .iter_mut()
        .flat_map(|r| std::mem::take(&mut r.saved_chunks))
        .map(|(pos, snap)| (pos, Some(snap)))
        .collect();
    for (chunk, snapshot) in &saved {
        if let Some(snapshot) = snapshot {
            pipeline::ingest_chunk(
                host,
                dim,
                *chunk,
                snapshot,
                octree_ingest::IngestSource::Saved,
            );
        }
    }
    if !saved.is_empty() {
        ingest_chunk_snapshots(host, dim, &saved, dirty_sections);
    }

    // G4 full 3D chunks (trees & features in detail-0 near/mid ring).
    // Prepare their authored octants now; the cache-first batch retains no raw snapshots.
    let mut generated: Vec<([i32; 2], Option<mod_api::ChunkSnapshot>)> = Vec::new();
    for r in results.iter_mut() {
        let chunks = std::mem::take(&mut r.generated_chunks);
        if chunks.is_empty() {
            continue;
        }
        for (pos, snap) in &chunks {
            pipeline::ingest_chunk(host, dim, *pos, snap, octree_ingest::IngestSource::Distant);
        }
        generated.extend(chunks.into_iter().map(|(pos, snap)| (pos, Some(snap))));
    }
    if !generated.is_empty() {
        ingest_chunk_snapshots(host, dim, &generated, dirty_sections);
    }

    for res in &results {
        pipeline::ingest_distant(
            host,
            dim,
            res.pos.detail,
            res.pos.block_origin(),
            &res.cells,
            &res.samples,
        );
    }

    let mut block_id_cache: HashMap<mod_api::BlockKey, &str> = HashMap::new();
    let mut st = rt.inner.lock().unwrap();
    let mut had_error = false;
    {
        let store = st.dimensions.entry(dim.to_owned()).or_default();

        for res in results {
            let step = res.pos.cell_size();
            let skirt = (step * 4).clamp(8, 32);
            let mut updated_any = false;
            let mut non_empty_count = 0u64;

            {
                let sec = store.get_or_create_mut(res.pos);
                for (&(cx, cz), sample) in res.cells.iter().zip(res.samples.iter()) {
                    if sec.get(cx, cz).source == SOURCE_LOADED_CHUNK {
                        continue;
                    }
                    let Some(col) = sample else {
                        continue;
                    };
                    let block_id = *block_id_cache
                        .entry(col.block)
                        .or_insert_with(|| col.block.id(host).unwrap_or("minecraft:stone"));
                    let solid_rgb = block_color(block_id, &col.biome, 0);
                    let fluid_rgb = water_surface_color(&col.biome, 0);
                    // Water attenuates sky light by 1 per block of depth (vanilla `WaterFluid.getLightBlock = 1`).
                    let solid_sky = match col.water {
                        Some(wy) if wy > col.height => {
                            15u8.saturating_sub((wy - col.height).min(15) as u8)
                        }
                        _ => 15,
                    };
                    let lod_col = LodColumn::from_solid_and_fluid(
                        col.height,
                        col.height - skirt,
                        solid_rgb,
                        col.water,
                        fluid_rgb,
                        solid_sky,
                        0,
                        SOURCE_DISTANT_SURFACE,
                    )
                    .with_fluid_light(15, 0)
                    .with_material(classify_dh_material(block_id));
                    if sec.set_cell(cx, cz, lod_col) {
                        updated_any = true;
                    }
                    non_empty_count += 1;
                }
            }

            store.generated_columns = store.generated_columns.saturating_add(non_empty_count);
            match res.outcome {
                crate::gen_worker::GenOutcome::Complete => {
                    store.sampled_distant_sections.insert(res.pos);
                }
                crate::gen_worker::GenOutcome::Error => {
                    had_error = true;
                }
                crate::gen_worker::GenOutcome::Cancelled => {}
            }
            if updated_any {
                dirty_sections.insert(res.pos);
            }
        }
    }
    if had_error {
        st.next_distant_sample.insert(
            dim.to_owned(),
            std::time::Instant::now() + DISTANT_SAMPLE_MAX_COOLDOWN,
        );
    }
}
