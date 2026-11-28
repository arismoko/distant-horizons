//! Public mod-to-mod API for Distant Sunrises on Sunrise.
//!
//! External Sunrise mods depend on `distantsunrises` via `[dependencies]` in `sunrise-mod.toml`
//! (or optionally via `mod_api::if_installed!`) to:
//! - Override Distant Sunrises' background world generation for a dimension
//!   ([`register_world_generator`], [`LodWorldGenerator`]);
//! - Supply server-sent LOD sections directly from a network protocol or external cache
//!   ([`register_remote_lod_provider`], [`RemoteLodProvider`], [`submit_lod_section`],
//!   [`submit_lod_chunk`], [`submit_dh_v2_dto`], [`submit_dh_v2_bytes`]);
//! - Query or invalidate cached LOD data ([`get_column_data`], [`invalidate_lod_section`]).
//!
//! All submission, registration, and query functions are thread-safe and may be called from
//! network threads, background workers, or game ticks.

use crate::colors::{biome_water_color, block_color, is_skippable_for_lod, is_water_block};
use crate::config::DEFAULT_MAX_EXTRA_SPANS;
use crate::dh_v2::{
    DH_SECTION_MIN_DETAIL, DecodedDhV2Section, DhV2Error, DhV2FullDataDto, pack_dh_section_pos,
    unpack_dh_section_pos,
};
use crate::lod::{
    ExtraSpan, LodColumn, MAX_DETAIL, SECTION_CELLS, SECTION_SIZE, SOURCE_LOADED_CHUNK, SectionPos,
};
use crate::render_bridge::{DhState, runtime};
use std::collections::HashSet;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Semantic version of the public `distantsunrises::api` module.
pub const API_VERSION: (u32, u32, u32) = (1, 0, 0);

/// Returns the semantic version `(major, minor, patch)` of the Distant Sunrises Sunrise API.
pub const fn api_version() -> (u32, u32, u32) {
    API_VERSION
}

/// Returns `true` when Distant Sunrises LOD rendering is currently enabled.
pub fn is_active() -> bool {
    runtime().enabled_flag.load(Ordering::Relaxed)
}

/// Default build minimum Y for standard Minecraft dimensions when a caller does not know the
/// dimension's custom `min_y`.
pub fn default_level_min_y(dimension: &str) -> i32 {
    match dimension {
        "minecraft:the_nether" | "minecraft:the_end" => 0,
        _ => -64,
    }
}

/// Public coordinate key for a `64 x 64` cell LOD section at detail level `0..=MAX_DETAIL`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LodSectionPos {
    pub detail: u8,
    pub x: i32,
    pub z: i32,
}

impl LodSectionPos {
    pub const fn new(detail: u8, x: i32, z: i32) -> Self {
        Self { detail, x, z }
    }

    pub const fn from_block(detail: u8, block_x: i32, block_z: i32) -> Self {
        let p = SectionPos::from_block(detail, block_x, block_z);
        Self {
            detail: p.detail,
            x: p.x,
            z: p.z,
        }
    }

    pub const fn block_origin(self) -> (i32, i32) {
        SectionPos::new(self.detail, self.x, self.z).block_origin()
    }

    pub const fn cell_size(self) -> i32 {
        SectionPos::new(self.detail, self.x, self.z).cell_size()
    }

    pub const fn block_span(self) -> i32 {
        SectionPos::new(self.detail, self.x, self.z).block_span()
    }

    /// Packs into upstream DH's 64-bit `DhSectionPos` wire integer (`section_detail = detail + 6`).
    pub const fn to_dh_packed(self) -> i64 {
        pack_dh_section_pos(
            self.detail.saturating_add(DH_SECTION_MIN_DETAIL),
            self.x,
            self.z,
        )
    }

    /// Unpacks upstream DH's 64-bit `DhSectionPos` wire integer.
    pub const fn from_dh_packed(packed: i64) -> Option<Self> {
        let (sec_detail, x, z) = unpack_dh_section_pos(packed);
        if sec_detail < DH_SECTION_MIN_DETAIL {
            return None;
        }
        let detail = sec_detail - DH_SECTION_MIN_DETAIL;
        if detail > MAX_DETAIL {
            return None;
        }
        Some(Self { detail, x, z })
    }
}

impl From<SectionPos> for LodSectionPos {
    fn from(p: SectionPos) -> Self {
        Self {
            detail: p.detail,
            x: p.x,
            z: p.z,
        }
    }
}

impl From<LodSectionPos> for SectionPos {
    fn from(p: LodSectionPos) -> Self {
        Self {
            detail: p.detail,
            x: p.x,
            z: p.z,
        }
    }
}

/// Errors returned by `distantsunrises::api` submission calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DhApiError {
    InvalidDetail(u8),
    InvalidColumnCount { expected: usize, got: usize },
    Decode(DhV2Error),
}

impl std::fmt::Display for DhApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDetail(d) => write!(f, "LOD detail {d} exceeds MAX_DETAIL ({MAX_DETAIL})"),
            Self::InvalidColumnCount { expected, got } => {
                write!(f, "expected {expected} columns, got {got}")
            }
            Self::Decode(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for DhApiError {}

impl From<DhV2Error> for DhApiError {
    fn from(value: DhV2Error) -> Self {
        Self::Decode(value)
    }
}

/// Target generation stage requested from a [`LodWorldGenerator`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LodGenStep {
    Empty,
    StructureStart,
    Biomes,
    Noise,
    Surface,
    Features,
    Light,
}

/// One vertical terrain or fluid span in a column (ordered top-to-bottom).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LodDataPoint {
    /// Inclusive bottom block Y.
    pub bottom_y: i32,
    /// Exclusive top block Y (`top_y > bottom_y`).
    pub top_y: i32,
    /// Block identifier (e.g. `"minecraft:stone"`, `"minecraft:water"`).
    pub block_id: String,
    /// Biome identifier (e.g. `"minecraft:plains"`).
    pub biome_id: String,
    /// Optional explicit RGB override; when `None`, DH resolves the color from `block_id` and
    /// `biome_id`.
    pub rgb_override: Option<[u8; 3]>,
    pub sky_light: u8,
    pub block_light: u8,
    pub is_fluid: bool,
}

impl LodDataPoint {
    /// Convenience constructor for a solid terrain span.
    pub fn solid(
        bottom_y: i32,
        top_y: i32,
        block_id: impl Into<String>,
        biome_id: impl Into<String>,
        sky_light: u8,
        block_light: u8,
    ) -> Self {
        Self {
            bottom_y,
            top_y,
            block_id: block_id.into(),
            biome_id: biome_id.into(),
            rgb_override: None,
            sky_light: sky_light.min(15),
            block_light: block_light.min(15),
            is_fluid: false,
        }
    }

    /// Convenience constructor for a translucent fluid span (such as water).
    pub fn fluid(
        bottom_y: i32,
        top_y: i32,
        block_id: impl Into<String>,
        biome_id: impl Into<String>,
        sky_light: u8,
        block_light: u8,
    ) -> Self {
        Self {
            bottom_y,
            top_y,
            block_id: block_id.into(),
            biome_id: biome_id.into(),
            rgb_override: None,
            sky_light: sky_light.min(15),
            block_light: block_light.min(15),
            is_fluid: true,
        }
    }

    pub fn with_rgb(mut self, rgb: [u8; 3]) -> Self {
        self.rgb_override = Some(rgb);
        self
    }
}

/// Full `16 x 16` chunk of LOD datapoints at detail 0 (`columns[lz * 16 + lx]`).
#[derive(Clone, Debug)]
pub struct LodChunk {
    pub chunk_x: i32,
    pub chunk_z: i32,
    /// 256 columns indexed by `lz * 16 + lx`, each ordered top-to-bottom.
    pub columns: Vec<Vec<LodDataPoint>>,
}

impl LodChunk {
    pub fn new(chunk_x: i32, chunk_z: i32) -> Self {
        Self {
            chunk_x,
            chunk_z,
            columns: vec![Vec::new(); 256],
        }
    }

    #[inline]
    pub fn column_mut(&mut self, lx: usize, lz: usize) -> &mut Vec<LodDataPoint> {
        &mut self.columns[lz * 16 + lx]
    }
}

/// Full `64 x 64` cell LOD section (`columns[cz * 64 + cx]`) at any detail level `0..=MAX_DETAIL`.
#[derive(Clone, Debug)]
pub struct LodSectionData {
    pub pos: LodSectionPos,
    /// 4096 columns indexed by `cz * 64 + cx`, each ordered top-to-bottom.
    pub columns: Vec<Vec<LodDataPoint>>,
    pub timestamp_ms: i64,
}

impl LodSectionData {
    pub fn new(pos: LodSectionPos) -> Self {
        Self {
            pos,
            columns: vec![Vec::new(); SECTION_CELLS],
            timestamp_ms: 0,
        }
    }

    #[inline]
    pub fn column_mut(&mut self, cx: usize, cz: usize) -> &mut Vec<LodDataPoint> {
        &mut self.columns[cz * SECTION_SIZE + cx]
    }
}

/// Generation request passed to a registered [`LodWorldGenerator`].
#[derive(Clone, Debug)]
pub struct LodGenRequest {
    pub dimension: String,
    pub pos: LodSectionPos,
    pub target_step: LodGenStep,
}

/// Payload emitted by a [`LodWorldGenerator`] during [`LodWorldGenerator::generate`].
#[derive(Clone, Debug)]
pub enum LodGenOutput {
    Chunk(LodChunk),
    Section(LodSectionData),
    DhV2Dto {
        dto: DhV2FullDataDto,
        level_min_y: i32,
    },
}

/// World-generator override trait matching upstream DH's `IDhApiWorldGenerator`.
///
/// Mods implementing custom terrain (for example, Terra, Terralith overrides, or custom dimensions)
/// register an `Arc<dyn LodWorldGenerator>` via [`register_world_generator`].
pub trait LodWorldGenerator: Send + Sync + 'static {
    /// Higher priority wins when multiple generators could apply; default is `0`.
    fn priority(&self) -> i32 {
        0
    }

    /// Inclusive LOD detail range this generator can produce; default covers all detail levels.
    fn detail_range(&self) -> RangeInclusive<u8> {
        0..=MAX_DETAIL
    }

    /// Optional hook called immediately before a generation batch begins.
    fn pre_task_start(&self) {}

    /// Generates LOD data for `request`, invoking `emit` for each produced section, chunk, or
    /// `FullDataSourceV2DTO`.
    fn generate(&self, request: &LodGenRequest, emit: &mut dyn FnMut(LodGenOutput));

    /// Called when this generator is unregistered or replaced.
    fn close(&self) {}
}

/// Client-side pull hook for mods that fetch LOD sections from a remote server or web cache.
///
/// Distant Sunrises calls [`RemoteLodProvider::request_section`] for missing visible sections;
/// the provider responds asynchronously (on any thread) via [`submit_dh_v2_bytes`],
/// [`submit_dh_v2_dto`], [`submit_lod_section`], or [`submit_lod_chunk`].
pub trait RemoteLodProvider: Send + Sync + 'static {
    fn request_section(&self, dimension: &str, pos: LodSectionPos, client_timestamp_ms: i64);

    fn cancel_section(&self, _dimension: &str, _pos: LodSectionPos) {}

    fn close(&self) {}
}

enum InstallOutcome {
    Installed(Option<Arc<dyn LodWorldGenerator>>),
    Rejected,
}

impl DhState {
    fn rewind_legacy_cursors(&mut self, dimension: &str) {
        for ((dim, _), cache) in &mut self.desired_cache {
            if dim == dimension {
                cache.disk_cursor = 0;
                cache.sample_cursor = 0;
            }
        }
    }

    fn install_world_generator(
        &mut self,
        dimension: &str,
        generator: &Arc<dyn LodWorldGenerator>,
    ) -> InstallOutcome {
        if let Some(existing) = self.world_generators.get(dimension)
            && existing.priority() > generator.priority()
        {
            return InstallOutcome::Rejected;
        }
        let replaced = self
            .world_generators
            .insert(dimension.to_owned(), Arc::clone(generator));
        self.world_generator_epoch = self.world_generator_epoch.wrapping_add(1);
        self.rewind_legacy_cursors(dimension);
        InstallOutcome::Installed(replaced)
    }

    fn remove_world_generator(&mut self, dimension: &str) -> Option<Arc<dyn LodWorldGenerator>> {
        let removed = self.world_generators.remove(dimension)?;
        self.world_generator_epoch = self.world_generator_epoch.wrapping_add(1);
        self.rewind_legacy_cursors(dimension);
        Some(removed)
    }
}

/// Registers a custom [`LodWorldGenerator`] for `dimension`. Replaces any previously registered
/// generator with lower or equal priority (calling [`LodWorldGenerator::close`] on the replaced
/// generator). Returns `true` if installed. Newly eligible legacy cache rows are reconsidered
/// without waiting for a player to move.
pub fn register_world_generator(dimension: &str, generator: Arc<dyn LodWorldGenerator>) -> bool {
    let InstallOutcome::Installed(replaced) = runtime()
        .inner
        .lock()
        .unwrap()
        .install_world_generator(dimension, &generator)
    else {
        return false;
    };
    if let Some(old) = replaced {
        old.close();
    }
    true
}

/// Removes any registered [`LodWorldGenerator`] for `dimension`, calling `close()` if present
/// and reconsidering the desired sections for built-in scheduling.
pub fn unregister_world_generator(dimension: &str) -> bool {
    let removed = runtime()
        .inner
        .lock()
        .unwrap()
        .remove_world_generator(dimension);
    if let Some(old) = removed {
        old.close();
        true
    } else {
        false
    }
}

/// Disables or re-enables Distant Sunrises' built-in distant generator for `dimension`.
pub fn disable_builtin_generation(dimension: &str, disabled: bool) {
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    if disabled {
        st.disabled_builtin_gen.insert(dimension.to_owned());
    } else {
        st.disabled_builtin_gen.remove(dimension);
    }
}

/// Registers a client-side [`RemoteLodProvider`] for `dimension`.
pub fn register_remote_lod_provider(dimension: &str, provider: Arc<dyn RemoteLodProvider>) {
    let rt = runtime();
    let replaced = {
        let mut st = rt.inner.lock().unwrap();
        st.remote_requested.remove(dimension);
        st.remote_providers.insert(dimension.to_owned(), provider)
    };
    if let Some(old) = replaced {
        old.close();
    }
}

/// Unregisters the client-side [`RemoteLodProvider`] for `dimension`.
pub fn unregister_remote_lod_provider(dimension: &str) -> bool {
    let rt = runtime();
    let removed = {
        let mut st = rt.inner.lock().unwrap();
        st.remote_requested.remove(dimension);
        st.remote_providers.remove(dimension)
    };
    if let Some(old) = removed {
        old.close();
        true
    } else {
        false
    }
}

/// Disables or re-enables Distant Sunrises' built-in multiplayer LOD network handling for
/// `dimension`.
pub fn disable_builtin_network(dimension: &str, disabled: bool) {
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    if disabled {
        st.disabled_builtin_net.insert(dimension.to_owned());
    } else {
        st.disabled_builtin_net.remove(dimension);
    }
}

/// Submits a `64 x 64` [`LodSectionData`] directly into Distant Sunrises' LOD store, propagates
/// parent detail levels, queues meshing, and schedules disk persistence if a save directory is
/// active. Returns the number of cells modified.
pub fn submit_lod_section(dimension: &str, section: LodSectionData) -> Result<usize, DhApiError> {
    if section.pos.detail > MAX_DETAIL {
        return Err(DhApiError::InvalidDetail(section.pos.detail));
    }
    if section.columns.len() != SECTION_CELLS {
        return Err(DhApiError::InvalidColumnCount {
            expected: SECTION_CELLS,
            got: section.columns.len(),
        });
    }

    let pos = SectionPos::from(section.pos);
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let save_root = st.world_dir.clone();
    let store = st.dimensions.entry(dimension.to_owned()).or_default();

    let mut modified = 0usize;
    {
        let sec = store.get_or_create_mut(pos);
        for cz in 0..SECTION_SIZE {
            for cx in 0..SECTION_SIZE {
                let pts = &section.columns[cz * SECTION_SIZE + cx];
                let (col, extra) = datapoints_to_lod_column(pts);
                if !col.is_empty() && sec.set_cell_with_spans(cx, cz, col, extra) {
                    modified += 1;
                }
            }
        }
    }

    if modified > 0 {
        let mut dirty = HashSet::new();
        dirty.insert(pos);
        for parent in store.propagate_up(pos) {
            dirty.insert(parent);
        }
        queue_save(rt, dimension, save_root.as_ref(), store, &dirty);
    }
    Ok(modified)
}

/// Submits a `16 x 16` [`LodChunk`] at detail 0 into Distant Sunrises' LOD store, propagates
/// parent detail levels, and queues meshing/persistence. Returns the number of cells modified.
pub fn submit_lod_chunk(dimension: &str, chunk: LodChunk) -> Result<usize, DhApiError> {
    if chunk.columns.len() != 256 {
        return Err(DhApiError::InvalidColumnCount {
            expected: 256,
            got: chunk.columns.len(),
        });
    }

    let bx0 = chunk.chunk_x << 4;
    let bz0 = chunk.chunk_z << 4;
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let save_root = st.world_dir.clone();
    let store = st.dimensions.entry(dimension.to_owned()).or_default();

    let mut dirty = HashSet::new();
    let mut modified = 0usize;
    for lz in 0..16 {
        for lx in 0..16 {
            let pts = &chunk.columns[lz * 16 + lx];
            let (col, extra) = datapoints_to_lod_column(pts);
            if col.is_empty() {
                continue;
            }
            let bx = bx0 + lx as i32;
            let bz = bz0 + lz as i32;
            let pos = SectionPos::from_block(0, bx, bz);
            let (cx, cz) = pos.local_cell(bx, bz);
            if store
                .get_or_create_mut(pos)
                .set_cell_with_spans(cx, cz, col, extra)
            {
                dirty.insert(pos);
                modified += 1;
            }
        }
    }

    if !dirty.is_empty() {
        let mut all_dirty = dirty.clone();
        for &pos in &dirty {
            for parent in store.propagate_up(pos) {
                all_dirty.insert(parent);
            }
        }
        queue_save(rt, dimension, save_root.as_ref(), store, &all_dirty);
    }
    Ok(modified)
}

/// Decodes and ingests an upstream Distant Sunrises [`DhV2FullDataDto`] into `dimension`.
pub fn submit_dh_v2_dto(
    dimension: &str,
    dto: &DhV2FullDataDto,
    level_min_y: i32,
) -> Result<usize, DhApiError> {
    let decoded = dto.decode_section(level_min_y)?;
    ingest_decoded_dh_v2(dimension, decoded)
}

/// Parses big-endian `FullDataSourceV2DTO` wire bytes, decompresses, and ingests into `dimension`.
pub fn submit_dh_v2_bytes(
    dimension: &str,
    wire_bytes: &[u8],
    level_min_y: i32,
) -> Result<usize, DhApiError> {
    let dto = DhV2FullDataDto::decode_wire(wire_bytes)?;
    submit_dh_v2_dto(dimension, &dto, level_min_y)
}

/// Removes the legacy cached row and clears terminal scheduling for this target.
/// Octree fallback geometry and its coverage are retained. Returns whether a legacy row was removed.
pub fn invalidate_lod_section(dimension: &str, pos: LodSectionPos) -> bool {
    let sec_pos = SectionPos::from(pos);
    let rt = runtime();
    rt.gen_pool.invalidate_section(dimension, sec_pos);
    let mut st = rt.inner.lock().unwrap();
    crate::pipeline::invalidate_coarse_target(dimension, sec_pos);
    let mut removed = false;
    if let Some(store) = st.dimensions.get_mut(dimension) {
        removed |= store.sections.remove(&sec_pos).is_some();
        store.legacy_terminal_targets.remove(&sec_pos);
        store.disk_checked.remove(&sec_pos);
    }
    st.rewind_legacy_cursors(dimension);
    if let Some(requested) = st.remote_requested.get_mut(dimension) {
        requested.remove(&sec_pos);
    }
    removed
}

/// Reads the highest-detail cached [`LodColumn`] covering world block `(block_x, block_z)`.
pub fn get_column_data(dimension: &str, block_x: i32, block_z: i32) -> Option<LodColumn> {
    let rt = runtime();
    let st = rt.inner.lock().unwrap();
    let store = st.dimensions.get(dimension)?;
    for detail in 0..=MAX_DETAIL {
        let pos = SectionPos::from_block(detail, block_x, block_z);
        if let Some(sec) = store.sections.get(&pos) {
            let (cx, cz) = pos.local_cell(block_x, block_z);
            let col = sec.get(cx, cz);
            if !col.is_empty() {
                return Some(col);
            }
        }
    }
    None
}

/// Returns how many cells (`0..=4096`) are currently populated in `pos` for `dimension`.
pub fn get_section_populated_count(dimension: &str, pos: LodSectionPos) -> usize {
    let sec_pos = SectionPos::from(pos);
    let rt = runtime();
    let st = rt.inner.lock().unwrap();
    st.dimensions
        .get(dimension)
        .and_then(|s| s.sections.get(&sec_pos))
        .map(|s| s.populated as usize)
        .unwrap_or(0)
}

/// Internal helper used by `lib.rs` to ingest any [`LodGenOutput`] produced by a registered
/// [`LodWorldGenerator`].
pub(crate) fn ingest_gen_output(
    dimension: &str,
    output: LodGenOutput,
) -> Result<usize, DhApiError> {
    match output {
        LodGenOutput::Chunk(chunk) => submit_lod_chunk(dimension, chunk),
        LodGenOutput::Section(section) => submit_lod_section(dimension, section),
        LodGenOutput::DhV2Dto { dto, level_min_y } => {
            submit_dh_v2_dto(dimension, &dto, level_min_y)
        }
    }
}

fn ingest_decoded_dh_v2(dimension: &str, decoded: DecodedDhV2Section) -> Result<usize, DhApiError> {
    if decoded.lod_detail > MAX_DETAIL {
        return Err(DhApiError::InvalidDetail(decoded.lod_detail));
    }

    let pos = LodSectionPos::new(decoded.lod_detail, decoded.section_x, decoded.section_z);
    let mut section = LodSectionData::new(pos);
    section.timestamp_ms = decoded.last_modified_unix_ms;

    for (idx, col_pts) in decoded.columns.iter().enumerate() {
        let out_col = &mut section.columns[idx];
        out_col.reserve(col_pts.len());
        for pt in col_pts {
            if pt.height == 0 {
                continue;
            }
            let Some(entry) = decoded.mapping.get(pt.mapping_id as usize) else {
                continue;
            };
            let is_fluid = is_water_block(&entry.block_id);
            if !is_fluid && is_skippable_for_lod(&entry.block_id) {
                continue;
            }
            out_col.push(LodDataPoint {
                bottom_y: pt.bottom_y,
                top_y: pt.top_y(),
                block_id: entry.block_id.clone(),
                biome_id: entry.biome_id.clone(),
                rgb_override: None,
                sky_light: pt.sky_light.min(15),
                block_light: pt.block_light.min(15),
                is_fluid,
            });
        }
    }

    submit_lod_section(dimension, section)
}

fn datapoints_to_lod_column(points: &[LodDataPoint]) -> (LodColumn, Vec<ExtraSpan>) {
    if points.is_empty() {
        return (LodColumn::EMPTY, Vec::new());
    }

    // Ensure top-to-bottom ordering without modifying caller data.
    let mut ordered: Vec<&LodDataPoint> = points
        .iter()
        .filter(|p| {
            p.top_y > p.bottom_y
                && (p.is_fluid || is_water_block(&p.block_id) || !is_skippable_for_lod(&p.block_id))
        })
        .collect();
    if ordered.is_empty() {
        return (LodColumn::EMPTY, Vec::new());
    }
    ordered.sort_by_key(|pt| std::cmp::Reverse(pt.top_y));

    let mut top_fluid: Option<&LodDataPoint> = None;
    let mut top_solid_idx: Option<usize> = None;

    for (idx, pt) in ordered.iter().enumerate() {
        let fluid = pt.is_fluid || is_water_block(&pt.block_id);
        if fluid {
            if top_fluid.is_none() && top_solid_idx.is_none() {
                top_fluid = Some(pt);
            }
        } else {
            top_solid_idx = Some(idx);
            break;
        }
    }

    let (primary, next_idx) = if let Some(si) = top_solid_idx {
        let solid = ordered[si];
        let solid_rgb = solid
            .rgb_override
            .unwrap_or_else(|| block_color(&solid.block_id, &solid.biome_id, 0));
        let (fluid_top, fluid_rgb, sky, block) = if let Some(fl) = top_fluid {
            let rgb = fl
                .rgb_override
                .unwrap_or_else(|| biome_water_color(&fl.biome_id, 0));
            (Some(fl.top_y), rgb, fl.sky_light, fl.block_light)
        } else {
            (None, [45, 95, 200], solid.sky_light, solid.block_light)
        };
        let col = LodColumn::from_solid_and_fluid(
            solid.top_y,
            solid.bottom_y,
            solid_rgb,
            fluid_top,
            fluid_rgb,
            sky,
            block,
            SOURCE_LOADED_CHUNK,
        );
        (col, si + 1)
    } else if let Some(fl) = top_fluid {
        let fluid_rgb = fl
            .rgb_override
            .unwrap_or_else(|| biome_water_color(&fl.biome_id, 0));
        let col = LodColumn::from_solid_and_fluid(
            fl.bottom_y,
            fl.bottom_y - 4,
            block_color("minecraft:sand", &fl.biome_id, 0),
            Some(fl.top_y),
            fluid_rgb,
            fl.sky_light,
            fl.block_light,
            SOURCE_LOADED_CHUNK,
        );
        (col, ordered.len())
    } else {
        return (LodColumn::EMPTY, Vec::new());
    };

    let mut extra = Vec::new();
    if next_idx < ordered.len() {
        for pt in &ordered[next_idx..] {
            if extra.len() >= DEFAULT_MAX_EXTRA_SPANS {
                break;
            }
            let fluid = pt.is_fluid || is_water_block(&pt.block_id);
            let rgb = pt.rgb_override.unwrap_or_else(|| {
                if fluid {
                    biome_water_color(&pt.biome_id, 0)
                } else {
                    block_color(&pt.block_id, &pt.biome_id, 0)
                }
            });
            extra.push(ExtraSpan::new(
                pt.top_y,
                pt.bottom_y,
                rgb,
                pt.sky_light,
                pt.block_light,
                fluid,
            ));
        }
    }

    (primary, extra)
}

fn queue_save(
    rt: &Arc<crate::render_bridge::DhRuntime>,
    dimension: &str,
    save_root: Option<&std::path::PathBuf>,
    store: &mut crate::lod::DimensionLodStore,
    dirty: &HashSet<SectionPos>,
) {
    for &pos in dirty {
        if let Some(sec) = store.sections.get_mut(&pos)
            && sec.dirty_save
            && sec.populated > 0
            && let Some(dir) = save_root
        {
            sec.dirty_save = false;
            rt.io
                .request_save(dir.clone(), dimension.to_owned(), sec.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dh_v2::{BLOCK_STATE_SEPARATOR, DhV2CompressionMode, write_varint, zigzag_encode};
    use std::sync::atomic::AtomicUsize;

    struct DummyGenerator {
        calls: AtomicUsize,
        priority: i32,
    }

    impl LodWorldGenerator for DummyGenerator {
        fn priority(&self) -> i32 {
            self.priority
        }

        fn generate(&self, request: &LodGenRequest, emit: &mut dyn FnMut(LodGenOutput)) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut sec = LodSectionData::new(request.pos);
            sec.column_mut(2, 3).push(LodDataPoint::solid(
                60,
                80,
                "minecraft:stone",
                "minecraft:plains",
                15,
                0,
            ));
            emit(LodGenOutput::Section(sec));
        }
    }

    #[test]
    fn api_chunk_submit_query_propagate_and_invalidate() {
        let dim = "test:api_chunk_dim";
        let mut chunk = LodChunk::new(1, -1); // blocks x=16..32, z=-16..0
        chunk.column_mut(4, 5).push(LodDataPoint::fluid(
            62,
            66,
            "minecraft:water",
            "minecraft:ocean",
            15,
            0,
        ));
        chunk.column_mut(4, 5).push(LodDataPoint::solid(
            40,
            62,
            "minecraft:sand",
            "minecraft:ocean",
            0,
            0,
        ));
        chunk.column_mut(4, 5).push(LodDataPoint::solid(
            10,
            30,
            "minecraft:stone",
            "minecraft:ocean",
            0,
            4,
        ));

        let modified = submit_lod_chunk(dim, chunk).expect("submit chunk");
        assert_eq!(modified, 1);

        let col = get_column_data(dim, 20, -11).expect("column present");
        assert_eq!(col.solid_top_y, 62);
        assert_eq!(col.fluid_top_y, 66);

        let pos0 = LodSectionPos::from_block(0, 20, -11);
        assert_eq!(get_section_populated_count(dim, pos0), 1);
        // Parent detail 1 should also be populated by quadtree propagation.
        let pos1 = LodSectionPos::from_block(1, 20, -11);
        assert!(get_section_populated_count(dim, pos1) >= 1);

        assert!(invalidate_lod_section(dim, pos0));
        assert_eq!(get_section_populated_count(dim, pos0), 0);
        let _ = invalidate_lod_section(dim, pos1);
    }

    #[test]
    fn api_world_generator_registration_respects_priority() {
        let dim = "test:api_gen_dim";
        let low = Arc::new(DummyGenerator {
            calls: AtomicUsize::new(0),
            priority: 1,
        });
        let high = Arc::new(DummyGenerator {
            calls: AtomicUsize::new(0),
            priority: 10,
        });
        assert!(register_world_generator(dim, high.clone()));
        assert!(!register_world_generator(dim, low.clone()));

        let req = LodGenRequest {
            dimension: dim.to_owned(),
            pos: LodSectionPos::new(0, 0, 0),
            target_step: LodGenStep::Surface,
        };
        high.generate(&req, &mut |out| {
            let n = ingest_gen_output(dim, out).expect("ingest");
            assert_eq!(n, 1);
        });
        assert_eq!(high.calls.load(Ordering::SeqCst), 1);
        assert!(get_column_data(dim, 2, 3).is_some());
        assert!(unregister_world_generator(dim));
        assert!(invalidate_lod_section(dim, LodSectionPos::new(0, 0, 0)));
    }

    #[test]
    fn stationary_player_revisits_legacy_d1_disk_row_after_generator_registration() {
        let dim = "test:late-generator";
        let key = (dim.to_owned(), 1);
        let other_key = ("test:other-dimension".to_owned(), 1);
        let target = SectionPos::from_block(1, 384, 0);
        let cam = [0.0, 64.0, 0.0];
        let cfg = crate::config::DhConfig::default();
        let mut st = DhState::default();
        for key in [&key, &other_key] {
            let cache = crate::ensure_desired_cache(&mut st, key.clone(), cam, cfg);
            cache.list = Arc::from([target]);
            cache.sample_cursor = 1;
            cache.octree_disk_cursor = 1;
        }
        let list = Arc::clone(&st.desired_cache[&key].list);
        let mut legacy = crate::lod::DimensionLodStore::default();
        let cache = st.desired_cache.get_mut(&key).unwrap();
        assert_eq!(crate::next_legacy_disk_load(cache, &mut legacy, None), None);
        assert_eq!(cache.disk_cursor, 1);
        assert!(!legacy.disk_checked.contains(&target));
        st.desired_cache.get_mut(&other_key).unwrap().disk_cursor = 1;

        let generator: Arc<dyn LodWorldGenerator> = Arc::new(DummyGenerator {
            calls: AtomicUsize::new(0),
            priority: 0,
        });
        assert!(matches!(
            st.install_world_generator(dim, &generator),
            InstallOutcome::Installed(None)
        ));
        let range = st.world_generators[dim].detail_range();
        let cache = crate::ensure_desired_cache(&mut st, key.clone(), cam, cfg);
        assert!(Arc::ptr_eq(&cache.list, &list));
        assert_eq!(
            (
                cache.disk_cursor,
                cache.sample_cursor,
                cache.octree_disk_cursor
            ),
            (0, 0, 1)
        );
        assert_eq!(
            crate::next_legacy_disk_load(cache, &mut legacy, Some(&range)),
            Some(target)
        );
        assert!(legacy.disk_checked.contains(&target));
        cache.sample_cursor = 1;

        assert!(matches!(
            st.install_world_generator(dim, &generator),
            InstallOutcome::Installed(Some(_))
        ));
        let cache = st.desired_cache.get_mut(&key).unwrap();
        assert_eq!((cache.disk_cursor, cache.sample_cursor), (0, 0));
        // Rewinding does not duplicate a disk request already made for this row.
        assert_eq!(
            crate::next_legacy_disk_load(cache, &mut legacy, Some(&range)),
            None
        );
        assert_eq!(cache.disk_cursor, 1);
        cache.sample_cursor = 1;

        assert!(st.remove_world_generator(dim).is_some());
        let cache = &st.desired_cache[&key];
        assert_eq!(
            (
                cache.disk_cursor,
                cache.sample_cursor,
                cache.octree_disk_cursor
            ),
            (0, 0, 1)
        );
        let other = &st.desired_cache[&other_key];
        assert_eq!((other.disk_cursor, other.sample_cursor), (1, 1));
    }

    #[test]
    fn same_arc_reregistration_rejects_stale_range_for_both_cursors() {
        use crate::octree::{OctreeSection, OctreeSectionPos};
        use std::sync::Mutex;
        use std::sync::atomic::AtomicU8;

        struct MutableGenerator(AtomicU8);
        impl LodWorldGenerator for MutableGenerator {
            fn detail_range(&self) -> RangeInclusive<u8> {
                self.0.load(Ordering::Relaxed)..=2
            }

            fn generate(&self, _: &LodGenRequest, _: &mut dyn FnMut(LodGenOutput)) {}
        }

        let dim = "test:mutable-generator";
        let key = (dim.to_owned(), 1);
        let d1 = SectionPos::from_block(1, 384, 0);
        let cfg = crate::config::DhConfig::default();
        let mutable = Arc::new(MutableGenerator(AtomicU8::new(2)));
        let generator: Arc<dyn LodWorldGenerator> = mutable.clone();
        let mut st = DhState::default();
        crate::ensure_desired_cache(&mut st, key.clone(), [0.0, 64.0, 0.0], cfg).list =
            Arc::from([d1]);
        assert!(matches!(
            st.install_world_generator(dim, &generator),
            InstallOutcome::Installed(None)
        ));
        let state = Mutex::new(st);

        let mut octree = crate::octree_ingest::OctreeIngestStore::new();
        octree.set_y_bounds((0, 64));
        for dz in 0..2 {
            for dx in 0..2 {
                let pos = OctreeSectionPos::new(1, d1.x * 2 + dx, 0, d1.z * 2 + dz);
                assert!(octree.adopt_section(OctreeSection::new(pos), true));
            }
        }
        let snapshot = || {
            let (generator, epoch) = {
                let st = state.lock().unwrap();
                (
                    Arc::clone(&st.world_generators[dim]),
                    st.world_generator_epoch,
                )
            };
            let range = generator.detail_range();
            (generator, epoch, range)
        };
        let mut legacy = crate::lod::DimensionLodStore::default();
        let mut scan = |generator: &Arc<dyn LodWorldGenerator>,
                        epoch,
                        range: RangeInclusive<u8>| {
            let disk = crate::lock_current_world_generator(&state, epoch).and_then(|mut st| {
                let cache = st.desired_cache.get_mut(&key).unwrap();
                crate::next_legacy_disk_load(cache, &mut legacy, Some(&range))
            });
            let sample = crate::lock_current_world_generator(&state, epoch).and_then(|mut st| {
                let cache = st.desired_cache.get_mut(&key).unwrap();
                crate::pick_distant_target(
                    cache,
                    &legacy,
                    Some((generator, range)),
                    &HashSet::new(),
                    Some(&octree),
                )
            });
            (disk, sample)
        };
        let cursors = || {
            let st = state.lock().unwrap();
            let cache = &st.desired_cache[&key];
            (cache.disk_cursor, cache.sample_cursor)
        };

        let (old_generator, old_epoch, old_range) = snapshot();
        assert_eq!(old_range, 2..=2);
        let (disk, sample) = scan(&old_generator, old_epoch, old_range.clone());
        assert!(disk.is_none() && sample.is_none());
        assert_eq!(cursors(), (1, 1));

        mutable.0.store(1, Ordering::Relaxed);
        let InstallOutcome::Installed(Some(replaced)) = state
            .lock()
            .unwrap()
            .install_world_generator(dim, &old_generator)
        else {
            panic!("expected to replace the registered generator");
        };
        assert!(Arc::ptr_eq(&replaced, &old_generator));
        let (disk, sample) = scan(&old_generator, old_epoch, old_range);
        assert!(disk.is_none() && sample.is_none());
        assert_eq!(cursors(), (0, 0));

        let (fresh_generator, fresh_epoch, fresh_range) = snapshot();
        assert_eq!(fresh_range, 1..=2);
        let (disk, sample) = scan(&fresh_generator, fresh_epoch, fresh_range);
        assert_eq!(disk, Some(d1));
        assert!(matches!(sample, Some((pos, crate::DistantTarget::Custom(_))) if pos == d1));
    }

    #[test]
    fn api_submit_dh_v2_bytes_ingests_and_queries_column() {
        let dim = "test:api_v2_dim";
        let min_y = -64;

        // Build a minimal V2 mapping + interior blob with cell (4, 5) populated.
        let mut mapping = Vec::new();
        mapping.extend_from_slice(&2i32.to_be_bytes());
        for s in [
            format!("minecraft:plains{BLOCK_STATE_SEPARATOR}minecraft:air"),
            format!("minecraft:forest{BLOCK_STATE_SEPARATOR}minecraft:grass_block[snowy=false]"),
        ] {
            mapping.extend_from_slice(&(s.len() as u16).to_be_bytes());
            mapping.extend_from_slice(s.as_bytes());
        }

        let mut interior = Vec::new();
        // 62 * 62 columns; only (x=4, z=5) has 1 datapoint
        for x in 1..63usize {
            for z in 1..63usize {
                write_varint(&mut interior, u32::from(x == 4 && z == 5));
            }
        }
        // id=1, has_light=1, has_y_delta=1
        write_varint(&mut interior, (1u32 << 2) | 0b11);
        // height = 10
        write_varint(&mut interior, 10);
        // bottom_y = 70 => rel_bottom = 134, expected = 0 - 10 = -10, delta = 144
        write_varint(&mut interior, zigzag_encode(144));
        // light: block=0, sky=15
        interior.push(0x0F);

        let dto = DhV2FullDataDto {
            pos: LodSectionPos::new(0, 2, 3).to_dh_packed(),
            data_checksum: 7,
            compressed_data: interior,
            compressed_north_adj: Vec::new(),
            compressed_south_adj: Vec::new(),
            compressed_east_adj: Vec::new(),
            compressed_west_adj: Vec::new(),
            compressed_column_gen_step: Vec::new(),
            compressed_world_compression_mode: Vec::new(),
            compressed_mapping: mapping,
            data_format_version: 3,
            compression_mode: DhV2CompressionMode::Uncompressed,
            apply_to_parent: true,
            apply_to_children: false,
            regenerate: false,
            last_modified_unix_ms: 1234,
            created_unix_ms: 1000,
        };

        let wire = dto.encode_wire();
        let modified = submit_dh_v2_bytes(dim, &wire, min_y).expect("submit v2 bytes");
        assert_eq!(modified, 1);

        let bx = 2 * 64 + 4;
        let bz = 3 * 64 + 5;
        let col = get_column_data(dim, bx, bz).expect("v2 column present");
        assert_eq!(col.solid_top_y, 80);
        assert_eq!(col.solid_bottom_y, 70);
        assert_eq!(LodColumn::sky_light(col.solid_rgba), 15);
        assert!(invalidate_lod_section(dim, LodSectionPos::new(0, 2, 3)));
    }
}
