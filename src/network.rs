//! Built-in multiplayer LOD synchronization over Sunrise's optional, droppable, network-thread
//! bidirectional play payload channel (`distantsunrises:lod_sync`).
//!
//! - **Optional**: joining vanilla servers or servers without Distant Sunrises is never blocked;
//!   the client checks `host.has_channel(&lod_sync_channel_id())` before sending requests.
//! - **Network-thread & droppable**: Zstd compression/decompression and RLE encoding run on
//!   Sunrise's `HandlerThread::Network`, and congestion drops stale LOD packets cleanly instead
//!   of disconnecting players.
//! - **Singleplayer zero-overhead**: integrated singleplayer skips network sync because both
//!   sides share [`crate::render_bridge::DhRuntime`] in-process.
//! - **Overridable**: third-party network mods can call
//!   [`crate::api::disable_builtin_network`] and register a [`crate::api::RemoteLodProvider`]
//!   instead.

use crate::lod::{DimensionLodStore, LodColumn, LodSection, MAX_DETAIL, SECTION_SIZE, SectionPos};
use crate::render_bridge::runtime;
use crate::storage::{decode_section, encode_section};
use mod_api::{
    ClientHost, HandlerThread, IPayloadContext, IPayloadHandler, PayloadBytes, PayloadHandleError,
    PayloadId, PlayerRef, Registrar, ServerHost,
};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Namespaced channel identifier negotiated between Sunrise DH clients and servers.
pub const LOD_SYNC_CHANNEL: &str = "distantsunrises:lod_sync";
/// Wire schema version for [`LOD_SYNC_CHANNEL`].
pub const LOD_SYNC_VERSION: &str = "1";

const MSG_SECTION_REQUEST: u8 = 1;
const MSG_SECTION_DATA: u8 = 2;
const MSG_SESSION_CONFIG: u8 = 3;
const MSG_SESSION_CONFIG_REQUEST: u8 = 4;
const MSG_PARTIAL_UPDATE: u8 = 5;

/// Maximum wire bytes emitted for one section reply (well below vanilla's 1 MiB play limit).
const MAX_WIRE_PAYLOAD_BYTES: usize = 512 * 1024;
/// Maximum decompressed RLE section size accepted from a server.
const MAX_DECOMPRESSED_SECTION_BYTES: usize = 4 * 1024 * 1024;
/// Maximum dimension identifier bytes accepted on the wire.
const MAX_DIM_LEN: usize = 256;
/// Maximum new section requests issued per client tick to avoid bursting the link.
const MAX_REQUESTS_PER_TICK: usize = 4;
/// Maximum columns in a single partial delta update packet.
const MAX_PARTIAL_COLUMNS: usize = 512;

/// Server-enforced multiplayer session policy (matches upstream DH `SessionConfig`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionConfig {
    pub max_render_distance_chunks: u32,
    pub distant_generation_allowed: bool,
    pub partial_updates_enabled: bool,
    pub max_bytes_per_sec: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_render_distance_chunks: 256,
            distant_generation_allowed: true,
            partial_updates_enabled: true,
            max_bytes_per_sec: 2 * 1024 * 1024,
        }
    }
}

/// Incremental column delta inside one LOD section (matches upstream `FullDataPartialUpdateMessage`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialColumnUpdate {
    pub dimension: String,
    pub pos: SectionPos,
    pub revision: u64,
    /// `(lx, lz, column)` cells updated inside `pos`.
    pub columns: Vec<(u8, u8, LodColumn)>,
}

/// Per-player token-bucket rate and bandwidth limiter guarding server-side LOD streaming.
pub struct PlayerRateLimiter {
    max_req_per_sec: f64,
    max_bytes_per_sec: f64,
    buckets: HashMap<String, (f64, f64, Instant)>,
}

impl PlayerRateLimiter {
    pub fn new(max_req_per_sec: u32, max_bytes_per_sec: u32) -> Self {
        Self {
            max_req_per_sec: max_req_per_sec.max(1) as f64,
            max_bytes_per_sec: max_bytes_per_sec.max(1024) as f64,
            buckets: HashMap::new(),
        }
    }

    /// Consume 1 request token at `now`. Returns `false` if the player exceeded their request budget.
    pub fn allow_request(&mut self, player_key: &str, now: Instant) -> bool {
        let entry = self.buckets.entry(player_key.to_owned()).or_insert((
            self.max_req_per_sec,
            self.max_bytes_per_sec,
            now,
        ));
        let dt = now.saturating_duration_since(entry.2).as_secs_f64();
        entry.0 = (entry.0 + dt * self.max_req_per_sec).min(self.max_req_per_sec);
        entry.1 = (entry.1 + dt * self.max_bytes_per_sec).min(self.max_bytes_per_sec);
        entry.2 = now;
        if entry.0 >= 1.0 {
            entry.0 -= 1.0;
            true
        } else {
            false
        }
    }

    /// Consume `bytes` from the player's bandwidth bucket at `now`.
    pub fn allow_bytes(&mut self, player_key: &str, bytes: usize, now: Instant) -> bool {
        let entry = self.buckets.entry(player_key.to_owned()).or_insert((
            self.max_req_per_sec,
            self.max_bytes_per_sec,
            now,
        ));
        let dt = now.saturating_duration_since(entry.2).as_secs_f64();
        entry.0 = (entry.0 + dt * self.max_req_per_sec).min(self.max_req_per_sec);
        entry.1 = (entry.1 + dt * self.max_bytes_per_sec).min(self.max_bytes_per_sec);
        entry.2 = now;
        let cost = bytes as f64;
        if entry.1 >= cost {
            entry.1 -= cost;
            true
        } else {
            false
        }
    }
}

fn rate_limiter() -> &'static Mutex<PlayerRateLimiter> {
    static LIMITER: OnceLock<Mutex<PlayerRateLimiter>> = OnceLock::new();
    LIMITER.get_or_init(|| Mutex::new(PlayerRateLimiter::new(32, 2 * 1024 * 1024)))
}

/// Sanitize a multiplayer server key + dimension name into a collision-free cache folder slug
/// (disambiguates multi-world / proxy sub-servers).
pub fn server_storage_subdir(server_key: &str, dimension: &str) -> String {
    let clean = |s: &str| -> String {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    };
    format!("{}__{}", clean(server_key), clean(dimension))
}

/// Derive a collision-free, path-traversal-safe local disk cache root for a remote multiplayer
/// server (`<instance_dir>/distant_sunrises_remote/<percent-encoded server_key>`).
pub fn remote_cache_root(instance_dir: &Path, server_key: &str) -> PathBuf {
    let mut slug = String::with_capacity(server_key.len());
    for b in server_key.bytes() {
        if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' {
            slug.push(b as char);
        } else {
            use std::fmt::Write as _;
            let _ = write!(&mut slug, "%{b:02X}");
        }
    }
    if slug.is_empty() {
        slug.push_str("remote");
    }
    instance_dir.join("distant_sunrises_remote").join(slug)
}

/// Pending column changes for one detail-0 section to broadcast to remote DH clients.
#[derive(Clone, Debug, Default)]
pub struct SectionColumnDelta {
    pub revision: u64,
    pub columns: Vec<(u8, u8, LodColumn)>,
}

/// Broadcast incremental LOD column deltas from a published server (LAN host or dedicated server)
/// to remote players currently in `dim` whose connections negotiated `distantsunrises:lod_sync`.
pub fn broadcast_partial_updates(
    host: &mut dyn ServerHost,
    dim: &str,
    deltas: &HashMap<SectionPos, SectionColumnDelta>,
) -> u32 {
    if deltas.is_empty() || !host.server_is_published() {
        return 0;
    }
    {
        let st = runtime().inner.lock().unwrap();
        if st.disabled_builtin_net.contains(dim) {
            return 0;
        }
    }

    let dim_id = mod_api::Identifier::parse(dim).ok();
    let targets: Vec<PlayerRef> = host
        .players()
        .server_players()
        .into_iter()
        .filter(|&p| {
            !p.is_singleplayer_owner(host) && dim_id.is_some() && p.dimension(host) == dim_id
        })
        .collect();
    if targets.is_empty() {
        return 0;
    }

    let channel = lod_sync_channel_id();
    let now = Instant::now();
    let mut sent = 0u32;

    for (&pos, delta) in deltas {
        for chunk in delta.columns.chunks(MAX_PARTIAL_COLUMNS) {
            let upd = PartialColumnUpdate {
                dimension: dim.to_owned(),
                pos,
                revision: delta.revision,
                columns: chunk.to_vec(),
            };
            let Some(wire) = encode_partial_update(&upd) else {
                continue;
            };
            let allowed: Vec<PlayerRef> = {
                let mut lim = rate_limiter().lock().unwrap();
                targets
                    .iter()
                    .copied()
                    .filter(|p| lim.allow_bytes(&format!("{p:?}"), wire.len(), now))
                    .collect()
            };
            if !allowed.is_empty() {
                sent = sent.saturating_add(host.players_send_payload(
                    &allowed,
                    &channel,
                    PayloadBytes(wire),
                ));
            }
        }
    }
    sent
}

/// Encode a `SessionConfig` handshake packet (`server -> client`).
pub fn encode_session_config(cfg: SessionConfig) -> Vec<u8> {
    let mut out = Vec::with_capacity(11);
    out.push(MSG_SESSION_CONFIG);
    out.extend_from_slice(&cfg.max_render_distance_chunks.to_le_bytes());
    out.push(u8::from(cfg.distant_generation_allowed));
    out.push(u8::from(cfg.partial_updates_enabled));
    out.extend_from_slice(&cfg.max_bytes_per_sec.to_le_bytes());
    out
}

/// Decode a `SessionConfig` handshake packet.
pub fn decode_session_config(bytes: &[u8]) -> Option<SessionConfig> {
    if bytes.len() != 11 || bytes[0] != MSG_SESSION_CONFIG {
        return None;
    }
    Some(SessionConfig {
        max_render_distance_chunks: u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]])
            .clamp(32, 512),
        distant_generation_allowed: bytes[5] != 0,
        partial_updates_enabled: bytes[6] != 0,
        max_bytes_per_sec: u32::from_le_bytes([bytes[7], bytes[8], bytes[9], bytes[10]])
            .max(16 * 1024),
    })
}

/// Encode a client -> server `SessionConfig` request packet.
pub fn encode_session_config_request() -> Vec<u8> {
    vec![MSG_SESSION_CONFIG_REQUEST]
}

/// Encode an incremental column delta (`FullDataPartialUpdateMessage` counterpart).
pub fn encode_partial_update(update: &PartialColumnUpdate) -> Option<Vec<u8>> {
    let dim_bytes = update.dimension.as_bytes();
    if dim_bytes.is_empty()
        || dim_bytes.len() > MAX_DIM_LEN
        || update.pos.detail > MAX_DETAIL
        || update.columns.is_empty()
        || update.columns.len() > MAX_PARTIAL_COLUMNS
    {
        return None;
    }
    let mut raw = Vec::with_capacity(2 + update.columns.len() * 20);
    raw.extend_from_slice(&(update.columns.len() as u16).to_le_bytes());
    for &(lx, lz, col) in &update.columns {
        if (lx as usize) >= SECTION_SIZE || (lz as usize) >= SECTION_SIZE {
            return None;
        }
        raw.push(lx);
        raw.push(lz);
        raw.extend_from_slice(&col.solid_top_y.to_le_bytes());
        raw.extend_from_slice(&col.solid_bottom_y.to_le_bytes());
        raw.extend_from_slice(&col.solid_rgba);
        raw.extend_from_slice(&col.fluid_top_y.to_le_bytes());
        raw.extend_from_slice(&col.fluid_rgba);
        raw.push(col.source);
        raw.push(col.material_id);
    }
    let compressed = ruzstd::encoding::compress_to_vec(
        raw.as_slice(),
        ruzstd::encoding::CompressionLevel::Fastest,
    );
    let total = 1 + 2 + dim_bytes.len() + 1 + 4 + 4 + 8 + compressed.len();
    if total > MAX_WIRE_PAYLOAD_BYTES {
        return None;
    }
    let mut out = Vec::with_capacity(total);
    out.push(MSG_PARTIAL_UPDATE);
    out.extend_from_slice(&(dim_bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(dim_bytes);
    out.push(update.pos.detail);
    out.extend_from_slice(&update.pos.x.to_le_bytes());
    out.extend_from_slice(&update.pos.z.to_le_bytes());
    out.extend_from_slice(&update.revision.to_le_bytes());
    out.extend_from_slice(&compressed);
    Some(out)
}

/// Decode an incremental column delta (`FullDataPartialUpdateMessage` counterpart).
pub fn decode_partial_update(bytes: &[u8]) -> Option<PartialColumnUpdate> {
    if bytes.len() < 20 || bytes[0] != MSG_PARTIAL_UPDATE {
        return None;
    }
    let dim_len = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
    if dim_len == 0 || dim_len > MAX_DIM_LEN || bytes.len() < 3 + dim_len + 17 {
        return None;
    }
    let dimension = std::str::from_utf8(&bytes[3..3 + dim_len]).ok()?.to_owned();
    let h = &bytes[3 + dim_len..];
    let detail = h[0];
    if detail > MAX_DETAIL {
        return None;
    }
    let x = i32::from_le_bytes([h[1], h[2], h[3], h[4]]);
    let z = i32::from_le_bytes([h[5], h[6], h[7], h[8]]);
    let revision = u64::from_le_bytes([h[9], h[10], h[11], h[12], h[13], h[14], h[15], h[16]]);
    let compressed = &h[17..];

    let decoder = ruzstd::decoding::StreamingDecoder::new(compressed).ok()?;
    let mut raw = Vec::new();
    decoder
        .take((2 + MAX_PARTIAL_COLUMNS * 18 + 1) as u64)
        .read_to_end(&mut raw)
        .ok()?;
    if raw.len() < 2 {
        return None;
    }
    let count = u16::from_le_bytes([raw[0], raw[1]]) as usize;
    if count == 0 || count > MAX_PARTIAL_COLUMNS || raw.len() != 2 + count * 18 {
        return None;
    }
    let mut columns = Vec::with_capacity(count);
    let mut off = 2usize;
    for _ in 0..count {
        let lx = raw[off];
        let lz = raw[off + 1];
        if (lx as usize) >= SECTION_SIZE || (lz as usize) >= SECTION_SIZE {
            return None;
        }
        let col = LodColumn {
            solid_top_y: i16::from_le_bytes([raw[off + 2], raw[off + 3]]),
            solid_bottom_y: i16::from_le_bytes([raw[off + 4], raw[off + 5]]),
            solid_rgba: [raw[off + 6], raw[off + 7], raw[off + 8], raw[off + 9]],
            fluid_top_y: i16::from_le_bytes([raw[off + 10], raw[off + 11]]),
            fluid_rgba: [raw[off + 12], raw[off + 13], raw[off + 14], raw[off + 15]],
            source: raw[off + 16],
            material_id: raw[off + 17],
        };
        columns.push((lx, lz, col));
        off += 18;
    }
    Some(PartialColumnUpdate {
        dimension,
        pos: SectionPos::new(detail, x, z),
        revision,
        columns,
    })
}

pub fn lod_sync_channel_id() -> PayloadId {
    PayloadId(LOD_SYNC_CHANNEL.to_owned())
}

/// Registers the optional bidirectional `distantsunrises:lod_sync` channel during
/// `Hooks::register`.
pub fn register_payloads(registrar: &mut dyn Registrar) {
    registrar
        .payload_handlers()
        .registrar(LOD_SYNC_VERSION)
        .optional()
        .executes_on(HandlerThread::Network)
        .droppable()
        .play_bidirectional(
            lod_sync_channel_id(),
            Arc::new(ServerLodSyncHandler),
            Some(Arc::new(ClientLodSyncHandler)),
        );
}

/// Encodes a client -> server section request frame.
pub fn encode_section_request(dimension: &str, pos: SectionPos, client_revision: u64) -> Vec<u8> {
    let dim_bytes = dimension.as_bytes();
    let dim_len = dim_bytes.len().min(MAX_DIM_LEN);
    let mut out = Vec::with_capacity(1 + 2 + dim_len + 1 + 4 + 4 + 8);
    out.push(MSG_SECTION_REQUEST);
    out.extend_from_slice(&(dim_len as u16).to_le_bytes());
    out.extend_from_slice(&dim_bytes[..dim_len]);
    out.push(pos.detail);
    out.extend_from_slice(&pos.x.to_le_bytes());
    out.extend_from_slice(&pos.z.to_le_bytes());
    out.extend_from_slice(&client_revision.to_le_bytes());
    out
}

/// Decodes a client -> server section request frame.
pub fn decode_section_request(bytes: &[u8]) -> Option<(String, SectionPos, u64)> {
    if bytes.len() < 3 || bytes[0] != MSG_SECTION_REQUEST {
        return None;
    }
    let dim_len = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
    if dim_len == 0 || dim_len > MAX_DIM_LEN || bytes.len() != 3 + dim_len + 17 {
        return None;
    }
    let dim = std::str::from_utf8(&bytes[3..3 + dim_len]).ok()?.to_owned();
    let rest = &bytes[3 + dim_len..];
    let detail = rest[0];
    if detail > MAX_DETAIL {
        return None;
    }
    let x = i32::from_le_bytes([rest[1], rest[2], rest[3], rest[4]]);
    let z = i32::from_le_bytes([rest[5], rest[6], rest[7], rest[8]]);
    let rev = u64::from_le_bytes([
        rest[9], rest[10], rest[11], rest[12], rest[13], rest[14], rest[15], rest[16],
    ]);
    Some((dim, SectionPos::new(detail, x, z), rev))
}

/// Encodes a server -> client Zstd-compressed LOD section response frame.
pub fn encode_section_data(dimension: &str, section: &LodSection) -> Option<Vec<u8>> {
    let dim_bytes = dimension.as_bytes();
    if dim_bytes.is_empty() || dim_bytes.len() > MAX_DIM_LEN {
        return None;
    }
    let raw = encode_section(section);
    let compressed = ruzstd::encoding::compress_to_vec(
        raw.as_slice(),
        ruzstd::encoding::CompressionLevel::Fastest,
    );
    let total = 1 + 2 + dim_bytes.len() + 8 + compressed.len();
    if total > MAX_WIRE_PAYLOAD_BYTES {
        return None;
    }
    let mut out = Vec::with_capacity(total);
    out.push(MSG_SECTION_DATA);
    out.extend_from_slice(&(dim_bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(dim_bytes);
    out.extend_from_slice(&section.revision.to_le_bytes());
    out.extend_from_slice(&compressed);
    Some(out)
}

/// Decodes a server -> client Zstd-compressed LOD section response frame.
pub fn decode_section_data(bytes: &[u8]) -> Option<(String, LodSection)> {
    if bytes.len() < 11 || bytes[0] != MSG_SECTION_DATA {
        return None;
    }
    let dim_len = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
    if dim_len == 0 || dim_len > MAX_DIM_LEN || bytes.len() < 3 + dim_len + 8 {
        return None;
    }
    let dim = std::str::from_utf8(&bytes[3..3 + dim_len]).ok()?.to_owned();
    let rev_start = 3 + dim_len;
    let revision = u64::from_le_bytes([
        bytes[rev_start],
        bytes[rev_start + 1],
        bytes[rev_start + 2],
        bytes[rev_start + 3],
        bytes[rev_start + 4],
        bytes[rev_start + 5],
        bytes[rev_start + 6],
        bytes[rev_start + 7],
    ]);
    let compressed = &bytes[rev_start + 8..];
    let decoder = ruzstd::decoding::StreamingDecoder::new(compressed).ok()?;
    let mut raw = Vec::new();
    decoder
        .take(MAX_DECOMPRESSED_SECTION_BYTES as u64 + 1)
        .read_to_end(&mut raw)
        .ok()?;
    if raw.len() > MAX_DECOMPRESSED_SECTION_BYTES {
        return None;
    }
    let mut section = decode_section(&raw)?;
    if section.pos.detail > MAX_DETAIL {
        return None;
    }
    section.revision = revision;
    Some((dim, section))
}

/// Called from `Hooks::client_tick_pre` on remote multiplayer connections to pull missing
/// visible LOD sections from a DH-enabled Sunrise server.
pub fn poll_remote_sections(host: &mut dyn ClientHost) {
    let is_remote = host
        .client_session()
        .client_server()
        .map(|s| !s.integrated_local)
        .unwrap_or(false);
    if !is_remote {
        return;
    }
    let channel = lod_sync_channel_id();
    if !host.has_channel(&channel) {
        return;
    }

    let rt = runtime();
    let need_cfg_req = {
        let mut st = rt.inner.lock().unwrap();
        if !st.remote_config_requested {
            st.remote_config_requested = true;
            true
        } else {
            false
        }
    };
    if need_cfg_req {
        let _ =
            host.send_payload_to_server(&channel, PayloadBytes(encode_session_config_request()));
    }

    let requests: Vec<(String, SectionPos, u64)> = {
        let mut st = rt.inner.lock().unwrap();
        if !st.config.enabled || !st.has_camera {
            return;
        }
        let dim = st.active_dimension.clone();
        if st.disabled_builtin_net.contains(&dim) || st.remote_providers.contains_key(&dim) {
            return;
        }
        let desired = DimensionLodStore::desired_sections_coarse_to_fine(
            st.camera_pos[0],
            st.camera_pos[2],
            st.config,
        );
        // Trigger local disk cache loads for unseen remote sections first.
        if let Some(cache_dir) = st.remote_cache_dir.clone() {
            let store = st.dimensions.entry(dim.clone()).or_default();
            let mut loads = 0usize;
            for &pos in &desired {
                if store.disk_checked.insert(pos) {
                    rt.io.request_load(cache_dir.clone(), dim.clone(), pos);
                    loads += 1;
                    if loads >= 8 {
                        break;
                    }
                }
            }
        }

        let mut batch = Vec::with_capacity(MAX_REQUESTS_PER_TICK);
        for pos in desired {
            let existing_rev = st
                .dimensions
                .get(&dim)
                .and_then(|d| d.sections.get(&pos))
                .filter(|s| s.populated > 0)
                .map(|s| s.revision)
                .unwrap_or(0);
            if st
                .remote_requested
                .entry(dim.clone())
                .or_default()
                .insert(pos)
            {
                batch.push((dim.clone(), pos, existing_rev));
                if batch.len() >= MAX_REQUESTS_PER_TICK {
                    break;
                }
            }
        }
        batch
    };

    for (dim, pos, rev) in requests {
        let bytes = encode_section_request(&dim, pos, rev);
        let _ = host.send_payload_to_server(&channel, PayloadBytes(bytes));
    }
}

struct ServerLodSyncHandler;

impl IPayloadHandler for ServerLodSyncHandler {
    fn handle(
        &self,
        payload: PayloadBytes,
        context: &mut dyn IPayloadContext,
    ) -> Result<(), PayloadHandleError> {
        if payload.0.first() == Some(&MSG_SESSION_CONFIG_REQUEST) {
            let cfg = SessionConfig::default();
            let _ = context.reply(
                &lod_sync_channel_id(),
                PayloadBytes(encode_session_config(cfg)),
            );
            return Ok(());
        }

        let Some((dim, pos, client_rev)) = decode_section_request(&payload.0) else {
            return Ok(());
        };

        let sender_key = context
            .player()
            .map(|p| format!("{p:?}"))
            .unwrap_or_else(|_| "global".to_owned());
        let now = Instant::now();
        if !rate_limiter()
            .lock()
            .unwrap()
            .allow_request(&sender_key, now)
        {
            return Ok(());
        }

        let rt = runtime();
        let section_opt = {
            let st = rt.inner.lock().unwrap();
            if st.disabled_builtin_net.contains(&dim) {
                return Ok(());
            }
            st.dimensions
                .get(&dim)
                .and_then(|d| d.sections.get(&pos))
                .filter(|s| s.populated > 0 && s.revision != client_rev)
                .cloned()
        };
        if let Some(sec) = section_opt
            && let Some(wire) = encode_section_data(&dim, &sec)
            && rate_limiter()
                .lock()
                .unwrap()
                .allow_bytes(&sender_key, wire.len(), now)
        {
            let _ = context.reply(&lod_sync_channel_id(), PayloadBytes(wire));
        }
        Ok(())
    }
}

struct ClientLodSyncHandler;

impl IPayloadHandler for ClientLodSyncHandler {
    fn handle(
        &self,
        payload: PayloadBytes,
        _context: &mut dyn IPayloadContext,
    ) -> Result<(), PayloadHandleError> {
        match payload.0.first().copied() {
            Some(MSG_SECTION_DATA) => {
                if let Some((dim, section)) = decode_section_data(&payload.0) {
                    ingest_remote_section(&dim, section);
                }
            }
            Some(MSG_SESSION_CONFIG) => {
                if let Some(scfg) = decode_session_config(&payload.0) {
                    let rt = runtime();
                    let mut st = rt.inner.lock().unwrap();
                    st.remote_session_config = Some(scfg);
                }
            }
            Some(MSG_PARTIAL_UPDATE) => {
                if let Some(update) = decode_partial_update(&payload.0) {
                    ingest_partial_update(update);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn ingest_partial_update(update: PartialColumnUpdate) {
    let pos = update.pos;
    let dim = update.dimension;
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let cache_dir = st.remote_cache_dir.clone();
    let max_extra = st.config.max_extra_spans();
    let store = st.dimensions.entry(dim.clone()).or_default();
    let sec = store.get_or_create_mut(pos);
    let mut changed = false;
    for (lx, lz, col) in update.columns {
        if sec.set_cell(lx as usize, lz as usize, col) {
            changed = true;
        }
    }
    sec.revision = sec.revision.max(update.revision);
    if !changed {
        return;
    }
    let mut to_mesh = HashSet::new();
    to_mesh.insert(pos);
    for parent in store.propagate_up_with_quality(pos, max_extra) {
        to_mesh.insert(parent);
    }
    for mpos in to_mesh {
        if let Some(s) = store.sections.get_mut(&mpos)
            && s.populated > 0
        {
            s.dirty_mesh = false;
            rt.mesher.submit(dim.clone(), s.clone());
            if let Some(ref dir) = cache_dir {
                s.dirty_save = false;
                rt.io.request_save(dir.clone(), dim.clone(), s.clone());
            }
        }
    }
}

fn ingest_remote_section(dim: &str, section: LodSection) {
    if section.populated == 0 {
        return;
    }
    let pos = section.pos;
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let cache_dir = st.remote_cache_dir.clone();
    let max_extra = st.config.max_extra_spans();
    let store = st.dimensions.entry(dim.to_owned()).or_default();
    store.sections.insert(pos, section);

    let mut to_mesh = HashSet::new();
    to_mesh.insert(pos);
    for parent in store.propagate_up_with_quality(pos, max_extra) {
        to_mesh.insert(parent);
    }
    for mpos in to_mesh {
        if let Some(sec) = store.sections.get_mut(&mpos)
            && sec.populated > 0
        {
            sec.dirty_mesh = false;
            rt.mesher.submit(dim.to_owned(), sec.clone());
            if let Some(ref dir) = cache_dir {
                sec.dirty_save = false;
                rt.io.request_save(dir.clone(), dim.to_owned(), sec.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod::{ExtraSpan, LodColumn, SOURCE_LOADED_CHUNK};

    #[test]
    fn section_request_and_compressed_data_roundtrip() {
        let pos = SectionPos::new(2, -7, 13);
        let req = encode_section_request("minecraft:overworld", pos, 99);
        assert_eq!(
            decode_section_request(&req),
            Some(("minecraft:overworld".to_owned(), pos, 99))
        );
        // Reject invalid detail > MAX_DETAIL
        let mut bad_req = req.clone();
        let detail_idx = 3 + "minecraft:overworld".len();
        bad_req[detail_idx] = MAX_DETAIL + 1;
        assert_eq!(decode_section_request(&bad_req), None);

        // Build section with primary surface + cave extra span
        let mut sec = LodSection::new(pos);
        sec.revision = 42;
        let col = LodColumn::from_solid_and_fluid(
            70,
            60,
            [90, 140, 70],
            Some(74),
            [40, 90, 200],
            15,
            0,
            SOURCE_LOADED_CHUNK,
        );
        sec.set_cell_with_spans(
            10,
            12,
            col,
            vec![ExtraSpan::new(40, 20, [110, 110, 110], 0, 5, false)],
        );

        let wire = encode_section_data("minecraft:overworld", &sec).expect("encode section");
        let (dim, decoded) = decode_section_data(&wire).expect("decode section");
        assert_eq!(dim, "minecraft:overworld");
        assert_eq!(decoded.pos, pos);
        assert_eq!(decoded.revision, sec.revision);
        assert_eq!(decoded.get(10, 12), col);
        assert_eq!(decoded.extra_spans_at(10, 12).len(), 1);
        assert_eq!(decoded.extra_spans_at(10, 12)[0].top_y, 40);
    }

    #[test]
    fn session_config_partial_update_and_rate_limiter_work() {
        let cfg = SessionConfig {
            max_render_distance_chunks: 192,
            distant_generation_allowed: false,
            partial_updates_enabled: true,
            max_bytes_per_sec: 512 * 1024,
        };
        assert_eq!(
            decode_session_config(&encode_session_config(cfg)),
            Some(cfg)
        );

        let col = LodColumn::from_solid_and_fluid(
            80,
            70,
            [100, 180, 90],
            None,
            [0, 0, 0],
            15,
            2,
            SOURCE_LOADED_CHUNK,
        );
        let upd = PartialColumnUpdate {
            dimension: "minecraft:overworld".to_owned(),
            pos: SectionPos::new(0, 3, -4),
            revision: 77,
            columns: vec![(5, 9, col), (6, 9, col)],
        };
        let wire = encode_partial_update(&upd).expect("encode partial update");
        let dec = decode_partial_update(&wire).expect("decode partial update");
        assert_eq!(dec, upd);

        let now = Instant::now();
        let mut lim = PlayerRateLimiter::new(2, 1024);
        assert!(lim.allow_request("p1", now));
        assert!(lim.allow_request("p1", now));
        assert!(!lim.allow_request("p1", now));
        assert!(lim.allow_bytes("p1", 1024, now));
        assert!(!lim.allow_bytes("p1", 1, now));

        assert_eq!(
            server_storage_subdir("play.example.com:25565", "minecraft:the_nether"),
            "play_example_com_25565__minecraft_the_nether"
        );
        let root_a = remote_cache_root(Path::new("/inst"), "a:b");
        let root_b = remote_cache_root(Path::new("/inst"), "a_b");
        assert_ne!(root_a, root_b);
        assert!(!root_a.to_string_lossy().contains(".."));
    }
}
