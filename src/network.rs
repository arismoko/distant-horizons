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

/// DH-exact server-to-client 32³ octree LOD sync over one raw Sunrise play payload channel.
///
/// # Why raw bytes, not a typed `Wire` payload
/// Typed payloads cap every list/`Vec<u8>` at 4096 entries and every string at 4096 bytes
/// (`sunrise-mod-vocab/src/wire/limits.rs`: `PAYLOAD_MAX_LIST_ENTRIES`, `PAYLOAD_MAX_STRING_BYTES`),
/// and encode little-endian. DH's split chunk is 1,048,000 bytes and its codec is Netty
/// big-endian, so the DH layout can only ride a raw channel (`PayloadRegistrar::play_bidirectional`
/// with [`PayloadBytes`]).
///
/// # Size limits
/// A payload on a channel the receiver registered is bounded by the frame limits: 2,097,151 bytes
/// per frame and 8,388,608 uncompressed. A payload sent to a receiver that does not know the
/// channel is held to vanilla's `DiscardedPayload` limits instead: 1,048,576 bytes clientbound
/// (`ClientboundCustomPayloadPacket.MAX_PAYLOAD_SIZE`) and 32,767 serverbound
/// (`ServerboundCustomPayloadPacket.MAX_PAYLOAD_SIZE`). DH's 1,048,000-byte split is kept so a
/// chunk also fits the 1,048,576 limit when the receiver has not registered the channel.
///
/// # Framing (DH `AbstractPluginPacketSender.java`)
/// `u16 protocol version` (`:142`, `:58`), `u16 message id` (`:148`, decode `:100`), message body,
/// all big-endian (Netty `ByteBuf`). Message ids are the registration order of
/// `MessageRegistry.java:88-93` (`id = size + 1`): SessionConfig 4 (`:63`), Exception 6 (`:67`),
/// FullDataSourceRequest 7 (`:70`), FullDataSourceResponse 8 (`:71`), FullDataPartialUpdate 9
/// (`:72`), FullDataSplit 10 (`:73`). `CloseReason`, `LevelInit`, `RequestLevelInit` and `Cancel`
/// are not carried: the level key is the Sunrise dimension identifier inside each message and a
/// stale request is simply answered or ignored.
///
/// Reference root: `coreSubProjects/core/src/main/java/com/seibel/distanthorizons/core/`
/// (abbreviated `DH:` below).
pub mod octree_sync {
    use crate::dh_v2::{DhV2CompressionMode, decompress_blob};
    use crate::octree::{
        MAX_OCTREE_DETAIL, OctreeSection, OctreeSectionPos, RleSection, Run, SECTION_VOLUME, Voxel,
    };
    use mod_api::{
        ClientHost, HandlerThread, IPayloadContext, IPayloadHandler, PayloadBytes,
        PayloadHandleError, PayloadId, PlayerRef, Registrar, ServerHost,
    };
    use std::collections::{HashMap, VecDeque};
    use std::hash::Hash;
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    /// Raw play channel carrying every octree sync message.
    pub const OCTREE_SYNC_CHANNEL: &str = "distantsunrises:octree_sync";
    /// Negotiated channel version (the raw channel has no typed schema).
    pub const OCTREE_SYNC_VERSION: &str = "1";
    /// Our protocol number, written where DH writes `ModInfo.PROTOCOL_VERSION`
    /// (`DH: api/.../coreapi/ModInfo.java:34` = 17). Not DH's value: the payloads differ.
    pub const PROTOCOL_VERSION: u16 = 1;

    /// `FullDataPayloadSender.FULL_DATA_SPLIT_SIZE_IN_BYTES` ("1 Mebibyte minus 576 bytes for
    /// other info"), `DH: multiplayer/fullData/FullDataPayloadSender.java:21`.
    pub const FULL_DATA_SPLIT_SIZE_IN_BYTES: usize = 1_048_000;
    /// `FullDataPayloadSender.TICK_RATE`, `FullDataPayloadSender.java:18`; also Minecraft's tick rate.
    pub const TICK_RATE: i64 = 20;
    /// Vanilla clientbound custom-payload body limit for a channel the receiver does not know (see module docs).
    pub const CLIENTBOUND_BODY_LIMIT: usize = 1_048_576;
    /// Vanilla serverbound custom-payload body limit for a channel the receiver does not know (see module docs).
    pub const SERVERBOUND_BODY_LIMIT: usize = 32_767;
    /// Bytes a split message adds around its chunk: protocol u16, id u16, bufferId i32, size i32,
    /// isFirst bool (`FullDataSplitMessage.java:63-68`).
    pub const SPLIT_OVERHEAD_BYTES: usize = 2 + 2 + 4 + 4 + 1;
    const _: () =
        assert!(FULL_DATA_SPLIT_SIZE_IN_BYTES + SPLIT_OVERHEAD_BYTES <= CLIENTBOUND_BODY_LIMIT);

    /// A receiver drops a half-received buffer 30 s after its last access
    /// (`FullDataPayloadReceiver.java:23-25`, `expireAfterAccess(30, SECONDS)`).
    pub const RECEIVE_BUFFER_TTL: Duration = Duration::from_secs(30);
    /// Our hardening (DH has none): the largest reassembled payload buffer a client accepts.
    pub const MAX_PAYLOAD_BUFFER_BYTES: usize = 8 * 1024 * 1024;
    /// Our hardening: the largest decompressed blob. A section is at most
    /// `4 + 32768*8 + 4 + 32768*4` bytes of palette and runs.
    pub const MAX_RAW_SECTION_BYTES: usize = 4 + SECTION_VOLUME * 8 + 4 + SECTION_VOLUME * 4;
    const MAX_MAPPING_ENTRIES: usize = 2 * SECTION_VOLUME;
    /// Our DTO layout version (DH's `FullDataSourceV2DTO.DATA_FORMAT` is 1..=3, `:61-66`).
    pub const OCTREE_DTO_FORMAT_V1: u8 = 1;

    // ------------------------------------------------------------------ errors

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum SyncError {
        Truncated,
        Trailing,
        BadProtocol(u16),
        BadMessageId(u16),
        BadString,
        StringTooLong,
        BadPos,
        BadEnum(u8),
        BadLength,
        BufferTooLarge,
        Compression(String),
        UnsupportedEncode(u8),
        ChecksumMismatch,
        Octree(crate::octree::OctreeError),
    }

    // ------------------------------------------------------------------ byte codec (Netty big-endian)

    struct Reader<'a>(&'a [u8]);

    impl<'a> Reader<'a> {
        fn bytes(&mut self, n: usize) -> Result<&'a [u8], SyncError> {
            if self.0.len() < n {
                return Err(SyncError::Truncated);
            }
            let (head, tail) = self.0.split_at(n);
            self.0 = tail;
            Ok(head)
        }
        fn u8(&mut self) -> Result<u8, SyncError> {
            Ok(self.bytes(1)?[0])
        }
        fn u16(&mut self) -> Result<u16, SyncError> {
            Ok(u16::from_be_bytes(
                self.bytes(2)?.try_into().expect("2 bytes"),
            ))
        }
        fn i32(&mut self) -> Result<i32, SyncError> {
            Ok(i32::from_be_bytes(
                self.bytes(4)?.try_into().expect("4 bytes"),
            ))
        }
        fn u32(&mut self) -> Result<u32, SyncError> {
            Ok(u32::from_be_bytes(
                self.bytes(4)?.try_into().expect("4 bytes"),
            ))
        }
        fn i64(&mut self) -> Result<i64, SyncError> {
            Ok(i64::from_be_bytes(
                self.bytes(8)?.try_into().expect("8 bytes"),
            ))
        }
        fn u64(&mut self) -> Result<u64, SyncError> {
            Ok(u64::from_be_bytes(
                self.bytes(8)?.try_into().expect("8 bytes"),
            ))
        }
        /// `ByteBuf.readBoolean`: any non-zero byte is true.
        fn bool(&mut self) -> Result<bool, SyncError> {
            Ok(self.u8()? != 0)
        }
        /// `INetworkObject.readStringStatic`: unsigned-short length, then UTF-8 (`INetworkObject.java:91-96`).
        fn string(&mut self) -> Result<String, SyncError> {
            let len = self.u16()? as usize;
            String::from_utf8(self.bytes(len)?.to_vec()).map_err(|_| SyncError::BadString)
        }
        /// A non-negative `int` length that the remaining bytes can hold at `item` bytes each.
        fn len(&mut self, item: usize) -> Result<usize, SyncError> {
            let len = usize::try_from(self.i32()?).map_err(|_| SyncError::BadLength)?;
            if len
                .checked_mul(item.max(1))
                .is_none_or(|need| need > self.0.len())
            {
                return Err(SyncError::BadLength);
            }
            Ok(len)
        }
        fn finish(self) -> Result<(), SyncError> {
            if self.0.is_empty() {
                Ok(())
            } else {
                Err(SyncError::Trailing)
            }
        }
    }

    fn put_u16(out: &mut Vec<u8>, v: u16) {
        out.extend_from_slice(&v.to_be_bytes());
    }
    fn put_i32(out: &mut Vec<u8>, v: i32) {
        out.extend_from_slice(&v.to_be_bytes());
    }
    fn put_i64(out: &mut Vec<u8>, v: i64) {
        out.extend_from_slice(&v.to_be_bytes());
    }
    fn put_bool(out: &mut Vec<u8>, v: bool) {
        out.push(u8::from(v));
    }
    /// `INetworkObject.writeStringStatic` (`INetworkObject.java:84-89`).
    fn put_string(out: &mut Vec<u8>, s: &str) -> Result<(), SyncError> {
        let len = u16::try_from(s.len()).map_err(|_| SyncError::StringTooLong)?;
        put_u16(out, len);
        out.extend_from_slice(s.as_bytes());
        Ok(())
    }

    // ------------------------------------------------------------------ section position on the wire

    /// DH sends `long sectionPos` (`FullDataSourceRequestMessage.java:129,163`). DH's position is
    /// 2D; ours is 3D, so the one `long` is Voxy's world-section id
    /// (`detail << 60 | (y & 0xFF) << 52 | (z & 0xFFFFFF) << 28 | (x & 0xFFFFFF) << 4`, see
    /// [`OctreeSectionPos::key`]). Refuses a position that does not survive the round trip
    /// (|x|,|z| < 2^23 sections, y in -128..=127, detail <= `MAX_OCTREE_DETAIL`).
    pub fn pos_to_wire(pos: OctreeSectionPos) -> Result<i64, SyncError> {
        let wire = pos.key() as i64;
        if pos_from_wire(wire)? != pos {
            return Err(SyncError::BadPos);
        }
        Ok(wire)
    }

    /// Inverse of [`pos_to_wire`]; the low four bits must be zero.
    pub fn pos_from_wire(wire: i64) -> Result<OctreeSectionPos, SyncError> {
        let k = wire as u64;
        let detail = (k >> 60) as u8;
        if detail > MAX_OCTREE_DETAIL || k & 0xF != 0 {
            return Err(SyncError::BadPos);
        }
        let y = i32::from(((k >> 52) & 0xFF) as u8 as i8);
        let z = ((((k >> 28) & 0xFF_FFFF) as i32) << 8) >> 8;
        let x = ((((k >> 4) & 0xFF_FFFF) as i32) << 8) >> 8;
        Ok(OctreeSectionPos::new(detail, x, y, z))
    }

    // ------------------------------------------------------------------ session config

    /// `EDhApiGeneratorPlan` in declaration order (the enum codec writes the ordinal byte,
    /// `INetworkObject.java:189-192`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(u8)]
    pub enum GeneratorPlan {
        SurfaceThenChunks = 0,
        SurfaceOnly = 1,
        ChunksOnly = 2,
        Disabled = 3,
    }

    impl GeneratorPlan {
        fn from_ordinal(v: u8) -> Result<Self, SyncError> {
            Ok(match v {
                0 => Self::SurfaceThenChunks,
                1 => Self::SurfaceOnly,
                2 => Self::ChunksOnly,
                3 => Self::Disabled,
                other => return Err(SyncError::BadEnum(other)),
            })
        }
        fn generation_enabled(self) -> bool {
            self != Self::Disabled
        }
        fn chunk_gen(self) -> bool {
            matches!(self, Self::SurfaceThenChunks | Self::ChunksOnly)
        }
        fn surface_gen(self) -> bool {
            matches!(self, Self::SurfaceThenChunks | Self::SurfaceOnly)
        }
    }

    /// `SessionConfig` (`DH: multiplayer/config/SessionConfig.java`). Field order is the
    /// registration order of `:37-82`, which is the transmission order (`:35`, `:190`). Defaults
    /// and ranges are `Config.java` `(min, default, max)`; decode clamps to the range (our
    /// hardening, DH trusts the peer).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SessionConfig {
        /// `Config.java:1492-1494` default `SURFACE_THEN_CHUNKS`.
        pub generator_plan: GeneratorPlan,
        /// `Config.Server.maxGenerationRequestDistance` (256, 4096, 4096), `Config.java:2053`.
        pub max_generation_request_distance: i32,
        /// `generationCenterChunkX` (MIN, 0, MAX), `Config.java:1633`.
        pub generation_center_chunk_x: i32,
        /// `generationCenterChunkZ` (MIN, 0, MAX), `Config.java:1641`.
        pub generation_center_chunk_z: i32,
        /// `generationMaxChunkRadius` (0, 0, MAX), `Config.java:1649`.
        pub generation_max_chunk_radius: i32,
        /// `generationRequestRateLimit` (1, 20, 100), `Config.java:2044`.
        pub generation_request_rate_limit: i32,
        /// `enableRealTimeUpdates` default true, `Config.java:2063`.
        pub enable_real_time_updates: bool,
        /// `realTimeUpdateDistanceRadiusInChunks` (32, 256, 4096), `Config.java:2071`.
        pub real_time_update_distance_radius_in_chunks: i32,
        /// `synchronizeOnLoad` default true, `Config.java:2081`.
        pub synchronize_on_load: bool,
        /// `maxSyncOnLoadRequestDistance` (256, 4096, 4096), `Config.java:2098`.
        pub max_sync_on_load_request_distance: i32,
        /// `syncOnLoadRateLimit` (1, 50, 100), `Config.java:2089`.
        pub sync_on_load_rate_limit: i32,
        /// `playerBandwidthLimit` in KB/s (0, 500, 1_000_000); 0 disables, `Config.java:2109`.
        pub player_bandwidth_limit: i32,
    }

    impl Default for SessionConfig {
        fn default() -> Self {
            Self {
                generator_plan: GeneratorPlan::SurfaceThenChunks,
                max_generation_request_distance: 4096,
                generation_center_chunk_x: 0,
                generation_center_chunk_z: 0,
                generation_max_chunk_radius: 0,
                generation_request_rate_limit: 20,
                enable_real_time_updates: true,
                real_time_update_distance_radius_in_chunks: 256,
                synchronize_on_load: true,
                max_sync_on_load_request_distance: 4096,
                sync_on_load_rate_limit: 50,
                player_bandwidth_limit: 500,
            }
        }
    }

    impl SessionConfig {
        /// The value both sides end up with: `SessionConfig.getValue` applies each entry's
        /// constrainer as `apply(constrainingConfig.value, ownValue)` (`SessionConfig.java:161-163`),
        /// so `client` is the constraining side and `server` the owner
        /// (`ServerPlayerState.java:62-65`). Constrainers: plan `:37-57`, `min` for distances and
        /// rate limits, `logicalAnd` for the two enable flags, server wins for the generation
        /// bounds (`:60-62`), bandwidth `:72-82`.
        pub fn negotiate(client: &Self, server: &Self) -> Self {
            let bandwidth =
                if client.player_bandwidth_limit == 0 && server.player_bandwidth_limit == 0 {
                    0
                } else {
                    let pick = |v: i32| if v > 0 { v } else { i32::MAX };
                    pick(client.player_bandwidth_limit).min(pick(server.player_bandwidth_limit))
                };
            Self {
                generator_plan: merge_plan(client.generator_plan, server.generator_plan),
                max_generation_request_distance: client
                    .max_generation_request_distance
                    .min(server.max_generation_request_distance),
                generation_center_chunk_x: server.generation_center_chunk_x,
                generation_center_chunk_z: server.generation_center_chunk_z,
                generation_max_chunk_radius: server.generation_max_chunk_radius,
                generation_request_rate_limit: client
                    .generation_request_rate_limit
                    .min(server.generation_request_rate_limit),
                enable_real_time_updates: client.enable_real_time_updates
                    && server.enable_real_time_updates,
                real_time_update_distance_radius_in_chunks: client
                    .real_time_update_distance_radius_in_chunks
                    .min(server.real_time_update_distance_radius_in_chunks),
                synchronize_on_load: client.synchronize_on_load && server.synchronize_on_load,
                max_sync_on_load_request_distance: client
                    .max_sync_on_load_request_distance
                    .min(server.max_sync_on_load_request_distance),
                sync_on_load_rate_limit: client
                    .sync_on_load_rate_limit
                    .min(server.sync_on_load_rate_limit),
                player_bandwidth_limit: bandwidth,
            }
        }

        fn encode(&self, out: &mut Vec<u8>) {
            out.push(self.generator_plan as u8);
            put_i32(out, self.max_generation_request_distance);
            put_i32(out, self.generation_center_chunk_x);
            put_i32(out, self.generation_center_chunk_z);
            put_i32(out, self.generation_max_chunk_radius);
            put_i32(out, self.generation_request_rate_limit);
            put_bool(out, self.enable_real_time_updates);
            put_i32(out, self.real_time_update_distance_radius_in_chunks);
            put_bool(out, self.synchronize_on_load);
            put_i32(out, self.max_sync_on_load_request_distance);
            put_i32(out, self.sync_on_load_rate_limit);
            put_i32(out, self.player_bandwidth_limit);
        }

        fn decode(r: &mut Reader<'_>) -> Result<Self, SyncError> {
            Ok(Self {
                generator_plan: GeneratorPlan::from_ordinal(r.u8()?)?,
                max_generation_request_distance: r.i32()?.clamp(256, 4096),
                generation_center_chunk_x: r.i32()?,
                generation_center_chunk_z: r.i32()?,
                generation_max_chunk_radius: r.i32()?.max(0),
                generation_request_rate_limit: r.i32()?.clamp(1, 100),
                enable_real_time_updates: r.bool()?,
                real_time_update_distance_radius_in_chunks: r.i32()?.clamp(32, 4096),
                synchronize_on_load: r.bool()?,
                max_sync_on_load_request_distance: r.i32()?.clamp(256, 4096),
                sync_on_load_rate_limit: r.i32()?.clamp(1, 100),
                player_bandwidth_limit: r.i32()?.clamp(0, 1_000_000),
            })
        }
    }

    fn merge_plan(client: GeneratorPlan, server: GeneratorPlan) -> GeneratorPlan {
        use GeneratorPlan::*;
        if client == Disabled || server == Disabled {
            return Disabled;
        }
        let surface = client.surface_gen() && server.surface_gen();
        let chunks = client.chunk_gen() && server.chunk_gen();
        for plan in [SurfaceThenChunks, SurfaceOnly, ChunksOnly, Disabled] {
            if plan.generation_enabled()
                && plan.surface_gen() == surface
                && plan.chunk_gen() == chunks
            {
                return plan;
            }
        }
        server
    }

    // ------------------------------------------------------------------ messages

    const ID_SESSION_CONFIG: u16 = 4;
    const ID_EXCEPTION: u16 = 6;
    const ID_REQUEST: u16 = 7;
    const ID_RESPONSE: u16 = 8;
    const ID_PARTIAL_UPDATE: u16 = 9;
    const ID_SPLIT: u16 = 10;

    /// `ExceptionMessage.EXCEPTION_LIST` order (`ExceptionMessage.java`): the index is the wire value.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(i32)]
    pub enum ExceptionKind {
        RateLimited = 0,
        RequestOutOfRange = 1,
        RequestRejected = 2,
        SectionRequiresSplitting = 3,
    }

    /// `ExceptionMessage` (trackable: answers one request by `future_id`).
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ExceptionMessage {
        pub future_id: i32,
        pub kind: ExceptionKind,
        pub message: String,
    }

    /// `FullDataSourceRequestMessage` (client -> server), `FullDataSourceRequestMessage.java:159-176`.
    /// `client_timestamp` is only present when requesting changes (`:131-133`).
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct OctreeSectionRequest {
        pub future_id: i32,
        /// The Sunrise dimension identifier (DH: `ILevelWrapper.getDhIdentifier`).
        pub level: String,
        pub pos: OctreeSectionPos,
        /// The client's known revision/timestamp of this section.
        pub client_timestamp: Option<i64>,
    }

    /// One beacon beam (`BeaconBeamDTO.java:54-64`: block pos as three ints, ARGB int).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct BeaconBeam {
        pub x: i32,
        pub y: i32,
        pub z: i32,
        pub argb: i32,
    }

    static LAST_BUFFER_ID: AtomicI32 = AtomicI32::new(0);

    /// `FullDataPayload` (`DH: multiplayer/fullData/FullDataPayload.java:67-79`): the buffer id
    /// (`:70`, allocated from `lastBufferId.getAndIncrement()` at `:24`/`:42`) and the beacon
    /// beams as an int-counted collection (`:71`). The DTO bytes travel in split messages.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct OctreePayload {
        pub buffer_id: i32,
        pub beacon_beams: Vec<BeaconBeam>,
    }

    impl OctreePayload {
        /// Allocate a buffer id and encode `dto` as the buffer the split messages will carry.
        pub fn new(
            dto: &OctreeSectionDto,
            beacon_beams: Vec<BeaconBeam>,
        ) -> Result<(Self, Vec<u8>), SyncError> {
            let buffer = dto.encode()?;
            let buffer_id = LAST_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
            Ok((
                Self {
                    buffer_id,
                    beacon_beams,
                },
                buffer,
            ))
        }

        fn encode(&self, out: &mut Vec<u8>) {
            put_i32(out, self.buffer_id);
            put_i32(out, self.beacon_beams.len() as i32);
            for b in &self.beacon_beams {
                put_i32(out, b.x);
                put_i32(out, b.y);
                put_i32(out, b.z);
                put_i32(out, b.argb);
            }
        }

        fn decode(r: &mut Reader<'_>) -> Result<Self, SyncError> {
            let buffer_id = r.i32()?;
            let count = r.len(16)?;
            let mut beacon_beams = Vec::with_capacity(count);
            for _ in 0..count {
                beacon_beams.push(BeaconBeam {
                    x: r.i32()?,
                    y: r.i32()?,
                    z: r.i32()?,
                    argb: r.i32()?,
                });
            }
            Ok(Self {
                buffer_id,
                beacon_beams,
            })
        }
    }

    /// `FullDataSourceResponseMessage` (server -> client): `None` means "no data / client is up
    /// to date" (`FullDataSourceResponseMessage.java:222-223,251-261`), the 304-like answer.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct OctreeSectionResponse {
        pub future_id: i32,
        pub payload: Option<OctreePayload>,
    }

    /// `FullDataPartialUpdateMessage` (server -> client live edit), `:332-344`.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct OctreePartialUpdate {
        pub level: String,
        pub payload: OctreePayload,
    }

    /// `FullDataSplitMessage`: one chunk of a payload buffer (`FullDataSplitMessage.java:60-80`).
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct SplitMessage {
        pub buffer_id: i32,
        pub buffer: Vec<u8>,
        pub is_first: bool,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Message {
        SessionConfig(SessionConfig),
        Exception(ExceptionMessage),
        Request(OctreeSectionRequest),
        Response(OctreeSectionResponse),
        PartialUpdate(OctreePartialUpdate),
        Split(SplitMessage),
    }

    impl Message {
        pub fn encode(&self) -> Result<Vec<u8>, SyncError> {
            let mut out = Vec::new();
            put_u16(&mut out, PROTOCOL_VERSION);
            match self {
                Message::SessionConfig(cfg) => {
                    put_u16(&mut out, ID_SESSION_CONFIG);
                    cfg.encode(&mut out);
                }
                Message::Exception(e) => {
                    put_u16(&mut out, ID_EXCEPTION);
                    put_i32(&mut out, e.future_id);
                    put_i32(&mut out, e.kind as i32);
                    put_string(&mut out, &e.message)?;
                }
                Message::Request(r) => {
                    put_u16(&mut out, ID_REQUEST);
                    put_i32(&mut out, r.future_id);
                    put_string(&mut out, &r.level)?;
                    put_i64(&mut out, pos_to_wire(r.pos)?);
                    put_bool(&mut out, r.client_timestamp.is_some());
                    if let Some(ts) = r.client_timestamp {
                        put_i64(&mut out, ts);
                    }
                }
                Message::Response(r) => {
                    put_u16(&mut out, ID_RESPONSE);
                    put_i32(&mut out, r.future_id);
                    put_bool(&mut out, r.payload.is_some());
                    if let Some(p) = &r.payload {
                        p.encode(&mut out);
                    }
                }
                Message::PartialUpdate(u) => {
                    put_u16(&mut out, ID_PARTIAL_UPDATE);
                    put_string(&mut out, &u.level)?;
                    u.payload.encode(&mut out);
                }
                Message::Split(s) => {
                    put_u16(&mut out, ID_SPLIT);
                    put_i32(&mut out, s.buffer_id);
                    put_i32(&mut out, s.buffer.len() as i32);
                    out.extend_from_slice(&s.buffer);
                    put_bool(&mut out, s.is_first);
                }
            }
            Ok(out)
        }

        pub fn decode(bytes: &[u8]) -> Result<Self, SyncError> {
            let mut r = Reader(bytes);
            let protocol = r.u16()?;
            if protocol != PROTOCOL_VERSION {
                return Err(SyncError::BadProtocol(protocol));
            }
            let msg = match r.u16()? {
                ID_SESSION_CONFIG => Message::SessionConfig(SessionConfig::decode(&mut r)?),
                ID_EXCEPTION => {
                    let future_id = r.i32()?;
                    let kind = match r.i32()? {
                        0 => ExceptionKind::RateLimited,
                        1 => ExceptionKind::RequestOutOfRange,
                        2 => ExceptionKind::RequestRejected,
                        3 => ExceptionKind::SectionRequiresSplitting,
                        _ => return Err(SyncError::BadLength),
                    };
                    Message::Exception(ExceptionMessage {
                        future_id,
                        kind,
                        message: r.string()?,
                    })
                }
                ID_REQUEST => {
                    let future_id = r.i32()?;
                    let level = r.string()?;
                    let pos = pos_from_wire(r.i64()?)?;
                    let client_timestamp = if r.bool()? { Some(r.i64()?) } else { None };
                    Message::Request(OctreeSectionRequest {
                        future_id,
                        level,
                        pos,
                        client_timestamp,
                    })
                }
                ID_RESPONSE => {
                    let future_id = r.i32()?;
                    let payload = if r.bool()? {
                        Some(OctreePayload::decode(&mut r)?)
                    } else {
                        None
                    };
                    Message::Response(OctreeSectionResponse { future_id, payload })
                }
                ID_PARTIAL_UPDATE => {
                    let level = r.string()?;
                    Message::PartialUpdate(OctreePartialUpdate {
                        level,
                        payload: OctreePayload::decode(&mut r)?,
                    })
                }
                ID_SPLIT => {
                    let buffer_id = r.i32()?;
                    let size = r.len(1)?;
                    if size > FULL_DATA_SPLIT_SIZE_IN_BYTES {
                        return Err(SyncError::BadLength);
                    }
                    let buffer = r.bytes(size)?.to_vec();
                    Message::Split(SplitMessage {
                        buffer_id,
                        buffer,
                        is_first: r.bool()?,
                    })
                }
                other => return Err(SyncError::BadMessageId(other)),
            };
            r.finish()?;
            Ok(msg)
        }
    }

    // ------------------------------------------------------------------ section DTO

    /// One `(id -> name)` mapping entry shipped with a section. A voxel's block and biome ids are
    /// session-local mapper ids (`Voxel` bit layout), so the receiver needs the names, exactly as
    /// DH ships `FullDataPointIdMap` (`FullDataSourceV2DTO.java:740-747`).
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct MappingEntry {
        /// 0 = block state, 1 = biome.
        pub kind: u8,
        pub id: u32,
        pub name: String,
    }

    /// The octree counterpart of `FullDataSourceV2DTO` (`DH: sql/dto/FullDataSourceV2DTO.java`).
    /// Wire layout (`encode` `:763-801`, big-endian): `long pos`, `int checksum`, `int len + data`,
    /// `int len + mapping`, `byte dataFormatVersion`, `byte compressionMode`, three bools
    /// (applyToParent/applyToChildren/regenerate, always false here), `long lastModified`,
    /// `long created`. DH's four adjacent-data, generation-step and world-compression blobs are
    /// dropped: the mesher derives borders from neighbouring sections and a section has one
    /// generation state.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct OctreeSectionDto {
        pub pos: OctreeSectionPos,
        /// CRC-32 of the uncompressed data blob (DH: `dataSource.hashCode()`, `:130`).
        pub checksum: i32,
        pub data: Vec<u8>,
        pub mapping: Vec<u8>,
        pub format_version: u8,
        /// `EDhApiDataCompressionMode` value of both blobs (`EDhApiDataCompressionMode.java:45-87`:
        /// UNCOMPRESSED 0, LZ4 1, Z_STD_STREAM 2, LZMA2 3, Z_STD_BLOCK 4).
        pub compression: DhV2CompressionMode,
        /// The section's revision, used as DH's `lastModifiedUnixDateTime` for the 304-like check.
        pub last_modified: i64,
        pub created: i64,
    }

    /// `Config.Common.LodBuilding.dataCompression` default (`Config.java:1687-1688`).
    pub const DEFAULT_COMPRESSION: DhV2CompressionMode = DhV2CompressionMode::ZstdBlock;

    fn compress(raw: &[u8], mode: DhV2CompressionMode) -> Result<Vec<u8>, SyncError> {
        match mode {
            DhV2CompressionMode::Uncompressed => Ok(raw.to_vec()),
            DhV2CompressionMode::ZstdBlock | DhV2CompressionMode::ZstdStream => Ok(
                ruzstd::encoding::compress_to_vec(raw, ruzstd::encoding::CompressionLevel::Fastest),
            ),
            // Decodable (`decompress_blob`) but not encodable with the vendored crates.
            DhV2CompressionMode::Lz4 | DhV2CompressionMode::Lzma2 => {
                Err(SyncError::UnsupportedEncode(mode as u8))
            }
        }
    }

    fn decompress(data: &[u8], mode: DhV2CompressionMode) -> Result<Vec<u8>, SyncError> {
        let raw =
            decompress_blob(data, mode).map_err(|e| SyncError::Compression(format!("{e:?}")))?;
        if raw.len() > MAX_RAW_SECTION_BYTES.max(MAX_MAPPING_ENTRIES * (1 + 4 + 2 + 255)) {
            return Err(SyncError::BadLength);
        }
        Ok(raw)
    }

    impl OctreeSectionDto {
        /// Serialize `section` (palette + runs from [`OctreeSection::to_rle`]) and `mapping`.
        pub fn from_section(
            section: &OctreeSection,
            mapping: &[MappingEntry],
            mode: DhV2CompressionMode,
            created: i64,
        ) -> Result<Self, SyncError> {
            let rle = section.to_rle();
            let mut raw = Vec::with_capacity(8 + rle.palette.len() * 8 + rle.runs.len() * 4);
            put_i32(&mut raw, rle.palette.len() as i32);
            for v in &rle.palette {
                raw.extend_from_slice(&v.0.to_be_bytes());
            }
            put_i32(&mut raw, rle.runs.len() as i32);
            for run in &rle.runs {
                put_u16(&mut raw, run.palette_index);
                put_u16(&mut raw, run.length);
            }
            let mut map_raw = Vec::new();
            put_i32(&mut map_raw, mapping.len() as i32);
            for e in mapping {
                map_raw.push(e.kind);
                put_i32(&mut map_raw, e.id as i32);
                put_string(&mut map_raw, &e.name)?;
            }
            Ok(Self {
                pos: section.pos,
                checksum: crc32fast::hash(&raw) as i32,
                data: compress(&raw, mode)?,
                mapping: compress(&map_raw, mode)?,
                format_version: OCTREE_DTO_FORMAT_V1,
                compression: mode,
                last_modified: section.revision as i64,
                created,
            })
        }

        /// Decompress, verify the checksum and rebuild the section (`FullDataSourceV2DTO.decode`
        /// `:804-`). Malformed data is an `Err`, never a panic.
        pub fn into_section(&self) -> Result<(OctreeSection, Vec<MappingEntry>), SyncError> {
            if self.format_version != OCTREE_DTO_FORMAT_V1 {
                return Err(SyncError::BadEnum(self.format_version));
            }
            let raw = decompress(&self.data, self.compression)?;
            if crc32fast::hash(&raw) as i32 != self.checksum {
                return Err(SyncError::ChecksumMismatch);
            }
            let mut r = Reader(&raw);
            let palette_len = r.len(8)?;
            if palette_len == 0 || palette_len > SECTION_VOLUME {
                return Err(SyncError::BadLength);
            }
            let mut palette = Vec::with_capacity(palette_len);
            for _ in 0..palette_len {
                palette.push(Voxel(r.u64()?));
            }
            let run_count = r.len(4)?;
            if run_count == 0 || run_count > SECTION_VOLUME {
                return Err(SyncError::BadLength);
            }
            let mut runs = Vec::with_capacity(run_count);
            for _ in 0..run_count {
                runs.push(Run {
                    palette_index: r.u16()?,
                    length: r.u16()?,
                });
            }
            r.finish()?;
            let mut section = OctreeSection::from_rle(&RleSection {
                pos: self.pos,
                palette,
                runs,
            })
            .map_err(SyncError::Octree)?;
            section.revision = self.last_modified as u64;

            let map_raw = decompress(&self.mapping, self.compression)?;
            let mut m = Reader(&map_raw);
            let count = m.len(7)?;
            if count > MAX_MAPPING_ENTRIES {
                return Err(SyncError::BadLength);
            }
            let mut mapping = Vec::with_capacity(count);
            for _ in 0..count {
                mapping.push(MappingEntry {
                    kind: m.u8()?,
                    id: m.u32()?,
                    name: m.string()?,
                });
            }
            m.finish()?;
            Ok((section, mapping))
        }

        /// The buffer the split messages carry.
        pub fn encode(&self) -> Result<Vec<u8>, SyncError> {
            let mut out = Vec::with_capacity(self.data.len() + self.mapping.len() + 48);
            put_i64(&mut out, pos_to_wire(self.pos)?);
            put_i32(&mut out, self.checksum);
            put_i32(&mut out, self.data.len() as i32);
            out.extend_from_slice(&self.data);
            put_i32(&mut out, self.mapping.len() as i32);
            out.extend_from_slice(&self.mapping);
            out.push(self.format_version);
            out.push(self.compression as u8);
            put_bool(&mut out, false);
            put_bool(&mut out, false);
            put_bool(&mut out, false);
            put_i64(&mut out, self.last_modified);
            put_i64(&mut out, self.created);
            Ok(out)
        }

        pub fn decode(buffer: &[u8]) -> Result<Self, SyncError> {
            let mut r = Reader(buffer);
            let pos = pos_from_wire(r.i64()?)?;
            let checksum = r.i32()?;
            let n = r.len(1)?;
            let data = r.bytes(n)?.to_vec();
            let n = r.len(1)?;
            let mapping = r.bytes(n)?.to_vec();
            let format_version = r.u8()?;
            let mode = r.u8()?;
            let compression =
                DhV2CompressionMode::from_byte(mode).map_err(|_| SyncError::BadEnum(mode))?;
            r.bool()?;
            r.bool()?;
            r.bool()?;
            let last_modified = r.i64()?;
            let created = r.i64()?;
            r.finish()?;
            Ok(Self {
                pos,
                checksum,
                data,
                mapping,
                format_version,
                compression,
                last_modified,
                created,
            })
        }
    }

    /// The 304-like rule of `FullDataSourceRequestHandler.java:100-111`: a missing client
    /// timestamp counts as -1, and no server data or `server <= client` answers with an empty
    /// response instead of data.
    pub fn is_up_to_date(server_timestamp: Option<i64>, client_timestamp: Option<i64>) -> bool {
        match server_timestamp {
            None => true,
            Some(server) => server <= client_timestamp.unwrap_or(-1),
        }
    }

    // ------------------------------------------------------------------ splitting, pacing, reassembly

    struct PendingTransfer {
        buffer_id: i32,
        buffer: Vec<u8>,
        offset: usize,
        final_message: Vec<u8>,
    }

    /// `FullDataPayloadSender`: a per-player queue that drains at the player's bandwidth limit
    /// once per tick (`FullDataPayloadSender.java:54-95`).
    #[derive(Default)]
    pub struct PayloadSender {
        queue: VecDeque<PendingTransfer>,
    }

    impl PayloadSender {
        /// Queue `buffer` for chunked sending; `final_message` (the encoded response or partial
        /// update) is emitted right after its last chunk (`sendInChunks`, `:49-52`).
        pub fn send_in_chunks(&mut self, buffer_id: i32, buffer: Vec<u8>, final_message: Vec<u8>) {
            self.queue.push_back(PendingTransfer {
                buffer_id,
                buffer,
                offset: 0,
                final_message,
            });
        }

        pub fn is_idle(&self) -> bool {
            self.queue.is_empty()
        }

        /// One 20 Hz tick. `max_kbps` is the effective `playerBandwidthLimit` (0 = unlimited).
        /// Returns the encoded messages to send now, in order. Per tick the budget is
        /// `kbps * 1000 / 20 + 1` bytes (`:60-62`) and each chunk is
        /// `min(budget, 1_048_000, remaining)` (`:74`).
        pub fn tick(&mut self, max_kbps: i32) -> Vec<Vec<u8>> {
            let mut out = Vec::new();
            let mut budget: i64 = if max_kbps > 0 {
                i64::from(max_kbps) * 1000 / TICK_RATE + 1
            } else {
                i64::from(i32::MAX)
            };
            while budget > 0 {
                let Some(pending) = self.queue.front_mut() else {
                    break;
                };
                let remaining = pending.buffer.len() - pending.offset;
                let chunk = (budget as usize)
                    .min(FULL_DATA_SPLIT_SIZE_IN_BYTES)
                    .min(remaining);
                let split = Message::Split(SplitMessage {
                    buffer_id: pending.buffer_id,
                    buffer: pending.buffer[pending.offset..pending.offset + chunk].to_vec(),
                    is_first: pending.offset == 0,
                });
                if let Ok(bytes) = split.encode() {
                    out.push(bytes);
                }
                pending.offset += chunk;
                budget -= chunk as i64;
                if pending.offset == pending.buffer.len()
                    && let Some(done) = self.queue.pop_front()
                {
                    out.push(done.final_message);
                }
            }
            out
        }
    }

    /// `FullDataPayloadReceiver`: reassembles split messages by buffer id
    /// (`FullDataPayloadReceiver.java:33-54`). A first chunk starts a new buffer, a non-first
    /// chunk with no buffer is dropped (`:42-46`), buffers expire after 30 s of no access (`:24`).
    #[derive(Default)]
    pub struct SplitReceiver {
        buffers: HashMap<i32, (Vec<u8>, Instant)>,
    }

    impl SplitReceiver {
        pub fn receive(&mut self, msg: SplitMessage, now: Instant) -> Result<(), SyncError> {
            self.expire(now);
            let entry = if msg.is_first {
                self.buffers.insert(msg.buffer_id, (Vec::new(), now));
                self.buffers.get_mut(&msg.buffer_id)
            } else {
                self.buffers.get_mut(&msg.buffer_id)
            };
            let Some((buf, touched)) = entry else {
                return Ok(());
            };
            if buf.len() + msg.buffer.len() > MAX_PAYLOAD_BUFFER_BYTES {
                self.buffers.remove(&msg.buffer_id);
                return Err(SyncError::BufferTooLarge);
            }
            buf.extend_from_slice(&msg.buffer);
            *touched = now;
            Ok(())
        }

        /// Remove and return the completed buffer (`decodeDataSource`, `:56-71`).
        pub fn take(&mut self, buffer_id: i32) -> Option<Vec<u8>> {
            self.buffers.remove(&buffer_id).map(|(buf, _)| buf)
        }

        pub fn expire(&mut self, now: Instant) {
            self.buffers.retain(|_, (_, touched)| {
                now.saturating_duration_since(*touched) < RECEIVE_BUFFER_TTL
            });
        }
    }

    // ------------------------------------------------------------------ rate limiting

    /// `SupplierBasedRateAndConcurrencyLimiter` (`DH: util/ratelimiting/...java:36-53`): a
    /// concurrency limit (pending count may not exceed the rate, `SupplierBasedConcurrencyLimiter`
    /// `:38-47`) checked first, then a Guava `RateLimiter` (`SupplierBasedRateLimiter.java:24,48-58`):
    /// `max_rate` permits per second, the first permit free, idle time stores up to one second of
    /// permits. A failed rate check releases the concurrency slot.
    #[derive(Default)]
    pub struct RateAndConcurrencyLimiter {
        tokens: f64,
        last: Option<Instant>,
        pending: i32,
    }

    impl RateAndConcurrencyLimiter {
        pub fn try_acquire(&mut self, now: Instant, max_rate: i32) -> bool {
            let max_rate = max_rate.max(1);
            if self.pending + 1 > max_rate {
                return false;
            }
            self.pending += 1;
            let rate = f64::from(max_rate);
            self.tokens = match self.last {
                None => 1.0,
                Some(last) => (self.tokens
                    + now.saturating_duration_since(last).as_secs_f64() * rate)
                    .min(rate),
            };
            self.last = Some(now);
            if self.tokens >= 1.0 {
                self.tokens -= 1.0;
                true
            } else {
                self.pending -= 1;
                false
            }
        }

        pub fn release(&mut self) {
            self.pending = (self.pending - 1).max(0);
        }
    }

    /// `ServerPlayerState.RateLimiterSet` (`ServerPlayerState.java:133-147`), one per player and level.
    #[derive(Default)]
    pub struct LimiterSet {
        pub generation_request: RateAndConcurrencyLimiter,
        pub sync_on_load: RateAndConcurrencyLimiter,
    }

    /// Which limiter slot an admitted request holds; give it back with
    /// [`PlayerSyncState::release`] when the request is answered.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Admission {
        Generation,
        SyncOnLoad,
    }

    /// Chebyshev distance in chunks from a block position to a section, `DH:
    /// DhSectionPos.getChebyshevSignedBlockDistance(..) / 16` (`AbstractDhServerLevel.java:146`),
    /// clamped to 0 inside the section.
    pub fn chebyshev_chunk_distance(pos: OctreeSectionPos, block_x: i32, block_z: i32) -> i32 {
        let [ox, _, oz] = pos.block_origin();
        let span = pos.block_span();
        let axis = |p: i32, o: i32| {
            if p < o {
                o - p
            } else if p >= o + span {
                p - (o + span) + 1
            } else {
                0
            }
        };
        axis(block_x, ox).max(axis(block_z, oz)) / 16
    }

    /// `ServerPlayerState`: negotiated session config, per-level limiters, and the chunked sender.
    pub struct PlayerSyncState {
        server_config: SessionConfig,
        effective: Option<SessionConfig>,
        limiters: HashMap<String, LimiterSet>,
        pub sender: PayloadSender,
    }

    impl PlayerSyncState {
        pub fn new(server_config: SessionConfig) -> Self {
            Self {
                server_config,
                effective: None,
                limiters: HashMap::new(),
                sender: PayloadSender::default(),
            }
        }

        /// `isReady` (`ServerPlayerState.java:44`): the client sent its config.
        pub fn is_ready(&self) -> bool {
            self.effective.is_some()
        }

        pub fn effective(&self) -> Option<&SessionConfig> {
            self.effective.as_ref()
        }

        /// The client's `SessionConfigMessage` constrains ours; the reply is the result
        /// (`ServerPlayerState.java:62-65,104-110`).
        pub fn on_client_config(&mut self, client: &SessionConfig) -> SessionConfig {
            let merged = SessionConfig::negotiate(client, &self.server_config);
            self.effective = Some(merged);
            merged
        }

        /// Request admission in the order of `AbstractDhServerLevel.java:148-182` and
        /// `FullDataSourceRequestHandler.java:63-73,153-165`. The world-generation bounds and
        /// surface-only splitting checks (`AbstractDhServerLevel.java:158-173`) belong to the
        /// generator and are not repeated here.
        pub fn admit(
            &mut self,
            req: &OctreeSectionRequest,
            distance_chunks: i32,
            now: Instant,
        ) -> Result<Admission, ExceptionMessage> {
            let fail = |kind, message: String| ExceptionMessage {
                future_id: req.future_id,
                kind,
                message,
            };
            let Some(cfg) = self.effective else {
                return Err(fail(
                    ExceptionKind::RequestRejected,
                    "Session config not negotiated.".into(),
                ));
            };
            let limiters = self.limiters.entry(req.level.clone()).or_default();
            if req.client_timestamp.is_none() {
                if distance_chunks > cfg.max_generation_request_distance {
                    return Err(fail(
                        ExceptionKind::RequestOutOfRange,
                        format!(
                            "Distance too large: {distance_chunks} > {}",
                            cfg.max_generation_request_distance
                        ),
                    ));
                }
                if !cfg.generator_plan.generation_enabled() {
                    return Err(fail(
                        ExceptionKind::RequestRejected,
                        "Operation is disabled in config.".into(),
                    ));
                }
                if !limiters
                    .generation_request
                    .try_acquire(now, cfg.generation_request_rate_limit)
                {
                    return Err(fail(
                        ExceptionKind::RateLimited,
                        format!(
                            "Full data request rate limit: {}",
                            cfg.generation_request_rate_limit
                        ),
                    ));
                }
                Ok(Admission::Generation)
            } else {
                if distance_chunks > cfg.max_sync_on_load_request_distance {
                    return Err(fail(
                        ExceptionKind::RequestOutOfRange,
                        format!(
                            "Distance too large: {distance_chunks} > {}",
                            cfg.max_sync_on_load_request_distance
                        ),
                    ));
                }
                if !cfg.synchronize_on_load {
                    return Err(fail(
                        ExceptionKind::RequestRejected,
                        "Operation is disabled in config.".into(),
                    ));
                }
                if !limiters
                    .sync_on_load
                    .try_acquire(now, cfg.sync_on_load_rate_limit)
                {
                    return Err(fail(
                        ExceptionKind::RateLimited,
                        format!("Sync on login rate limit: {}", cfg.sync_on_load_rate_limit),
                    ));
                }
                Ok(Admission::SyncOnLoad)
            }
        }

        pub fn release(&mut self, level: &str, admission: Admission) {
            if let Some(set) = self.limiters.get_mut(level) {
                match admission {
                    Admission::Generation => set.generation_request.release(),
                    Admission::SyncOnLoad => set.sync_on_load.release(),
                }
            }
        }

        /// `AbstractDhServerLevel.java:262-272`: real-time updates must be enabled and the
        /// section within `realTimeUpdateDistanceRadiusInChunks` of the player.
        pub fn wants_partial_update(
            &self,
            pos: OctreeSectionPos,
            block_x: i32,
            block_z: i32,
        ) -> bool {
            self.effective.is_some_and(|cfg| {
                cfg.enable_real_time_updates
                    && chebyshev_chunk_distance(pos, block_x, block_z)
                        <= cfg.real_time_update_distance_radius_in_chunks
            })
        }

        fn bandwidth_kbps(&self) -> i32 {
            self.effective
                .unwrap_or(self.server_config)
                .player_bandwidth_limit
        }
    }

    /// What the server backend provides to the sync layer.
    pub trait ServerSource<K>: Send + Sync {
        /// `getTimestampForPos` (`FullDataSourceRequestHandler.java:104`).
        fn section_timestamp(&self, level: &str, pos: OctreeSectionPos) -> Option<i64>;
        fn section_dto(&self, level: &str, pos: OctreeSectionPos) -> Option<OctreeSectionDto>;
        /// The player's block position, used for the distance checks.
        fn player_block_xz(&self, player: K) -> Option<(i32, i32)>;
    }

    /// All players' sync state on the server.
    pub struct ServerSyncRegistry<K> {
        server_config: SessionConfig,
        players: HashMap<K, PlayerSyncState>,
    }

    impl<K: Copy + Eq + Hash> ServerSyncRegistry<K> {
        pub fn new(server_config: SessionConfig) -> Self {
            Self {
                server_config,
                players: HashMap::new(),
            }
        }

        pub fn remove_player(&mut self, player: K) {
            self.players.remove(&player);
        }

        pub fn player(&self, player: K) -> Option<&PlayerSyncState> {
            self.players.get(&player)
        }

        /// Handle one clientbound-visible message from `player`; returns immediate replies.
        /// Section data is not returned: it is queued on the player's [`PayloadSender`].
        pub fn handle(
            &mut self,
            player: K,
            msg: Message,
            now: Instant,
            source: &dyn ServerSource<K>,
        ) -> Vec<Message> {
            let server_config = self.server_config;
            let state = self
                .players
                .entry(player)
                .or_insert_with(|| PlayerSyncState::new(server_config));
            match msg {
                Message::SessionConfig(client) => {
                    vec![Message::SessionConfig(state.on_client_config(&client))]
                }
                Message::Request(req) => {
                    let (px, pz) = source.player_block_xz(player).unwrap_or((0, 0));
                    let distance = chebyshev_chunk_distance(req.pos, px, pz);
                    let admission = match state.admit(&req, distance, now) {
                        Ok(a) => a,
                        Err(e) => return vec![Message::Exception(e)],
                    };
                    let not_modified = Message::Response(OctreeSectionResponse {
                        future_id: req.future_id,
                        payload: None,
                    });
                    let timestamp = source.section_timestamp(&req.level, req.pos);
                    let dto = if is_up_to_date(timestamp, req.client_timestamp) {
                        None
                    } else {
                        source.section_dto(&req.level, req.pos)
                    };
                    state.release(&req.level, admission);
                    let Some(dto) = dto else {
                        return vec![not_modified];
                    };
                    match OctreePayload::new(&dto, Vec::new()) {
                        Ok((payload, buffer)) => {
                            let buffer_id = payload.buffer_id;
                            let response = Message::Response(OctreeSectionResponse {
                                future_id: req.future_id,
                                payload: Some(payload),
                            });
                            match response.encode() {
                                Ok(final_message) => {
                                    state
                                        .sender
                                        .send_in_chunks(buffer_id, buffer, final_message);
                                    Vec::new()
                                }
                                Err(_) => vec![not_modified],
                            }
                        }
                        Err(_) => vec![not_modified],
                    }
                }
                _ => Vec::new(),
            }
        }

        /// Queue one live edit for every `(player, block_x, block_z)` in `players` (already
        /// restricted to the edited level) that wants it; one shared buffer id like
        /// `AbstractDhServerLevel.java:252-274`. Returns how many players were queued.
        pub fn queue_partial_update(
            &mut self,
            level: &str,
            dto: &OctreeSectionDto,
            beacon_beams: Vec<BeaconBeam>,
            players: &[(K, i32, i32)],
        ) -> Result<usize, SyncError> {
            let (payload, buffer) = OctreePayload::new(dto, beacon_beams)?;
            let buffer_id = payload.buffer_id;
            let final_message = Message::PartialUpdate(OctreePartialUpdate {
                level: level.to_owned(),
                payload,
            })
            .encode()?;
            let mut queued = 0;
            for &(key, x, z) in players {
                if let Some(state) = self.players.get_mut(&key)
                    && state.wants_partial_update(dto.pos, x, z)
                {
                    state
                        .sender
                        .send_in_chunks(buffer_id, buffer.clone(), final_message.clone());
                    queued += 1;
                }
            }
            Ok(queued)
        }

        /// One 20 Hz server tick: the messages each player's bandwidth budget allows now.
        pub fn tick(&mut self) -> Vec<(K, Vec<Vec<u8>>)> {
            self.players
                .iter_mut()
                .filter_map(|(&key, state)| {
                    let kbps = state.bandwidth_kbps();
                    let msgs = state.sender.tick(kbps);
                    (!msgs.is_empty()).then_some((key, msgs))
                })
                .collect()
        }
    }

    // ------------------------------------------------------------------ client

    /// What a client learns from the server.
    #[derive(Clone, Debug)]
    pub enum ClientEvent {
        Config(SessionConfig),
        Section {
            level: String,
            section: OctreeSection,
            mapping: Vec<MappingEntry>,
        },
        UpToDate {
            level: String,
            pos: OctreeSectionPos,
        },
        PartialUpdate {
            level: String,
            section: OctreeSection,
            mapping: Vec<MappingEntry>,
        },
        Failed {
            level: String,
            pos: OctreeSectionPos,
            kind: ExceptionKind,
            message: String,
        },
        /// A response whose buffer never arrived (dropped chunk or expired).
        Lost {
            level: String,
            pos: OctreeSectionPos,
        },
    }

    #[derive(Default)]
    pub struct ClientSession {
        next_future: i32,
        pending: HashMap<i32, (String, OctreeSectionPos)>,
        receiver: SplitReceiver,
        pub config: Option<SessionConfig>,
    }

    impl ClientSession {
        /// The client's own config message, sent first (`ServerPlayerState.java:62-65`).
        pub fn config_message(local: &SessionConfig) -> Result<Vec<u8>, SyncError> {
            Message::SessionConfig(*local).encode()
        }

        /// Encode a request; `known_timestamp` is the cached section's revision, if any.
        pub fn request(
            &mut self,
            level: &str,
            pos: OctreeSectionPos,
            known_timestamp: Option<i64>,
        ) -> Result<Vec<u8>, SyncError> {
            let future_id = self.next_future;
            let bytes = Message::Request(OctreeSectionRequest {
                future_id,
                level: level.to_owned(),
                pos,
                client_timestamp: known_timestamp,
            })
            .encode()?;
            self.next_future = self.next_future.wrapping_add(1);
            self.pending.insert(future_id, (level.to_owned(), pos));
            Ok(bytes)
        }

        pub fn on_message(
            &mut self,
            bytes: &[u8],
            now: Instant,
        ) -> Result<Vec<ClientEvent>, SyncError> {
            Ok(match Message::decode(bytes)? {
                Message::SessionConfig(cfg) => {
                    self.config = Some(cfg);
                    vec![ClientEvent::Config(cfg)]
                }
                Message::Split(split) => {
                    self.receiver.receive(split, now)?;
                    Vec::new()
                }
                Message::Exception(e) => match self.pending.remove(&e.future_id) {
                    Some((level, pos)) => {
                        vec![ClientEvent::Failed {
                            level,
                            pos,
                            kind: e.kind,
                            message: e.message,
                        }]
                    }
                    None => Vec::new(),
                },
                Message::Response(resp) => {
                    let Some((level, pos)) = self.pending.remove(&resp.future_id) else {
                        return Ok(Vec::new());
                    };
                    match resp.payload {
                        None => vec![ClientEvent::UpToDate { level, pos }],
                        Some(payload) => match self.receiver.take(payload.buffer_id) {
                            None => vec![ClientEvent::Lost { level, pos }],
                            Some(buffer) => {
                                let (section, mapping) =
                                    OctreeSectionDto::decode(&buffer)?.into_section()?;
                                vec![ClientEvent::Section {
                                    level,
                                    section,
                                    mapping,
                                }]
                            }
                        },
                    }
                }
                Message::PartialUpdate(update) => {
                    match self.receiver.take(update.payload.buffer_id) {
                        None => Vec::new(),
                        Some(buffer) => {
                            let (section, mapping) =
                                OctreeSectionDto::decode(&buffer)?.into_section()?;
                            vec![ClientEvent::PartialUpdate {
                                level: update.level,
                                section,
                                mapping,
                            }]
                        }
                    }
                }
                Message::Request(_) => Vec::new(),
            })
        }
    }

    // ------------------------------------------------------------------ Sunrise payload glue

    pub fn octree_sync_channel_id() -> PayloadId {
        PayloadId(OCTREE_SYNC_CHANNEL.to_owned())
    }

    fn server_registry() -> &'static Mutex<ServerSyncRegistry<PlayerRef>> {
        static REG: OnceLock<Mutex<ServerSyncRegistry<PlayerRef>>> = OnceLock::new();
        REG.get_or_init(|| Mutex::new(ServerSyncRegistry::new(SessionConfig::default())))
    }

    fn server_source() -> &'static OnceLock<Arc<dyn ServerSource<PlayerRef>>> {
        static SRC: OnceLock<Arc<dyn ServerSource<PlayerRef>>> = OnceLock::new();
        &SRC
    }

    fn client_session() -> &'static Mutex<ClientSession> {
        static SESSION: OnceLock<Mutex<ClientSession>> = OnceLock::new();
        SESSION.get_or_init(|| Mutex::new(ClientSession::default()))
    }

    /// Events decoded by the client handler, drained by the render side.
    pub fn drain_client_events() -> Vec<ClientEvent> {
        std::mem::take(&mut *client_events().lock().unwrap())
    }

    fn client_events() -> &'static Mutex<Vec<ClientEvent>> {
        static EVENTS: OnceLock<Mutex<Vec<ClientEvent>>> = OnceLock::new();
        EVENTS.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Install the server backend; the first install wins.
    pub fn install_server_source(source: Arc<dyn ServerSource<PlayerRef>>) {
        let _ = server_source().set(source);
    }

    /// Forget a player's state on logout (`ServerPlayerState` lives per connection in DH).
    pub fn remove_player(player: PlayerRef) {
        server_registry().lock().unwrap().remove_player(player);
    }

    /// Declare the optional raw play channel during `Hooks::register`. Not `droppable`: a dropped
    /// middle chunk would corrupt a buffer.
    pub fn register_payloads(registrar: &mut dyn Registrar) {
        registrar
            .payload_handlers()
            .registrar(OCTREE_SYNC_VERSION)
            .optional()
            .executes_on(HandlerThread::Network)
            .play_bidirectional(
                octree_sync_channel_id(),
                Arc::new(ServerHandler),
                Some(Arc::new(ClientHandler)),
            );
    }

    struct ServerHandler;

    impl IPayloadHandler for ServerHandler {
        fn handle(
            &self,
            payload: PayloadBytes,
            context: &mut dyn IPayloadContext,
        ) -> Result<(), PayloadHandleError> {
            let failed = |e: String| PayloadHandleError::Failed(e);
            let msg = Message::decode(&payload.0).map_err(|e| failed(format!("{e:?}")))?;
            let player = context.player().map_err(|e| failed(format!("{e:?}")))?;
            let Some(source) = server_source().get() else {
                return Ok(());
            };
            let replies = server_registry().lock().unwrap().handle(
                player,
                msg,
                Instant::now(),
                source.as_ref(),
            );
            for reply in replies {
                if let Ok(bytes) = reply.encode() {
                    let _ = context.reply(&octree_sync_channel_id(), PayloadBytes(bytes));
                }
            }
            Ok(())
        }
    }

    struct ClientHandler;

    impl IPayloadHandler for ClientHandler {
        fn handle(
            &self,
            payload: PayloadBytes,
            _context: &mut dyn IPayloadContext,
        ) -> Result<(), PayloadHandleError> {
            let events = client_session()
                .lock()
                .unwrap()
                .on_message(&payload.0, Instant::now())
                .map_err(|e| PayloadHandleError::Failed(format!("{e:?}")))?;
            client_events().lock().unwrap().extend(events);
            Ok(())
        }
    }

    static CONFIG_SENT: AtomicBool = AtomicBool::new(false);

    /// Send our session config once per connection (`ServerPlayerState.java:62-65`).
    pub fn client_send_config_once(host: &mut dyn ClientHost) {
        if CONFIG_SENT.swap(true, Ordering::Relaxed) {
            return;
        }
        if let Ok(bytes) = ClientSession::config_message(&SessionConfig::default()) {
            let _ = host.send_payload_to_server(&octree_sync_channel_id(), PayloadBytes(bytes));
        }
    }

    /// Forget the per-connection state when the client leaves a server.
    pub fn client_reset() {
        CONFIG_SENT.store(false, Ordering::Relaxed);
        *client_session().lock().unwrap() = ClientSession::default();
        client_events().lock().unwrap().clear();
    }

    /// Send one request from the client tick.
    pub fn client_send_request(
        host: &mut dyn ClientHost,
        level: &str,
        pos: OctreeSectionPos,
        known_timestamp: Option<i64>,
    ) -> bool {
        let Ok(bytes) = client_session()
            .lock()
            .unwrap()
            .request(level, pos, known_timestamp)
        else {
            return false;
        };
        host.send_payload_to_server(&octree_sync_channel_id(), PayloadBytes(bytes))
            .is_ok()
    }

    /// Drain every player's budget for this server tick (20 Hz).
    pub fn server_tick(host: &mut dyn ServerHost) -> u32 {
        let out = server_registry().lock().unwrap().tick();
        let mut sent = 0;
        for (player, messages) in out {
            for message in messages {
                sent += host.players_send_payload(
                    &[player],
                    &octree_sync_channel_id(),
                    PayloadBytes(message),
                );
            }
        }
        sent
    }

    // ------------------------------------------------------------------ tests

    #[cfg(test)]
    mod tests {
        use super::*;

        const LEVEL: &str = "minecraft:overworld";

        fn varied_section(pos: OctreeSectionPos, distinct: u32) -> OctreeSection {
            let mut s = OctreeSection::new(pos);
            for i in 0..distinct as usize {
                s.set(
                    i & 31,
                    i >> 10,
                    (i >> 5) & 31,
                    Voxel::new(i as u32 + 1, (i % 7) as u16, 15, 3),
                );
            }
            s.revision = 77;
            s
        }

        fn deliver(
            session: &mut ClientSession,
            wire: Vec<Vec<u8>>,
            now: Instant,
        ) -> Vec<ClientEvent> {
            wire.iter()
                .flat_map(|m| session.on_message(m, now).unwrap())
                .collect()
        }

        struct Fake {
            dto: OctreeSectionDto,
        }
        impl ServerSource<u32> for Fake {
            fn section_timestamp(&self, _: &str, _: OctreeSectionPos) -> Option<i64> {
                Some(self.dto.last_modified)
            }
            fn section_dto(&self, _: &str, _: OctreeSectionPos) -> Option<OctreeSectionDto> {
                Some(self.dto.clone())
            }
            fn player_block_xz(&self, _: u32) -> Option<(i32, i32)> {
                Some((0, 0))
            }
        }

        #[test]
        fn request_roundtrips_with_and_without_timestamp_and_negative_pos() {
            for ts in [None, Some(123_456_789_i64)] {
                let req = OctreeSectionRequest {
                    future_id: -5,
                    level: LEVEL.into(),
                    pos: OctreeSectionPos::new(3, -1000, -7, 42),
                    client_timestamp: ts,
                };
                let wire = Message::Request(req.clone()).encode().unwrap();
                assert!(wire.len() < SERVERBOUND_BODY_LIMIT);
                assert_eq!(&wire[..4], &[0, 1, 0, 7]); // protocol 1, id 7, big-endian
                assert_eq!(Message::decode(&wire).unwrap(), Message::Request(req));
            }
            // Positions outside the key's range, bad detail and trailing bytes are refused.
            assert_eq!(
                pos_to_wire(OctreeSectionPos::new(0, 0, 200, 0)),
                Err(SyncError::BadPos)
            );
            assert_eq!(pos_from_wire((5u64 << 60) as i64), Err(SyncError::BadPos));
            let mut wire = Message::Request(OctreeSectionRequest {
                future_id: 0,
                level: LEVEL.into(),
                pos: OctreeSectionPos::new(0, 0, 0, 0),
                client_timestamp: None,
            })
            .encode()
            .unwrap();
            wire.push(0);
            assert_eq!(Message::decode(&wire), Err(SyncError::Trailing));
        }

        #[test]
        fn response_roundtrips_end_to_end_for_each_encodable_mode() {
            for mode in [
                DhV2CompressionMode::Uncompressed,
                DEFAULT_COMPRESSION,
                DhV2CompressionMode::ZstdStream,
            ] {
                let pos = OctreeSectionPos::new(1, 4, -2, 9);
                let section = varied_section(pos, 500);
                let mapping = vec![
                    MappingEntry {
                        kind: 0,
                        id: 1,
                        name: "minecraft:stone".into(),
                    },
                    MappingEntry {
                        kind: 1,
                        id: 0,
                        name: "minecraft:plains".into(),
                    },
                ];
                let dto = OctreeSectionDto::from_section(&section, &mapping, mode, 5).unwrap();
                assert_eq!(
                    OctreeSectionDto::decode(&dto.encode().unwrap()).unwrap(),
                    dto
                );

                let source = Fake { dto };
                let mut registry = ServerSyncRegistry::<u32>::new(SessionConfig::default());
                let mut client = ClientSession::default();
                let now = Instant::now();
                let cfg = registry.handle(
                    1,
                    Message::SessionConfig(SessionConfig::default()),
                    now,
                    &source,
                );
                assert_eq!(cfg.len(), 1);
                let req = client.request(LEVEL, pos, None).unwrap();
                assert!(
                    registry
                        .handle(1, Message::decode(&req).unwrap(), now, &source)
                        .is_empty()
                );
                let wire = registry.tick().pop().unwrap().1;
                let events = deliver(&mut client, wire, now);
                let [
                    ClientEvent::Section {
                        level,
                        section: got,
                        mapping: got_map,
                    },
                ] = events.as_slice()
                else {
                    panic!("{events:?}");
                };
                assert_eq!(level, LEVEL);
                assert_eq!(got.to_dense(), section.to_dense());
                assert_eq!((got.pos, got.revision), (pos, 77));
                assert_eq!(got_map, &mapping);
            }
        }

        #[test]
        fn up_to_date_request_gets_an_empty_response() {
            assert!(is_up_to_date(None, None));
            assert!(is_up_to_date(Some(5), Some(5)));
            assert!(!is_up_to_date(Some(6), Some(5)));
            assert!(!is_up_to_date(Some(0), None));
            let pos = OctreeSectionPos::new(0, 0, 0, 0);
            let dto = OctreeSectionDto::from_section(
                &varied_section(pos, 3),
                &[],
                DEFAULT_COMPRESSION,
                0,
            )
            .unwrap();
            let source = Fake { dto };
            let mut registry = ServerSyncRegistry::<u32>::new(SessionConfig::default());
            let mut client = ClientSession::default();
            let now = Instant::now();
            registry.handle(
                1,
                Message::SessionConfig(SessionConfig::default()),
                now,
                &source,
            );
            let req = client.request(LEVEL, pos, Some(77)).unwrap();
            let replies = registry.handle(1, Message::decode(&req).unwrap(), now, &source);
            let events = client
                .on_message(&replies[0].encode().unwrap(), now)
                .unwrap();
            assert!(matches!(events.as_slice(), [ClientEvent::UpToDate { .. }]));
        }

        #[test]
        fn large_buffer_splits_at_dh_size_and_reassembles() {
            // 2.5 MB of incompressible bytes: chunks of 1_048_000, 1_048_000 and 404_000.
            let mut x = 0x9E37_79B9_u32;
            let buffer: Vec<u8> = (0..2_500_000)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    x as u8
                })
                .collect();
            let mut sender = PayloadSender::default();
            sender.send_in_chunks(9, buffer.clone(), vec![0xAA]);
            let wire = sender.tick(0);
            assert!(sender.is_idle());
            assert_eq!(wire.len(), 4); // 3 split messages + the final message
            assert_eq!(wire[3], vec![0xAA]);
            let mut receiver = SplitReceiver::default();
            let now = Instant::now();
            let mut sizes = Vec::new();
            for (i, w) in wire[..3].iter().enumerate() {
                assert!(w.len() <= CLIENTBOUND_BODY_LIMIT);
                let Message::Split(s) = Message::decode(w).unwrap() else {
                    panic!()
                };
                assert_eq!(s.is_first, i == 0);
                sizes.push(s.buffer.len());
                receiver.receive(s, now).unwrap();
            }
            assert_eq!(sizes, vec![1_048_000, 1_048_000, 404_000]);
            assert_eq!(receiver.take(9).unwrap(), buffer);
            assert!(receiver.take(9).is_none());
        }

        #[test]
        fn real_32_cubed_section_with_distinct_voxels_survives_chunked_pacing() {
            // 32768 distinct voxels: ~390 KB of palette and runs, uncompressed.
            let pos = OctreeSectionPos::new(0, 2, 1, -3);
            let section = varied_section(pos, SECTION_VOLUME as u32);
            let dto =
                OctreeSectionDto::from_section(&section, &[], DhV2CompressionMode::Uncompressed, 0)
                    .unwrap();
            let buffer_len = dto.encode().unwrap().len();
            assert!(buffer_len > 380_000);
            let source = Fake { dto };
            let mut registry = ServerSyncRegistry::<u32>::new(SessionConfig::default());
            let mut client = ClientSession::default();
            let t0 = Instant::now();
            registry.handle(
                1,
                Message::SessionConfig(SessionConfig::default()),
                t0,
                &source,
            );
            let req = client.request(LEVEL, pos, None).unwrap();
            registry.handle(1, Message::decode(&req).unwrap(), t0, &source);
            // 500 KB/s => 25_001 bytes per 20 Hz tick, so several ticks.
            let mut ticks = 0;
            let mut events = Vec::new();
            while events.is_empty() {
                ticks += 1;
                assert!(ticks < 100);
                for (_, wire) in registry.tick() {
                    events.extend(deliver(&mut client, wire, t0));
                }
            }
            assert_eq!(ticks, buffer_len.div_ceil(25_001));
            let [ClientEvent::Section { section: got, .. }] = events.as_slice() else {
                panic!()
            };
            assert_eq!(got.to_dense(), section.to_dense());
        }

        #[test]
        fn dropped_chunk_is_detected() {
            let pos = OctreeSectionPos::new(0, 0, 0, 0);
            let section = varied_section(pos, 4000);
            let dto =
                OctreeSectionDto::from_section(&section, &[], DhV2CompressionMode::Uncompressed, 0)
                    .unwrap();
            let source = Fake { dto };
            let mut registry = ServerSyncRegistry::<u32>::new(SessionConfig::default());
            let mut client = ClientSession::default();
            let now = Instant::now();
            registry.handle(
                1,
                Message::SessionConfig(SessionConfig::default()),
                now,
                &source,
            );
            let req = client.request(LEVEL, pos, None).unwrap();
            registry.handle(1, Message::decode(&req).unwrap(), now, &source);
            let mut all = Vec::new();
            while all.is_empty() || !registry.player(1).unwrap().sender.is_idle() {
                all.extend(registry.tick().into_iter().flat_map(|(_, w)| w));
            }
            all.remove(1); // lose a middle chunk
            let mut failure = None;
            for w in &all {
                match client.on_message(w, now) {
                    Ok(_) => {}
                    Err(e) => failure = Some(e),
                }
            }
            assert!(failure.is_some());
        }

        #[test]
        fn partial_update_roundtrips_and_respects_real_time_settings() {
            let pos = OctreeSectionPos::new(0, 10, 0, 0);
            let section = varied_section(pos, 40);
            let dto =
                OctreeSectionDto::from_section(&section, &[], DEFAULT_COMPRESSION, 0).unwrap();
            let beams = vec![BeaconBeam {
                x: 1,
                y: 2,
                z: 3,
                argb: -16711936,
            }];
            let (payload, _) = OctreePayload::new(&dto, beams).unwrap();
            let msg = Message::PartialUpdate(OctreePartialUpdate {
                level: LEVEL.into(),
                payload,
            });
            assert_eq!(Message::decode(&msg.encode().unwrap()).unwrap(), msg);

            let mut registry = ServerSyncRegistry::<u32>::new(SessionConfig::default());
            let source = Fake { dto: dto.clone() };
            let now = Instant::now();
            for p in [1, 2] {
                registry.handle(
                    p,
                    Message::SessionConfig(SessionConfig::default()),
                    now,
                    &source,
                );
            }
            // Section at x blocks 320..352: player 1 is next to it, player 2 is >256 chunks away.
            let queued = registry
                .queue_partial_update(
                    LEVEL,
                    &dto,
                    Vec::new(),
                    &[(1, 330, 0), (2, 100_000, 0), (3, 0, 0)],
                )
                .unwrap();
            assert_eq!(queued, 1);
            let mut client = ClientSession::default();
            let wire = registry.tick().pop().unwrap().1;
            let events = deliver(&mut client, wire, now);
            let [
                ClientEvent::PartialUpdate {
                    level,
                    section: got,
                    ..
                },
            ] = events.as_slice()
            else {
                panic!()
            };
            assert_eq!(level, LEVEL);
            assert_eq!(got.to_dense(), section.to_dense());
        }

        #[test]
        fn session_config_roundtrips_and_negotiates_like_dh() {
            let d = SessionConfig::default();
            assert_eq!(
                (
                    d.generation_request_rate_limit,
                    d.sync_on_load_rate_limit,
                    d.player_bandwidth_limit
                ),
                (20, 50, 500)
            );
            let wire = Message::SessionConfig(d).encode().unwrap();
            assert_eq!(Message::decode(&wire).unwrap(), Message::SessionConfig(d));

            let client = SessionConfig {
                generator_plan: GeneratorPlan::SurfaceOnly,
                max_generation_request_distance: 512,
                generation_request_rate_limit: 5,
                enable_real_time_updates: false,
                sync_on_load_rate_limit: 100,
                player_bandwidth_limit: 0,
                ..d
            };
            let server = SessionConfig {
                generation_center_chunk_x: 9,
                player_bandwidth_limit: 800,
                ..d
            };
            let m = SessionConfig::negotiate(&client, &server);
            assert_eq!(m.generator_plan, GeneratorPlan::SurfaceOnly);
            assert_eq!(m.max_generation_request_distance, 512);
            assert_eq!(m.generation_request_rate_limit, 5);
            assert!(!m.enable_real_time_updates);
            assert_eq!(m.sync_on_load_rate_limit, 50);
            assert_eq!(m.generation_center_chunk_x, 9);
            assert_eq!(m.player_bandwidth_limit, 800);
            let off = SessionConfig {
                player_bandwidth_limit: 0,
                ..d
            };
            assert_eq!(
                SessionConfig::negotiate(&off, &off).player_bandwidth_limit,
                0
            );
            let dis = SessionConfig {
                generator_plan: GeneratorPlan::Disabled,
                ..d
            };
            assert_eq!(
                SessionConfig::negotiate(&d, &dis).generator_plan,
                GeneratorPlan::Disabled
            );
        }

        #[test]
        fn per_player_rate_limiter_matches_dh_semantics() {
            let t0 = Instant::now();
            let mut lim = RateAndConcurrencyLimiter::default();
            // Rate 2/s: the first permit is free, then one token per half second.
            assert!(lim.try_acquire(t0, 2));
            assert!(!lim.try_acquire(t0, 2));
            lim.release();
            assert!(lim.try_acquire(t0 + Duration::from_millis(500), 2));
            // Idle time stores at most one second of permits (2), the concurrency cap is 2.
            lim.release();
            let later = t0 + Duration::from_secs(60);
            assert!(lim.try_acquire(later, 2));
            assert!(lim.try_acquire(later, 2));
            assert!(!lim.try_acquire(later, 2)); // concurrency: two pending
            lim.release();
            assert!(!lim.try_acquire(later, 2)); // no tokens left, slot given back
            lim.release();

            // Per player and per level through admission.
            let mut registry = ServerSyncRegistry::<u32>::new(SessionConfig {
                generation_request_rate_limit: 1,
                ..SessionConfig::default()
            });
            let pos = OctreeSectionPos::new(0, 0, 0, 0);
            let dto = OctreeSectionDto::from_section(
                &varied_section(pos, 1),
                &[],
                DEFAULT_COMPRESSION,
                0,
            )
            .unwrap();
            let source = Fake { dto };
            for p in [1, 2] {
                registry.handle(
                    p,
                    Message::SessionConfig(SessionConfig::default()),
                    t0,
                    &source,
                );
            }
            let req = |future_id| {
                Message::Request(OctreeSectionRequest {
                    future_id,
                    level: LEVEL.into(),
                    pos,
                    client_timestamp: None,
                })
            };
            assert!(registry.handle(1, req(1), t0, &source).is_empty());
            let denied = registry.handle(1, req(2), t0, &source);
            assert!(matches!(
                denied.as_slice(),
                [Message::Exception(ExceptionMessage {
                    future_id: 2,
                    kind: ExceptionKind::RateLimited,
                    ..
                })]
            ));
            assert!(registry.handle(2, req(1), t0, &source).is_empty()); // another player is unaffected
            // Before the client's config arrives a request is rejected.
            let mut fresh = ServerSyncRegistry::<u32>::new(SessionConfig::default());
            let early = fresh.handle(9, req(1), t0, &source);
            assert!(matches!(
                early.as_slice(),
                [Message::Exception(ExceptionMessage {
                    kind: ExceptionKind::RequestRejected,
                    ..
                })]
            ));
        }

        #[test]
        fn chebyshev_distance_and_range_limit() {
            let pos = OctreeSectionPos::new(0, 0, 0, 0);
            assert_eq!(chebyshev_chunk_distance(pos, 10, 10), 0);
            assert_eq!(chebyshev_chunk_distance(pos, 32 + 160, 0), 10);
            let mut st = PlayerSyncState::new(SessionConfig::default());
            st.on_client_config(&SessionConfig::default());
            let req = OctreeSectionRequest {
                future_id: 1,
                level: LEVEL.into(),
                pos,
                client_timestamp: None,
            };
            let err = st.admit(&req, 5000, Instant::now()).unwrap_err();
            assert_eq!(err.kind, ExceptionKind::RequestOutOfRange);
        }
    }
}
