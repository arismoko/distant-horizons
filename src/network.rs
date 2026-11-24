//! Built-in multiplayer LOD synchronization over Sunrise's optional, droppable, network-thread
//! bidirectional play payload channel (`distanthorizons:lod_sync`).
//!
//! - **Optional**: joining vanilla servers or servers without Distant Horizons is never blocked;
//!   the client checks `host.has_channel(&lod_sync_channel_id())` before sending requests.
//! - **Network-thread & droppable**: Zstd compression/decompression and RLE encoding run on
//!   Sunrise's `HandlerThread::Network`, and congestion drops stale LOD packets cleanly instead
//!   of disconnecting players.
//! - **Singleplayer zero-overhead**: integrated singleplayer skips network sync because both
//!   sides share [`crate::render_bridge::DhRuntime`] in-process.
//! - **Overridable**: third-party network mods can call
//!   [`crate::api::disable_builtin_network`] and register a [`crate::api::RemoteLodProvider`]
//!   instead.

use crate::lod::{DimensionLodStore, LodSection, MAX_DETAIL, SectionPos};
use crate::render_bridge::runtime;
use crate::storage::{decode_section, encode_section};
use mod_api::{
    ClientHost, HandlerThread, IPayloadContext, IPayloadHandler, PayloadBytes, PayloadHandleError,
    PayloadId, Registrar,
};
use std::collections::HashSet;
use std::io::Read;
use std::sync::Arc;

/// Namespaced channel identifier negotiated between Sunrise DH clients and servers.
pub const LOD_SYNC_CHANNEL: &str = "distanthorizons:lod_sync";
/// Wire schema version for [`LOD_SYNC_CHANNEL`].
pub const LOD_SYNC_VERSION: &str = "1";

const MSG_SECTION_REQUEST: u8 = 1;
const MSG_SECTION_DATA: u8 = 2;

/// Maximum wire bytes emitted for one section reply (well below vanilla's 1 MiB play limit).
const MAX_WIRE_PAYLOAD_BYTES: usize = 512 * 1024;
/// Maximum decompressed RLE section size accepted from a server.
const MAX_DECOMPRESSED_SECTION_BYTES: usize = 4 * 1024 * 1024;
/// Maximum dimension identifier bytes accepted on the wire.
const MAX_DIM_LEN: usize = 256;
/// Maximum new section requests issued per client tick to avoid bursting the link.
const MAX_REQUESTS_PER_TICK: usize = 4;

pub fn lod_sync_channel_id() -> PayloadId {
    PayloadId(LOD_SYNC_CHANNEL.to_owned())
}

/// Registers the optional bidirectional `distanthorizons:lod_sync` channel during
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
        let mut batch = Vec::with_capacity(MAX_REQUESTS_PER_TICK);
        for pos in desired {
            let existing_rev = st
                .dimensions
                .get(&dim)
                .and_then(|d| d.sections.get(&pos))
                .filter(|s| s.populated > 0)
                .map(|s| s.revision);
            if existing_rev.is_some() {
                continue;
            }
            if st
                .remote_requested
                .entry(dim.clone())
                .or_default()
                .insert(pos)
            {
                batch.push((dim.clone(), pos, 0));
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
        let Some((dim, pos, client_rev)) = decode_section_request(&payload.0) else {
            return Ok(());
        };
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
        let Some((dim, section)) = decode_section_data(&payload.0) else {
            return Ok(());
        };
        ingest_remote_section(&dim, section);
        Ok(())
    }
}

fn ingest_remote_section(dim: &str, section: LodSection) {
    if section.populated == 0 {
        return;
    }
    let pos = section.pos;
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
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
}
