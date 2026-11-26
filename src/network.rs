//! Multiplayer LOD synchronization: server-to-client 32³ octree sections over one raw Sunrise
//! play payload channel (`distantsunrises:octree_sync`). Joining vanilla servers or servers
//! without Distant Sunrises is never blocked; the client checks `host.has_channel(..)` first.

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
        ClientHost, HandlerThread, IPayloadContext, IPayloadHandler, PayloadAccepted, PayloadBytes,
        PayloadHandleError, PayloadId, PayloadSendError, PlayerRef, Registrar, ServerHost,
    };
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::hash::Hash;
    use std::sync::atomic::{AtomicI32, Ordering};
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
            Self::from_rle(&section.to_rle(), section.revision, mapping, mode, created)
        }

        pub(crate) fn from_rle(
            rle: &RleSection,
            revision: u64,
            mapping: &[MappingEntry],
            mode: DhV2CompressionMode,
            created: i64,
        ) -> Result<Self, SyncError> {
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
                pos: rle.pos,
                checksum: crc32fast::hash(&raw) as i32,
                data: compress(&raw, mode)?,
                mapping: compress(&map_raw, mode)?,
                format_version: OCTREE_DTO_FORMAT_V1,
                compression: mode,
                last_modified: revision as i64,
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

    type SectionKey = (String, OctreeSectionPos);
    type EncodedUpdates = HashMap<(SectionKey, u64), Option<EncodedTransfer>>;

    struct TransferRevision {
        key: SectionKey,
        revision: u64,
    }

    #[derive(Clone)]
    struct EncodedTransfer {
        buffer_id: i32,
        buffer: Arc<[u8]>,
        final_message: Arc<[u8]>,
    }

    struct PendingTransfer {
        encoded: EncodedTransfer,
        offset: usize,
        revision: Option<TransferRevision>,
        // Full replies hold their slot even if level retirement clears the receipt revision.
        admission: Option<Admission>,
    }

    impl PendingTransfer {
        fn frame(&self, budget: usize) -> Option<(Vec<u8>, usize)> {
            let remaining = self.encoded.buffer.len() - self.offset;
            if remaining == 0 {
                return Some((self.encoded.final_message.to_vec(), 0));
            }
            let chunk = budget.min(FULL_DATA_SPLIT_SIZE_IN_BYTES).min(remaining);
            if chunk == 0 {
                return None;
            }
            let split = Message::Split(SplitMessage {
                buffer_id: self.encoded.buffer_id,
                buffer: self.encoded.buffer[self.offset..self.offset + chunk].to_vec(),
                is_first: self.offset == 0,
            });
            Some((split.encode().ok()?, chunk))
        }
    }

    /// `FullDataPayloadSender`: one ordered, bandwidth-paced queue per player. Advance a split
    /// only on local queue acceptance; retire a transfer on final acceptance or terminal refusal.
    #[derive(Default)]
    pub struct PayloadSender {
        queue: VecDeque<PendingTransfer>,
    }

    impl PayloadSender {
        pub fn is_idle(&self) -> bool {
            self.queue.is_empty()
        }

        /// Effective `playerBandwidthLimit` (0 = unlimited), per 20 Hz tick
        /// (`FullDataPayloadSender.java:60-74`). Only split bytes count toward the budget.
        fn budget(max_kbps: i32) -> usize {
            if max_kbps > 0 {
                (i64::from(max_kbps) * 1000 / TICK_RATE + 1) as usize
            } else {
                i32::MAX as usize
            }
        }
    }

    /// `FullDataPayloadReceiver`: reassembles split messages by buffer id
    /// (`FullDataPayloadReceiver.java:33-54`). A first chunk starts a new buffer, a non-first
    /// chunk with no buffer is dropped (`:42-46`), buffers expire after 30 s of no access (`:24`).
    #[derive(Default)]
    pub struct SplitReceiver {
        buffers: HashMap<i32, (Vec<u8>, Instant)>,
    }

    /// Our hardening (DH has none): the most half-received buffers a receiver holds at once. A
    /// further first chunk is dropped, so a hostile peer cannot pin `count * 8 MiB` in the 30 s TTL.
    pub const MAX_PENDING_BUFFERS: usize = 8;

    impl SplitReceiver {
        pub fn receive(&mut self, msg: SplitMessage, now: Instant) -> Result<(), SyncError> {
            self.expire(now);
            let entry = if msg.is_first {
                if !self.buffers.contains_key(&msg.buffer_id)
                    && self.buffers.len() >= MAX_PENDING_BUFFERS
                {
                    return Ok(());
                }
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

    /// `ServerPlayerState.RateLimiterSet` (`ServerPlayerState.java:133-147`), one per player. The
    /// set is keyed by the player alone: the level in a request is client-supplied and must not
    /// create limiter state.
    #[derive(Default)]
    pub struct LimiterSet {
        pub generation_request: RateAndConcurrencyLimiter,
        pub sync_on_load: RateAndConcurrencyLimiter,
    }

    /// Which limiter slot an admitted request holds. An immediate reply releases it; a queued
    /// full reply owns it until the transfer leaves the mod queue.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Admission {
        Generation,
        SyncOnLoad,
    }

    /// Chebyshev distance in chunks from a block position to a section, `DH:
    /// DhSectionPos.getChebyshevSignedBlockDistance(..) / 16` (`AbstractDhServerLevel.java:146`),
    /// clamped to 0 inside the section.
    pub fn chebyshev_chunk_distance(pos: OctreeSectionPos, block_x: i32, block_z: i32) -> i64 {
        let span = i64::from(pos.block_span());
        let axis = |block: i32, section: i32| {
            let block = i64::from(block);
            let origin = i64::from(section) * span;
            let end = origin + span;
            if block < origin {
                origin - block
            } else if block >= end {
                block - end + 1
            } else {
                0
            }
        };
        axis(block_x, pos.x).max(axis(block_z, pos.z)) / 16
    }

    #[derive(Default)]
    struct Receipt {
        sent: Option<u64>,
        queued_full: usize,
    }

    /// `ServerPlayerState`: negotiated config, limiters, and revisions eligible for live updates.
    pub struct PlayerSyncState {
        server_config: SessionConfig,
        effective: Option<SessionConfig>,
        limiters: LimiterSet,
        pub sender: PayloadSender,
        receipts: HashMap<SectionKey, Receipt>,
        deferred: HashMap<SectionKey, u64>,
    }

    impl PlayerSyncState {
        pub fn new(server_config: SessionConfig) -> Self {
            Self {
                server_config,
                effective: None,
                limiters: LimiterSet::default(),
                sender: PayloadSender::default(),
                receipts: HashMap::new(),
                deferred: HashMap::new(),
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
            distance_chunks: i64,
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
            let limiters = &mut self.limiters;
            if req.client_timestamp.is_none() {
                if distance_chunks > i64::from(cfg.max_generation_request_distance) {
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
                if distance_chunks > i64::from(cfg.max_sync_on_load_request_distance) {
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

        pub fn release(&mut self, admission: Admission) {
            match admission {
                Admission::Generation => self.limiters.generation_request.release(),
                Admission::SyncOnLoad => self.limiters.sync_on_load.release(),
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
                        <= i64::from(cfg.real_time_update_distance_radius_in_chunks)
            })
        }

        fn bandwidth_kbps(&self) -> i32 {
            self.effective
                .unwrap_or(self.server_config)
                .player_bandwidth_limit
        }

        fn partial_in_range(&self, key: &SectionKey, location: Option<&PlayerLocation>) -> bool {
            location.is_some_and(|p| {
                p.level == key.0 && self.wants_partial_update(key.1, p.block_x, p.block_z)
            })
        }

        fn prepare_transfer<K>(
            &mut self,
            location: Option<&PlayerLocation>,
            source: &dyn ServerSource<K>,
            encoded: &mut EncodedUpdates,
            budget: usize,
        ) {
            // Recheck eligibility at the start. Once the receiver has any split bytes, finish
            // the transfer even if the player moves, rather than strand a receive buffer.
            if let Some(front) = self.sender.queue.front()
                && front.offset == 0
                && front.admission.is_none()
                && let Some(rev) = &front.revision
            {
                if !self.partial_in_range(&rev.key, location) {
                    self.deferred.remove(&rev.key);
                    self.sender.queue.pop_front();
                } else if self.sender.queue.len() > 1
                    || self.deferred.get(&rev.key) != Some(&rev.revision)
                {
                    self.sender.queue.pop_front();
                }
            }
            if budget == 0 || !self.sender.is_idle() {
                return;
            }
            while let Some((key, &revision)) = self.deferred.iter().next() {
                let key = key.clone();
                if self.partial_in_range(&key, location) {
                    let transfer = encoded.entry((key.clone(), revision)).or_insert_with(|| {
                        let dto = source.section_dto(&key.0, key.1)?;
                        if dto.last_modified as u64 != revision {
                            return None;
                        }
                        let (payload, buffer) = OctreePayload::new(&dto, Vec::new()).ok()?;
                        let buffer_id = payload.buffer_id;
                        let final_message = Message::PartialUpdate(OctreePartialUpdate {
                            level: key.0.clone(),
                            payload,
                        })
                        .encode()
                        .ok()?;
                        Some(EncodedTransfer {
                            buffer_id,
                            buffer: buffer.into(),
                            final_message: final_message.into(),
                        })
                    });
                    if let Some(transfer) = transfer {
                        self.sender.queue.push_back(PendingTransfer {
                            encoded: transfer.clone(),
                            offset: 0,
                            revision: Some(TransferRevision { key, revision }),
                            admission: None,
                        });
                        break;
                    }
                }
                self.deferred.remove(&key);
            }
        }

        fn finish_transfer(&mut self, transfer: PendingTransfer, accepted: bool) {
            let partial = transfer.admission.is_none();
            if let Some(admission) = transfer.admission {
                self.release(admission);
            }
            let Some(TransferRevision { key, revision }) = transfer.revision else {
                return;
            };
            let receipt = self.receipts.get_mut(&key).unwrap();
            if !partial {
                receipt.queued_full -= 1;
            }
            if accepted {
                receipt.sent = Some(receipt.sent.map_or(revision, |sent| sent.max(revision)));
            }
            if receipt.sent.is_none() && receipt.queued_full == 0 {
                // A permanently refused initial reply must not subscribe this key.
                self.receipts.remove(&key);
                self.deferred.remove(&key);
            } else if self
                .deferred
                .get(&key)
                .is_some_and(|&pending| (accepted || partial) && pending <= revision)
            {
                self.deferred.remove(&key);
            }
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

    pub(crate) struct PlayerLocation {
        level: String,
        block_x: i32,
        block_z: i32,
    }

    impl PlayerLocation {
        pub(crate) fn new(level: String, pose: [f64; 3]) -> Self {
            // Match EntityRef::block_position without resolving the already known pose again.
            Self {
                level,
                block_x: pose[0].floor() as i32,
                block_z: pose[2].floor() as i32,
            }
        }
    }

    pub(crate) type PlayerLocations<K> = HashMap<K, PlayerLocation>;

    /// All players' sync state on the server.
    pub struct ServerSyncRegistry<K> {
        server_config: SessionConfig,
        open: bool,
        players: HashMap<K, PlayerSyncState>,
    }

    impl<K: Copy + Eq + Hash> ServerSyncRegistry<K> {
        pub fn new(server_config: SessionConfig) -> Self {
            Self {
                server_config,
                open: false,
                players: HashMap::new(),
            }
        }

        fn set_open(&mut self, open: bool) {
            self.players.clear();
            self.open = open;
        }

        pub fn remove_player(&mut self, player: K) {
            self.players.remove(&player);
        }

        fn receive(
            &mut self,
            player: impl FnOnce() -> Result<K, mod_api::PayloadContextError>,
            message: Message,
            now: Instant,
            source: &dyn ServerSource<K>,
        ) -> Result<Vec<Message>, mod_api::PayloadContextError> {
            if !self.open {
                return Ok(Vec::new());
            }
            // Resolve the live play epoch under the same lock as logout. Negotiation stays
            // lazy: a remote connection can send config as soon as its PlayerRef is bound,
            // before player_logged_in. The server closes intake before orderly stop logout.
            Ok(self.handle(player()?, message, now, source))
        }

        pub fn player(&self, player: K) -> Option<&PlayerSyncState> {
            self.players.get(&player)
        }

        fn remove_level(&mut self, level: &str) {
            for state in self.players.values_mut() {
                state.receipts.retain(|(d, _), _| d != level);
                state.deferred.retain(|(d, _), _| d != level);
                state.sender.queue.retain_mut(|transfer| {
                    if let Some(rev) = &transfer.revision
                        && rev.key.0 == level
                    {
                        if transfer.admission.is_none() {
                            return false;
                        }
                        // Keep the response and its admission, but do not reestablish a receipt
                        // for the retired level.
                        transfer.revision = None;
                    }
                    true
                });
            }
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
                    let timestamp = source.section_timestamp(&req.level, req.pos);
                    if !is_up_to_date(timestamp, req.client_timestamp)
                        && let Some(dto) = source.section_dto(&req.level, req.pos)
                        && let Ok((payload, buffer)) = OctreePayload::new(&dto, Vec::new())
                    {
                        let buffer_id = payload.buffer_id;
                        let response = Message::Response(OctreeSectionResponse {
                            future_id: req.future_id,
                            payload: Some(payload),
                        });
                        if let Ok(final_message) = response.encode() {
                            let key = (req.level, req.pos);
                            state.receipts.entry(key.clone()).or_default().queued_full += 1;
                            state.sender.queue.push_back(PendingTransfer {
                                encoded: EncodedTransfer {
                                    buffer_id,
                                    buffer: buffer.into(),
                                    final_message: final_message.into(),
                                },
                                offset: 0,
                                revision: Some(TransferRevision {
                                    key,
                                    revision: dto.last_modified as u64,
                                }),
                                admission: Some(admission),
                            });
                            return Vec::new();
                        }
                    }
                    state.release(admission);
                    vec![Message::Response(OctreeSectionResponse {
                        future_id: req.future_id,
                        payload: None,
                    })]
                }
                _ => Vec::new(),
            }
        }

        /// Record only actual mutations, not disk adoption or render dirtiness. A pending full
        /// response can qualify a key, but it must clear the sender before a partial can start.
        fn queue_partial_update(
            &mut self,
            level: &str,
            changes: impl IntoIterator<Item = (OctreeSectionPos, u64)>,
            players: &PlayerLocations<K>,
        ) {
            for (pos, revision) in changes {
                let key = (level.to_owned(), pos);
                for (&player, location) in players {
                    if let Some(state) = self.players.get_mut(&player)
                        && state.partial_in_range(&key, Some(location))
                        && let Some(receipt) = state.receipts.get(&key)
                        && receipt.sent.is_none_or(|sent| revision > sent)
                    {
                        state
                            .deferred
                            .entry(key.clone())
                            .and_modify(|rev| {
                                *rev = (*rev).max(revision);
                            })
                            .or_insert(revision);
                    }
                }
            }
        }

        /// Pace full replies and partials through the same queue. No host send holds the
        /// registry: a network callback can append full replies while a frame is in flight.
        fn tick(
            registry: &Mutex<Self>,
            locations: &PlayerLocations<K>,
            source: &dyn ServerSource<K>,
            mut send: impl FnMut(K, Vec<u8>) -> Result<(), PayloadSendError>,
        ) -> u32 {
            let players: Vec<_> = registry
                .lock()
                .unwrap()
                .players
                .iter()
                .map(|(&k, state)| (k, PayloadSender::budget(state.bandwidth_kbps())))
                .collect();
            let mut encoded = EncodedUpdates::new();
            let mut sent = 0;
            for (player, mut budget) in players {
                loop {
                    let (id, bytes, chunk) = {
                        let mut registry = registry.lock().unwrap();
                        let Some(state) = registry.players.get_mut(&player) else {
                            break;
                        };
                        let location = locations.get(&player);
                        state.prepare_transfer(location, source, &mut encoded, budget);
                        let Some(front) = state.sender.queue.front() else {
                            break;
                        };
                        let Some((bytes, chunk)) = front.frame(budget) else {
                            break;
                        };
                        (front.encoded.buffer_id, bytes, chunk)
                    };
                    let result = send(player, bytes);
                    let accepted = result.is_ok();
                    if accepted {
                        sent += 1;
                        budget -= chunk;
                    }
                    if matches!(result, Err(PayloadSendError::Backpressure)) {
                        // Leave the offset/final message in place. An unstarted partial can
                        // still be superseded by a newer revision or an intervening full reply.
                        break;
                    }
                    let mut registry = registry.lock().unwrap();
                    let Some(state) = registry.players.get_mut(&player) else {
                        break;
                    };
                    if let Some(front) = state.sender.queue.front_mut()
                        && front.encoded.buffer_id == id
                    {
                        if accepted && chunk != 0 {
                            front.offset += chunk;
                        } else {
                            let done = state.sender.queue.pop_front().unwrap();
                            state.finish_transfer(done, accepted);
                        }
                    } else {
                        break;
                    }
                    if !accepted {
                        break;
                    }
                }
            }
            sent
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
        /// A full response without a usable section.
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

        /// Encode and track a request for callers driving the protocol directly. Live sends
        /// track it only after the connection has accepted its bytes.
        pub fn request(
            &mut self,
            level: &str,
            pos: OctreeSectionPos,
            known_timestamp: Option<i64>,
        ) -> Result<Vec<u8>, SyncError> {
            let bytes = self.encode_request(level, pos, known_timestamp)?;
            self.track_request(level, pos);
            Ok(bytes)
        }

        fn encode_request(
            &self,
            level: &str,
            pos: OctreeSectionPos,
            known_timestamp: Option<i64>,
        ) -> Result<Vec<u8>, SyncError> {
            Message::Request(OctreeSectionRequest {
                future_id: self.next_future,
                level: level.to_owned(),
                pos,
                client_timestamp: known_timestamp,
            })
            .encode()
        }

        fn track_request(&mut self, level: &str, pos: OctreeSectionPos) {
            self.pending
                .insert(self.next_future, (level.to_owned(), pos));
            self.next_future = self.next_future.wrapping_add(1);
        }

        fn retain_pending_requests(&self, level: &str, requested: &mut HashSet<OctreeSectionPos>) {
            *requested = self
                .pending
                .values()
                .filter_map(|(dim, pos)| (dim == level && requested.contains(pos)).then_some(*pos))
                .collect();
        }

        pub fn on_message(
            &mut self,
            bytes: &[u8],
            now: Instant,
        ) -> Result<Vec<ClientEvent>, SyncError> {
            let mut events = Vec::new();
            self.receive_events(bytes, now, &mut events)?;
            Ok(events)
        }

        fn receive_events(
            &mut self,
            bytes: &[u8],
            now: Instant,
            events: &mut Vec<ClientEvent>,
        ) -> Result<(), SyncError> {
            match Message::decode(bytes)? {
                Message::SessionConfig(cfg) => {
                    self.config = Some(cfg);
                    events.push(ClientEvent::Config(cfg));
                }
                Message::Split(split) => self.receiver.receive(split, now)?,
                Message::Exception(e) => {
                    if let Some((level, pos)) = self.pending.remove(&e.future_id) {
                        events.push(ClientEvent::Failed {
                            level,
                            pos,
                            kind: e.kind,
                            message: e.message,
                        });
                    }
                }
                Message::Response(resp) => {
                    let Some((level, pos)) = self.pending.remove(&resp.future_id) else {
                        return Ok(());
                    };
                    match resp.payload {
                        None => events.push(ClientEvent::UpToDate { level, pos }),
                        Some(payload) => match self.receiver.take(payload.buffer_id) {
                            None => events.push(ClientEvent::Lost { level, pos }),
                            Some(buffer) => match OctreeSectionDto::decode(&buffer)
                                .and_then(|dto| dto.into_section())
                            {
                                Ok((section, mapping)) => events.push(ClientEvent::Section {
                                    level,
                                    section,
                                    mapping,
                                }),
                                Err(err) => {
                                    // The future is already retired. Queue completion even when
                                    // returning the diagnostic to the payload handler so the
                                    // pipeline can remove its marker.
                                    events.push(ClientEvent::Lost { level, pos });
                                    return Err(err);
                                }
                            },
                        },
                    }
                }
                Message::PartialUpdate(update) => {
                    if let Some(buffer) = self.receiver.take(update.payload.buffer_id) {
                        let (section, mapping) =
                            OctreeSectionDto::decode(&buffer)?.into_section()?;
                        events.push(ClientEvent::PartialUpdate {
                            level: update.level,
                            section,
                            mapping,
                        });
                    }
                }
                Message::Request(_) => {}
            }
            Ok(())
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

    #[derive(Default)]
    struct ClientIntake {
        open: bool,
        session: ClientSession,
        events: Vec<ClientEvent>,
    }

    impl ClientIntake {
        fn receive(
            &mut self,
            bytes: &[u8],
            now: Instant,
            epoch_live: impl FnOnce() -> bool,
        ) -> Result<(), SyncError> {
            // Check under the same lock as logout and rearm, before touching any protocol state.
            if self.open && epoch_live() {
                self.session.receive_events(bytes, now, &mut self.events)?;
            }
            Ok(())
        }

        fn configure(
            &mut self,
            send: impl FnOnce(PayloadBytes) -> Result<PayloadAccepted, PayloadSendError>,
        ) -> bool {
            if self.open {
                return true;
            }
            let Ok(bytes) = ClientSession::config_message(&SessionConfig::default()) else {
                return false;
            };
            if matches!(send(PayloadBytes(bytes)), Ok(PayloadAccepted::Queued)) {
                self.open = true;
            }
            self.open
        }

        fn send_request(
            &mut self,
            level: &str,
            pos: OctreeSectionPos,
            known_timestamp: Option<i64>,
            send: impl FnOnce(PayloadBytes) -> Result<PayloadAccepted, PayloadSendError>,
        ) -> bool {
            if !self.open {
                return false;
            }
            let Ok(bytes) = self.session.encode_request(level, pos, known_timestamp) else {
                return false;
            };
            if !matches!(send(PayloadBytes(bytes)), Ok(PayloadAccepted::Queued)) {
                return false;
            }
            self.session.track_request(level, pos);
            true
        }

        fn close(&mut self) {
            *self = Self::default();
        }
    }

    fn client_intake() -> &'static Mutex<ClientIntake> {
        static INTAKE: OnceLock<Mutex<ClientIntake>> = OnceLock::new();
        INTAKE.get_or_init(|| Mutex::new(ClientIntake::default()))
    }

    /// Keep only marks with an outstanding future in the current dimension. The callback
    /// removes the future and enqueues its event under the same lock. A reply already queued
    /// here needs no mark: client_sync applies queued events before selecting requests.
    pub(crate) fn retain_client_pending_requests(
        level: &str,
        requested: &mut HashSet<OctreeSectionPos>,
    ) {
        client_intake()
            .lock()
            .unwrap()
            .session
            .retain_pending_requests(level, requested);
    }

    /// Events decoded by the network executor, drained by the render side.
    pub fn drain_client_events() -> Vec<ClientEvent> {
        std::mem::take(&mut client_intake().lock().unwrap().events)
    }

    /// Install the server backend; the first install wins.
    pub fn install_server_source(source: Arc<dyn ServerSource<PlayerRef>>) {
        let _ = server_source().set(source);
    }

    pub(crate) fn set_server_open(open: bool) {
        server_registry().lock().unwrap().set_open(open);
    }

    /// Forget a player's state on logout (`ServerPlayerState` lives per connection in DH).
    pub fn remove_player(player: PlayerRef) {
        server_registry().lock().unwrap().remove_player(player);
    }

    pub(crate) fn remove_level(level: &str) {
        server_registry().lock().unwrap().remove_level(level);
    }

    pub(crate) fn publish_changes(
        level: &str,
        changes: impl IntoIterator<Item = (OctreeSectionPos, u64)>,
        locations: &PlayerLocations<PlayerRef>,
    ) {
        server_registry()
            .lock()
            .unwrap()
            .queue_partial_update(level, changes, locations);
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
            let Some(source) = server_source().get() else {
                return Ok(());
            };
            let replies = server_registry()
                .lock()
                .unwrap()
                .receive(|| context.player(), msg, Instant::now(), source.as_ref())
                .map_err(|e| failed(format!("{e:?}")))?;
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
            context: &mut dyn IPayloadContext,
        ) -> Result<(), PayloadHandleError> {
            client_intake()
                .lock()
                .unwrap()
                .receive(&payload.0, Instant::now(), || context.player().is_ok())
                .map_err(|e| PayloadHandleError::Failed(format!("{e:?}")))
        }
    }

    /// Open the current play world's intake by locally queueing its initial session config
    /// (`ServerPlayerState.java:62-65`); true once queued, and on every later call. A refused
    /// send leaves intake closed and the next eligible tick tries the same config again.
    /// `pipeline::client_sync` calls this before it demands any section.
    pub fn client_configure<H: ClientHost + ?Sized>(host: &mut H) -> bool {
        let channel = octree_sync_channel_id();
        configure_via(client_intake(), host, |host, bytes| {
            host.send_payload_to_server(&channel, bytes)
        })
    }

    fn configure_via<H: ?Sized>(
        intake: &Mutex<ClientIntake>,
        host: &mut H,
        send: impl FnOnce(&mut H, PayloadBytes) -> Result<PayloadAccepted, PayloadSendError>,
    ) -> bool {
        // The host send is nonblocking and cannot call the raw handler inline. Hold the same
        // guard as receive so a fast reply cannot arrive before intake is opened.
        intake.lock().unwrap().configure(|bytes| send(host, bytes))
    }

    /// Forget the per-play-world state at a lifetime boundary.
    pub fn client_reset() {
        client_intake().lock().unwrap().close();
    }

    /// Queue one request from the client tick. Only local acceptance records a pending future.
    pub fn client_send_request(
        host: &mut dyn ClientHost,
        level: &str,
        pos: OctreeSectionPos,
        known_timestamp: Option<i64>,
    ) -> bool {
        let channel = octree_sync_channel_id();
        // Hold intake across the narrow nonblocking send and publish the accepted future before
        // a raw reply can acquire the guard. The raw callback never acquires PIPELINE.
        client_intake()
            .lock()
            .unwrap()
            .send_request(level, pos, known_timestamp, |bytes| {
                host.send_payload_to_server(&channel, bytes)
            })
    }

    /// Drain each player's per-tick budget on an eligible published server's Overworld post-tick.
    pub(crate) fn server_tick(
        host: &mut dyn ServerHost,
        locations: &PlayerLocations<PlayerRef>,
    ) -> u32 {
        let Some(source) = server_source().get() else {
            return 0;
        };
        let channel = octree_sync_channel_id();
        ServerSyncRegistry::tick(
            server_registry(),
            locations,
            source.as_ref(),
            |player, message| {
                player
                    .send_payload(host, &channel, PayloadBytes(message))
                    .map(|_| ())
            },
        )
    }

    // ------------------------------------------------------------------ tests

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::atomic::AtomicUsize;

        const LEVEL: &str = "minecraft:overworld";

        #[test]
        fn full_response_without_transfer_queues_lost_and_retires_the_future() {
            // Covers the real missing-transfer branch only; the decode-error branch is not
            // exercised because no corrupted packet is crafted.
            let now = Instant::now();
            let pos = OctreeSectionPos::new(0, 2, 1, -3);
            let mut intake = ClientIntake::default();
            assert!(intake.configure(|_| Ok(PayloadAccepted::Queued)));
            assert!(intake.send_request(LEVEL, pos, None, |_| Ok(PayloadAccepted::Queued)));
            assert_eq!(
                intake.session.pending.get(&0),
                Some(&(LEVEL.to_owned(), pos))
            );
            let response = Message::Response(OctreeSectionResponse {
                future_id: 0,
                payload: Some(OctreePayload {
                    buffer_id: 4242,
                    beacon_beams: Vec::new(),
                }),
            })
            .encode()
            .unwrap();

            intake.receive(&response, now, || true).unwrap();
            assert!(intake.session.pending.is_empty());
            assert!(matches!(
                intake.events.as_slice(),
                [ClientEvent::Lost { level, pos: lost }] if level == LEVEL && *lost == pos
            ));

            intake.events.clear();
            intake.receive(&response, now, || true).unwrap();
            assert!(intake.events.is_empty(), "a retired future completes once");
        }

        #[test]
        fn initial_remote_login_preserves_installed_dimension_and_configures_once() {
            let state = Mutex::new(crate::render_bridge::DhState {
                active_dimension: "minecraft:the_nether".into(),
                ..Default::default()
            });
            let mut intake = ClientIntake::default();
            crate::reset_world(
                &state,
                crate::WorldBoundary::ClientLogin(Some(false)),
                || intake.close(),
                || {},
            );
            assert_eq!(
                state.lock().unwrap().active_dimension,
                "minecraft:the_nether"
            );
            assert!(intake.configure(|bytes| {
                assert!(matches!(
                    Message::decode(&bytes.0),
                    Ok(Message::SessionConfig(_))
                ));
                Ok(PayloadAccepted::Queued)
            }));
            assert!(intake.configure(|_| panic!("already configured")));
        }

        #[test]
        fn client_gate_stays_closed_until_the_same_config_is_queued_once() {
            #[derive(Default)]
            struct Link {
                accept: bool,
                sent: Vec<Vec<u8>>,
            }
            let intake = Mutex::new(ClientIntake::default());
            let tick = |link: &mut Link| {
                configure_via(&intake, link, |link, bytes| {
                    link.sent.push(bytes.0);
                    if link.accept {
                        Ok(PayloadAccepted::Queued)
                    } else {
                        Err(PayloadSendError::Backpressure)
                    }
                })
            };
            let mut link = Link::default();
            for refused in 1..=2 {
                assert!(!tick(&mut link), "refused config must not open the gate");
                assert_eq!(link.sent.len(), refused);
                assert!(!intake.lock().unwrap().open);
            }
            assert_eq!(link.sent[0], link.sent[1]);

            link.accept = true;
            assert!(tick(&mut link));
            assert_eq!(link.sent.len(), 3);
            assert_eq!(link.sent[2], link.sent[0]);
            assert!(intake.lock().unwrap().open);
            assert!(tick(&mut link));
            assert_eq!(
                link.sent.len(),
                3,
                "an open play world never resends config"
            );
        }

        #[test]
        fn refused_config_keeps_intake_closed_until_the_same_config_is_queued() {
            let now = Instant::now();
            let pos = OctreeSectionPos::new(0, 0, 0, 0);
            let source = Fake::of(&varied_section(pos, 1), DhV2CompressionMode::Uncompressed);
            let mut registry = ServerSyncRegistry::new(SessionConfig::default());
            registry.set_open(true);
            let mut intake = ClientIntake::default();
            let mut refused = Vec::new();
            assert!(!intake.configure(|bytes| {
                refused = bytes.0;
                Err(PayloadSendError::Backpressure)
            }));
            assert!(!intake.open);
            assert!(intake.session.pending.is_empty());
            assert!(registry.player(1).is_none());

            let mut replies = Vec::new();
            assert!(intake.configure(|bytes| {
                assert_eq!(bytes.0, refused);
                let config = Message::decode(&bytes.0).unwrap();
                assert!(matches!(config, Message::SessionConfig(_)));
                replies = registry.receive(|| Ok(1), config, now, &source).unwrap();
                Ok(PayloadAccepted::Queued)
            }));
            assert!(
                intake.session.config.is_none(),
                "requests need not wait for the reply"
            );
            let request = intake.session.request(LEVEL, pos, None).unwrap();
            assert!(
                registry
                    .receive(|| Ok(1), Message::decode(&request).unwrap(), now, &source)
                    .unwrap()
                    .is_empty()
            );
            assert!(!registry.player(1).unwrap().sender.is_idle());
            let [reply] = replies.as_slice() else {
                panic!("expected one config reply")
            };
            intake
                .receive(&reply.encode().unwrap(), now, || true)
                .unwrap();
            assert!(intake.session.config.is_some());
            assert!(intake.configure(|_| panic!("already configured")));
        }

        #[test]
        fn remote_logout_closes_intake_before_the_transport_retires() {
            check_remote_intake_reset(crate::WorldBoundary::ClientLogout(Some(false)), "");
        }

        #[test]
        fn backend_login_rearms_config_and_rejects_old_raw_payloads() {
            check_remote_intake_reset(crate::WorldBoundary::ClientLogin(Some(false)), LEVEL);
        }

        fn check_remote_intake_reset(boundary: crate::WorldBoundary, expected_dimension: &str) {
            let now = Instant::now();
            let cfg = Message::SessionConfig(SessionConfig::default())
                .encode()
                .unwrap();
            let split = Message::Split(SplitMessage {
                buffer_id: 7,
                is_first: true,
                buffer: vec![1, 2, 3],
            })
            .encode()
            .unwrap();
            let pos = OctreeSectionPos::new(0, 0, 0, 0);
            let mut intake = ClientIntake::default();
            assert!(intake.configure(|bytes| {
                assert_eq!(bytes.0, cfg);
                Ok(PayloadAccepted::Queued)
            }));
            assert!(intake.configure(|_| panic!("already configured")));
            intake.receive(&cfg, now, || true).unwrap();
            intake.receive(&split, now, || true).unwrap();
            intake.session.request(LEVEL, pos, None).unwrap();
            assert!(intake.session.config.is_some());
            assert_eq!(intake.events.len(), 1);
            assert_eq!(intake.session.receiver.buffers.len(), 1);

            let state = Mutex::new(crate::render_bridge::DhState {
                active_dimension: LEVEL.into(),
                ..Default::default()
            });
            crate::reset_world(&state, boundary, || intake.close(), || {});
            assert_eq!(state.lock().unwrap().active_dimension, expected_dimension);
            // Even if the old transport is still live (pre-teardown logout), closed intake
            // must reject it. The backend switch's old epoch is already inactive by Login.
            for bytes in [&cfg, &split] {
                intake
                    .receive(bytes, now, || {
                        panic!("closed intake must not consult the epoch")
                    })
                    .unwrap();
            }
            assert!(intake.events.is_empty());
            assert!(intake.session.pending.is_empty());
            assert!(intake.session.receiver.buffers.is_empty());
            assert!(!intake.configure(|_| Err(PayloadSendError::Backpressure)));
            intake
                .receive(&cfg, now, || panic!("refused config opened intake"))
                .unwrap();

            assert!(intake.configure(|bytes| {
                assert_eq!(bytes.0, cfg);
                Ok(PayloadAccepted::Queued)
            }));
            assert!(intake.configure(|_| panic!("already configured")));
            intake.session.request(LEVEL, pos, None).unwrap();
            let old = Message::Response(OctreeSectionResponse {
                future_id: 0,
                payload: None,
            })
            .encode()
            .unwrap();
            for bytes in [&cfg, &split, &old] {
                intake.receive(bytes, now, || false).unwrap();
            }
            assert!(intake.session.config.is_none());
            assert_eq!(intake.session.pending.get(&0).unwrap().1, pos);
            assert!(intake.session.receiver.buffers.is_empty());
            assert!(intake.events.is_empty());
            intake.receive(&cfg, now, || true).unwrap();
            assert!(intake.session.config.is_some());
        }

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
            dto_reads: AtomicUsize,
        }
        impl Fake {
            fn new(dto: OctreeSectionDto) -> Self {
                Self {
                    dto,
                    dto_reads: AtomicUsize::new(0),
                }
            }

            fn of(section: &OctreeSection, mode: DhV2CompressionMode) -> Self {
                Self::new(OctreeSectionDto::from_section(section, &[], mode, 0).unwrap())
            }
        }

        impl ServerSource<u32> for Fake {
            fn section_timestamp(&self, _: &str, _: OctreeSectionPos) -> Option<i64> {
                Some(self.dto.last_modified)
            }
            fn section_dto(&self, _: &str, _: OctreeSectionPos) -> Option<OctreeSectionDto> {
                self.dto_reads.fetch_add(1, Ordering::Relaxed);
                Some(self.dto.clone())
            }
            fn player_block_xz(&self, _: u32) -> Option<(i32, i32)> {
                Some((0, 0))
            }
        }

        fn wire_tick(
            registry: &mut ServerSyncRegistry<u32>,
            locations: &PlayerLocations<u32>,
            source: &Fake,
        ) -> HashMap<u32, Vec<Vec<u8>>> {
            let config = registry.server_config;
            let locked = Mutex::new(std::mem::replace(registry, ServerSyncRegistry::new(config)));
            let mut wire = HashMap::<_, Vec<_>>::new();
            ServerSyncRegistry::tick(&locked, locations, source, |player, bytes| {
                wire.entry(player).or_default().push(bytes);
                Ok(())
            });
            *registry = locked.into_inner().unwrap();
            wire
        }

        #[test]
        fn refused_scan_and_gpu_requests_remain_selectable_until_they_queue() {
            use crate::octree_ingest::OctreeIngestStore;
            use crate::pipeline::request_remote_sections;

            let now = Instant::now();
            for from_gpu in [false, true] {
                let mut store = OctreeIngestStore::default();
                let resident = OctreeSectionPos::new(0, 0, -2, 0);
                store.replace_section(varied_section(resident, 1));
                let (pos, gpu_demand) = if from_gpu {
                    (OctreeSectionPos::new(1, -2, 0, 1), vec![(1, [-2, 0, 1])])
                } else {
                    (OctreeSectionPos::new(0, 0, -1, 0), Vec::new())
                };
                let section = varied_section(pos, 1);
                let source = Fake::of(&section, DhV2CompressionMode::Uncompressed);
                let mut registry = ServerSyncRegistry::new(SessionConfig::default());
                registry.set_open(true);
                let mut intake = ClientIntake::default();
                assert!(
                    !intake
                        .send_request(LEVEL, pos, None, |_| panic!("closed intake sent a request"))
                );
                assert!(intake.configure(|cfg| {
                    let msg = Message::decode(&cfg.0).unwrap();
                    assert_eq!(
                        registry.receive(|| Ok(1), msg, now, &source).unwrap().len(),
                        1
                    );
                    Ok(PayloadAccepted::Queued)
                }));

                let mut requested = HashSet::new();
                let mut refused = Vec::new();
                request_remote_sections(
                    Some(&store),
                    &mut requested,
                    [0.0; 3],
                    0,
                    1,
                    gpu_demand.clone(),
                    |key| {
                        assert_eq!(key, pos);
                        intake.send_request(LEVEL, key, None, |bytes| {
                            refused = bytes.0;
                            Err(PayloadSendError::Backpressure)
                        })
                    },
                );
                assert!(
                    !refused.is_empty(),
                    "selected key reached the encoder and send"
                );
                assert!(requested.is_empty());
                assert!(intake.session.pending.is_empty());
                assert!(registry.player(1).unwrap().sender.is_idle());

                request_remote_sections(
                    Some(&store),
                    &mut requested,
                    [0.0; 3],
                    0,
                    1,
                    gpu_demand,
                    |key| {
                        assert_eq!(key, pos);
                        intake.send_request(LEVEL, key, None, |bytes| {
                            assert_eq!(bytes.0, refused);
                            let msg = Message::decode(&bytes.0).unwrap();
                            let Message::Request(req) = &msg else {
                                panic!("expected a request")
                            };
                            assert_eq!(req.pos, pos);
                            assert_eq!(req.level, LEVEL);
                            assert_eq!(req.future_id, 0);
                            assert!(
                                registry
                                    .receive(|| Ok(1), msg, now, &source)
                                    .unwrap()
                                    .is_empty()
                            );
                            Ok(PayloadAccepted::Queued)
                        })
                    },
                );
                assert_eq!(requested, HashSet::from([pos]));
                assert_eq!(
                    intake.session.pending.get(&0).unwrap(),
                    &(LEVEL.to_owned(), pos)
                );
                assert!(!registry.player(1).unwrap().sender.is_idle());
                let wire = wire_tick(&mut registry, &HashMap::new(), &source)
                    .remove(&1)
                    .unwrap();
                for bytes in &wire {
                    intake.receive(bytes, now, || true).unwrap();
                }
                assert!(intake.session.pending.is_empty());
                let [
                    ClientEvent::Section {
                        level,
                        section: received,
                        ..
                    },
                ] = intake.events.as_slice()
                else {
                    panic!("accepted request did not produce a section");
                };
                assert_eq!(level, LEVEL);
                assert_eq!(received.to_rle(), section.to_rle());
                assert_eq!(received.revision, section.revision);
            }
        }

        #[test]
        fn clearing_request_marks_preserves_pending_future_but_reselects_missing_completed_key() {
            use crate::octree_ingest::OctreeIngestStore;
            use crate::pipeline::request_remote_sections;

            let now = Instant::now();
            let pos = OctreeSectionPos::new(1, 0, 0, 0);
            let demand = [(pos.detail, [pos.x, pos.y, pos.z])];
            let mut intake = ClientIntake::default();
            let source = Fake::of(&varied_section(pos, 1), DhV2CompressionMode::Uncompressed);
            let mut registry = ServerSyncRegistry::new(SessionConfig::default());
            registry.set_open(true);
            assert!(intake.configure(|cfg| {
                registry
                    .receive(|| Ok(1), Message::decode(&cfg.0).unwrap(), now, &source)
                    .unwrap();
                Ok(PayloadAccepted::Queued)
            }));
            assert!(intake.send_request(LEVEL, pos, None, |bytes| {
                registry
                    .receive(|| Ok(1), Message::decode(&bytes.0).unwrap(), now, &source)
                    .unwrap();
                Ok(PayloadAccepted::Queued)
            }));
            let mut requested = HashSet::from([pos]);
            intake
                .session
                .retain_pending_requests(LEVEL, &mut requested);
            request_remote_sections(None, &mut requested, [0.0; 3], 0, 1, demand, |key| {
                assert_ne!(
                    key, pos,
                    "clear must not send a second future for the pending key"
                );
                false
            });
            assert_eq!(requested, HashSet::from([pos]));
            assert_eq!(intake.session.next_future, 1);
            assert_eq!(
                intake.session.pending.get(&0),
                Some(&(LEVEL.to_owned(), pos))
            );

            for bytes in &wire_tick(&mut registry, &HashMap::new(), &source)[&1] {
                intake.receive(bytes, now, || true).unwrap();
            }
            assert!(intake.session.pending.is_empty());
            // The callback has finished, but client_sync has not drained the valid reply.
            // Clearing at this boundary must not discard the event.
            let mut queued_marks = requested.clone();
            intake
                .session
                .retain_pending_requests(LEVEL, &mut queued_marks);
            assert!(queued_marks.is_empty());
            let events = std::mem::take(&mut intake.events);
            let [ClientEvent::Section { level, section, .. }] = events.as_slice() else {
                panic!("clear lost the already queued reply");
            };
            assert_eq!(level, LEVEL);
            assert_eq!(section.pos, pos);

            let mut store = OctreeIngestStore::new();
            store.replace_section(section.clone());
            request_remote_sections(
                Some(&store),
                &mut queued_marks,
                [0.0; 3],
                0,
                1,
                demand,
                |key| {
                    assert_ne!(key, pos, "the queued reply satisfies refresh");
                    false
                },
            );

            // A clear after adoption instead drops the resident copy and its completed mark.
            intake
                .session
                .retain_pending_requests(LEVEL, &mut requested);
            assert_eq!(store.evict_sections_except(None), 1);
            request_remote_sections(
                Some(&store),
                &mut requested,
                [0.0; 3],
                0,
                1,
                demand,
                |key| {
                    assert_eq!(key, pos);
                    intake.send_request(LEVEL, key, None, |bytes| {
                        let Message::Request(req) = Message::decode(&bytes.0).unwrap() else {
                            panic!("expected a fresh request");
                        };
                        assert_eq!(req.future_id, 1);
                        assert_eq!(req.pos, pos);
                        Ok(PayloadAccepted::Queued)
                    })
                },
            );
            assert_eq!(requested, HashSet::from([pos]));
            assert_eq!(
                intake.session.pending.get(&1),
                Some(&(LEVEL.to_owned(), pos))
            );
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

                let source = Fake::new(dto);
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
                let wire = wire_tick(&mut registry, &HashMap::new(), &source)
                    .remove(&1)
                    .unwrap();
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
            let source = Fake::new(dto);
            let cfg = SessionConfig {
                sync_on_load_rate_limit: 1,
                ..SessionConfig::default()
            };
            let mut registry = ServerSyncRegistry::<u32>::new(cfg);
            let mut client = ClientSession::default();
            let now = Instant::now();
            registry.handle(1, Message::SessionConfig(cfg), now, &source);
            let req = client.request(LEVEL, pos, Some(77)).unwrap();
            let replies = registry.handle(1, Message::decode(&req).unwrap(), now, &source);
            let events = client
                .on_message(&replies[0].encode().unwrap(), now)
                .unwrap();
            assert!(matches!(events.as_slice(), [ClientEvent::UpToDate { .. }]));
            assert_eq!(source.dto_reads.load(Ordering::Relaxed), 0);
            let req = Message::decode(&client.request(LEVEL, pos, Some(0)).unwrap()).unwrap();
            assert!(
                registry
                    .handle(1, req, now + Duration::from_secs(1), &source)
                    .is_empty()
            );
        }

        #[test]
        fn split_receiver_caps_pending_buffers() {
            let now = Instant::now();
            let mut receiver = SplitReceiver::default();
            for id in 0..(MAX_PENDING_BUFFERS as i32 + 4) {
                receiver
                    .receive(
                        SplitMessage {
                            buffer_id: id,
                            buffer: vec![1, 2, 3],
                            is_first: true,
                        },
                        now,
                    )
                    .unwrap();
            }
            assert!(receiver.take(0).is_some());
            assert!(receiver.take(MAX_PENDING_BUFFERS as i32).is_none());
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
            let cfg = SessionConfig {
                player_bandwidth_limit: 0,
                ..SessionConfig::default()
            };
            let mut registry = ServerSyncRegistry::new(cfg);
            let mut state = PlayerSyncState::new(cfg);
            state.sender.queue.push_back(PendingTransfer {
                encoded: EncodedTransfer {
                    buffer_id: 9,
                    buffer: buffer.clone().into(),
                    final_message: vec![0xAA].into(),
                },
                offset: 0,
                revision: None,
                admission: None,
            });
            registry.players.insert(1, state);
            let source = Fake::of(
                &OctreeSection::new(OctreeSectionPos::new(0, 0, 0, 0)),
                DEFAULT_COMPRESSION,
            );
            let wire = wire_tick(&mut registry, &HashMap::new(), &source)
                .remove(&1)
                .unwrap();
            assert!(registry.players[&1].sender.is_idle());
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
            let source = Fake::new(dto);
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
                for (_, wire) in wire_tick(&mut registry, &HashMap::new(), &source) {
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
            let source = Fake::new(dto);
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
                all.extend(
                    wire_tick(&mut registry, &HashMap::new(), &source)
                        .into_values()
                        .flatten(),
                );
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
            let pos = OctreeSectionPos::new(0, 128, 0, 0); // starts at block x=4096
            let mut section = varied_section(pos, 40);
            let mut source = Fake::of(&section, DEFAULT_COMPRESSION);
            let beams = vec![BeaconBeam {
                x: 1,
                y: 2,
                z: 3,
                argb: -16711936,
            }];
            let (payload, _) = OctreePayload::new(&source.dto, beams).unwrap();
            let msg = Message::PartialUpdate(OctreePartialUpdate {
                level: LEVEL.into(),
                payload,
            });
            assert_eq!(Message::decode(&msg.encode().unwrap()).unwrap(), msg);

            const NETHER: &str = "minecraft:the_nether";
            let mut registry = ServerSyncRegistry::<u32>::new(SessionConfig::default());
            let now = Instant::now();
            let mut client = ClientSession::default();
            for p in [1, 2, 4, 5, 6, 7, 8] {
                let cfg = SessionConfig {
                    enable_real_time_updates: p != 5,
                    ..SessionConfig::default()
                };
                registry.handle(p, Message::SessionConfig(cfg), now, &source);
                if p == 6 {
                    continue;
                }
                let level = if p == 8 { NETHER } else { LEVEL };
                let timestamp = (p == 7).then_some(source.dto.last_modified);
                let request = if p == 1 {
                    Message::decode(&client.request(level, pos, timestamp).unwrap()).unwrap()
                } else {
                    Message::Request(OctreeSectionRequest {
                        future_id: 0,
                        level: level.into(),
                        pos,
                        client_timestamp: timestamp,
                    })
                };
                let replies = registry.handle(p, request, now, &source);
                if p == 7 {
                    assert!(matches!(
                        replies.as_slice(),
                        [Message::Response(OctreeSectionResponse {
                            payload: None,
                            ..
                        })]
                    ));
                } else {
                    assert!(replies.is_empty());
                }
            }
            // Only 1 qualifies. For 2, flooring x=-15.5 to -16 puts the section 257 chunks
            // away; a truncating cast would put it inside the 256-chunk radius. 3 is
            // unnegotiated, 4 in another dimension, 5 opted out, 6 unserved, 7 unchanged,
            // and 8 received the same position in a different level.
            let near = [4100.25, 64.0, 1.25];
            let locations = HashMap::from([
                (1, PlayerLocation::new(LEVEL.into(), near)),
                (2, PlayerLocation::new(LEVEL.into(), [-15.5, 64.0, 1.25])),
                (3, PlayerLocation::new(LEVEL.into(), near)),
                (4, PlayerLocation::new(NETHER.into(), near)),
                (5, PlayerLocation::new(LEVEL.into(), near)),
                (6, PlayerLocation::new(LEVEL.into(), near)),
                (7, PlayerLocation::new(LEVEL.into(), near)),
                (8, PlayerLocation::new(LEVEL.into(), near)),
            ]);
            let initial = wire_tick(&mut registry, &locations, &source)
                .remove(&1)
                .unwrap();
            assert!(matches!(
                deliver(&mut client, initial, now).as_slice(),
                [ClientEvent::Section { .. }]
            ));

            section.set(0, 5, 0, Voxel::new(2, 0, 15, 0)); // refinement
            let refined = section.clone();
            section.set(0, 5, 0, Voxel::new(3, 0, 15, 0)); // live edit
            let mut cleared = OctreeSection::new(pos); // an authored clearing revision
            cleared.revision = section.revision + 1;
            for change in [refined, section, cleared] {
                source = Fake::of(&change, DEFAULT_COMPRESSION);
                registry.queue_partial_update(LEVEL, [(pos, change.revision)], &locations);
                let mut out = wire_tick(&mut registry, &locations, &source);
                assert_eq!(out.len(), 1);
                let events = deliver(&mut client, out.remove(&1).unwrap(), now);
                let [
                    ClientEvent::PartialUpdate {
                        level,
                        section: got,
                        ..
                    },
                ] = events.as_slice()
                else {
                    panic!("{events:?}")
                };
                assert_eq!(level, LEVEL);
                assert_eq!(got.revision, change.revision);
                assert_eq!(got.to_dense(), change.to_dense());

                registry.queue_partial_update(LEVEL, [(pos, change.revision)], &locations);
                assert!(wire_tick(&mut registry, &locations, &source).is_empty());
            }

            registry.remove_level(LEVEL);
            registry.queue_partial_update(
                LEVEL,
                [(pos, source.dto.last_modified as u64 + 1)],
                &locations,
            );
            assert!(registry.players[&1].receipts.is_empty());
            assert!(registry.players[&1].deferred.is_empty());
            assert!(
                registry.players[&8]
                    .receipts
                    .contains_key(&(NETHER.to_owned(), pos))
            );
        }

        #[test]
        fn backpressure_preserves_split_order_and_the_newest_deferred_revision() {
            let pos = OctreeSectionPos::new(0, 0, 0, 0);
            let cfg = SessionConfig {
                player_bandwidth_limit: 1,
                ..SessionConfig::default()
            };
            let mut section = varied_section(pos, 60);
            let mode = DhV2CompressionMode::Uncompressed;
            let mut source = Fake::of(&section, mode);
            let mut state = ServerSyncRegistry::<u32>::new(cfg);
            let locations = HashMap::from([(1, PlayerLocation::new(LEVEL.into(), [0.0; 3]))]);
            let mut client = ClientSession::default();
            let now = Instant::now();
            state.handle(1, Message::SessionConfig(cfg), now, &source);
            let req = Message::decode(&client.request(LEVEL, pos, None).unwrap()).unwrap();
            state.handle(1, req, now, &source);
            let registry = Mutex::new(state);

            let mut wire = Vec::new();
            // The first full is still queued when refinement arrives.
            section.revision = 78;
            source = Fake::of(&section, mode);
            registry
                .lock()
                .unwrap()
                .queue_partial_update(LEVEL, [(pos, 78)], &locations);
            let mut request =
                Some(Message::decode(&client.request(LEVEL, pos, None).unwrap()).unwrap());
            let later = now + Duration::from_secs(1);
            for _ in 0..100 {
                ServerSyncRegistry::tick(&registry, &locations, &source, |player, bytes| {
                    if let Message::Split(_) = Message::decode(&bytes).unwrap()
                        && registry.lock().unwrap().players[&player]
                            .sender
                            .queue
                            .front()
                            .unwrap()
                            .admission
                            .is_none()
                    {
                        // This partial has not started. An off-thread request appends a full
                        // reply while the registry lock is released around the refused send.
                        registry.lock().unwrap().handle(
                            player,
                            request.take().unwrap(),
                            later,
                            &source,
                        );
                        return Err(PayloadSendError::Backpressure);
                    }
                    wire.push(bytes);
                    Ok(())
                });
                if request.is_none() {
                    break;
                }
            }
            assert!(request.is_none());

            // Coalesce 79 away while the admitted 78 full reply goes first. Then pause the
            // final frame of the 80 partial after its split bytes were accepted.
            for revision in [79, 80] {
                registry
                    .lock()
                    .unwrap()
                    .queue_partial_update(LEVEL, [(pos, revision)], &locations);
            }
            section.revision = 80;
            source = Fake::of(&section, mode);
            let mut refused_final = false;
            for _ in 0..100 {
                ServerSyncRegistry::tick(&registry, &locations, &source, |_, bytes| {
                    match Message::decode(&bytes).unwrap() {
                        Message::Split(split) => assert!(split.buffer.len() <= 51),
                        Message::PartialUpdate(_) => {
                            refused_final = true;
                            return Err(PayloadSendError::Backpressure);
                        }
                        _ => {}
                    }
                    wire.push(bytes);
                    Ok(())
                });
                if refused_final {
                    break;
                }
            }
            assert!(refused_final);

            section.revision = 81;
            source = Fake::of(&section, mode);
            registry
                .lock()
                .unwrap()
                .queue_partial_update(LEVEL, [(pos, 81)], &locations);
            for _ in 0..100 {
                if ServerSyncRegistry::tick(&registry, &locations, &source, |_, bytes| {
                    wire.push(bytes);
                    Ok(())
                }) == 0
                {
                    break;
                }
            }
            let events = deliver(&mut client, wire, now);
            let revisions: Vec<_> = events
                .iter()
                .map(|event| match event {
                    ClientEvent::Section { section, .. } => (false, section.revision),
                    ClientEvent::PartialUpdate { section, .. } => (true, section.revision),
                    _ => panic!("{event:?}"),
                })
                .collect();
            assert_eq!(
                revisions,
                [(false, 77), (false, 78), (true, 80), (true, 81)]
            );
        }

        #[test]
        fn pressured_full_replies_keep_both_admission_limits_until_the_final_frame() {
            let now = Instant::now();
            let pos = OctreeSectionPos::new(0, 0, 0, 0);
            let source = Fake::of(&varied_section(pos, 4), DhV2CompressionMode::Uncompressed);
            let cfg = SessionConfig {
                generation_request_rate_limit: 2,
                sync_on_load_rate_limit: 3,
                ..SessionConfig::default()
            };
            let mut state = ServerSyncRegistry::new(cfg);
            state.set_open(true);
            state.handle(1, Message::SessionConfig(cfg), now, &source);
            let registry = Mutex::new(state);
            let locations = PlayerLocations::new();
            let mut client = ClientSession::default();
            let mut submit = |known, second| {
                let bytes = client.request(LEVEL, pos, known).unwrap();
                let message = Message::decode(&bytes).unwrap();
                let Message::Request(req) = &message else {
                    panic!("{message:?}");
                };
                let future_id = req.future_id;
                let replies = registry
                    .lock()
                    .unwrap()
                    .receive(
                        || Ok(1),
                        message,
                        now + Duration::from_secs(second),
                        &source,
                    )
                    .unwrap();
                (future_id, replies)
            };
            let assert_denied = |(id, replies): (i32, Vec<Message>)| {
                assert!(matches!(
                    replies.as_slice(),
                    [Message::Exception(ExceptionMessage {
                        future_id,
                        kind: ExceptionKind::RateLimited,
                        ..
                    })] if *future_id == id
                ));
            };
            let footprint = || {
                let state = registry.lock().unwrap();
                let queue = &state.players[&1].sender.queue;
                (
                    queue.len(),
                    queue
                        .iter()
                        .map(|t| t.encoded.buffer.len() + t.encoded.final_message.len())
                        .sum::<usize>(),
                )
            };

            // Two generation and three sync-on-load admissions, mixed in one FIFO. Give each
            // request fresh tokens so this exercises concurrency rather than the rate check.
            let mut admitted = Vec::new();
            for (second, known) in [None, Some(0), None, Some(0), Some(0)]
                .into_iter()
                .enumerate()
            {
                let (id, replies) = submit(known, second as u64);
                assert!(replies.is_empty());
                admitted.push(id);
            }
            let full = footprint();
            assert_eq!(full.0, 5);
            assert!(full.1 > 0);
            assert_eq!(source.dto_reads.load(Ordering::Relaxed), full.0);
            let mut stalled_frame = None;
            for second in 5..=9 {
                let mut attempted = None;
                assert_eq!(
                    ServerSyncRegistry::tick(&registry, &locations, &source, |_, frame| {
                        assert!(attempted.replace(frame).is_none());
                        Err(PayloadSendError::Backpressure)
                    }),
                    0
                );
                let frame = attempted.unwrap();
                if let Some(stalled) = &stalled_frame {
                    assert_eq!(stalled, &frame);
                } else {
                    stalled_frame = Some(frame);
                }
                assert_denied(submit(None, second));
                assert_denied(submit(Some(0), second));
                assert_eq!(footprint(), full);
                assert_eq!(source.dto_reads.load(Ordering::Relaxed), full.0);
            }

            let mut wire = Vec::new();
            let mut final_refused = None;
            ServerSyncRegistry::tick(&registry, &locations, &source, |_, frame| {
                match Message::decode(&frame).unwrap() {
                    Message::Split(_) => wire.push(frame),
                    Message::Response(response) => {
                        assert_eq!(response.future_id, admitted[0]);
                        final_refused = Some(frame);
                        return Err(PayloadSendError::Backpressure);
                    }
                    other => panic!("{other:?}"),
                }
                Ok(())
            });
            assert!(!wire.is_empty());
            let final_refused = final_refused.unwrap();
            assert_denied(submit(None, 10));
            assert_denied(submit(Some(0), 10));
            assert_eq!(footprint(), full);
            assert_eq!(source.dto_reads.load(Ordering::Relaxed), full.0);

            // Accept only the first final marker. That opens one slot and only in its limiter.
            assert_eq!(
                ServerSyncRegistry::tick(&registry, &locations, &source, |_, frame| {
                    if frame == final_refused {
                        wire.push(frame);
                        Ok(())
                    } else {
                        assert!(matches!(
                            Message::decode(&frame).unwrap(),
                            Message::Split(_)
                        ));
                        Err(PayloadSendError::Backpressure)
                    }
                }),
                1
            );
            let (id, replies) = submit(None, 11);
            assert!(replies.is_empty());
            admitted.push(id);
            assert_denied(submit(None, 11));
            assert_denied(submit(Some(0), 11));
            assert_eq!(footprint(), full);

            // The remaining transfers fit in one tick at the ordinary bandwidth budget.
            ServerSyncRegistry::tick(&registry, &locations, &source, |_, frame| {
                wire.push(frame);
                Ok(())
            });
            assert_eq!(footprint(), (0, 0));
            for known in [None, Some(0)] {
                let (id, replies) = submit(known, 12);
                assert!(replies.is_empty());
                admitted.push(id);
            }
            ServerSyncRegistry::tick(&registry, &locations, &source, |_, frame| {
                wire.push(frame);
                Ok(())
            });
            assert_eq!(footprint(), (0, 0));
            assert_eq!(source.dto_reads.load(Ordering::Relaxed), admitted.len());
            let order: Vec<_> = wire
                .iter()
                .filter_map(|frame| match Message::decode(frame).unwrap() {
                    Message::Response(response) => Some(response.future_id),
                    Message::Split(_) => None,
                    other => panic!("{other:?}"),
                })
                .collect();
            assert_eq!(order, admitted);
            let events = deliver(&mut client, wire, now + Duration::from_secs(12));
            assert_eq!(events.len(), admitted.len());
            assert!(events.iter().all(|e| matches!(
                e,
                ClientEvent::Section { level, section, .. }
                    if level == LEVEL && section.pos == pos && section.revision == 77
            )));
        }

        #[test]
        fn refused_full_and_retired_player_do_not_receive_updates() {
            let now = Instant::now();
            let pos = OctreeSectionPos::new(0, 0, 0, 0);
            let source = Fake::of(&varied_section(pos, 4), DEFAULT_COMPRESSION);
            let cfg = SessionConfig {
                generation_request_rate_limit: 1,
                ..SessionConfig::default()
            };
            let mut state = ServerSyncRegistry::new(cfg);
            state.set_open(true);
            for player in [1, 2] {
                // A valid play context can negotiate before the main-thread login notification.
                assert_eq!(
                    state
                        .receive(|| Ok(player), Message::SessionConfig(cfg), now, &source)
                        .unwrap()
                        .len(),
                    1
                );
                let request = Message::Request(OctreeSectionRequest {
                    future_id: 0,
                    level: LEVEL.into(),
                    pos,
                    client_timestamp: None,
                });
                assert!(
                    state
                        .receive(|| Ok(player), request, now, &source)
                        .unwrap()
                        .is_empty()
                );
            }
            let locations = HashMap::from([
                (1, PlayerLocation::new(LEVEL.into(), [0.0; 3])),
                (2, PlayerLocation::new(LEVEL.into(), [0.0; 3])),
            ]);
            state.queue_partial_update(LEVEL, [(pos, 78)], &locations);
            let registry = Mutex::new(state);
            ServerSyncRegistry::tick(&registry, &locations, &source, |player, _| {
                if player == 1 {
                    Err(PayloadSendError::UnsupportedPeer(octree_sync_channel_id()))
                } else {
                    Err(PayloadSendError::Backpressure)
                }
            });
            let mut state = registry.into_inner().unwrap();
            assert!(state.players[&1].receipts.is_empty());
            assert!(state.players[&1].deferred.is_empty());
            assert!(state.players[&1].sender.is_idle());
            assert!(!state.players[&2].deferred.is_empty());

            let request = |future_id| {
                Message::Request(OctreeSectionRequest {
                    future_id,
                    level: LEVEL.into(),
                    pos,
                    client_timestamp: None,
                })
            };
            let later = now + Duration::from_secs(1);
            // Terminal refusal frees player 1's reservation; level retirement must not free
            // player 2's, because their original full reply is still queued.
            state.remove_level(LEVEL);
            assert!(state.handle(1, request(1), later, &source).is_empty());
            assert!(matches!(
                state.handle(2, request(1), later, &source).as_slice(),
                [Message::Exception(ExceptionMessage {
                    future_id: 1,
                    kind: ExceptionKind::RateLimited,
                    ..
                })]
            ));
            let registry = Mutex::new(state);
            let mut final_ids = Vec::new();
            ServerSyncRegistry::tick(&registry, &locations, &source, |player, bytes| {
                if player == 1 {
                    // Leave an admitted response in state for the server-stop check.
                    return Err(PayloadSendError::Backpressure);
                }
                if let Message::Response(response) = Message::decode(&bytes).unwrap() {
                    final_ids.push(response.future_id);
                }
                Ok(())
            });
            assert_eq!(final_ids, [0]);
            let mut state = registry.into_inner().unwrap();
            let mut edit = varied_section(pos, 4);
            edit.revision = 78;
            let edited = Fake::of(&edit, DEFAULT_COMPRESSION);
            state.queue_partial_update(LEVEL, [(pos, edit.revision)], &locations);
            let registry = Mutex::new(state);
            assert_eq!(
                ServerSyncRegistry::tick(&registry, &locations, &edited, |player, _| {
                    assert_eq!(player, 1, "the retired reply must not resubscribe player 2");
                    Err(PayloadSendError::Backpressure)
                }),
                0
            );
            let mut state = registry.into_inner().unwrap();
            assert!(
                state
                    .handle(2, request(2), later + Duration::from_secs(1), &edited)
                    .is_empty()
            );
            state.remove_player(2);
            assert!(state.player(2).is_none());
            assert!(
                state
                    .receive(
                        || Err(mod_api::PayloadContextError::Disconnected),
                        Message::SessionConfig(cfg),
                        now,
                        &source
                    )
                    .is_err()
            );
            assert!(state.player(2).is_none());
            state.set_open(false);
            assert!(
                state
                    .receive(
                        || panic!("stopping server must not resolve the old epoch"),
                        Message::SessionConfig(cfg),
                        now,
                        &source
                    )
                    .unwrap()
                    .is_empty()
            );
            assert!(state.players.is_empty());
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
            let source = Fake::new(dto);
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
            for ((x, z), expected) in [
                ((0, 0), 0),
                ((31, 31), 0),
                ((10, 10), 0),
                ((-15, 32), 0),
                ((-16, 32), 1),
                ((0, 46), 0),
                ((0, 47), 1),
                ((32 + 160, 0), 10),
                ((0, -160), 10),
            ] {
                assert_eq!(chebyshev_chunk_distance(pos, x, z), expected);
            }
            let cfg = SessionConfig::default();
            let mut st = PlayerSyncState::new(cfg);
            st.on_client_config(&cfg);
            for (timestamp, expected) in [
                (None, Admission::Generation),
                (Some(5), Admission::SyncOnLoad),
            ] {
                let req = OctreeSectionRequest {
                    future_id: 1,
                    level: LEVEL.into(),
                    pos,
                    client_timestamp: timestamp,
                };
                let now = Instant::now();
                let edge = 32 + (4096 * 16 - 1);
                let accepted = st
                    .admit(&req, chebyshev_chunk_distance(pos, edge, 0), now)
                    .unwrap();
                assert_eq!(accepted, expected);
                st.release(accepted);
                let failure = st
                    .admit(&req, chebyshev_chunk_distance(pos, edge + 16, 0), now)
                    .unwrap_err();
                assert_eq!(failure.kind, ExceptionKind::RequestOutOfRange);
            }
        }

        #[test]
        fn signed_wire_coordinate_bounds_remain_far_on_both_axes() {
            let now = Instant::now();
            let cfg = SessionConfig::default();
            let origin = OctreeSectionPos::new(0, 0, 0, 0);
            let source = Fake::of(
                &varied_section(origin, 1),
                DhV2CompressionMode::Uncompressed,
            );
            let mut registry = ServerSyncRegistry::new(cfg);
            registry.handle(1, Message::SessionConfig(cfg), now, &source);
            let span = i64::from(32) << MAX_OCTREE_DETAIL;
            let expected = (((1i64 << 23) - 1) * span) / 16;
            for coordinate in [-(1 << 23), (1 << 23) - 1] {
                for (x, z) in [(coordinate, 0), (0, coordinate)] {
                    let pos = OctreeSectionPos::new(MAX_OCTREE_DETAIL, x, 0, z);
                    for timestamp in [None, Some(5)] {
                        let bytes = Message::Request(OctreeSectionRequest {
                            future_id: 1,
                            level: LEVEL.into(),
                            pos,
                            client_timestamp: timestamp,
                        })
                        .encode()
                        .unwrap();
                        let message = Message::decode(&bytes).unwrap();
                        let Message::Request(req) = &message else {
                            panic!("expected a request")
                        };
                        assert_eq!(req.pos, pos);
                        assert_eq!(chebyshev_chunk_distance(req.pos, 0, 0), expected);
                        let result = registry.handle(1, message, now, &source);
                        assert!(matches!(
                            result.as_slice(),
                            [Message::Exception(ExceptionMessage {
                                kind: ExceptionKind::RequestOutOfRange,
                                ..
                            })]
                        ));
                        assert!(!registry.player(1).unwrap().wants_partial_update(pos, 0, 0));
                    }
                }
            }
            assert_eq!(source.dto_reads.load(Ordering::Relaxed), 0);
        }
    }
}
