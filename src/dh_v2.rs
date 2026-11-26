//! Wire codec and decoder for Distant Sunrises' `FullDataSourceV2DTO` and `DhSectionPos`
//! formats, allowing server-sent LOD payloads in upstream DH's format to be ingested directly
//! by external Sunrise mods via [`crate::api::submit_dh_v2_bytes`] and
//! [`crate::api::submit_dh_v2_dto`].

use crate::lod::{SECTION_CELLS, SECTION_SIZE};
use std::io::Read;

/// Section detail offset in upstream DH `DhSectionPos`: section detail `6` equals LOD detail `0`
/// (a `64 x 64` block = `4 x 4` chunk section).
pub const DH_SECTION_MIN_DETAIL: u8 = 6;

/// Separator between serialized biome and blockstate strings in `FullDataPointIdMap`.
pub const BLOCK_STATE_SEPARATOR: &str = "_DH-BSW_";

/// Safety limit on decompressed payload size per DTO blob (64 MiB).
const MAX_DECOMPRESSED_BYTES: usize = 64 * 1024 * 1024;
/// Safety limit on mapping entries per section.
const MAX_MAPPING_ENTRIES: usize = 1 << 20;
/// Safety limit on datapoints per vertical column.
const MAX_COLUMN_POINTS: usize = 4096;

/// Compression mode matching upstream `EDhApiDataCompressionMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DhV2CompressionMode {
    Uncompressed = 0,
    Lz4 = 1,
    ZstdStream = 2,
    Lzma2 = 3,
    ZstdBlock = 4,
}

impl DhV2CompressionMode {
    pub fn from_byte(v: u8) -> Result<Self, DhV2Error> {
        match v {
            0 => Ok(Self::Uncompressed),
            1 => Ok(Self::Lz4),
            2 => Ok(Self::ZstdStream),
            3 => Ok(Self::Lzma2),
            4 => Ok(Self::ZstdBlock),
            other => Err(DhV2Error::UnsupportedCompression(other)),
        }
    }
}

/// Errors returned when decoding a `FullDataSourceV2DTO` wire payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DhV2Error {
    Truncated,
    InvalidLength(String),
    UnsupportedFormatVersion(u8),
    UnsupportedCompression(u8),
    DecompressFailed(String),
    CorruptData(String),
}

impl std::fmt::Display for DhV2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "unexpected end of DH V2 DTO buffer"),
            Self::InvalidLength(msg) => write!(f, "invalid DH V2 length: {msg}"),
            Self::UnsupportedFormatVersion(v) => {
                write!(f, "unsupported DH V2 data format version {v}")
            }
            Self::UnsupportedCompression(c) => {
                write!(f, "unsupported DH V2 compression mode {c}")
            }
            Self::DecompressFailed(msg) => write!(f, "DH V2 decompression failed: {msg}"),
            Self::CorruptData(msg) => write!(f, "corrupt DH V2 payload: {msg}"),
        }
    }
}

impl std::error::Error for DhV2Error {}

/// Packs `(section_detail, x, z)` into upstream DH's 64-bit `DhSectionPos` representation:
/// bits `0..8` = `section_detail` (`lod_detail + 6`), bits `8..36` = signed 28-bit `x`,
/// bits `36..64` = signed 28-bit `z`.
pub const fn pack_dh_section_pos(section_detail: u8, x: i32, z: i32) -> i64 {
    let mut data: u64 = (section_detail & 0x7F) as u64;
    data |= ((x as u32 & 0x0FFF_FFFF) as u64) << 8;
    data |= ((z as u32 & 0x0FFF_FFFF) as u64) << 36;
    data as i64
}

/// Unpacks upstream DH's 64-bit `DhSectionPos` into `(section_detail, x, z)`.
pub const fn unpack_dh_section_pos(packed: i64) -> (u8, i32, i32) {
    let raw = packed as u64;
    let detail = (raw & 0x7F) as u8;
    let x = (((raw >> 8) as i32) << 4) >> 4;
    let z = (((raw >> 36) as i32) << 4) >> 4;
    (detail, x, z)
}

/// Decoded single datapoint from a `FullDataSourceV2` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodedDhDataPoint {
    pub mapping_id: u32,
    pub bottom_y: i32,
    pub height: u16,
    pub block_light: u8,
    pub sky_light: u8,
}

impl DecodedDhDataPoint {
    #[inline]
    pub const fn top_y(self) -> i32 {
        self.bottom_y + self.height as i32
    }
}

/// One `(biome_id, block_id)` pair from `FullDataPointIdMap`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DhMappingEntry {
    pub biome_id: String,
    pub block_id: String,
    pub raw_block_state: String,
}

/// Decoded `64 x 64` section ready to ingest into [`crate::lod::DimensionLodStore`].
#[derive(Clone, Debug)]
pub struct DecodedDhV2Section {
    /// LOD detail level (`0` = 1x1 block cells = `64x64` blocks per section).
    pub lod_detail: u8,
    pub section_x: i32,
    pub section_z: i32,
    pub mapping: Vec<DhMappingEntry>,
    /// Columns indexed by `cz * 64 + cx` (matching `LodSection::index`), each ordered top-to-bottom.
    pub columns: Vec<Vec<DecodedDhDataPoint>>,
    pub last_modified_unix_ms: i64,
    pub created_unix_ms: i64,
}

/// Raw wire container matching upstream Distant Sunrises `FullDataSourceV2DTO`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DhV2FullDataDto {
    pub pos: i64,
    pub data_checksum: i32,
    pub compressed_data: Vec<u8>,
    pub compressed_north_adj: Vec<u8>,
    pub compressed_south_adj: Vec<u8>,
    pub compressed_east_adj: Vec<u8>,
    pub compressed_west_adj: Vec<u8>,
    pub compressed_column_gen_step: Vec<u8>,
    pub compressed_world_compression_mode: Vec<u8>,
    pub compressed_mapping: Vec<u8>,
    pub data_format_version: u8,
    pub compression_mode: DhV2CompressionMode,
    pub apply_to_parent: bool,
    pub apply_to_children: bool,
    pub regenerate: bool,
    pub last_modified_unix_ms: i64,
    pub created_unix_ms: i64,
}

impl DhV2FullDataDto {
    /// Decodes a big-endian wire payload matching `FullDataSourceV2DTO.encode(ByteBuf)`.
    pub fn decode_wire(bytes: &[u8]) -> Result<Self, DhV2Error> {
        let mut cur = ByteCursor::new(bytes);
        let pos = cur.read_i64()?;
        let data_checksum = cur.read_i32()?;
        let compressed_data = cur.read_blob("data")?;
        let compressed_north_adj = cur.read_blob("north_adj")?;
        let compressed_south_adj = cur.read_blob("south_adj")?;
        let compressed_east_adj = cur.read_blob("east_adj")?;
        let compressed_west_adj = cur.read_blob("west_adj")?;
        let compressed_column_gen_step = cur.read_blob("column_gen_step")?;
        let compressed_world_compression_mode = cur.read_blob("world_compression_mode")?;
        let compressed_mapping = cur.read_blob("mapping")?;
        let data_format_version = cur.read_u8()?;
        let compression_mode = DhV2CompressionMode::from_byte(cur.read_u8()?)?;
        let apply_to_parent = cur.read_u8()? != 0;
        let apply_to_children = cur.read_u8()? != 0;
        let regenerate = cur.read_u8()? != 0;
        let last_modified_unix_ms = cur.read_i64()?;
        let created_unix_ms = cur.read_i64()?;

        Ok(Self {
            pos,
            data_checksum,
            compressed_data,
            compressed_north_adj,
            compressed_south_adj,
            compressed_east_adj,
            compressed_west_adj,
            compressed_column_gen_step,
            compressed_world_compression_mode,
            compressed_mapping,
            data_format_version,
            compression_mode,
            apply_to_parent,
            apply_to_children,
            regenerate,
            last_modified_unix_ms,
            created_unix_ms,
        })
    }

    /// Encodes this DTO into the exact big-endian wire format of `FullDataSourceV2DTO.encode`.
    pub fn encode_wire(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            64 + self.compressed_data.len()
                + self.compressed_north_adj.len()
                + self.compressed_south_adj.len()
                + self.compressed_east_adj.len()
                + self.compressed_west_adj.len()
                + self.compressed_column_gen_step.len()
                + self.compressed_world_compression_mode.len()
                + self.compressed_mapping.len(),
        );
        out.extend_from_slice(&self.pos.to_be_bytes());
        out.extend_from_slice(&self.data_checksum.to_be_bytes());
        write_blob(&mut out, &self.compressed_data);
        write_blob(&mut out, &self.compressed_north_adj);
        write_blob(&mut out, &self.compressed_south_adj);
        write_blob(&mut out, &self.compressed_east_adj);
        write_blob(&mut out, &self.compressed_west_adj);
        write_blob(&mut out, &self.compressed_column_gen_step);
        write_blob(&mut out, &self.compressed_world_compression_mode);
        write_blob(&mut out, &self.compressed_mapping);
        out.push(self.data_format_version);
        out.push(self.compression_mode as u8);
        out.push(u8::from(self.apply_to_parent));
        out.push(u8::from(self.apply_to_children));
        out.push(u8::from(self.regenerate));
        out.extend_from_slice(&self.last_modified_unix_ms.to_be_bytes());
        out.extend_from_slice(&self.created_unix_ms.to_be_bytes());
        out
    }

    /// Decompresses and decodes the ID mapping and all `64 x 64` terrain columns.
    /// `level_min_y` is the dimension's minimum build Y (e.g., `-64` for Overworld, `0` for
    /// Nether/End), used to turn DH's relative `bottomY` values into world block Y coordinates.
    pub fn decode_section(&self, level_min_y: i32) -> Result<DecodedDhV2Section, DhV2Error> {
        let (section_detail, section_x, section_z) = unpack_dh_section_pos(self.pos);
        if section_detail < DH_SECTION_MIN_DETAIL {
            return Err(DhV2Error::CorruptData(format!(
                "section detail {section_detail} is below minimum {DH_SECTION_MIN_DETAIL}"
            )));
        }
        let lod_detail = section_detail - DH_SECTION_MIN_DETAIL;

        let mapping_raw = decompress_blob(&self.compressed_mapping, self.compression_mode)?;
        let mapping = decode_mapping(&mapping_raw)?;

        let mut columns = vec![Vec::new(); SECTION_CELLS];

        match self.data_format_version {
            1 => {
                if !self.compressed_data.is_empty() {
                    let raw = decompress_blob(&self.compressed_data, self.compression_mode)?;
                    decode_data_v1(&raw, level_min_y, &mut columns)?;
                }
            }
            2 | 3 => {
                if !self.compressed_data.is_empty() {
                    let raw = decompress_blob(&self.compressed_data, self.compression_mode)?;
                    decode_data_v2_region(
                        &raw,
                        1,
                        SECTION_SIZE - 1,
                        1,
                        SECTION_SIZE - 1,
                        level_min_y,
                        &mut columns,
                    )?;
                }
                // Border strips (matching `FullDataMinMaxPosUtil`):
                // NORTH: x 0..64, z 0..1
                // SOUTH: x 0..64, z 63..64
                // EAST:  x 63..64, z 0..64
                // WEST:  x 0..1, z 0..64
                let strips: [(&[u8], usize, usize, usize, usize); 4] = [
                    (&self.compressed_north_adj, 0, SECTION_SIZE, 0, 1),
                    (
                        &self.compressed_south_adj,
                        0,
                        SECTION_SIZE,
                        SECTION_SIZE - 1,
                        SECTION_SIZE,
                    ),
                    (
                        &self.compressed_east_adj,
                        SECTION_SIZE - 1,
                        SECTION_SIZE,
                        0,
                        SECTION_SIZE,
                    ),
                    (&self.compressed_west_adj, 0, 1, 0, SECTION_SIZE),
                ];
                for (blob, min_x, max_x, min_z, max_z) in strips {
                    if !blob.is_empty() {
                        let raw = decompress_blob(blob, self.compression_mode)?;
                        decode_data_v2_region(
                            &raw,
                            min_x,
                            max_x,
                            min_z,
                            max_z,
                            level_min_y,
                            &mut columns,
                        )?;
                    }
                }
            }
            other => return Err(DhV2Error::UnsupportedFormatVersion(other)),
        }

        Ok(DecodedDhV2Section {
            lod_detail,
            section_x,
            section_z,
            mapping,
            columns,
            last_modified_unix_ms: self.last_modified_unix_ms,
            created_unix_ms: self.created_unix_ms,
        })
    }
}

fn write_blob(out: &mut Vec<u8>, blob: &[u8]) {
    let len = i32::try_from(blob.len()).unwrap_or(i32::MAX);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(blob);
}

struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], DhV2Error> {
        let end = self.pos.checked_add(len).ok_or(DhV2Error::Truncated)?;
        if end > self.data.len() {
            return Err(DhV2Error::Truncated);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, DhV2Error> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, DhV2Error> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16, DhV2Error> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_be_bytes([b[0], b[1]]))
    }

    fn read_i32(&mut self) -> Result<i32, DhV2Error> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i64(&mut self) -> Result<i64, DhV2Error> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_blob(&mut self, name: &str) -> Result<Vec<u8>, DhV2Error> {
        let len = self.read_i32()?;
        if len < 0 || (len as usize) > MAX_DECOMPRESSED_BYTES {
            return Err(DhV2Error::InvalidLength(format!("{name} length {len}")));
        }
        Ok(self.read_bytes(len as usize)?.to_vec())
    }

    fn read_varint(&mut self) -> Result<u32, DhV2Error> {
        let mut value: u32 = 0;
        let mut shift: u32 = 0;
        loop {
            if shift >= 32 {
                return Err(DhV2Error::CorruptData("varint exceeds 32 bits".into()));
            }
            let b = self.read_u8()?;
            value |= ((b & 0x7F) as u32) << shift;
            if (b & 0x80) == 0 {
                return Ok(value);
            }
            shift += 7;
        }
    }
}

#[inline]
const fn zigzag_decode(n: u32) -> i32 {
    ((n >> 1) as i32) ^ -((n & 1) as i32)
}

#[inline]
pub const fn zigzag_encode(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

pub fn write_varint(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push(((value & 0x7F) as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn decode_mapping(raw: &[u8]) -> Result<Vec<DhMappingEntry>, DhV2Error> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let mut cur = ByteCursor::new(raw);
    let count = cur.read_i32()?;
    if count < 0 || (count as usize) > MAX_MAPPING_ENTRIES {
        return Err(DhV2Error::InvalidLength(format!("mapping count {count}")));
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let utf_len = cur.read_u16()? as usize;
        let bytes = cur.read_bytes(utf_len)?;
        let s = String::from_utf8_lossy(bytes);
        let (biome_raw, block_raw) = match s.split_once(BLOCK_STATE_SEPARATOR) {
            Some((b, blk)) => (b, blk),
            None => ("minecraft:plains", s.as_ref()),
        };
        let biome_id = normalize_identifier(biome_raw, "minecraft:plains");
        let block_id = extract_base_block_id(block_raw);
        out.push(DhMappingEntry {
            biome_id,
            block_id,
            raw_block_state: block_raw.to_owned(),
        });
    }
    Ok(out)
}

fn normalize_identifier(raw: &str, fallback: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return fallback.to_owned();
    }
    // DH sometimes prefixes serialized wrappers with metadata before a namespace or uses `AIR`.
    if trimmed.eq_ignore_ascii_case("AIR") {
        return "minecraft:air".to_owned();
    }
    if trimmed.contains(':') {
        trimmed.to_owned()
    } else {
        format!("minecraft:{trimmed}")
    }
}

fn extract_base_block_id(raw_block_state: &str) -> String {
    let trimmed = raw_block_state.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("AIR") {
        return "minecraft:air".to_owned();
    }
    // Strip Minecraft blockstate properties `[key=val,...]` or DH wrapper suffixes (`_STATE_...`).
    let without_props = trimmed.split('[').next().unwrap_or(trimmed);
    let without_dh_state = without_props
        .split("_STATE_")
        .next()
        .unwrap_or(without_props);
    normalize_identifier(without_dh_state, "minecraft:air")
}

/// Decodes upstream `FullDataSourceV2DTO.readBlobToDataSourceDataArrayV1`.
fn decode_data_v1(
    raw: &[u8],
    level_min_y: i32,
    columns: &mut [Vec<DecodedDhDataPoint>],
) -> Result<(), DhV2Error> {
    let mut cur = ByteCursor::new(raw);
    for x in 0..SECTION_SIZE {
        for z in 0..SECTION_SIZE {
            let len = cur.read_i16()?;
            if len < 0 || (len as usize) > MAX_COLUMN_POINTS {
                return Err(DhV2Error::InvalidLength(format!(
                    "v1 column ({x}, {z}) length {len}"
                )));
            }
            let mut col = Vec::with_capacity(len as usize);
            for _ in 0..len {
                let packed = cur.read_i64()? as u64;
                let mapping_id = (packed & 0x7FFF_FFFF) as u32;
                let height = ((packed >> 32) & 0xFFF) as u16;
                let rel_min_y = ((packed >> 44) & 0xFFF) as i32;
                let sky_light = ((packed >> 56) & 0xF) as u8;
                let block_light = ((packed >> 60) & 0xF) as u8;
                col.push(DecodedDhDataPoint {
                    mapping_id,
                    bottom_y: level_min_y + rel_min_y,
                    height,
                    block_light,
                    sky_light,
                });
            }
            columns[z * SECTION_SIZE + x] = col;
        }
    }
    Ok(())
}

/// Decodes upstream `FullDataSourceV2DTO.readBlobToDataSourceDataArrayV2` for a rectangular
/// `(min_x..max_x, min_z..max_z)` slice.
fn decode_data_v2_region(
    raw: &[u8],
    min_x: usize,
    max_x: usize,
    min_z: usize,
    max_z: usize,
    level_min_y: i32,
    columns: &mut [Vec<DecodedDhDataPoint>],
) -> Result<(), DhV2Error> {
    let mut cur = ByteCursor::new(raw);
    let nx = max_x - min_x;
    let nz = max_z - min_z;
    let total_cols = nx * nz;

    // 1. Column lengths
    let mut lengths = Vec::with_capacity(total_cols);
    let mut total_points = 0usize;
    for _ in 0..total_cols {
        let len = cur.read_varint()? as usize;
        if len > MAX_COLUMN_POINTS {
            return Err(DhV2Error::InvalidLength(format!("v2 column length {len}")));
        }
        lengths.push(len);
        total_points = total_points.saturating_add(len);
    }

    // 2. IDs + flags (`bit 0 = has_y_delta`, `bit 1 = has_light`)
    let mut ids = Vec::with_capacity(total_points);
    let mut flags = Vec::with_capacity(total_points);
    for _ in 0..total_points {
        let encoded = cur.read_varint()?;
        ids.push(encoded >> 2);
        flags.push((encoded & 0x3) as u8);
    }

    // 3. Heights
    let mut heights = Vec::with_capacity(total_points);
    for _ in 0..total_points {
        let h = cur.read_varint()?;
        heights.push(h.min(u16::MAX as u32) as u16);
    }

    // 4. Relative bottom Y (delta-coded across the entire slice, matching DH's `previousBottomY`)
    let mut bottom_ys = Vec::with_capacity(total_points);
    let mut previous_bottom_y: i32 = 0;
    for i in 0..total_points {
        let error = if (flags[i] & 1) != 0 {
            zigzag_decode(cur.read_varint()?)
        } else {
            0
        };
        let bottom_y = previous_bottom_y - (heights[i] as i32) + error;
        bottom_ys.push(level_min_y + bottom_y);
        previous_bottom_y = bottom_y;
    }

    // 5. Lights (packed byte `(block_light << 4) | sky_light` only when `has_light` flag is set)
    let mut lights = Vec::with_capacity(total_points);
    for &flag in &flags {
        if (flag & 2) != 0 {
            let packed = cur.read_u8()?;
            let sky = packed & 0x0F;
            let block = (packed >> 4) & 0x0F;
            lights.push((block, sky));
        } else {
            lights.push((0, 0));
        }
    }

    // Scatter into `columns[z * 64 + x]`
    let mut col_idx = 0usize;
    let mut pt_idx = 0usize;
    for x in min_x..max_x {
        for z in min_z..max_z {
            let len = lengths[col_idx];
            col_idx += 1;
            let mut col = Vec::with_capacity(len);
            for _ in 0..len {
                let (block_light, sky_light) = lights[pt_idx];
                col.push(DecodedDhDataPoint {
                    mapping_id: ids[pt_idx],
                    bottom_y: bottom_ys[pt_idx],
                    height: heights[pt_idx],
                    block_light,
                    sky_light,
                });
                pt_idx += 1;
            }
            columns[z * SECTION_SIZE + x] = col;
        }
    }

    Ok(())
}

/// Decompresses a single DH V2 blob according to `mode`, with a 64 MiB output limit.
pub fn decompress_blob(data: &[u8], mode: DhV2CompressionMode) -> Result<Vec<u8>, DhV2Error> {
    decompress_blob_with_limit(data, mode, MAX_DECOMPRESSED_BYTES)
}

/// Decodes at most `limit` output bytes. `InvalidLength` means the output limit was exceeded.
pub(crate) fn decompress_blob_with_limit(
    data: &[u8],
    mode: DhV2CompressionMode,
    limit: usize,
) -> Result<Vec<u8>, DhV2Error> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    match mode {
        DhV2CompressionMode::Uncompressed => {
            if data.len() > limit {
                return Err(DhV2Error::InvalidLength(
                    "uncompressed blob exceeds limit".into(),
                ));
            }
            Ok(data.to_vec())
        }
        DhV2CompressionMode::Lz4 => decompress_lz4_frame(data, limit),
        DhV2CompressionMode::ZstdBlock | DhV2CompressionMode::ZstdStream => {
            let decoder = ruzstd::decoding::StreamingDecoder::new(data)
                .map_err(|e| DhV2Error::DecompressFailed(e.to_string()))?;
            read_bounded(decoder, limit, "zstd")
        }
        DhV2CompressionMode::Lzma2 => {
            let decoder = xz4rust::XzDecoder::in_heap_with_alloc_dict_size(1 << 20, 16 << 20);
            let buf_size = std::num::NonZeroUsize::new(1 << 16).expect("non-zero");
            let reader = xz4rust::XzReader::new_with_buffer_size_and_decoder(
                std::io::Cursor::new(data.to_vec()),
                buf_size,
                decoder,
            );
            read_bounded(reader, limit, "xz")
        }
    }
}

fn read_bounded(mut reader: impl Read, limit: usize, codec: &str) -> Result<Vec<u8>, DhV2Error> {
    let mut out = Vec::new();
    reader
        .by_ref()
        .take(limit as u64)
        .read_to_end(&mut out)
        .map_err(|e| DhV2Error::DecompressFailed(e.to_string()))?;
    if out.len() == limit {
        // Take stops at the limit even if there is more output. Probe without growing `out`.
        loop {
            match reader.read(&mut [0]) {
                Ok(0) => break,
                Ok(_) => {
                    return Err(DhV2Error::InvalidLength(format!(
                        "{codec} decompressed blob exceeds limit"
                    )));
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(DhV2Error::DecompressFailed(e.to_string())),
            }
        }
    }
    Ok(out)
}

/// Decodes a standard LZ4 frame (`0x184D2204`, produced by Java `LZ4FrameOutputStream`) using
/// `lz4_flex::block::decompress` with zero extra crates.
fn decompress_lz4_frame(data: &[u8], limit: usize) -> Result<Vec<u8>, DhV2Error> {
    if data.len() < 7 {
        return Err(DhV2Error::DecompressFailed("lz4 frame too short".into()));
    }
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 0x184D_2204 {
        return Err(DhV2Error::DecompressFailed(format!(
            "unexpected lz4 frame magic 0x{magic:08x}"
        )));
    }
    let flg = data[4];
    let bd = data[5];
    let version = (flg >> 6) & 0x3;
    if version != 1 {
        return Err(DhV2Error::DecompressFailed(format!(
            "unsupported lz4 frame version {version}"
        )));
    }
    let block_checksum = (flg & 0x10) != 0;
    let content_size = (flg & 0x08) != 0;
    let content_checksum = (flg & 0x04) != 0;
    let dict_id = (flg & 0x01) != 0;
    let block_max_id = (bd >> 4) & 0x7;
    let block_max_size: usize = match block_max_id {
        4 => 64 * 1024,
        5 => 256 * 1024,
        6 => 1024 * 1024,
        7 => 4 * 1024 * 1024,
        other => {
            return Err(DhV2Error::DecompressFailed(format!(
                "invalid lz4 block max id {other}"
            )));
        }
    };

    let mut pos = 6usize;
    if content_size {
        pos += 8;
    }
    if dict_id {
        pos += 4;
    }
    // Header checksum byte
    pos += 1;
    if pos > data.len() {
        return Err(DhV2Error::DecompressFailed(
            "lz4 frame header truncated".into(),
        ));
    }

    let mut out = Vec::new();
    loop {
        if pos + 4 > data.len() {
            return Err(DhV2Error::DecompressFailed(
                "lz4 frame truncated before end mark".into(),
            ));
        }
        let block_hdr =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        if block_hdr == 0 {
            if content_checksum && pos + 4 <= data.len() {
                pos += 4;
            }
            let _ = pos;
            break;
        }
        let uncompressed = (block_hdr & 0x8000_0000) != 0;
        let block_len = (block_hdr & 0x7FFF_FFFF) as usize;
        if pos + block_len > data.len() {
            return Err(DhV2Error::DecompressFailed(
                "lz4 frame block truncated".into(),
            ));
        }
        let block_bytes = &data[pos..pos + block_len];
        pos += block_len;
        if block_checksum {
            if pos + 4 > data.len() {
                return Err(DhV2Error::DecompressFailed(
                    "lz4 block checksum truncated".into(),
                ));
            }
            pos += 4;
        }
        let remaining = limit - out.len();
        if uncompressed {
            if block_bytes.len() > remaining {
                return Err(DhV2Error::InvalidLength(
                    "lz4 decompressed blob exceeds limit".into(),
                ));
            }
            out.extend_from_slice(block_bytes);
        } else {
            // With safe-decode, lz4_flex 0.14.0 uses a fixed-size output slice and
            // reports OutputTooSmall before a literal or match would exceed it.
            let decoded = lz4_flex::block::decompress(block_bytes, block_max_size.min(remaining))
                .map_err(|e| match e {
                lz4_flex::block::DecompressError::OutputTooSmall { expected, .. }
                    if remaining < block_max_size && expected <= block_max_size =>
                {
                    DhV2Error::InvalidLength("lz4 decompressed blob exceeds limit".into())
                }
                _ => DhV2Error::DecompressFailed(e.to_string()),
            })?;
            out.extend_from_slice(&decoded);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper that encodes a rectangular slice using DH's V2/V3 columnar format (matching
    /// `FullDataSourceV2DTO.writeDataSourceDataArrayToBlobV2`) for roundtrip testing.
    fn encode_v2_region(
        cols: &[Vec<DecodedDhDataPoint>],
        min_x: usize,
        max_x: usize,
        min_z: usize,
        max_z: usize,
        level_min_y: i32,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        // 1. lengths
        for x in min_x..max_x {
            for z in min_z..max_z {
                write_varint(&mut out, cols[z * SECTION_SIZE + x].len() as u32);
            }
        }
        // 2. ids + flags
        let mut prev_bottom = 0i32;
        for x in min_x..max_x {
            for z in min_z..max_z {
                for pt in &cols[z * SECTION_SIZE + x] {
                    let rel_bottom = pt.bottom_y - level_min_y;
                    let expected = prev_bottom - pt.height as i32;
                    let has_y_delta = u32::from(rel_bottom != expected);
                    let has_light = u32::from((pt.block_light | pt.sky_light) != 0);
                    write_varint(
                        &mut out,
                        (pt.mapping_id << 2) | (has_light << 1) | has_y_delta,
                    );
                    prev_bottom = rel_bottom;
                }
            }
        }
        // 3. heights
        for x in min_x..max_x {
            for z in min_z..max_z {
                for pt in &cols[z * SECTION_SIZE + x] {
                    write_varint(&mut out, pt.height as u32);
                }
            }
        }
        // 4. bottomY deltas
        let mut prev_bottom = 0i32;
        for x in min_x..max_x {
            for z in min_z..max_z {
                for pt in &cols[z * SECTION_SIZE + x] {
                    let rel_bottom = pt.bottom_y - level_min_y;
                    let expected = prev_bottom - pt.height as i32;
                    if rel_bottom != expected {
                        write_varint(&mut out, zigzag_encode(rel_bottom - expected));
                    }
                    prev_bottom = rel_bottom;
                }
            }
        }
        // 5. lights
        for x in min_x..max_x {
            for z in min_z..max_z {
                for pt in &cols[z * SECTION_SIZE + x] {
                    let packed = (pt.block_light << 4) | (pt.sky_light & 0xF);
                    if packed != 0 {
                        out.push(packed);
                    }
                }
            }
        }
        out
    }

    fn encode_mapping(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(entries.len() as i32).to_be_bytes());
        for (biome, block) in entries {
            let s = format!("{biome}{BLOCK_STATE_SEPARATOR}{block}");
            out.extend_from_slice(&(s.len() as u16).to_be_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        out
    }

    #[test]
    fn dh_section_pos_pack_unpack_preserves_signed_coords_and_detail() {
        for (detail, x, z) in [
            (6u8, 0i32, 0i32),
            (7, -1, -1),
            (10, 12345, -67890),
            (12, -(1 << 27), (1 << 27) - 1),
        ] {
            let packed = pack_dh_section_pos(detail, x, z);
            assert_eq!(unpack_dh_section_pos(packed), (detail, x, z));
        }
    }

    #[test]
    fn full_data_source_v2_dto_wire_and_zstd_decode_roundtrip() {
        let level_min_y = -64;
        let mut cols = vec![Vec::new(); SECTION_CELLS];
        // Interior column (10, 20): water over stone + cave overhang below
        cols[20 * SECTION_SIZE + 10] = vec![
            DecodedDhDataPoint {
                mapping_id: 2,
                bottom_y: 58,
                height: 5,
                block_light: 0,
                sky_light: 15,
            },
            DecodedDhDataPoint {
                mapping_id: 1,
                bottom_y: 40,
                height: 18,
                block_light: 0,
                sky_light: 0,
            },
            DecodedDhDataPoint {
                mapping_id: 0,
                bottom_y: 25,
                height: 15,
                block_light: 4,
                sky_light: 0,
            },
            DecodedDhDataPoint {
                mapping_id: 1,
                bottom_y: -64,
                height: 89,
                block_light: 0,
                sky_light: 0,
            },
        ];
        // North border column (5, 0)
        cols[5] = vec![DecodedDhDataPoint {
            mapping_id: 1,
            bottom_y: 0,
            height: 70,
            block_light: 2,
            sky_light: 12,
        }];

        let mapping_raw = encode_mapping(&[
            ("minecraft:plains", "minecraft:air"),
            ("minecraft:plains", "minecraft:stone"),
            ("minecraft:ocean", "minecraft:water[level=0]"),
        ]);
        let interior_raw = encode_v2_region(&cols, 1, 63, 1, 63, level_min_y);
        let north_raw = encode_v2_region(&cols, 0, 64, 0, 1, level_min_y);

        let compress = |b: &[u8]| {
            ruzstd::encoding::compress_to_vec(b, ruzstd::encoding::CompressionLevel::Fastest)
        };

        let dto = DhV2FullDataDto {
            pos: pack_dh_section_pos(DH_SECTION_MIN_DETAIL + 1, -3, 4),
            data_checksum: 42,
            compressed_data: compress(&interior_raw),
            compressed_north_adj: compress(&north_raw),
            compressed_south_adj: Vec::new(),
            compressed_east_adj: Vec::new(),
            compressed_west_adj: Vec::new(),
            compressed_column_gen_step: Vec::new(),
            compressed_world_compression_mode: Vec::new(),
            compressed_mapping: compress(&mapping_raw),
            data_format_version: 3,
            compression_mode: DhV2CompressionMode::ZstdBlock,
            apply_to_parent: true,
            apply_to_children: false,
            regenerate: false,
            last_modified_unix_ms: 1_700_000_000_000,
            created_unix_ms: 1_699_000_000_000,
        };

        let wire = dto.encode_wire();
        let parsed = DhV2FullDataDto::decode_wire(&wire).expect("decode wire");
        assert_eq!(parsed, dto);

        let decoded = parsed.decode_section(level_min_y).expect("decode section");
        assert_eq!(decoded.lod_detail, 1);
        assert_eq!(decoded.section_x, -3);
        assert_eq!(decoded.section_z, 4);
        assert_eq!(decoded.mapping.len(), 3);
        assert_eq!(decoded.mapping[2].biome_id, "minecraft:ocean");
        assert_eq!(decoded.mapping[2].block_id, "minecraft:water");
        assert_eq!(
            decoded.columns[20 * SECTION_SIZE + 10],
            cols[20 * SECTION_SIZE + 10]
        );
        assert_eq!(decoded.columns[5], cols[5]);
    }

    #[test]
    fn bounded_reader_stops_after_one_overflow_byte() {
        struct CountingReader<R> {
            inner: R,
            emitted: usize,
        }
        impl<R: Read> Read for CountingReader<R> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = self.inner.read(buf)?;
                self.emitted += n;
                Ok(n)
            }
        }

        let payload = [b'a'; 64];
        let blob = ruzstd::encoding::compress_to_vec(
            payload.as_slice(),
            ruzstd::encoding::CompressionLevel::Fastest,
        );
        let reader = || CountingReader {
            inner: ruzstd::decoding::StreamingDecoder::new(blob.as_slice()).unwrap(),
            emitted: 0,
        };
        assert_eq!(
            read_bounded(reader(), payload.len(), "zstd").unwrap(),
            payload
        );
        let mut oversized = reader();
        assert!(matches!(
            read_bounded(&mut oversized, 32, "zstd"),
            Err(DhV2Error::InvalidLength(_))
        ));
        assert_eq!(oversized.emitted, 33);
    }

    #[test]
    fn decompress_lz4_frame_handles_compressed_and_uncompressed_blocks() {
        let raw_block = b"raw";
        let payload =
            b"minecraft:plains_DH-BSW_minecraft:stone_repeated_repeated_repeated_repeated";
        let compressed_block = lz4_flex::block::compress(payload);

        let mut frame = vec![0x04, 0x22, 0x4d, 0x18, 0x60, 0x40, 0x82];
        frame.extend_from_slice(&(0x8000_0000 | raw_block.len() as u32).to_le_bytes());
        frame.extend_from_slice(raw_block);
        frame.extend_from_slice(&(compressed_block.len() as u32).to_le_bytes());
        frame.extend_from_slice(&compressed_block);
        frame.extend_from_slice(&0u32.to_le_bytes());

        let mode = DhV2CompressionMode::Lz4;
        let expected = [raw_block.as_slice(), payload.as_slice()].concat();
        assert_eq!(decompress_blob(&frame, mode).unwrap(), expected);
        assert_eq!(
            decompress_blob_with_limit(&frame, mode, expected.len()).unwrap(),
            expected
        );
        for limit in [raw_block.len() - 1, expected.len() - 1] {
            assert!(matches!(
                decompress_blob_with_limit(&frame, mode, limit),
                Err(DhV2Error::InvalidLength(_))
            ));
        }
    }

    #[test]
    fn lz4_compressed_block_stops_before_validating_past_output_limit() {
        let frame = [
            0x04, 0x22, 0x4d, 0x18, 0x60, 0x40, 0x82, // frame header
            5, 0, 0, 0, // compressed block size
            0x10, b'a', 1, 0,    // one literal and a four-byte match
            0x10, // missing a final literal (only reached if the match fit)
            0, 0, 0, 0, // end mark
        ];
        let mode = DhV2CompressionMode::Lz4;
        assert!(matches!(
            decompress_blob_with_limit(&frame, mode, 4),
            Err(DhV2Error::InvalidLength(_))
        ));
        assert!(matches!(
            decompress_blob_with_limit(&frame, mode, 5),
            Err(DhV2Error::DecompressFailed(_))
        ));
    }
}
