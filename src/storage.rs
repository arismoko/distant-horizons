//! Fast, crash-safe region-packed Zstd binary LOD storage (`DHREG002`) under
//! `<world_dir>/distant_sunrises/<dim>/d<detail>/r.<rx>.<rz>.dhreg`.
//!
//! Architecture:
//! - Packs `16 × 16 = 256` LOD sections per region file, cutting filesystem metadata/inode
//!   overhead by up to 256× compared to one-file-per-section storage while avoiding SQLite's
//!   SQL/B-tree/FFI overhead.
//! - Double-buffered, CRC32-checksummed, 512-byte-sector-aligned header pages (`Page A` at
//!   `0..4608`, `Page B` at `4608..9216`): each slot entry is 16 bytes `(offset, len, crc32)`,
//!   so no entry ever straddles a 512-byte sector boundary and any torn write or power loss
//!   falls back cleanly to the previous intact header page.
//! - Copy-on-write append updates: compressed `DHLOD002` records are appended at EOF and
//!   `sync_data()`'d before committing the alternate header page and syncing again.
//! - Streaming atomic compaction when dead overwritten space exceeds 2× live bytes: closes the
//!   handle before `rename` (Windows-safe), `sync_all()`s the temp file and parent directory,
//!   propagates read I/O errors instead of dropping slots, and removes failed handles from cache.
//! - Also stores compressed 32³ octree sections and their column footprints (`DHLOD003`)
//!   in `.dhoct` region files of the same crash-safe layout (`8 × 4 × 8` sections per region),
//!   saved and loaded through `LodIoWorker::save_octree_section` / `load_octree_section`.
//! - Coalescing background `LodIoWorker` with explicit `flush()` and clean `Drop` join so
//!   queued saves are never dropped on world unload or process exit.

use crate::dh_v2::{self, DhV2CompressionMode};
use crate::lod::{ExtraSpan, LodColumn, LodSection, MAX_DETAIL, SECTION_CELLS, SectionPos};
use crate::octree::{
    ColumnFootprint, MAX_OCTREE_DETAIL, OctreeError, OctreeRecord, OctreeSection, OctreeSectionPos,
    RleSection, Run, SECTION_VOLUME, Voxel,
};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

const MAGIC_V1: &[u8; 8] = b"DHLOD001";
const MAGIC_V2: &[u8; 8] = b"DHLOD002";
const REGION_MAGIC: &[u8; 8] = b"DHREG002";

/// `16 × 16` sections per `.dhreg` region file.
pub const REGION_SHIFT: i32 = 4;
pub const REGION_SIZE: usize = 1 << REGION_SHIFT;
pub const REGION_SLOTS: usize = REGION_SIZE * REGION_SIZE;

const HEADER_META_BYTES: usize = 512; // 1 sector: magic(8) + gen(8) + crc(4) + reserved(492)
const SLOT_ENTRY_BYTES: usize = 16; // u64 offset + u32 len + u32 crc32 (never straddles 512B sector)
const HEADER_PAGE_BYTES: usize = HEADER_META_BYTES + REGION_SLOTS * SLOT_ENTRY_BYTES; // 4608 bytes (9 sectors)
const REGION_HEADER_BYTES: usize = HEADER_PAGE_BYTES * 2; // 9216 bytes (18 sectors)

const MAX_DECOMPRESSED_SECTION_BYTES: usize = 4 * 1024 * 1024;
const MAX_COMPRESSED_RECORD_BYTES: usize = MAX_DECOMPRESSED_SECTION_BYTES + 4096;
const MAX_CACHED_REGIONS: usize = 32;
/// `Mapper` field widths (`BLOCK_MASK` 20 bits, `BIOME_MASK` 9 bits in `octree.rs`).
const BLOCK_ID_MAX: u32 = (1 << 20) - 1;
const BIOME_ID_MAX: u32 = 0x1FF;

static TMP_SEQ: AtomicU64 = AtomicU64::new(1);

/// Encode a `64 × 64` LOD section into compact RLE binary format (`DHLOD002`, primary columns
/// plus optional extra vertical spans).
pub fn encode_section(section: &LodSection) -> Vec<u8> {
    let mut out = Vec::with_capacity(2048);
    out.extend_from_slice(MAGIC_V2);
    out.push(section.pos.detail.min(MAX_DETAIL));
    out.extend_from_slice(&section.pos.x.to_le_bytes());
    out.extend_from_slice(&section.pos.z.to_le_bytes());

    let mut i = 0;
    while i < section.columns.len() {
        let col = section.columns[i];
        let mut run: u16 = 1;
        while i + (run as usize) < section.columns.len()
            && run < u16::MAX
            && section.columns[i + run as usize] == col
        {
            run += 1;
        }
        out.extend_from_slice(&run.to_le_bytes());
        out.extend_from_slice(&col.to_bytes());
        i += run as usize;
    }

    // Extra vertical spans tail (deterministic cell-index order), bounded to stay within
    // `MAX_DECOMPRESSED_SECTION_BYTES`.
    let mut keys: Vec<u16> = section
        .extra_spans
        .iter()
        .filter(|(_, spans)| !spans.is_empty())
        .map(|(&idx, _)| idx)
        .collect();
    keys.sort_unstable();

    let count_pos = out.len();
    out.extend_from_slice(&0u16.to_le_bytes());
    let mut written_cells: u16 = 0;
    for cell_idx in keys {
        let spans = &section.extra_spans[&cell_idx];
        let count = spans.len().min(u8::MAX as usize) as u8;
        let needed = 3 + (count as usize) * ExtraSpan::SERIALIZED_LEN;
        if out.len() + needed > MAX_DECOMPRESSED_SECTION_BYTES {
            break;
        }
        out.extend_from_slice(&cell_idx.to_le_bytes());
        out.push(count);
        for span in &spans[..count as usize] {
            out.extend_from_slice(&span.to_bytes());
        }
        written_cells += 1;
    }
    out[count_pos..count_pos + 2].copy_from_slice(&written_cells.to_le_bytes());
    out
}

/// Strictly decode a `DHLOD001` or `DHLOD002` binary LOD section payload.
/// Rejects invalid `detail > MAX_DETAIL`, zero/oversized RLE runs, truncated `DHLOD002` tails,
/// and trailing garbage.
pub fn decode_section(bytes: &[u8]) -> Option<LodSection> {
    if bytes.len() < 17 {
        return None;
    }
    let version = match &bytes[0..8] {
        m if m == MAGIC_V1 => 1u8,
        m if m == MAGIC_V2 => 2u8,
        _ => return None,
    };
    let detail = bytes[8];
    if detail > MAX_DETAIL {
        return None;
    }
    let x = i32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]);
    let z = i32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]);
    let pos = SectionPos::new(detail, x, z);

    let mut columns = Vec::with_capacity(SECTION_CELLS);
    let mut cursor = 17;
    while columns.len() < SECTION_CELLS {
        if cursor + 18 > bytes.len() {
            return None;
        }
        let run = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        let remaining = SECTION_CELLS - columns.len();
        if run == 0 || run > remaining {
            return None;
        }
        let mut col_bytes = [0u8; 16];
        col_bytes.copy_from_slice(&bytes[cursor + 2..cursor + 18]);
        let col = LodColumn::from_bytes(col_bytes);
        cursor += 18;
        columns.resize(columns.len() + run, col);
    }

    let mut extra_spans = HashMap::new();
    if version >= 2 {
        if cursor + 2 > bytes.len() {
            return None;
        }
        let cell_count = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        cursor += 2;
        for _ in 0..cell_count {
            if cursor + 3 > bytes.len() {
                return None;
            }
            let cell_idx = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
            let span_count = bytes[cursor + 2] as usize;
            cursor += 3;
            let byte_len = span_count * ExtraSpan::SERIALIZED_LEN;
            if cursor + byte_len > bytes.len() || (cell_idx as usize) >= SECTION_CELLS {
                return None;
            }
            let mut spans = Vec::with_capacity(span_count);
            for _ in 0..span_count {
                let mut sb = [0u8; ExtraSpan::SERIALIZED_LEN];
                sb.copy_from_slice(&bytes[cursor..cursor + ExtraSpan::SERIALIZED_LEN]);
                cursor += ExtraSpan::SERIALIZED_LEN;
                spans.push(ExtraSpan::from_bytes(sb));
            }
            if !spans.is_empty() {
                extra_spans.insert(cell_idx, spans);
            }
        }
    }

    if cursor != bytes.len() {
        return None;
    }

    Some(LodSection::from_columns_and_spans(
        pos,
        columns,
        extra_spans,
    ))
}

/// Compress an encoded `DHLOD002` payload with Zstd (fastest level) + leading CRC32 checksum.
/// Returns `(record_bytes, crc32)` or `None` if the encoded record exceeds the safe upper bound.
pub fn compress_section_record(section: &LodSection) -> Option<(Vec<u8>, u32)> {
    let raw = encode_section(section);
    if raw.len() > MAX_DECOMPRESSED_SECTION_BYTES {
        return None;
    }
    let compressed = ruzstd::encoding::compress_to_vec(
        raw.as_slice(),
        ruzstd::encoding::CompressionLevel::Fastest,
    );
    let total = 4usize.checked_add(compressed.len())?;
    if total > MAX_COMPRESSED_RECORD_BYTES || u32::try_from(total).is_err() {
        return None;
    }
    let crc = crc32fast::hash(&compressed);
    let mut record = Vec::with_capacity(total);
    record.extend_from_slice(&crc.to_le_bytes());
    record.extend_from_slice(&compressed);
    Some((record, crc))
}

/// Verify CRC32 (both embedded and slot-header CRC when provided), decompress Zstd, and decode a
/// stored region record.
pub fn decompress_section_record(
    record: &[u8],
    expected_slot_crc: Option<u32>,
) -> Option<LodSection> {
    if record.len() < 8 || record.len() > MAX_COMPRESSED_RECORD_BYTES {
        return None;
    }
    let embedded_crc = u32::from_le_bytes([record[0], record[1], record[2], record[3]]);
    if let Some(slot_crc) = expected_slot_crc
        && slot_crc != embedded_crc
    {
        return None;
    }
    let compressed = &record[4..];
    if crc32fast::hash(compressed) != embedded_crc {
        return None;
    }
    let decoder = ruzstd::decoding::StreamingDecoder::new(compressed).ok()?;
    let mut raw = Vec::new();
    decoder
        .take(MAX_DECOMPRESSED_SECTION_BYTES as u64 + 1)
        .read_to_end(&mut raw)
        .ok()?;
    if raw.len() > MAX_DECOMPRESSED_SECTION_BYTES {
        return None;
    }
    decode_section(&raw)
}

/// Collision-free, path-traversal-safe encoding of a dimension identifier (`%XX` for any byte
/// outside `[A-Za-z0-9-]`). Distinct IDs such as `"mod:a/b"` and `"mod:a_b"` never collide.
pub fn sanitize_dimension(dimension: &str) -> String {
    if dimension.is_empty() {
        return "_empty".to_owned();
    }
    let mut out = String::with_capacity(dimension.len() + 8);
    for b in dimension.bytes() {
        if b.is_ascii_alphanumeric() || b == b'-' {
            out.push(b as char);
        } else {
            use std::fmt::Write as _;
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

/// Legacy lossy folder name used before `DHREG002` (checked read-only when falling back to old
/// `.dhlod` files).
fn legacy_sanitize_dimension(dimension: &str) -> String {
    if dimension.is_empty() {
        return "_empty".to_owned();
    }
    dimension
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Region coordinates `(rx, rz)` and local slot index `0..256` for `pos`.
#[inline]
pub const fn section_region_slot(pos: SectionPos) -> (i32, i32, usize) {
    let rx = pos.x >> REGION_SHIFT;
    let rz = pos.z >> REGION_SHIFT;
    let lx = pos.x.rem_euclid(REGION_SIZE as i32) as usize;
    let lz = pos.z.rem_euclid(REGION_SIZE as i32) as usize;
    (rx, rz, lz * REGION_SIZE + lx)
}

/// Path to the `.dhreg` region file owning `pos`.
pub fn region_path(world_dir: &Path, dimension: &str, pos: SectionPos) -> PathBuf {
    let (rx, rz, _) = section_region_slot(pos);
    world_dir
        .join("distant_sunrises")
        .join(sanitize_dimension(dimension))
        .join(format!("d{}", pos.detail))
        .join(format!("r.{rx}.{rz}.dhreg"))
}

/// Legacy single-section `.dhlod` path (kept read-only for seamless migration of existing worlds).
pub fn legacy_section_path(world_dir: &Path, dimension: &str, pos: SectionPos) -> PathBuf {
    world_dir
        .join("distant_sunrises")
        .join(legacy_sanitize_dimension(dimension))
        .join(format!("d{}", pos.detail))
        .join(format!("s.{}.{}.dhlod", pos.x, pos.z))
}

// ---------------------------------------------------------------------------------------------
// DHLOD003: compressed 32³ octree section codec
// ---------------------------------------------------------------------------------------------
//
// The base layout follows Distant Horizons; references are relative to
// `coreSubProjects/` of `/box/shared/references/distant-horizons`:
// - `core/.../sql/dto/FullDataSourceV2DTO.java` (`DTO`): blob layout, format version, checksum.
// - `api/.../enums/config/EDhApiDataCompressionMode.java` (`MODE`): compression ids.
// - `core/.../sql/dto/util/VarintUtil.java` (`VARINT`): LEB128-style varints.
// - `core/.../dataObjects/fullData/FullDataPointIdMap.java` (`IDMAP`): block/biome string table.
// - `core/.../util/objects/dataStreams/DhDataInputStream.java` / `DhDataOutputStream.java`
//   (`STREAM`): which compressor wraps each blob.
//
// Record layout (big-endian like `DataOutputStream` / Netty `ByteBuf`, `DTO` `encode`
// lines 762-801; the DTO itself is a SQLite row, so the region slot plays the row's part):
//
// ```text
//  0  magic            "DHLOD003"
//  8  format_version   u8   = 4             Distant Sunrises version (adds the footprint)
//  9  compression      u8                   DTO compressionModeValue (lines 89, 132); MODE values
// 10  detail           u8   0..=4           `OctreeSectionPos::detail` (DTO packs it in `pos` long, line 765)
// 11  x, y, z          i32 each             rest of `OctreeSectionPos`
// 23  revision         u64                  `OctreeSection::revision` (DTO lastModifiedUnixDateTime, line 799)
// 31  data_len         u32  + data blob     DTO compressedDataByteArray (lines 75, 769-770)
//     mapping_len      u32  + mapping blob  DTO compressedMappingByteArray (lines 86, 789-790)
//     crc32            u32  over all preceding bytes   DTO dataChecksum (lines 73, 766)
// ```
//
// Data blob (compressed with `compression`, `STREAM` wrap rules): `VARINT` varints for
// `palette_len`; per palette entry `(block_id << 1) | has_light` then `biome_id`, then one packed
// light byte `block_light << 4 | sky_light` only when non-zero (the "lights only when lit" rule of
// `DTO` `writeDataSourceDataArrayToBlobV2` step 5, lines 501-522 and read side 622-648, with the
// id/flag varint of step 2, lines 424-455); then `run_count`, and per run `palette_index`,
// `length` (RLE over the x-fastest `voxel_index` order, the DTO's column-length + id scheme
// adapted from columns to a 32³ block). Finally, 128 footprint bytes: bit `z * 32 + x`,
// least-significant bit first, bytes in increasing bit order. There is no length prefix.
//
// Mapping blob: `IDMAP` `serialize` (lines 240-265): `i32` entry count, then per entry a `u16`
// length followed by the string bytes (`writeUTF`, read back at line 293), the string being
// `biome + "_DH-BSW_" + block` (`BlockBiomeWrapperPair.serialize`, line 156; separator
// `BLOCK_STATE_SEPARATOR_STRING`, `IDMAP` line 66). Entry `i` names palette entry `i` (DH: the
// list index is the datapoint id, `IDMAP` line 72). An empty half means "no name known".

/// Leading magic of a compressed octree record.
pub const MAGIC_V3: &[u8; 8] = b"DHLOD003";
/// Version 4 adds a fixed column footprint to the voxel data blob.
pub const OCTREE_FORMAT_VERSION: u8 = 4;
/// Default compressor: `Config.java` line 1688 sets `dataCompression` to `Z_STD_BLOCK` (4).
pub const DEFAULT_OCTREE_COMPRESSION: DhV2CompressionMode = DhV2CompressionMode::ZstdBlock;

const OCTREE_HEADER_BYTES: usize = 8 + 1 + 1 + 1 + 12 + 8;
/// Preserve the existing decompressed mapping cap; only the data blob gains a field.
const MAX_OCTREE_MAPPING_BYTES: usize = MAX_DECOMPRESSED_SECTION_BYTES;
const MAX_OCTREE_DATA_BYTES: usize = MAX_DECOMPRESSED_SECTION_BYTES + ColumnFootprint::BYTE_LEN;
/// LZ4 frame block size: `DhDataOutputStream` line 78 uses `BLOCKSIZE.SIZE_64KB`.
const LZ4_BLOCK_BYTES: usize = 64 * 1024;

/// Block and biome name tables carried beside a section (`IDMAP` strings, keyed by the ids the
/// section's [`Voxel`]s hold).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OctreeMapping {
    pub blocks: BTreeMap<u32, String>,
    pub biomes: BTreeMap<u16, String>,
}

/// Why an octree record was refused (decode) or could not be produced (encode).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OctreeCodecError {
    BadMagic,
    Truncated,
    ChecksumMismatch {
        stored: u32,
        computed: u32,
    },
    UnsupportedFormatVersion(u8),
    /// A compression id that `EDhApiDataCompressionMode` does not define (`MODE` lines 98-110).
    UnknownCompression(u8),
    /// A defined mode that cannot be written: `ZstdStream` (DH refuses it too, `STREAM`
    /// `DhDataOutputStream` lines 99-101) and `Lzma2` (no encoder in the dependency set).
    EncodeUnsupported(DhV2CompressionMode),
    Decompress(String),
    Oversized(&'static str),
    Corrupt(&'static str),
    Octree(OctreeError),
}

impl std::fmt::Display for OctreeCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a DHLOD003 record"),
            Self::Truncated => write!(f, "DHLOD003 record truncated"),
            Self::ChecksumMismatch { stored, computed } => {
                write!(
                    f,
                    "DHLOD003 crc32 mismatch (stored {stored:#x}, computed {computed:#x})"
                )
            }
            Self::UnsupportedFormatVersion(v) => write!(f, "unsupported DHLOD003 version {v}"),
            Self::UnknownCompression(v) => write!(f, "unknown compression mode {v}"),
            Self::EncodeUnsupported(m) => write!(f, "cannot encode with {m:?}"),
            Self::Decompress(m) => write!(f, "DHLOD003 decompression failed: {m}"),
            Self::Oversized(what) => write!(f, "DHLOD003 {what} exceeds its bound"),
            Self::Corrupt(what) => write!(f, "corrupt DHLOD003 record: {what}"),
            Self::Octree(e) => write!(f, "invalid octree section: {e:?}"),
        }
    }
}

impl std::error::Error for OctreeCodecError {}

/// `VARINT` `writeVarint` (lines 33-46): 7 bits per byte, low group first, high bit = more.
fn write_varint(out: &mut Vec<u8>, mut value: u32) {
    while value >= 128 {
        out.push((value as u8) | 128);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Bounds-checked cursor over a record or blob.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], OctreeCodecError> {
        let end = self.pos.checked_add(n).ok_or(OctreeCodecError::Truncated)?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or(OctreeCodecError::Truncated)?;
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, OctreeCodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, OctreeCodecError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, OctreeCodecError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32, OctreeCodecError> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64, OctreeCodecError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_be_bytes(a))
    }

    /// `VARINT` `readVarint` (lines 48-65). Stricter than Java on two points: a fifth byte
    /// that would overflow a non-negative `i32` is refused (Java silently wraps, while
    /// `writeVarint` line 35 only ever writes non-negative values).
    fn varint(&mut self) -> Result<u32, OctreeCodecError> {
        let mut value = 0u32;
        let mut shift = 0u32;
        loop {
            if shift >= 32 {
                return Err(OctreeCodecError::Corrupt("varint too long"));
            }
            let b = self.u8()?;
            let group = (b & 127) as u32;
            if shift == 28 && group > 0x07 {
                return Err(OctreeCodecError::Corrupt("varint overflows i32"));
            }
            value |= group << shift;
            shift += 7;
            if b & 128 == 0 {
                return Ok(value);
            }
        }
    }

    fn finished(&self) -> bool {
        self.pos == self.buf.len()
    }
}

/// xxHash32 of fewer than 16 bytes (all the LZ4 frame descriptor checksum needs).
fn xxh32_short(data: &[u8], seed: u32) -> u32 {
    const P1: u32 = 2_654_435_761;
    const P2: u32 = 2_246_822_519;
    const P3: u32 = 3_266_489_917;
    const P4: u32 = 668_265_263;
    const P5: u32 = 374_761_393;
    debug_assert!(data.len() < 16);
    let mut h = seed.wrapping_add(P5).wrapping_add(data.len() as u32);
    let (lanes, remainder) = data.as_chunks::<4>();
    for c in lanes {
        let lane = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        h = h
            .wrapping_add(lane.wrapping_mul(P3))
            .rotate_left(17)
            .wrapping_mul(P4);
    }
    for &b in remainder {
        h = h
            .wrapping_add((b as u32).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
    }
    h ^= h >> 15;
    h = h.wrapping_mul(P2);
    h ^= h >> 13;
    h = h.wrapping_mul(P3);
    h ^ (h >> 16)
}

/// LZ4 frame as Java's `LZ4FrameOutputStream` writes it for `DhDataOutputStream` lines 77-85:
/// 64 KiB blocks, `BLOCK_INDEPENDENCE` only (no block or content checksum, no content size).
fn lz4_frame_compress(raw: &[u8]) -> Vec<u8> {
    const FLG: u8 = 0x60; // version 01, block independence
    const BD: u8 = 0x40; // 64 KiB max block
    let mut out = Vec::with_capacity(raw.len() / 2 + 16);
    out.extend_from_slice(&0x184D_2204u32.to_le_bytes());
    out.push(FLG);
    out.push(BD);
    out.push((xxh32_short(&[FLG, BD], 0) >> 8) as u8);
    for chunk in raw.chunks(LZ4_BLOCK_BYTES) {
        let packed = lz4_flex::block::compress(chunk);
        if packed.len() < chunk.len() {
            out.extend_from_slice(&(packed.len() as u32).to_le_bytes());
            out.extend_from_slice(&packed);
        } else {
            out.extend_from_slice(&(chunk.len() as u32 | 0x8000_0000).to_le_bytes());
            out.extend_from_slice(chunk);
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

fn compress_blob(raw: &[u8], mode: DhV2CompressionMode) -> Result<Vec<u8>, OctreeCodecError> {
    match mode {
        DhV2CompressionMode::Uncompressed => Ok(raw.to_vec()),
        DhV2CompressionMode::Lz4 => Ok(lz4_frame_compress(raw)),
        DhV2CompressionMode::ZstdBlock => Ok(ruzstd::encoding::compress_to_vec(
            raw,
            ruzstd::encoding::CompressionLevel::Fastest,
        )),
        DhV2CompressionMode::ZstdStream | DhV2CompressionMode::Lzma2 => {
            Err(OctreeCodecError::EncodeUnsupported(mode))
        }
    }
}

fn decompress_octree_blob(
    blob: &[u8],
    mode: DhV2CompressionMode,
    limit: usize,
) -> Result<Vec<u8>, OctreeCodecError> {
    dh_v2::decompress_blob_with_limit(blob, mode, limit).map_err(|e| match e {
        dh_v2::DhV2Error::InvalidLength(_) => OctreeCodecError::Oversized("decompressed blob"),
        _ => OctreeCodecError::Decompress(e.to_string()),
    })
}

/// Encode `section` plus its name tables as one `DHLOD003` record using `mode`.
///
/// Voxels are canonicalised to the `Mapper` field layout (block 20 bits, biome 9 bits, light
/// byte); bits outside those fields are not stored.
pub fn encode_octree_section(
    section: &OctreeSection,
    mapping: &OctreeMapping,
    mode: DhV2CompressionMode,
) -> Result<Vec<u8>, OctreeCodecError> {
    encode_octree_record(&section.into(), mapping, mode)
}

fn encode_octree_record(
    section: &OctreeRecord,
    mapping: &OctreeMapping,
    mode: DhV2CompressionMode,
) -> Result<Vec<u8>, OctreeCodecError> {
    let rle = &section.rle;
    let mut data = Vec::with_capacity(
        64 + rle.palette.len() * 4 + rle.runs.len() * 4 + ColumnFootprint::BYTE_LEN,
    );
    write_varint(&mut data, rle.palette.len() as u32);
    for v in &rle.palette {
        let light = v.light();
        write_varint(&mut data, (v.block_id() << 1) | u32::from(light != 0));
        write_varint(&mut data, u32::from(v.biome_id()));
        if light != 0 {
            data.push(light);
        }
    }
    write_varint(&mut data, rle.runs.len() as u32);
    for r in &rle.runs {
        write_varint(&mut data, u32::from(r.palette_index));
        write_varint(&mut data, u32::from(r.length));
    }
    data.extend_from_slice(section.footprint.as_bytes());

    let mut names = Vec::with_capacity(4 + rle.palette.len() * 24);
    names.extend_from_slice(&(rle.palette.len() as i32).to_be_bytes());
    for v in &rle.palette {
        let biome = mapping.biomes.get(&v.biome_id()).map_or("", String::as_str);
        let block = mapping.blocks.get(&v.block_id()).map_or("", String::as_str);
        if biome.contains(dh_v2::BLOCK_STATE_SEPARATOR) {
            return Err(OctreeCodecError::Corrupt("biome name contains separator"));
        }
        let entry = format!("{biome}{}{block}", dh_v2::BLOCK_STATE_SEPARATOR);
        let len = u16::try_from(entry.len())
            .map_err(|_| OctreeCodecError::Oversized("mapping string"))?;
        names.extend_from_slice(&len.to_be_bytes());
        names.extend_from_slice(entry.as_bytes());
    }

    if data.len() > MAX_OCTREE_DATA_BYTES || names.len() > MAX_OCTREE_MAPPING_BYTES {
        return Err(OctreeCodecError::Oversized("blob"));
    }
    let data = compress_blob(&data, mode)?;
    let names = compress_blob(&names, mode)?;

    let total = OCTREE_HEADER_BYTES + 4 + data.len() + 4 + names.len() + 4;
    if total > MAX_COMPRESSED_RECORD_BYTES {
        return Err(OctreeCodecError::Oversized("record"));
    }
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC_V3);
    out.push(OCTREE_FORMAT_VERSION);
    out.push(mode as u8);
    out.push(rle.pos.detail);
    out.extend_from_slice(&rle.pos.x.to_be_bytes());
    out.extend_from_slice(&rle.pos.y.to_be_bytes());
    out.extend_from_slice(&rle.pos.z.to_be_bytes());
    out.extend_from_slice(&section.revision.to_be_bytes());
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(&data);
    out.extend_from_slice(&(names.len() as u32).to_be_bytes());
    out.extend_from_slice(&names);
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_be_bytes());
    Ok(out)
}

/// CRC32 stored in the trailer of a `DHLOD003` record, if the record is long enough to hold one.
pub fn octree_record_crc(record: &[u8]) -> Option<u32> {
    let tail = record.get(record.len().checked_sub(4)?..)?;
    Some(u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]))
}

/// Decode a `DHLOD003` record. The CRC32 is verified before anything is decompressed; every
/// length, id and run is bounds-checked, and malformed input is an `Err`, never a panic.
pub fn decode_octree_section(
    record: &[u8],
) -> Result<(OctreeSection, OctreeMapping), OctreeCodecError> {
    if record.len() < 8 || &record[..8] != MAGIC_V3 {
        return Err(OctreeCodecError::BadMagic);
    }
    if record.len() > MAX_COMPRESSED_RECORD_BYTES {
        return Err(OctreeCodecError::Oversized("record"));
    }
    if record.len() < OCTREE_HEADER_BYTES + 4 + 4 + 4 {
        return Err(OctreeCodecError::Truncated);
    }
    let (body, tail) = record.split_at(record.len() - 4);
    let stored = u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]);
    let computed = crc32fast::hash(body);
    if stored != computed {
        return Err(OctreeCodecError::ChecksumMismatch { stored, computed });
    }

    let mut cur = Cursor::new(&body[8..]);
    let version = cur.u8()?;
    if version != OCTREE_FORMAT_VERSION {
        return Err(OctreeCodecError::UnsupportedFormatVersion(version));
    }
    let mode_byte = cur.u8()?;
    let mode = DhV2CompressionMode::from_byte(mode_byte)
        .map_err(|_| OctreeCodecError::UnknownCompression(mode_byte))?;
    let detail = cur.u8()?;
    if detail > MAX_OCTREE_DETAIL {
        return Err(OctreeCodecError::Corrupt("detail out of range"));
    }
    let pos = OctreeSectionPos::new(detail, cur.i32()?, cur.i32()?, cur.i32()?);
    let revision = cur.u64()?;
    let data_len = cur.u32()? as usize;
    let data_blob = cur.take(data_len)?;
    let names_len = cur.u32()? as usize;
    let names_blob = cur.take(names_len)?;
    if !cur.finished() {
        return Err(OctreeCodecError::Corrupt("trailing bytes"));
    }

    // Data blob.
    let raw = decompress_octree_blob(data_blob, mode, MAX_OCTREE_DATA_BYTES)?;
    let mut d = Cursor::new(&raw);
    let palette_len = d.varint()? as usize;
    if palette_len == 0 {
        return Err(OctreeCodecError::Octree(OctreeError::EmptyPalette));
    }
    if palette_len > SECTION_VOLUME {
        return Err(OctreeCodecError::Oversized("palette"));
    }
    let mut palette = Vec::with_capacity(palette_len);
    for _ in 0..palette_len {
        let id_flags = d.varint()?;
        let block = id_flags >> 1;
        let biome = d.varint()?;
        if block > BLOCK_ID_MAX || biome > BIOME_ID_MAX {
            return Err(OctreeCodecError::Corrupt("voxel id out of range"));
        }
        let light = if id_flags & 1 == 1 {
            let l = d.u8()?;
            if l == 0 {
                return Err(OctreeCodecError::Corrupt("lit flag with zero light"));
            }
            l
        } else {
            0
        };
        palette.push(Voxel::new(block, biome as u16, light & 0xF, light >> 4));
    }
    let run_count = d.varint()? as usize;
    if run_count == 0 || run_count > SECTION_VOLUME {
        return Err(OctreeCodecError::Oversized("run list"));
    }
    let mut runs = Vec::with_capacity(run_count);
    let mut total = 0usize;
    for _ in 0..run_count {
        let palette_index = d.varint()?;
        let length = d.varint()?;
        if palette_index as usize >= palette_len {
            return Err(OctreeCodecError::Octree(
                OctreeError::PaletteIndexOutOfRange,
            ));
        }
        if length == 0 || length as usize > SECTION_VOLUME {
            return Err(OctreeCodecError::Corrupt("run length out of range"));
        }
        total += length as usize;
        if total > SECTION_VOLUME {
            return Err(OctreeCodecError::Octree(OctreeError::BadRunTotal));
        }
        runs.push(Run {
            palette_index: palette_index as u16,
            length: length as u16,
        });
    }
    let mut footprint = [0; ColumnFootprint::BYTE_LEN];
    footprint.copy_from_slice(d.take(ColumnFootprint::BYTE_LEN)?);
    if !d.finished() {
        return Err(OctreeCodecError::Corrupt("trailing data bytes"));
    }
    let mut section = OctreeSection::from_rle(&RleSection {
        pos,
        palette: palette.clone(),
        runs,
    })
    .map_err(OctreeCodecError::Octree)?;
    section.revision = revision;
    section.footprint = ColumnFootprint::from_bytes(footprint);

    // Mapping blob (`IDMAP` deserialize, lines 268-339).
    let raw = decompress_octree_blob(names_blob, mode, MAX_OCTREE_MAPPING_BYTES)?;
    let mut m = Cursor::new(&raw);
    let count = m.i32()?;
    if count < 0 {
        return Err(OctreeCodecError::Corrupt("negative mapping count"));
    }
    if count as usize != palette_len {
        return Err(OctreeCodecError::Corrupt(
            "mapping count differs from palette",
        ));
    }
    let mut mapping = OctreeMapping::default();
    for voxel in &palette {
        let len = m.u16()? as usize;
        let text = std::str::from_utf8(m.take(len)?)
            .map_err(|_| OctreeCodecError::Corrupt("mapping string is not utf-8"))?;
        let (biome, block) = text
            .split_once(dh_v2::BLOCK_STATE_SEPARATOR)
            .ok_or(OctreeCodecError::Corrupt("mapping string lacks separator"))?;
        if !block.is_empty() {
            let prev = mapping.blocks.insert(voxel.block_id(), block.to_owned());
            if prev.is_some_and(|p| p != block) {
                return Err(OctreeCodecError::Corrupt("conflicting block names"));
            }
        }
        if !biome.is_empty() {
            let prev = mapping.biomes.insert(voxel.biome_id(), biome.to_owned());
            if prev.is_some_and(|p| p != biome) {
                return Err(OctreeCodecError::Corrupt("conflicting biome names"));
            }
        }
    }
    if !m.finished() {
        return Err(OctreeCodecError::Corrupt("trailing mapping bytes"));
    }
    Ok((section, mapping))
}

/// A stored section of either on-disk generation.
#[derive(Debug)]
pub enum StoredSection {
    /// `DHLOD001` / `DHLOD002` column section (`decode_section`).
    Column(LodSection),
    /// `DHLOD003` 32³ octree section.
    Octree(OctreeSection, OctreeMapping),
}

/// Decode a record by its magic: `DHLOD003` octree records, or the older `DHLOD001/2` column
/// records through [`decode_section`].
pub fn decode_stored_section(bytes: &[u8]) -> Option<StoredSection> {
    if bytes.starts_with(MAGIC_V3) {
        let (s, m) = decode_octree_section(bytes).ok()?;
        Some(StoredSection::Octree(s, m))
    } else {
        decode_section(bytes).map(StoredSection::Column)
    }
}

/// Octree region shape: `8 × 4 × 8` sections (x, y, z) = 256 slots, the same slot count as the
/// column regions so the header pages are shared.
const OCT_REGION_SHIFT_X: i32 = 3;
const OCT_REGION_SHIFT_Y: i32 = 2;
const OCT_REGION_SHIFT_Z: i32 = 3;

/// Region coordinates `(rx, ry, rz)` and slot index `0..256` for an octree section.
pub const fn octree_region_slot(pos: OctreeSectionPos) -> (i32, i32, i32, usize) {
    let lx = (pos.x & ((1 << OCT_REGION_SHIFT_X) - 1)) as usize;
    let ly = (pos.y & ((1 << OCT_REGION_SHIFT_Y) - 1)) as usize;
    let lz = (pos.z & ((1 << OCT_REGION_SHIFT_Z) - 1)) as usize;
    (
        pos.x >> OCT_REGION_SHIFT_X,
        pos.y >> OCT_REGION_SHIFT_Y,
        pos.z >> OCT_REGION_SHIFT_Z,
        (ly << (OCT_REGION_SHIFT_X + OCT_REGION_SHIFT_Z)) | (lz << OCT_REGION_SHIFT_X) | lx,
    )
}

/// Path of the `.dhoct` region file owning `pos`:
/// `<world>/distant_sunrises/<dim>/o<detail>/r.<rx>.<ry>.<rz>.dhoct`.
pub fn octree_region_path(world_dir: &Path, dimension: &str, pos: OctreeSectionPos) -> PathBuf {
    let (rx, ry, rz, _) = octree_region_slot(pos);
    world_dir
        .join("distant_sunrises")
        .join(sanitize_dimension(dimension))
        .join(format!("o{}", pos.detail))
        .join(format!("r.{rx}.{ry}.{rz}.dhoct"))
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct SlotEntry {
    offset: u64,
    len: u32,
    crc32: u32,
}

fn header_page_crc(page: &[u8; HEADER_PAGE_BYTES]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&page[0..16]);
    hasher.update(&page[20..HEADER_PAGE_BYTES]);
    hasher.finalize()
}

fn encode_header_page(
    generation: u64,
    slots: &[SlotEntry; REGION_SLOTS],
) -> [u8; HEADER_PAGE_BYTES] {
    let mut page = [0u8; HEADER_PAGE_BYTES];
    page[0..8].copy_from_slice(REGION_MAGIC);
    page[8..16].copy_from_slice(&generation.to_le_bytes());
    for (i, slot) in slots.iter().enumerate() {
        let base = HEADER_META_BYTES + i * SLOT_ENTRY_BYTES;
        page[base..base + 8].copy_from_slice(&slot.offset.to_le_bytes());
        page[base + 8..base + 12].copy_from_slice(&slot.len.to_le_bytes());
        page[base + 12..base + 16].copy_from_slice(&slot.crc32.to_le_bytes());
    }
    let crc = header_page_crc(&page);
    page[16..20].copy_from_slice(&crc.to_le_bytes());
    page
}

fn parse_header_page(
    page: &[u8; HEADER_PAGE_BYTES],
    file_len: u64,
) -> Option<(u64, [SlotEntry; REGION_SLOTS], u64)> {
    if &page[0..8] != REGION_MAGIC {
        return None;
    }
    let generation = u64::from_le_bytes([
        page[8], page[9], page[10], page[11], page[12], page[13], page[14], page[15],
    ]);
    if generation == 0 {
        return None;
    }
    let stored_crc = u32::from_le_bytes([page[16], page[17], page[18], page[19]]);
    if header_page_crc(page) != stored_crc {
        return None;
    }

    let mut slots = [SlotEntry::default(); REGION_SLOTS];
    let mut intervals: Vec<(u64, u64)> = Vec::new();
    for (i, slot) in slots.iter_mut().enumerate() {
        let base = HEADER_META_BYTES + i * SLOT_ENTRY_BYTES;
        let offset = u64::from_le_bytes([
            page[base],
            page[base + 1],
            page[base + 2],
            page[base + 3],
            page[base + 4],
            page[base + 5],
            page[base + 6],
            page[base + 7],
        ]);
        let len = u32::from_le_bytes([
            page[base + 8],
            page[base + 9],
            page[base + 10],
            page[base + 11],
        ]);
        let crc32 = u32::from_le_bytes([
            page[base + 12],
            page[base + 13],
            page[base + 14],
            page[base + 15],
        ]);
        if len == 0 {
            continue;
        }
        if (len as usize) > MAX_COMPRESSED_RECORD_BYTES || offset < REGION_HEADER_BYTES as u64 {
            continue;
        }
        let Some(end) = offset.checked_add(len as u64) else {
            continue;
        };
        if end <= file_len {
            *slot = SlotEntry { offset, len, crc32 };
            intervals.push((offset, end));
        }
    }

    // Compute true non-overlapping live bytes so duplicate/corrupt ranges cannot skew compaction.
    intervals.sort_unstable();
    let mut live_bytes = 0u64;
    let mut cur_end = 0u64;
    for (start, end) in intervals {
        let s = start.max(cur_end);
        if end > s {
            live_bytes += end - s;
            cur_end = end;
        }
    }

    Some((generation, slots, live_bytes))
}

fn unique_tmp_path(path: &Path, tag: &str) -> PathBuf {
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_else(|| "region.dhreg".into());
    name.push(format!(".{tag}.{pid}.{seq}.tmp"));
    path.with_file_name(name)
}

fn sync_parent_dir(path: &Path) {
    #[cfg(unix)]
    if let Some(parent) = path.parent()
        && let Ok(dir) = File::open(parent)
    {
        let _ = dir.sync_all();
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// Atomically initialize a fresh, valid `DHREG002` region file via temp-file + `sync_all` + `rename`.
fn create_empty_region_atomic(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = unique_tmp_path(path, "init");
    let slots = [SlotEntry::default(); REGION_SLOTS];
    let page0 = encode_header_page(1, &slots);
    let page1 = [0u8; HEADER_PAGE_BYTES];

    let write_res = (|| -> io::Result<()> {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(&page0)?;
        f.write_all(&page1)?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    sync_parent_dir(path);
    Ok(())
}

struct OpenRegion {
    path: PathBuf,
    file: Option<File>,
    writable: bool,
    generation: u64,
    slots: [SlotEntry; REGION_SLOTS],
    file_len: u64,
    live_bytes: u64,
    last_used: u64,
}

impl OpenRegion {
    fn open(
        path: &Path,
        writable: bool,
        tick: u64,
        quarantine: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<Option<Self>> {
        let open_res = OpenOptions::new().read(true).write(writable).open(path);
        let mut file = match open_res {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                if !writable {
                    return Ok(None);
                }
                create_empty_region_atomic(path)?;
                OpenOptions::new().read(true).write(true).open(path)?
            }
            Err(e) => return Err(e),
        };

        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::Error::other("region path is not a regular file"));
        }
        let file_len = metadata.len();
        let parsed = if file_len >= REGION_HEADER_BYTES as u64 {
            let mut pages = [[0u8; HEADER_PAGE_BYTES]; 2];
            file.seek(SeekFrom::Start(0))?;
            let mut parsed = [None, None];
            for (page, out) in pages.iter_mut().zip(&mut parsed) {
                match file.read_exact(page) {
                    Ok(()) => *out = parse_header_page(page, file_len),
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e),
                }
            }
            let [a, b] = parsed;
            match (a, b) {
                (Some(pa), Some(pb)) => Some(if pb.0 > pa.0 { pb } else { pa }),
                (Some(pa), None) => Some(pa),
                (None, Some(pb)) => Some(pb),
                (None, None) => None,
            }
        } else {
            None
        };

        if let Some((generation, slots, live_bytes)) = parsed {
            return Ok(Some(Self {
                path: path.to_path_buf(),
                file: Some(file),
                writable,
                generation,
                slots,
                file_len,
                live_bytes,
                last_used: tick,
            }));
        }

        // Header is missing, truncated, or both header pages are corrupt.
        if !writable {
            return Ok(None);
        }

        // Reinitialize only after quarantine preserves the original at its new path.
        // If renaming fails, the cache refuses writes to this path until the disk session resets.
        drop(file);
        quarantine()?;
        create_empty_region_atomic(path)?;
        let fresh = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Some(Self {
            path: path.to_path_buf(),
            file: Some(fresh),
            writable: true,
            generation: 1,
            slots: [SlotEntry::default(); REGION_SLOTS],
            file_len: REGION_HEADER_BYTES as u64,
            live_bytes: 0,
            last_used: tick,
        }))
    }

    /// Raw bytes from the slot validated when this region opened.
    fn read_slot_bytes(&mut self, slot_idx: usize) -> io::Result<Option<(Vec<u8>, u32)>> {
        let entry = self.slots[slot_idx];
        if entry.len == 0 {
            return Ok(None);
        }
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("region handle closed"))?;
        let mut buf = vec![0u8; entry.len as usize];
        file.seek(SeekFrom::Start(entry.offset))?;
        match file.read_exact(&mut buf) {
            Ok(()) => Ok(Some((buf, entry.crc32))),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn read_slot(&mut self, slot_idx: usize, expected_pos: SectionPos) -> Option<LodSection> {
        let (buf, crc) = self.read_slot_bytes(slot_idx).ok()??;
        let sec = decompress_section_record(&buf, Some(crc))?;
        if sec.pos != expected_pos {
            return None;
        }
        Some(sec)
    }

    /// Copy-on-write append of `record` to EOF + `sync_data`, then commit the next generation to
    /// the alternating checksummed header page + `sync_data`. Compacts atomically when dead space
    /// exceeds 2× live data (and > 64 KiB).
    fn write_slot(&mut self, slot_idx: usize, record: &[u8], record_crc: u32) -> io::Result<()> {
        if !self.writable {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "region handle is read-only",
            ));
        }
        let new_len = u32::try_from(record.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "record exceeds u32"))?;
        let new_offset = self.file_len;

        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("region handle closed"))?;

        // 1. Append payload at EOF and sync data blocks before referencing it in any header page.
        file.seek(SeekFrom::Start(new_offset))?;
        file.write_all(record)?;
        file.sync_data()?;

        // 2. Build next-generation header page and write it to the alternating slot (`gen & 1`).
        let prev = self.slots[slot_idx];
        let next_gen = self.generation.saturating_add(1).max(1);
        let mut next_slots = self.slots;
        next_slots[slot_idx] = SlotEntry {
            offset: new_offset,
            len: new_len,
            crc32: record_crc,
        };
        let page = encode_header_page(next_gen, &next_slots);
        let page_offset = if (next_gen & 1) == 1 {
            0u64
        } else {
            HEADER_PAGE_BYTES as u64
        };
        file.seek(SeekFrom::Start(page_offset))?;
        file.write_all(&page)?;
        file.sync_data()?;

        // 3. Commit in-memory state only after both durable syncs succeeded.
        self.generation = next_gen;
        self.slots = next_slots;
        self.file_len = new_offset.saturating_add(new_len as u64);
        self.live_bytes = self
            .live_bytes
            .saturating_sub(prev.len as u64)
            .saturating_add(new_len as u64);

        // 4. Compact atomically when stale overwritten space exceeds 2× live data (> 64 KiB).
        let data_bytes = self.file_len.saturating_sub(REGION_HEADER_BYTES as u64);
        let dead_bytes = data_bytes.saturating_sub(self.live_bytes);
        if dead_bytes > 64 * 1024 && dead_bytes > self.live_bytes.saturating_mul(2) {
            self.compact()?;
        }
        Ok(())
    }

    /// Stream live records directly into a unique temp file, `sync_all`, close the live handle
    /// (Windows-safe), atomically rename into place, sync parent directory, and reopen.
    fn compact(&mut self) -> io::Result<()> {
        let tmp_path = unique_tmp_path(&self.path, "compact");
        let mut new_slots = [SlotEntry::default(); REGION_SLOTS];
        let mut cur_offset = REGION_HEADER_BYTES as u64;
        let mut new_live = 0u64;
        let next_gen = self.generation.saturating_add(1).max(1);

        let stream_res = (|| -> io::Result<()> {
            let src = self
                .file
                .as_mut()
                .ok_or_else(|| io::Error::other("region handle closed"))?;
            let mut tmp = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp_path)?;

            // Reserve double-header space; stream each live record directly to tmp (no full-file RAM buffer).
            tmp.seek(SeekFrom::Start(REGION_HEADER_BYTES as u64))?;
            let mut copy_buf = Vec::new();
            for (idx, entry) in self.slots.iter().enumerate() {
                if entry.len == 0 {
                    continue;
                }
                let len = entry.len as usize;
                if len > MAX_COMPRESSED_RECORD_BYTES {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "slot length exceeds bound during compaction",
                    ));
                }
                copy_buf.resize(len, 0);
                src.seek(SeekFrom::Start(entry.offset))?;
                // Propagate I/O errors — never silently drop live slots on short/failed reads.
                src.read_exact(&mut copy_buf)?;
                tmp.write_all(&copy_buf)?;
                new_slots[idx] = SlotEntry {
                    offset: cur_offset,
                    len: entry.len,
                    crc32: entry.crc32,
                };
                cur_offset = cur_offset.saturating_add(entry.len as u64);
                new_live = new_live.saturating_add(entry.len as u64);
            }

            let page = encode_header_page(next_gen, &new_slots);
            let empty_page = [0u8; HEADER_PAGE_BYTES];
            tmp.seek(SeekFrom::Start(0))?;
            if (next_gen & 1) == 1 {
                tmp.write_all(&page)?;
                tmp.write_all(&empty_page)?;
            } else {
                tmp.write_all(&empty_page)?;
                tmp.write_all(&page)?;
            }
            tmp.sync_all()?;
            Ok(())
        })();

        if let Err(e) = stream_res {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }

        // Close our live handle BEFORE renaming so Windows never fails with a sharing violation
        // and no stale unlinked inode can linger if reopen fails.
        self.file = None;
        if let Err(e) = fs::rename(&tmp_path, &self.path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }
        sync_parent_dir(&self.path);

        self.file = Some(OpenOptions::new().read(true).write(true).open(&self.path)?);
        self.writable = true;
        self.generation = next_gen;
        self.slots = new_slots;
        self.file_len = cur_offset;
        self.live_bytes = new_live;
        Ok(())
    }
}

/// A save refusal whose cause was already logged at the failed quarantine rename.
#[derive(Debug)]
struct RegionWriteRefused;

impl RegionWriteRefused {
    fn matches(error: &io::Error) -> bool {
        error.get_ref().is_some_and(|cause| cause.is::<Self>())
    }
}

impl std::fmt::Display for RegionWriteRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("region is read-only for this disk session after quarantine failed")
    }
}

impl std::error::Error for RegionWriteRefused {}

impl From<RegionWriteRefused> for io::Error {
    fn from(error: RegionWriteRefused) -> Self {
        Self::new(io::ErrorKind::PermissionDenied, error)
    }
}

#[cfg(test)]
type QuarantineRename = Box<dyn FnMut(&Path, &Path) -> io::Result<()> + Send>;

/// Region file handle + index cache owned by the background IO worker thread.
#[derive(Default)]
pub struct RegionCache {
    regions: HashMap<PathBuf, OpenRegion>,
    // Keep failed paths outside the handle LRU: only a disk session reset unblocks them.
    quarantine_failed: HashSet<PathBuf>,
    tick: u64,
    #[cfg(test)]
    quarantine_rename: Option<QuarantineRename>,
}

impl RegionCache {
    fn quarantine(&mut self, path: &Path) -> io::Result<()> {
        let corrupt = unique_tmp_path(path, "corrupt");
        #[cfg(test)]
        let result = self.quarantine_rename.as_mut().map_or_else(
            || fs::rename(path, &corrupt),
            |rename| rename(path, &corrupt),
        );
        #[cfg(not(test))]
        let result = fs::rename(path, &corrupt);

        if let Err(e) = result {
            self.quarantine_failed.insert(path.to_path_buf());
            eprintln!(
                "DH: failed to quarantine corrupt region {}: {e}; leaving it read-only for this disk session",
                path.display()
            );
            return Err(RegionWriteRefused.into());
        }
        eprintln!(
            "DH: quarantined corrupt region {} -> {}",
            path.display(),
            corrupt.display()
        );
        Ok(())
    }

    // Refuse before encoding or treating an empty octree slot as a no-op.
    fn check_write_allowed(&self, path: &Path) -> io::Result<()> {
        if self.quarantine_failed.contains(path) {
            return Err(RegionWriteRefused.into());
        }
        Ok(())
    }

    // Writable callers must have passed check_write_allowed for this path before calling.
    fn get_region(&mut self, path: &Path, writable: bool) -> io::Result<Option<&mut OpenRegion>> {
        if !writable && self.quarantine_failed.contains(path) {
            return Ok(None);
        }
        self.tick = self.tick.wrapping_add(1);
        let now = self.tick;

        // Upgrade a cached read-only handle when a write arrives.
        if writable
            && self
                .regions
                .get(path)
                .is_some_and(|r| !r.writable || r.file.is_none())
        {
            self.regions.remove(path);
        }

        if !self.regions.contains_key(path) {
            // Open BEFORE evicting so probing nonexistent files never evicts hot handles.
            let Some(reg) = OpenRegion::open(path, writable, now, || self.quarantine(path))? else {
                return Ok(None);
            };
            if self.regions.len() >= MAX_CACHED_REGIONS
                && let Some(evict_key) = self
                    .regions
                    .iter()
                    .min_by_key(|(_, r)| r.last_used)
                    .map(|(k, _)| k.clone())
            {
                self.regions.remove(&evict_key);
            }
            self.regions.insert(path.to_path_buf(), reg);
        }

        if let Some(reg) = self.regions.get_mut(path) {
            reg.last_used = now;
            Ok(Some(reg))
        } else {
            Ok(None)
        }
    }

    /// Store a column section, or return `PermissionDenied` if quarantine has failed for the
    /// region this disk session.
    pub fn save_section(
        &mut self,
        world_dir: &Path,
        dimension: &str,
        section: &LodSection,
    ) -> io::Result<()> {
        let path = region_path(world_dir, dimension, section.pos);
        self.check_write_allowed(&path)?;
        let (_, _, slot_idx) = section_region_slot(section.pos);
        let (record, crc) = compress_section_record(section).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "section record exceeds size limit",
            )
        })?;
        let Some(reg) = self.get_region(&path, true)? else {
            return Err(io::Error::other("failed to open region file"));
        };
        let res = reg.write_slot(slot_idx, &record, crc);
        if res.is_err() {
            // Drop any broken/unlinked handle from cache immediately.
            self.regions.remove(&path);
        }
        res
    }

    pub fn load_section(
        &mut self,
        world_dir: &Path,
        dimension: &str,
        pos: SectionPos,
    ) -> Option<LodSection> {
        let path = region_path(world_dir, dimension, pos);
        let (_, _, slot_idx) = section_region_slot(pos);
        if let Ok(Some(reg)) = self.get_region(&path, false)
            && let Some(sec) = reg.read_slot(slot_idx, pos)
        {
            return Some(sec);
        }
        // Fallback: seamlessly read legacy single-section `.dhlod` files if present.
        let legacy = legacy_section_path(world_dir, dimension, pos);
        let bytes = fs::read(legacy).ok()?;
        let sec = decode_section(&bytes)?;
        if sec.pos != pos {
            return None;
        }
        Some(sec)
    }
}

impl RegionCache {
    /// Encode and store one octree section in its `.dhoct` region slot (copy-on-write append,
    /// then header commit, as for column regions). All-air sections without known columns
    /// can clear existing data but do not allocate a new slot. Returns `PermissionDenied`
    /// if quarantine has failed for the region this disk session.
    pub fn save_octree_section(
        &mut self,
        world_dir: &Path,
        dimension: &str,
        section: &OctreeSection,
        mapping: &OctreeMapping,
        mode: DhV2CompressionMode,
    ) -> io::Result<()> {
        self.write_octree(world_dir, dimension, &section.into(), mapping, mode)
    }

    fn write_octree(
        &mut self,
        world_dir: &Path,
        dimension: &str,
        record: &OctreeRecord,
        mapping: &OctreeMapping,
        mode: DhV2CompressionMode,
    ) -> io::Result<()> {
        let pos = record.rle.pos;
        let path = octree_region_path(world_dir, dimension, pos);
        self.check_write_allowed(&path)?;
        let (_, _, _, slot_idx) = octree_region_slot(pos);
        // Probe without creating a region, after queue coalescing and any in-flight write.
        if record.is_untouched_air()
            && !self
                .get_region(&path, false)?
                .is_some_and(|reg| reg.slots[slot_idx].len != 0)
        {
            return Ok(());
        }
        let record = encode_octree_record(record, mapping, mode)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let crc = octree_record_crc(&record)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "record too short"))?;
        let Some(reg) = self.get_region(&path, true)? else {
            return Err(io::Error::other("failed to open region file"));
        };
        let res = reg.write_slot(slot_idx, &record, crc);
        if res.is_err() {
            self.regions.remove(&path);
        }
        res
    }

    /// Load one octree section. A missing or malformed slot is a miss; filesystem errors
    /// propagate so a caller cannot mistake an unread slot for available space.
    pub fn load_octree_section(
        &mut self,
        world_dir: &Path,
        dimension: &str,
        pos: OctreeSectionPos,
    ) -> OctreeRead {
        let path = octree_region_path(world_dir, dimension, pos);
        let (_, _, _, slot_idx) = octree_region_slot(pos);
        let Some(reg) = self.get_region(&path, false)? else {
            return Ok(None);
        };
        let Some((bytes, slot_crc)) = reg.read_slot_bytes(slot_idx)? else {
            return Ok(None);
        };
        if octree_record_crc(&bytes) != Some(slot_crc) {
            return Ok(None);
        }
        Ok(decode_octree_section(&bytes)
            .ok()
            .filter(|(section, _)| section.pos == pos))
    }
}

pub type OctreeRead = io::Result<Option<(OctreeSection, OctreeMapping)>>;

/// Completion (including misses or failures) of one tagged octree probe.
pub struct OctreeLoaded {
    pub world_dir: PathBuf,
    pub dimension: String,
    pub pos: OctreeSectionPos,
    pub request: u64,
    pub result: OctreeRead,
}

pub enum IoResponse {
    Loaded {
        dimension: String,
        section: LodSection,
    },
    Saved,
}

type SaveKey = (PathBuf, String, SectionPos);
type OctreeSaveKey = (PathBuf, String, OctreeSectionPos);

struct OctreeSaveJob {
    section: OctreeRecord,
    mapping: OctreeMapping,
    mode: DhV2CompressionMode,
}

/// Accepted writes that failed since the previous barrier: the first contextual error and the
/// total count. The worker logs each write when it fails, except an attempt refused by a
/// quarantine fence: it counts here but is not logged again, because the original failed rename
/// logged the fence. A barrier's caller therefore reports only the count.
#[derive(Debug)]
pub struct SaveFailure {
    section: String,
    cause: io::Error,
    count: usize,
}

impl std::fmt::Display for SaveFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} ({} failed cache operations)",
            self.section, self.cause, self.count
        )
    }
}

impl std::error::Error for SaveFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

impl SaveFailure {
    /// Failed writes since the previous barrier.
    pub fn count(&self) -> usize {
        self.count
    }

    /// One failed accepted write, logged now unless it is a quarantine-fenced attempt, which
    /// still counts but was already logged once at the failed rename.
    fn logged(section: String, cause: io::Error) -> Self {
        if !RegionWriteRefused::matches(&cause) {
            eprintln!("DH: {section}: {cause}");
        }
        Self {
            section,
            cause,
            count: 1,
        }
    }
}

#[derive(Default)]
struct IoQueueState {
    /// Coalesced pending saves (newest revision per `(world_dir, dimension, pos)` wins).
    pending_saves: HashMap<SaveKey, LodSection>,
    save_order: VecDeque<SaveKey>,
    pending_loads: VecDeque<(PathBuf, String, SectionPos)>,
    /// Coalesced pending octree saves (newest write per `(world_dir, dimension, pos)` wins).
    pending_octree_saves: HashMap<OctreeSaveKey, OctreeSaveJob>,
    octree_save_order: VecDeque<OctreeSaveKey>,
    pending_octree_loads: VecDeque<(OctreeSaveKey, u64)>,
    octree_responses: Vec<OctreeLoaded>,
    in_flight: usize,
    save_failure: Option<SaveFailure>,
    reset_regions: bool,
    shutdown: bool,
    responses: Vec<IoResponse>,
}

impl IoQueueState {
    fn record_failed_save(&mut self, failure: SaveFailure) {
        match &mut self.save_failure {
            Some(first) => first.count = first.count.saturating_add(1),
            None => self.save_failure = Some(failure),
        }
    }

    fn idle(&self) -> bool {
        self.pending_saves.is_empty()
            && self.pending_loads.is_empty()
            && self.pending_octree_saves.is_empty()
            && self.pending_octree_loads.is_empty()
            && self.in_flight == 0
            && !self.reset_regions
    }
}

/// An octree save was not admitted, as distinct from an accepted write that failed.
#[derive(Debug)]
pub(crate) enum SaveRefusal<'a> {
    NoWorker(&'a io::Error),
    Shutdown,
}

impl std::fmt::Display for SaveRefusal<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoWorker(cause) => write!(f, "I/O worker could not start: {cause}"),
            Self::Shutdown => f.write_str("I/O worker is shut down"),
        }
    }
}

struct IoShared {
    state: Mutex<IoQueueState>,
    wake_worker: Condvar,
    wake_flusher: Condvar,
}

/// Asynchronous disk IO worker for loading and saving LOD sections via region-packed `.dhreg`
/// files, with coalesced saves, explicit `flush()`, and clean thread join on `Drop`.
pub struct LodIoWorker {
    shared: Arc<IoShared>,
    handle: Mutex<Option<JoinHandle<()>>>,
    spawn_error: Option<io::Error>,
}

impl LodIoWorker {
    pub fn spawn() -> Self {
        Self::spawn_with_cache(RegionCache::default())
    }

    fn spawn_with_cache(cache: RegionCache) -> Self {
        Self::spawn_with(|shared| {
            thread::Builder::new()
                .name("dh-lod-io".into())
                .spawn(move || run_io_worker(shared, cache))
        })
    }

    fn spawn_with(spawn: impl FnOnce(Arc<IoShared>) -> io::Result<JoinHandle<()>>) -> Self {
        let shared = Arc::new(IoShared {
            state: Mutex::new(IoQueueState::default()),
            wake_worker: Condvar::new(),
            wake_flusher: Condvar::new(),
        });
        let (handle, spawn_error) = match spawn(Arc::clone(&shared)) {
            Ok(h) => (Some(h), None),
            Err(e) => {
                eprintln!("DH: failed to spawn dh-lod-io worker thread: {e}");
                // Nothing can serve work: refuse every request, as after shutdown, so no
                // barrier waits on or reports operations that were never accepted.
                shared.state.lock().unwrap().shutdown = true;
                (None, Some(e))
            }
        };
        Self {
            shared,
            handle: Mutex::new(handle),
            spawn_error,
        }
    }

    pub fn request_save(&self, world_dir: PathBuf, dimension: String, section: LodSection) {
        let mut st = self.shared.state.lock().unwrap();
        if st.shutdown {
            return;
        }
        let key = (world_dir, dimension, section.pos);
        if !st.pending_saves.contains_key(&key) {
            st.save_order.push_back(key.clone());
        }
        st.pending_saves.insert(key, section);
        self.shared.wake_worker.notify_one();
    }

    pub fn request_load(&self, world_dir: PathBuf, dimension: String, pos: SectionPos) {
        let mut st = self.shared.state.lock().unwrap();
        if st.shutdown {
            return;
        }
        st.pending_loads.push_back((world_dir, dimension, pos));
        self.shared.wake_worker.notify_one();
    }

    /// Queue an octree section for a coalesced background save with the default compression
    /// (`DEFAULT_OCTREE_COMPRESSION`). A newer save of the same position replaces a pending one.
    pub fn save_octree_section(
        &self,
        world_dir: PathBuf,
        dimension: String,
        section: OctreeSection,
        mapping: OctreeMapping,
    ) {
        self.save_octree_section_with(
            world_dir,
            dimension,
            section,
            mapping,
            DEFAULT_OCTREE_COMPRESSION,
        );
    }

    /// [`Self::save_octree_section`] with an explicit compression mode.
    pub fn save_octree_section_with(
        &self,
        world_dir: PathBuf,
        dimension: String,
        section: OctreeSection,
        mapping: OctreeMapping,
        mode: DhV2CompressionMode,
    ) {
        // This public queue keeps its fire-and-forget contract; refusal is observed only by
        // callers of `queue_octree_record`.
        let _ = self.queue_octree_record(world_dir, dimension, (&section).into(), mapping, mode);
    }

    /// A worker whose thread could not start: it refuses every request.
    #[cfg(test)]
    pub(crate) fn spawn_unavailable() -> Self {
        Self::spawn_with(|_| Err(io::Error::other("spawn refused")))
    }

    /// Queue already-canonical runs; compression and region writes stay on the worker.
    /// A refusal means the worker did not accept the record.
    pub(crate) fn queue_octree_record(
        &self,
        world_dir: PathBuf,
        dimension: String,
        section: OctreeRecord,
        mapping: OctreeMapping,
        mode: DhV2CompressionMode,
    ) -> Result<(), SaveRefusal<'_>> {
        let mut st = self.shared.state.lock().unwrap();
        if st.shutdown {
            return Err(self
                .spawn_error
                .as_ref()
                .map_or(SaveRefusal::Shutdown, SaveRefusal::NoWorker));
        }
        let key = (world_dir, dimension, section.rle.pos);
        if !st.pending_octree_saves.contains_key(&key) {
            st.octree_save_order.push_back(key.clone());
        }
        st.pending_octree_saves.insert(
            key,
            OctreeSaveJob {
                section,
                mapping,
                mode,
            },
        );
        self.shared.wake_worker.notify_one();
        Ok(())
    }

    /// Queue a tagged background probe; every completion arrives through
    /// [`Self::drain_octree_responses`]. Probes needed by mutations go ahead of the warm sweep.
    pub fn load_octree_section(
        &self,
        world_dir: PathBuf,
        dimension: String,
        pos: OctreeSectionPos,
        request: u64,
        for_write: bool,
    ) {
        let mut st = self.shared.state.lock().unwrap();
        if st.shutdown {
            return;
        }
        let job = ((world_dir, dimension, pos), request);
        if for_write {
            st.pending_octree_loads.push_front(job);
        } else {
            st.pending_octree_loads.push_back(job);
        }
        self.shared.wake_worker.notify_one();
    }

    pub fn drain_octree_responses(&self) -> Vec<OctreeLoaded> {
        let mut st = self.shared.state.lock().unwrap();
        std::mem::take(&mut st.octree_responses)
    }

    /// Drain queued I/O. A successful return means no write since the previous barrier failed.
    pub fn flush(&self) -> Result<(), SaveFailure> {
        let mut st = self.shared.state.lock().unwrap();
        while !st.idle() {
            st = self.shared.wake_flusher.wait(st).unwrap();
        }
        st.save_failure.take().map_or(Ok(()), Err)
    }

    /// Finish accepted I/O before discarding results and closing the old world's region handles.
    /// The lifetime owner must exclude new world work until this barrier returns.
    pub(crate) fn reset_world(&self) {
        let mut st = self.shared.state.lock().unwrap();
        if self.handle.lock().unwrap().is_none() {
            // No worker accepted anything, so there is nothing to finish or clear.
            return;
        }
        st.reset_regions = true;
        self.shared.wake_worker.notify_one();
        while st.reset_regions {
            st = self.shared.wake_flusher.wait(st).unwrap();
        }
        // Failed writes were logged when they happened, except quarantine-fenced attempts
        // (their fence was logged at the failed rename); a reset only discards the count.
        st.save_failure = None;
    }

    pub fn drain_responses(&self) -> Vec<IoResponse> {
        let mut st = self.shared.state.lock().unwrap();
        std::mem::take(&mut st.responses)
    }
}

impl Drop for LodIoWorker {
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.shutdown = true;
            self.shared.wake_worker.notify_all();
        }
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

fn run_io_worker(shared: Arc<IoShared>, mut cache: RegionCache) {
    loop {
        enum Task {
            Save(PathBuf, String, LodSection),
            Load(PathBuf, String, SectionPos),
            OctreeSave(OctreeSaveKey, OctreeSaveJob),
            OctreeLoad(OctreeSaveKey, u64),
            Exit,
        }

        let task = {
            let mut st = shared.state.lock().unwrap();
            loop {
                if let Some(key) = st.save_order.pop_front()
                    && let Some(section) = st.pending_saves.remove(&key)
                {
                    st.in_flight += 1;
                    break Task::Save(key.0, key.1, section);
                }
                if let Some((world_dir, dim, pos)) = st.pending_loads.pop_front() {
                    st.in_flight += 1;
                    break Task::Load(world_dir, dim, pos);
                }
                if let Some(key) = st.octree_save_order.pop_front()
                    && let Some(job) = st.pending_octree_saves.remove(&key)
                {
                    st.in_flight += 1;
                    break Task::OctreeSave(key, job);
                }
                if let Some((key, request)) = st.pending_octree_loads.pop_front() {
                    st.in_flight += 1;
                    break Task::OctreeLoad(key, request);
                }
                if st.reset_regions {
                    cache.regions.clear();
                    cache.quarantine_failed.clear();
                    st.responses.clear();
                    st.octree_responses.clear();
                    st.reset_regions = false;
                    shared.wake_flusher.notify_all();
                }
                if st.shutdown {
                    break Task::Exit;
                }
                st = shared.wake_worker.wait(st).unwrap();
            }
        };

        match task {
            Task::Exit => break,
            Task::OctreeSave((world_dir, dimension, pos), job) => {
                let failure = cache
                    .write_octree(&world_dir, &dimension, &job.section, &job.mapping, job.mode)
                    .err()
                    .map(|cause| {
                        SaveFailure::logged(
                            format!(
                                "failed to save octree section {pos:?} in {dimension} ({})",
                                world_dir.display()
                            ),
                            cause,
                        )
                    });
                let mut st = shared.state.lock().unwrap();
                st.in_flight -= 1;
                if let Some(failure) = failure {
                    st.record_failed_save(failure);
                }
                if st.idle() {
                    shared.wake_flusher.notify_all();
                }
            }
            Task::OctreeLoad((world_dir, dimension, pos), request) => {
                let result = cache.load_octree_section(&world_dir, &dimension, pos);
                let mut st = shared.state.lock().unwrap();
                st.in_flight -= 1;
                st.octree_responses.push(OctreeLoaded {
                    world_dir,
                    dimension,
                    pos,
                    request,
                    result,
                });
                if st.idle() {
                    shared.wake_flusher.notify_all();
                }
            }
            Task::Save(world_dir, dimension, section) => {
                let failure = cache
                    .save_section(&world_dir, &dimension, &section)
                    .err()
                    .map(|cause| {
                        SaveFailure::logged(
                            format!(
                                "failed to save section {:?} in {dimension} ({})",
                                section.pos,
                                world_dir.display()
                            ),
                            cause,
                        )
                    });
                let mut st = shared.state.lock().unwrap();
                st.in_flight -= 1;
                match failure {
                    None => st.responses.push(IoResponse::Saved),
                    Some(failure) => st.record_failed_save(failure),
                }
                if st.idle() {
                    shared.wake_flusher.notify_all();
                }
            }
            Task::Load(world_dir, dimension, pos) => {
                let loaded = cache.load_section(&world_dir, &dimension, pos);
                let mut st = shared.state.lock().unwrap();
                st.in_flight -= 1;
                if let Some(section) = loaded {
                    st.responses.push(IoResponse::Loaded { dimension, section });
                }
                if st.idle() {
                    shared.wake_flusher.notify_all();
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) struct TestTempDir(PathBuf);

#[cfg(test)]
impl TestTempDir {
    pub(crate) fn new(label: &str) -> Self {
        loop {
            let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "dh-storage-test-{label}-{}-{seq}",
                std::process::id()
            ));
            match fs::create_dir(&dir) {
                Ok(()) => return Self(dir),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create test directory {}: {e}", dir.display()),
            }
        }
    }
}

#[cfg(test)]
impl std::ops::Deref for TestTempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
impl AsRef<Path> for TestTempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
impl Drop for TestTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod::SOURCE_LOADED_CHUNK;

    fn sample_section(pos: SectionPos, top_y: i32) -> LodSection {
        let mut section = LodSection::new(pos);
        let grass = LodColumn::from_solid_and_fluid(
            top_y,
            top_y - 8,
            [145, 189, 89],
            None,
            [0, 0, 0],
            15,
            0,
            SOURCE_LOADED_CHUNK,
        );
        let water = LodColumn::from_solid_and_fluid(
            58,
            52,
            [219, 207, 163],
            Some(63),
            [63, 118, 228],
            15,
            0,
            SOURCE_LOADED_CHUNK,
        );
        for z in 0..64 {
            for x in 0..64 {
                section.set_cell(x, z, if x < 32 { grass } else { water });
            }
        }
        section
    }

    #[test]
    fn rle_section_roundtrip_and_strict_rejections() {
        let pos = SectionPos::new(1, -3, 7);
        let mut section = sample_section(pos, 68);
        let encoded = encode_section(&section);
        assert!(encoded.len() < 4096);
        let decoded = decode_section(&encoded).expect("valid encoded section");
        assert_eq!(decoded.pos, pos);
        assert_eq!(decoded.columns, section.columns);
        assert_eq!(decoded.populated, 4096);
        assert!(decoded.extra_spans.is_empty());

        let floor = ExtraSpan::new(36, 28, [96, 40, 40], 0, 12, false);
        section.set_cell_with_spans(5, 9, section.get(5, 9), vec![floor]);
        let encoded_v2 = encode_section(&section);
        let decoded_v2 = decode_section(&encoded_v2).expect("valid v2 section");
        assert_eq!(decoded_v2.extra_spans_at(5, 9), &[floor]);

        // Reject invalid detail > MAX_DETAIL
        let mut bad_detail = encoded_v2.clone();
        bad_detail[8] = MAX_DETAIL + 1;
        assert!(decode_section(&bad_detail).is_none());

        // Reject truncated DHLOD002 tail (missing extra-spans count)
        let truncated = &encoded[..encoded.len() - 1];
        assert!(decode_section(truncated).is_none());

        // Reject trailing garbage
        let mut trailing = encoded.clone();
        trailing.push(0xAA);
        assert!(decode_section(&trailing).is_none());
    }

    #[test]
    fn region_store_packs_sections_survives_torn_header_and_quarantines_corrupt_magic() {
        let tmp_dir = TestTempDir::new("region");

        let mut cache = RegionCache::default();
        let positions = [
            SectionPos::new(0, 0, 0),
            SectionPos::new(0, 15, 15),
            SectionPos::new(0, -1, -1),
            SectionPos::new(0, -16, -16),
        ];

        for (i, &pos) in positions.iter().enumerate() {
            let sec = sample_section(pos, 64 + i as i32);
            cache
                .save_section(&tmp_dir, "minecraft:overworld", &sec)
                .unwrap();
        }

        let reg0 = region_path(&tmp_dir, "minecraft:overworld", positions[0]);
        assert!(reg0.exists());
        assert_eq!(
            reg0,
            region_path(&tmp_dir, "minecraft:overworld", positions[1])
        );
        assert_eq!(
            region_path(&tmp_dir, "minecraft:overworld", positions[2]),
            region_path(&tmp_dir, "minecraft:overworld", positions[3])
        );

        // Cold read from disk
        let mut reader = RegionCache::default();
        for (i, &pos) in positions.iter().enumerate() {
            let loaded = reader
                .load_section(&tmp_dir, "minecraft:overworld", pos)
                .expect("section stored in region");
            assert_eq!(loaded.pos, pos);
            assert_eq!(loaded.get(0, 0).solid_top_y, 64 + i as i16);
        }

        // Overwrite slot 0 (writes alternating header page), then corrupt the NEWER header page
        // to simulate a torn write mid-header-commit and verify fallback to the older valid page!
        let updated = sample_section(positions[0], 120);
        cache
            .save_section(&tmp_dir, "minecraft:overworld", &updated)
            .unwrap();
        drop(cache);

        let mut raw_reg = fs::read(&reg0).unwrap();
        let gen_a = u64::from_le_bytes(raw_reg[8..16].try_into().unwrap());
        let gen_b = u64::from_le_bytes(
            raw_reg[HEADER_PAGE_BYTES + 8..HEADER_PAGE_BYTES + 16]
                .try_into()
                .unwrap(),
        );
        let newer_page_off = if gen_b > gen_a { HEADER_PAGE_BYTES } else { 0 };
        raw_reg[newer_page_off + 24] ^= 0xFF; // corrupt newer page
        fs::write(&reg0, &raw_reg).unwrap();

        let mut torn_reader = RegionCache::default();
        let recovered = torn_reader
            .load_section(&tmp_dir, "minecraft:overworld", positions[0])
            .expect("falls back cleanly to previous intact header page");
        assert_eq!(recovered.get(0, 0).solid_top_y, 64);

        // Corrupt BOTH header pages and verify read returns None (no mutation) while write
        // quarantines the corrupt file and recovers cleanly.
        let mut both_bad = fs::read(&reg0).unwrap();
        both_bad[0] ^= 0xFF;
        both_bad[HEADER_PAGE_BYTES] ^= 0xFF;
        fs::write(&reg0, &both_bad).unwrap();

        let mut ro_reader = RegionCache::default();
        assert!(
            ro_reader
                .load_section(&tmp_dir, "minecraft:overworld", positions[0])
                .is_none()
        );
        assert_eq!(
            fs::read(&reg0).unwrap(),
            both_bad,
            "read must not mutate corrupt file"
        );

        let mut writer = RegionCache::default();
        writer
            .save_section(&tmp_dir, "minecraft:overworld", &updated)
            .expect("write quarantines corrupt region and recovers");
        let after_recover = writer
            .load_section(&tmp_dir, "minecraft:overworld", positions[0])
            .unwrap();
        assert_eq!(after_recover.get(0, 0).solid_top_y, 120);

        // Corrupt CRC on record
        let (mut rec, crc) = compress_section_record(&updated).unwrap();
        rec[5] ^= 0xFF;
        assert!(decompress_section_record(&rec, Some(crc)).is_none());
    }

    #[test]
    fn sanitize_dimension_is_collision_free_and_io_worker_flushes_coalesced_saves() {
        assert_ne!(
            sanitize_dimension("mymod:a/b"),
            sanitize_dimension("mymod:a_b")
        );
        assert_ne!(
            sanitize_dimension("mymod:a_b"),
            sanitize_dimension("mymod_a:b")
        );

        let tmp_dir = TestTempDir::new("worker");
        let pos = SectionPos::new(0, 2, -3);
        let worker = LodIoWorker::spawn();
        worker.request_save(
            tmp_dir.to_path_buf(),
            "minecraft:overworld".into(),
            sample_section(pos, 70),
        );
        worker.request_save(
            tmp_dir.to_path_buf(),
            "minecraft:overworld".into(),
            sample_section(pos, 95),
        );
        worker.flush().unwrap();

        let mut reader = RegionCache::default();
        let loaded = reader
            .load_section(&tmp_dir, "minecraft:overworld", pos)
            .expect("flushed save present on disk");
        assert_eq!(loaded.get(0, 0).solid_top_y, 95);

        drop(worker);
    }

    // ---- DHLOD003 octree codec ----

    fn populated_octree(pos: OctreeSectionPos) -> (OctreeSection, OctreeMapping) {
        let mut sec = OctreeSection::new(pos);
        for z in 0..32 {
            for x in 0..32 {
                sec.fill_column(x, z, 0, 10, Voxel::new(1, 3, 15, 0));
            }
        }
        sec.fill_column(8, 8, 10, 20, Voxel::new(2, 3, 15, 0));
        for z in 6..11 {
            for x in 6..11 {
                sec.fill_column(x, z, 20, 22, Voxel::new(3, 4, 12, 7));
            }
        }
        sec.set(31, 31, 31, Voxel::new(0xABCDE, 0x1FF, 1, 2));
        sec.set(0, 31, 0, Voxel::air_with_light(15, 0));
        sec.footprint.set(8, 8);
        sec.footprint.set(31, 31);
        sec.revision = 0xDEAD_BEEF_0123;
        let mut mapping = OctreeMapping::default();
        mapping.blocks.insert(1, "minecraft:stone".into());
        mapping.blocks.insert(2, "minecraft:oak_log".into());
        mapping
            .blocks
            .insert(3, "minecraft:oak_leaves[persistent=false]".into());
        mapping.blocks.insert(0xABCDE, "mod:odd".into());
        mapping.biomes.insert(3, "minecraft:plains".into());
        mapping.biomes.insert(4, "minecraft:forest".into());
        (sec, mapping)
    }

    #[test]
    fn octree_roundtrips_in_every_encodable_mode() {
        let pos = OctreeSectionPos::new(2, -3, 1, 40);
        let (sec, mapping) = populated_octree(pos);
        for mode in [
            DhV2CompressionMode::Uncompressed,
            DhV2CompressionMode::Lz4,
            DhV2CompressionMode::ZstdBlock,
        ] {
            let bytes = encode_octree_section(&sec, &mapping, mode).unwrap();
            assert_eq!(&bytes[..8], MAGIC_V3);
            assert_eq!(bytes[9], mode as u8);
            let (back, back_map) = decode_octree_section(&bytes).unwrap();
            assert_eq!(back.pos, pos, "{mode:?}");
            assert_eq!(back.revision, sec.revision, "{mode:?}");
            assert_eq!(back.footprint, sec.footprint, "{mode:?}");
            assert_eq!(back.to_dense(), sec.to_dense(), "{mode:?}");
            assert_eq!(back_map, mapping, "{mode:?}");
            assert!(matches!(
                decode_stored_section(&bytes),
                Some(StoredSection::Octree(..))
            ));
        }
        let plain =
            encode_octree_section(&sec, &mapping, DhV2CompressionMode::Uncompressed).unwrap();
        let zstd = encode_octree_section(&sec, &mapping, DEFAULT_OCTREE_COMPRESSION).unwrap();
        assert!(zstd.len() < plain.len());
        assert_eq!(DEFAULT_OCTREE_COMPRESSION as u8, 4);
        assert_eq!(DhV2CompressionMode::Lz4 as u8, 1);
    }

    #[test]
    fn octree_blob_limit_is_inclusive_and_applies_to_both_blobs() {
        let mode = DhV2CompressionMode::Lz4;
        let payload = vec![0x5a; MAX_OCTREE_DATA_BYTES + 1];
        for limit in [MAX_OCTREE_DATA_BYTES, MAX_OCTREE_MAPPING_BYTES] {
            let exact = compress_blob(&payload[..limit], mode).unwrap();
            assert_eq!(
                decompress_octree_blob(&exact, mode, limit).unwrap(),
                &payload[..limit]
            );
        }
        let oversized_data = compress_blob(&payload, mode).unwrap();
        let oversized_mapping =
            compress_blob(&payload[..MAX_OCTREE_MAPPING_BYTES + 1], mode).unwrap();
        assert!(decompress_octree_blob(&oversized_mapping, mode, MAX_OCTREE_DATA_BYTES).is_ok());

        let (section, mapping) = populated_octree(OctreeSectionPos::new(0, 0, 0, 0));
        let record = encode_octree_section(&section, &mapping, mode).unwrap();
        let mut blobs = Cursor::new(&record[OCTREE_HEADER_BYTES..record.len() - 4]);
        let data_len = blobs.u32().unwrap() as usize;
        let data = blobs.take(data_len).unwrap();
        let names_len = blobs.u32().unwrap() as usize;
        let names = blobs.take(names_len).unwrap();
        for (data, names) in [
            (oversized_data.as_slice(), names),
            (data, oversized_mapping.as_slice()),
        ] {
            let mut record = record[..OCTREE_HEADER_BYTES].to_vec();
            for blob in [data, names] {
                record.extend_from_slice(&(blob.len() as u32).to_be_bytes());
                record.extend_from_slice(blob);
            }
            let crc = crc32fast::hash(&record);
            record.extend_from_slice(&crc.to_be_bytes());
            assert_eq!(
                decode_octree_section(&record).unwrap_err(),
                OctreeCodecError::Oversized("decompressed blob")
            );
        }
    }

    #[test]
    fn octree_uncompressed_footprint_is_the_fixed_x_fastest_data_tail() {
        let mut section = OctreeSection::new(OctreeSectionPos::new(0, 0, 0, 0));
        for (x, z) in [(0, 0), (7, 0), (8, 0), (31, 0), (2, 1), (4, 19), (31, 31)] {
            section.footprint.set(x, z);
        }
        let record = encode_octree_section(
            &section,
            &OctreeMapping::default(),
            DhV2CompressionMode::Uncompressed,
        )
        .unwrap();
        assert_eq!(record[8], 4);
        let mut body = Cursor::new(&record[OCTREE_HEADER_BYTES..]);
        // One palette entry (dark air), one run (index 0, length 32768).
        let voxel_bytes = [1u8, 0, 0, 1, 0, 0x80, 0x80, 0x02];
        let len = body.u32().unwrap() as usize;
        assert_eq!(len, voxel_bytes.len() + 128);
        let (voxels, footprint) = body.take(len).unwrap().split_at(voxel_bytes.len());
        assert_eq!(voxels, &voxel_bytes);
        let mut expected = [0u8; 128];
        expected[0] = 0x81;
        expected[1] = 0x01;
        expected[3] = 0x80;
        expected[4] = 0x04;
        expected[76] = 0x10;
        expected[127] = 0x80;
        assert_eq!(footprint, &expected);
    }

    fn replace_uncompressed_data_tail(mut record: Vec<u8>, remove: usize, tail: &[u8]) -> Vec<u8> {
        let len = OCTREE_HEADER_BYTES..OCTREE_HEADER_BYTES + 4;
        let old_len = u32::from_be_bytes(record[len.clone()].try_into().unwrap()) as usize;
        let end = len.end + old_len;
        drop(record.splice(end - remove..end, tail.iter().copied()));
        let new_len = (old_len - remove + tail.len()) as u32;
        record[len].copy_from_slice(&new_len.to_be_bytes());
        record
    }

    fn reseal_octree_record(mut record: Vec<u8>) -> Vec<u8> {
        let n = record.len() - 4;
        let crc = crc32fast::hash(&record[..n]);
        record[n..].copy_from_slice(&crc.to_be_bytes());
        record
    }

    #[test]
    fn octree_refuses_short_or_trailing_footprint_bytes_with_valid_crc() {
        let (section, mapping) = populated_octree(OctreeSectionPos::new(1, 5, -2, 9));
        let valid =
            encode_octree_section(&section, &mapping, DhV2CompressionMode::Uncompressed).unwrap();
        for (remove, tail, error) in [
            (1, &[][..], OctreeCodecError::Truncated),
            (
                0,
                &[0xAA][..],
                OctreeCodecError::Corrupt("trailing data bytes"),
            ),
        ] {
            let record =
                reseal_octree_record(replace_uncompressed_data_tail(valid.clone(), remove, tail));
            assert_eq!(decode_octree_section(&record).unwrap_err(), error);
        }
    }

    #[test]
    fn octree_refuses_modes_it_cannot_write() {
        let (sec, mapping) = populated_octree(OctreeSectionPos::new(0, 0, 0, 0));
        for mode in [DhV2CompressionMode::ZstdStream, DhV2CompressionMode::Lzma2] {
            assert_eq!(
                encode_octree_section(&sec, &mapping, mode).unwrap_err(),
                OctreeCodecError::EncodeUnsupported(mode)
            );
        }
    }

    #[test]
    fn lz4_frame_descriptor_checksum_matches_reference() {
        // `04 22 4D 18 60 40 82` is the frame header the reference lz4 tool writes for
        // independent 64 KiB blocks without checksums.
        let frame = lz4_frame_compress(b"hello hello hello hello");
        assert_eq!(&frame[..7], &[0x04, 0x22, 0x4D, 0x18, 0x60, 0x40, 0x82]);
        assert_eq!(&frame[frame.len() - 4..], &[0, 0, 0, 0]);
    }

    #[test]
    fn octree_rejects_corruption_and_bad_framing() {
        let (sec, mapping) = populated_octree(OctreeSectionPos::new(1, 5, -2, 9));
        for mode in [
            DhV2CompressionMode::Uncompressed,
            DhV2CompressionMode::Lz4,
            DhV2CompressionMode::ZstdBlock,
        ] {
            let bytes = encode_octree_section(&sec, &mapping, mode).unwrap();
            // Flip every byte after the magic in turn; no flip may decode.
            for i in 8..bytes.len() {
                let mut bad = bytes.clone();
                bad[i] ^= 0x40;
                assert!(decode_octree_section(&bad).is_err(), "{mode:?} byte {i}");
            }
            let mid = bytes.len() / 2;
            let mut bad = bytes.clone();
            bad[mid] ^= 1;
            assert!(matches!(
                decode_octree_section(&bad),
                Err(OctreeCodecError::ChecksumMismatch { .. })
            ));
            for cut in [0, 7, 8, 30, bytes.len() - 1] {
                assert!(decode_octree_section(&bytes[..cut]).is_err());
            }
            let mut long = bytes.clone();
            long.push(0);
            assert!(decode_octree_section(&long).is_err());
        }
        let mut bytes =
            encode_octree_section(&sec, &mapping, DhV2CompressionMode::Uncompressed).unwrap();
        bytes[0] = b'X';
        assert_eq!(
            decode_octree_section(&bytes).unwrap_err(),
            OctreeCodecError::BadMagic
        );
    }

    #[test]
    fn octree_rejects_valid_crc_with_bad_fields() {
        let (sec, mapping) = populated_octree(OctreeSectionPos::new(1, 5, -2, 9));
        let good =
            encode_octree_section(&sec, &mapping, DhV2CompressionMode::Uncompressed).unwrap();
        let mut v = good.clone();
        v[8] = 9; // format version
        assert_eq!(
            decode_octree_section(&reseal_octree_record(v)).unwrap_err(),
            OctreeCodecError::UnsupportedFormatVersion(9)
        );
        let mut v = good.clone();
        v[9] = 77; // compression id
        assert_eq!(
            decode_octree_section(&reseal_octree_record(v)).unwrap_err(),
            OctreeCodecError::UnknownCompression(77)
        );
        let mut v = good.clone();
        v[10] = MAX_OCTREE_DETAIL + 1;
        assert!(decode_octree_section(&reseal_octree_record(v)).is_err());
        let mut v = good;
        v[OCTREE_HEADER_BYTES..OCTREE_HEADER_BYTES + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(
            decode_octree_section(&reseal_octree_record(v)).unwrap_err(),
            OctreeCodecError::Truncated
        );
    }

    #[test]
    fn varint_matches_dh_and_rejects_overflow() {
        let mut out = Vec::new();
        for v in [0u32, 1, 127, 128, 300, 16_384, i32::MAX as u32] {
            out.clear();
            write_varint(&mut out, v);
            assert_eq!(Cursor::new(&out).varint().unwrap(), v);
        }
        out.clear();
        write_varint(&mut out, 300);
        assert_eq!(out, [0xAC, 0x02]);
        assert!(
            Cursor::new(&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F])
                .varint()
                .is_err()
        );
        assert!(
            Cursor::new(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x01])
                .varint()
                .is_err()
        );
        assert!(Cursor::new(&[0x80]).varint().is_err());
    }

    #[test]
    fn octree_region_slots_are_in_range_and_distinct() {
        let mut seen = std::collections::HashSet::new();
        for y in -4..4 {
            for z in -8..8 {
                for x in -8..8 {
                    let (rx, ry, rz, slot) = octree_region_slot(OctreeSectionPos::new(0, x, y, z));
                    assert!(slot < REGION_SLOTS);
                    assert!(seen.insert((rx, ry, rz, slot)));
                }
            }
        }
        let p = Path::new("/w");
        assert_ne!(
            octree_region_path(p, "d", OctreeSectionPos::new(0, 0, 0, 0)),
            octree_region_path(p, "d", OctreeSectionPos::new(1, 0, 0, 0))
        );
    }

    #[test]
    fn octree_region_file_persists_and_recovers() {
        let dir = TestTempDir::new("octree-region");
        let dim = "minecraft:overworld";
        let a_pos = OctreeSectionPos::new(0, 1, 0, 2);
        let b_pos = OctreeSectionPos::new(0, 2, 1, 3);
        assert_eq!(
            octree_region_path(&dir, dim, a_pos),
            octree_region_path(&dir, dim, b_pos)
        );
        let (a, a_map) = populated_octree(a_pos);
        let (mut b, b_map) = populated_octree(b_pos);
        b.set(3, 3, 3, Voxel::new(2, 3, 0, 0));

        let mut cache = RegionCache::default();
        cache
            .save_octree_section(&dir, dim, &a, &a_map, DhV2CompressionMode::ZstdBlock)
            .unwrap();
        cache
            .save_octree_section(&dir, dim, &b, &b_map, DhV2CompressionMode::Lz4)
            .unwrap();

        let mut cold = RegionCache::default();
        let (la, lm) = cold.load_octree_section(&dir, dim, a_pos).unwrap().unwrap();
        assert_eq!(la.to_dense(), a.to_dense());
        assert_eq!(lm, a_map);
        let (lb, _) = cold.load_octree_section(&dir, dim, b_pos).unwrap().unwrap();
        assert_eq!(lb.to_dense(), b.to_dense());
        assert!(
            cold.load_octree_section(&dir, dim, OctreeSectionPos::new(0, 3, 0, 3))
                .unwrap()
                .is_none()
        );
        assert!(
            cold.load_octree_section(&dir, "minecraft:the_nether", a_pos)
                .unwrap()
                .is_none()
        );

        // Torn header: the second save committed generation 3 into header page A (offset 0).
        // Wreck it and the older page (generation 2, holding only `a`) must take over.
        let path = octree_region_path(&dir, dim, a_pos);
        drop(cache);
        drop(cold);
        let mut f = OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all(&[0xFF; 64]).unwrap();
        drop(f);
        let mut recovered = RegionCache::default();
        assert!(
            recovered
                .load_octree_section(&dir, dim, a_pos)
                .unwrap()
                .is_some()
        );
        assert!(
            recovered
                .load_octree_section(&dir, dim, b_pos)
                .unwrap()
                .is_none()
        );

        // Corrupt magic on both pages: reads fail soft, the next save quarantines and restarts.
        drop(recovered);
        let mut f = OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
        f.seek(SeekFrom::Start(HEADER_PAGE_BYTES as u64)).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
        drop(f);
        let mut after = RegionCache::default();
        assert!(
            after
                .load_octree_section(&dir, dim, a_pos)
                .unwrap()
                .is_none()
        );
        after
            .save_octree_section(&dir, dim, &b, &b_map, DEFAULT_OCTREE_COMPRESSION)
            .unwrap();
        let quarantined = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.file_name().to_string_lossy().contains(".corrupt."));
        assert!(quarantined);
        assert!(
            after
                .load_octree_section(&dir, dim, b_pos)
                .unwrap()
                .is_some()
        );

        // A flipped payload byte fails the record CRC and reads as absent.
        drop(after);
        let mut raw = fs::read(&path).unwrap();
        let last = raw.len() - 8;
        raw[last] ^= 0xFF;
        fs::write(&path, &raw).unwrap();
        let mut bad = RegionCache::default();
        assert!(bad.load_octree_section(&dir, dim, b_pos).unwrap().is_none());
    }

    #[test]
    fn region_quarantine_failure_blocks_both_formats_until_worker_cache_reset() {
        let dir = TestTempDir::new("quarantine-failure");
        let dim = "minecraft:overworld";
        let column = SectionPos::new(0, 0, 0);
        let octree = OctreeSectionPos::new(0, 0, 0, 0);
        let column_path = region_path(&dir, dim, column);
        let octree_path = octree_region_path(&dir, dim, octree);
        let old_column = vec![0x33; REGION_HEADER_BYTES + 17];
        let old_octree = vec![0x55; REGION_HEADER_BYTES + 31];
        let originals = [(&column_path, &old_column), (&octree_path, &old_octree)];
        for (path, bytes) in originals {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }

        let attempts = Arc::new(Mutex::new(HashMap::<PathBuf, Vec<PathBuf>>::new()));
        let rename_attempts = Arc::clone(&attempts);
        let worker = LodIoWorker::spawn_with_cache(RegionCache {
            quarantine_rename: Some(Box::new(move |from, to| {
                let mut attempts = rename_attempts.lock().unwrap();
                let destinations = attempts.entry(from.to_path_buf()).or_default();
                destinations.push(to.to_path_buf());
                if destinations.len() == 1 {
                    return Err(io::Error::other("injected quarantine rename failure"));
                }
                fs::rename(from, to)
            })),
            ..RegionCache::default()
        });

        let read_misses = |column, octree| {
            worker.request_load(dir.to_path_buf(), dim.into(), column);
            worker.load_octree_section(dir.to_path_buf(), dim.into(), octree, 1, false);
            worker.flush().unwrap();
            assert!(worker.drain_responses().is_empty());
            let probes = worker.drain_octree_responses();
            assert_eq!(probes.len(), 1);
            assert!(probes.into_iter().next().unwrap().result.unwrap().is_none());
        };
        let queue_saves = |column, octree| {
            let column = sample_section(column, 88);
            let octree = OctreeSection::filled(octree, Voxel::new(2, 0, 12, 0));
            worker.request_save(dir.to_path_buf(), dim.into(), column.clone());
            worker.save_octree_section(
                dir.to_path_buf(),
                dim.into(),
                octree.clone(),
                OctreeMapping::default(),
            );
            (column, octree)
        };
        let assert_saved = |column: LodSection, octree: OctreeSection| {
            worker.flush().unwrap();
            assert!(matches!(
                worker.drain_responses().as_slice(),
                [IoResponse::Saved]
            ));
            worker.request_load(dir.to_path_buf(), dim.into(), column.pos);
            worker.load_octree_section(dir.to_path_buf(), dim.into(), octree.pos, 2, false);
            worker.flush().unwrap();
            assert!(matches!(
                worker.drain_responses().as_slice(),
                [IoResponse::Loaded { dimension, section }]
                    if dimension.as_str() == dim && section.pos == column.pos
                        && section.columns == column.columns
            ));
            let mut probes = worker.drain_octree_responses();
            assert_eq!(probes.len(), 1);
            let (loaded, _) = probes.pop().unwrap().result.unwrap().unwrap();
            assert_eq!(loaded.pos, octree.pos);
            assert_eq!(loaded.to_dense(), octree.to_dense());
        };
        let assert_first_failures_preserved = || {
            let attempts = attempts.lock().unwrap();
            assert_eq!(attempts.len(), originals.len());
            for (path, bytes) in originals {
                assert_eq!(attempts[path].len(), 1, "{}", path.display());
                assert!(!attempts[path][0].exists());
                assert_eq!(&fs::read(path).unwrap(), bytes);
            }
        };

        read_misses(column, octree);
        assert!(attempts.lock().unwrap().is_empty());
        for (path, bytes) in originals {
            assert_eq!(&fs::read(path).unwrap(), bytes);
        }
        queue_saves(column, octree);
        let failure = worker.flush().unwrap_err();
        assert_eq!(failure.count, 2);
        assert!(RegionWriteRefused::matches(&failure.cause));
        assert!(worker.drain_responses().is_empty(), "no Saved on refusal");
        assert_first_failures_preserved();
        read_misses(column, octree);

        // Open enough distinct region handles to force eviction without resetting the session.
        let mut empty_region = vec![0; REGION_HEADER_BYTES];
        empty_region[..HEADER_PAGE_BYTES].copy_from_slice(&encode_header_page(
            1,
            &[SlotEntry::default(); REGION_SLOTS],
        ));
        for i in 1..=MAX_CACHED_REGIONS + 1 {
            let pos = OctreeSectionPos::new(0, (i as i32) << OCT_REGION_SHIFT_X, 0, 0);
            fs::write(octree_region_path(&dir, dim, pos), &empty_region).unwrap();
            worker.load_octree_section(dir.to_path_buf(), dim.into(), pos, 3, false);
        }
        worker.flush().unwrap();
        let probes = worker.drain_octree_responses();
        assert_eq!(probes.len(), MAX_CACHED_REGIONS + 1);
        assert!(
            probes
                .into_iter()
                .all(|probe| probe.result.unwrap().is_none())
        );

        let column_neighbor = SectionPos::new(0, 1, 0);
        let octree_neighbor = OctreeSectionPos::new(0, 1, 0, 0);
        queue_saves(column, octree);
        queue_saves(column_neighbor, octree_neighbor);
        let failure = worker.flush().unwrap_err();
        assert_eq!(failure.count, 4);
        assert!(RegionWriteRefused::matches(&failure.cause));
        assert!(worker.drain_responses().is_empty(), "no Saved on retries");
        read_misses(column_neighbor, octree_neighbor);
        assert_first_failures_preserved();

        // The refusal is scoped to the exact file, not the dimension or file format.
        let (healthy_column, healthy_octree) = queue_saves(
            SectionPos::new(0, REGION_SIZE as i32, 0),
            OctreeSectionPos::new(0, 1 << OCT_REGION_SHIFT_X, 0, 0),
        );
        assert_saved(healthy_column, healthy_octree);
        assert_first_failures_preserved();

        worker.reset_world();
        read_misses(column, octree);
        assert_first_failures_preserved();
        let (new_column, new_octree) = queue_saves(column, octree);
        assert_saved(new_column, new_octree);
        let attempts = attempts.lock().unwrap();
        assert_eq!(attempts.len(), originals.len());
        for (path, bytes) in originals {
            let [failed, quarantined] = attempts[path].as_slice() else {
                panic!(
                    "expected one failed rename and one retry for {}",
                    path.display()
                );
            };
            assert!(!failed.exists());
            let prefix = format!("{}.corrupt.", path.file_name().unwrap().to_string_lossy());
            assert!(
                quarantined
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(&prefix)
            );
            assert_eq!(&fs::read(quarantined).unwrap(), bytes);
        }
    }

    #[test]
    fn worker_flush_reports_both_formats_without_losing_unrelated_writes() {
        let dir = TestTempDir::new("flush-save-failure");
        let dim = "minecraft:overworld";
        let column_bad = SectionPos::new(0, 0, 0);
        let column_good = SectionPos::new(0, REGION_SIZE as i32, 0);
        let octree_bad = OctreeSectionPos::new(0, 0, 0, 0);
        let octree_good = OctreeSectionPos::new(0, 1 << OCT_REGION_SHIFT_X, 0, 0);
        let worker = LodIoWorker::spawn();
        // Neither directory can be opened as a writable region file.
        fs::create_dir_all(region_path(&dir, dim, column_bad)).unwrap();
        fs::create_dir_all(octree_region_path(&dir, dim, octree_bad)).unwrap();
        // A second write into the unopenable region still reports as a barrier count.
        let column_bad_neighbor = SectionPos::new(0, 1, 0);
        for pos in [column_bad, column_bad_neighbor, column_good] {
            worker.request_save(dir.to_path_buf(), dim.into(), sample_section(pos, 95));
        }
        let failure = worker.flush().unwrap_err();
        assert_eq!(failure.count, 2);
        assert!(
            failure
                .section
                .contains(&format!("section {column_bad:?} in {dim}"))
        );
        assert!(matches!(
            worker.drain_responses().as_slice(),
            [IoResponse::Saved]
        ));

        let (bad, bad_mapping) = populated_octree(octree_bad);
        let (good, good_mapping) = populated_octree(octree_good);
        worker.save_octree_section(dir.to_path_buf(), dim.into(), bad, bad_mapping);
        worker.save_octree_section(dir.to_path_buf(), dim.into(), good.clone(), good_mapping);
        let failure = worker.flush().unwrap_err();
        assert_eq!(failure.count, 1);
        assert!(
            failure
                .section
                .contains(&format!("octree section {octree_bad:?} in {dim}"))
        );
        worker.flush().unwrap();
        drop(worker);

        let mut cold = RegionCache::default();
        let column = cold.load_section(&dir, dim, column_good).unwrap();
        assert_eq!(column.get(0, 0).solid_top_y, 95);
        let (octree, _) = cold
            .load_octree_section(&dir, dim, octree_good)
            .unwrap()
            .unwrap();
        assert_eq!(octree.to_dense(), good.to_dense());
    }

    #[test]
    fn worker_that_failed_to_spawn_refuses_work_without_repeated_failures() {
        let dir = TestTempDir::new("no-worker");
        let dim = "minecraft:overworld";
        let column = SectionPos::new(0, 0, 0);
        let octree = OctreeSectionPos::new(0, 0, 0, 0);
        let worker = LodIoWorker::spawn_unavailable();
        for _ in 0..2 {
            // The same backlog is offered at every barrier; none of it can be served.
            worker.request_save(dir.to_path_buf(), dim.into(), sample_section(column, 95));
            worker.request_load(dir.to_path_buf(), dim.into(), column);
            let (section, mapping) = populated_octree(octree);
            let refusal = worker.queue_octree_record(
                dir.to_path_buf(),
                dim.into(),
                (&section).into(),
                mapping,
                DEFAULT_OCTREE_COMPRESSION,
            );
            assert!(
                matches!(refusal, Err(SaveRefusal::NoWorker(cause)) if cause.to_string() == "spawn refused")
            );
            worker.load_octree_section(dir.to_path_buf(), dim.into(), octree, 1, true);
            worker.flush().unwrap();
            worker.reset_world();
            worker.flush().unwrap();
        }
        {
            let st = worker.shared.state.lock().unwrap();
            assert!(st.idle() && st.save_order.is_empty() && st.octree_save_order.is_empty());
            assert!(st.save_failure.is_none());
        }
        assert!(worker.drain_responses().is_empty(), "no fabricated Saved");
        assert!(worker.drain_octree_responses().is_empty());
        assert!(!region_path(&dir, dim, column).exists());
        assert!(!octree_region_path(&dir, dim, octree).exists());
    }

    #[test]
    fn octree_v3_slot_is_a_cache_miss_without_mutating_the_region() {
        let dir = TestTempDir::new("octree-v3");
        let dim = "minecraft:overworld";
        let pos = OctreeSectionPos::new(0, 1, 0, 2);
        let (section, mapping) = populated_octree(pos);
        // The v3 layout has the same voxel/mapping encoding but no footprint tail.
        let v4 =
            encode_octree_section(&section, &mapping, DhV2CompressionMode::Uncompressed).unwrap();
        let mut old = replace_uncompressed_data_tail(v4, 128, &[]);
        old[8] = 3;
        let old = reseal_octree_record(old);
        assert_eq!(
            decode_octree_section(&old).unwrap_err(),
            OctreeCodecError::UnsupportedFormatVersion(3)
        );

        let path = octree_region_path(&dir, dim, pos);
        {
            let mut cache = RegionCache::default();
            cache.check_write_allowed(&path).unwrap();
            let writer = cache.get_region(&path, true).unwrap().unwrap();
            let (_, _, _, slot) = octree_region_slot(pos);
            writer
                .write_slot(slot, &old, octree_record_crc(&old).unwrap())
                .unwrap();
        }
        let before = fs::read(&path).unwrap();
        assert!(
            RegionCache::default()
                .load_octree_section(&dir, dim, pos)
                .unwrap()
                .is_none()
        );
        assert_eq!(fs::read(&path).unwrap(), before);
    }

    #[test]
    fn octree_authored_all_air_reopens_but_untouched_air_never_allocates_a_slot() {
        let dir = TestTempDir::new("octree-authored-air");
        let dim = "minecraft:overworld";
        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let mut authored = OctreeSection::new(pos);
        authored.footprint.set(3, 19);
        authored.footprint.set(31, 31);
        authored.revision = 73;
        let mut untouched = OctreeSection::new(pos.neighbor(crate::octree::Face::PosX));
        // Only the footprint authorizes the extra slot, not the revision or voxel bytes.
        untouched.revision = authored.revision;
        let mapping = OctreeMapping::default();
        let path = octree_region_path(&dir, dim, pos);
        let before = {
            let mut writer = RegionCache::default();
            writer
                .save_octree_section(&dir, dim, &authored, &mapping, DEFAULT_OCTREE_COMPRESSION)
                .unwrap();
            let before = fs::read(&path).unwrap();
            writer
                .save_octree_section(&dir, dim, &untouched, &mapping, DEFAULT_OCTREE_COMPRESSION)
                .unwrap();
            before
        };
        assert_eq!(
            fs::read(&path).unwrap(),
            before,
            "no write for untouched air"
        );
        let mut reader = RegionCache::default();
        let (reopened, _) = reader.load_octree_section(&dir, dim, pos).unwrap().unwrap();
        assert_eq!(reopened.to_dense(), authored.to_dense());
        assert_eq!(reopened.footprint, authored.footprint);
        assert_eq!(reopened.revision, authored.revision);
        assert!(
            reader
                .load_octree_section(&dir, dim, untouched.pos)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn octree_worker_rewrites_existing_slots_with_air_without_allocating_new_ones() {
        let dir = TestTempDir::new("octree-air");
        let dim = "minecraft:overworld";
        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let neighbor = pos.neighbor(crate::octree::Face::PosX);
        let missing = OctreeSectionPos::new(0, 2, 0, 0);
        let absent_region = OctreeSectionPos::new(0, 8, 0, 0);
        let (old, mapping) = populated_octree(pos);
        let (other, other_mapping) = populated_octree(neighbor);
        let mut clear = OctreeSection::filled(pos, Voxel::air_with_light(15, 0));
        clear.revision = old.revision + 1;

        let worker = LodIoWorker::spawn();
        worker.save_octree_section(dir.to_path_buf(), dim.into(), old, mapping);
        worker.save_octree_section(dir.to_path_buf(), dim.into(), other.clone(), other_mapping);
        worker.flush().unwrap();
        for section in [
            clear.clone(),
            OctreeSection::new(missing),
            OctreeSection::new(absent_region),
        ] {
            worker.save_octree_section(
                dir.to_path_buf(),
                dim.into(),
                section,
                OctreeMapping::default(),
            );
        }
        worker.load_octree_section(dir.to_path_buf(), dim.into(), pos, 7, false);
        worker.flush().unwrap();
        let responses = worker.drain_octree_responses();
        assert_eq!(responses.len(), 1);
        let (reloaded, _) = responses
            .into_iter()
            .next()
            .unwrap()
            .result
            .unwrap()
            .unwrap();
        assert_eq!(reloaded.revision, clear.revision);
        assert_eq!(reloaded.to_dense(), clear.to_dense());
        drop(worker);

        let mut fresh = RegionCache::default();
        let (reopened, _) = fresh.load_octree_section(&dir, dim, pos).unwrap().unwrap();
        assert_eq!(reopened.revision, clear.revision);
        assert_eq!(reopened.to_dense(), clear.to_dense());
        assert!(reopened.footprint.is_empty());
        let (reopened_neighbor, _) = fresh
            .load_octree_section(&dir, dim, neighbor)
            .unwrap()
            .unwrap();
        assert_eq!(reopened_neighbor.to_dense(), other.to_dense());
        assert!(
            fresh
                .load_octree_section(&dir, dim, missing)
                .unwrap()
                .is_none()
        );
        assert!(!octree_region_path(&dir, dim, absent_region).exists());
        drop(fresh);
    }

    #[test]
    fn octree_pending_clear_supersedes_nonempty_write_before_load() {
        let dir = TestTempDir::new("octree-coalesced-air");
        let dim = "minecraft:overworld";
        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let (old, mapping) = populated_octree(pos);
        // Stage operations before running the real I/O loop so coalescing is deterministic.
        let worker = LodIoWorker {
            shared: Arc::new(IoShared {
                state: Mutex::new(IoQueueState::default()),
                wake_worker: Condvar::new(),
                wake_flusher: Condvar::new(),
            }),
            handle: Mutex::new(None),
            spawn_error: None,
        };
        worker.save_octree_section(dir.to_path_buf(), dim.into(), old.clone(), mapping);
        worker.load_octree_section(dir.to_path_buf(), dim.into(), pos, 7, false);
        worker.save_octree_section(
            dir.to_path_buf(),
            dim.into(),
            OctreeSection::new(pos),
            OctreeMapping::default(),
        );
        worker.shared.state.lock().unwrap().shutdown = true;
        assert!(matches!(
            worker.queue_octree_record(
                dir.to_path_buf(),
                dim.into(),
                (&old).into(),
                OctreeMapping::default(),
                DEFAULT_OCTREE_COMPRESSION,
            ),
            Err(SaveRefusal::Shutdown)
        ));
        run_io_worker(Arc::clone(&worker.shared), RegionCache::default());

        let got = worker.drain_octree_responses();
        assert_eq!(got.len(), 1);
        assert!(got.into_iter().next().unwrap().result.unwrap().is_none());
        assert!(!octree_region_path(&dir, dim, pos).exists());
    }

    #[test]
    fn reset_closes_cached_regions_after_writes_and_reopens_a_replacement() {
        let dir = TestTempDir::new("octree-reopen");
        let replacement_dir = TestTempDir::new("octree-replacement");
        let dim = "minecraft:overworld";
        let pos = OctreeSectionPos::new(0, 0, 0, 0);
        let worker = LodIoWorker::spawn();
        let first = OctreeSection::filled(pos, Voxel::new(1, 0, 12, 0));
        worker.save_octree_section(
            dir.to_path_buf(),
            dim.into(),
            first.clone(),
            OctreeMapping::default(),
        );
        worker.load_octree_section(dir.to_path_buf(), dim.into(), pos, 1, false);
        worker.reset_world();
        assert!(worker.drain_octree_responses().is_empty());
        let (written, _) = RegionCache::default()
            .load_octree_section(&dir, dim, pos)
            .unwrap()
            .unwrap();
        assert_eq!(written.to_dense(), first.to_dense());

        let replacement = OctreeSection::filled(pos, Voxel::new(2, 0, 10, 0));
        RegionCache::default()
            .save_octree_section(
                &replacement_dir,
                dim,
                &replacement,
                &OctreeMapping::default(),
                DEFAULT_OCTREE_COMPRESSION,
            )
            .unwrap();
        let path = octree_region_path(&dir, dim, pos);
        fs::rename(&path, path.with_extension("first-open")).unwrap();
        fs::rename(octree_region_path(&replacement_dir, dim, pos), path).unwrap();
        worker.load_octree_section(dir.to_path_buf(), dim.into(), pos, 2, false);
        worker.flush().unwrap();
        let result = worker.drain_octree_responses().pop().unwrap();
        assert_eq!(result.request, 2);
        assert_eq!(
            result.result.unwrap().unwrap().0.to_dense(),
            replacement.to_dense()
        );
    }

    #[test]
    fn octree_worker_coalesces_saves_and_flushes_on_drop() {
        let dir = TestTempDir::new("octree-worker");
        let dim = "minecraft:overworld";
        let pos = OctreeSectionPos::new(1, -1, 0, -1);
        let (first, mapping) = populated_octree(pos);
        let mut second = first.clone();
        second.set(4, 4, 4, Voxel::new(2, 3, 9, 9));

        let worker = LodIoWorker::spawn();
        worker.save_octree_section(dir.to_path_buf(), dim.into(), first, mapping.clone());
        worker.save_octree_section(
            dir.to_path_buf(),
            dim.into(),
            second.clone(),
            mapping.clone(),
        );
        worker.flush().unwrap();
        worker.load_octree_section(dir.to_path_buf(), dim.into(), pos, 7, false);
        worker.flush().unwrap();
        let got = worker.drain_octree_responses();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].dimension, dim);
        assert_eq!(got[0].request, 7);
        let (loaded, loaded_mapping) = got.into_iter().next().unwrap().result.unwrap().unwrap();
        assert_eq!(loaded.to_dense(), second.to_dense());
        assert_eq!(loaded_mapping, mapping);

        // A save still queued at drop is written before the worker thread exits.
        let other = OctreeSectionPos::new(1, -2, 0, -1);
        let (extra, extra_map) = populated_octree(other);
        worker.save_octree_section_with(
            dir.to_path_buf(),
            dim.into(),
            extra.clone(),
            extra_map,
            DhV2CompressionMode::Uncompressed,
        );
        drop(worker);
        let mut reader = RegionCache::default();
        let (loaded, _) = reader
            .load_octree_section(&dir, dim, other)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.to_dense(), extra.to_dense());
    }
}
