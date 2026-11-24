//! Fast, crash-safe region-packed Zstd binary LOD storage (`DHREG002`) under
//! `<world_dir>/distant_horizons/<dim>/d<detail>/r.<rx>.<rz>.dhreg`.
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
//! - Coalescing background `LodIoWorker` with explicit `flush()` and clean `Drop` join so
//!   queued saves are never dropped on world unload or process exit.

use crate::lod::{ExtraSpan, LodColumn, LodSection, MAX_DETAIL, SECTION_CELLS, SectionPos};
use std::collections::{HashMap, VecDeque};
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
        .join("distant_horizons")
        .join(sanitize_dimension(dimension))
        .join(format!("d{}", pos.detail))
        .join(format!("r.{rx}.{rz}.dhreg"))
}

/// Legacy single-section `.dhlod` path (kept read-only for seamless migration of existing worlds).
pub fn legacy_section_path(world_dir: &Path, dimension: &str, pos: SectionPos) -> PathBuf {
    world_dir
        .join("distant_horizons")
        .join(legacy_sanitize_dimension(dimension))
        .join(format!("d{}", pos.detail))
        .join(format!("s.{}.{}.dhlod", pos.x, pos.z))
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
    fn open(path: &Path, writable: bool, tick: u64) -> io::Result<Option<Self>> {
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

        let file_len = file.metadata()?.len();
        let parsed = if file_len >= REGION_HEADER_BYTES as u64 {
            let mut page_a = [0u8; HEADER_PAGE_BYTES];
            let mut page_b = [0u8; HEADER_PAGE_BYTES];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut page_a)?;
            file.read_exact(&mut page_b)?;
            let a = parse_header_page(&page_a, file_len);
            let b = parse_header_page(&page_b, file_len);
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

        // Quarantine the unrecoverable corrupt file before reinitializing so saves never brick
        // permanently and forensic data isn't silently overwritten.
        drop(file);
        let corrupt = unique_tmp_path(path, "corrupt");
        if let Err(e) = fs::rename(path, &corrupt) {
            eprintln!(
                "DH: failed to quarantine corrupt region {}: {e}",
                path.display()
            );
            let _ = fs::remove_file(path);
        } else {
            eprintln!(
                "DH: quarantined corrupt region {} -> {}",
                path.display(),
                corrupt.display()
            );
        }
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

    fn read_slot(&mut self, slot_idx: usize, expected_pos: SectionPos) -> Option<LodSection> {
        let entry = self.slots[slot_idx];
        if entry.len == 0
            || (entry.len as usize) > MAX_COMPRESSED_RECORD_BYTES
            || entry.offset < REGION_HEADER_BYTES as u64
        {
            return None;
        }
        let file = self.file.as_mut()?;
        let mut buf = vec![0u8; entry.len as usize];
        file.seek(SeekFrom::Start(entry.offset)).ok()?;
        file.read_exact(&mut buf).ok()?;
        let sec = decompress_section_record(&buf, Some(entry.crc32))?;
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

/// Region file handle + index cache owned by the background IO worker thread.
#[derive(Default)]
pub struct RegionCache {
    regions: HashMap<PathBuf, OpenRegion>,
    tick: u64,
}

impl RegionCache {
    fn get_region(&mut self, path: &Path, writable: bool) -> io::Result<Option<&mut OpenRegion>> {
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
            let Some(reg) = OpenRegion::open(path, writable, now)? else {
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

    pub fn save_section(
        &mut self,
        world_dir: &Path,
        dimension: &str,
        section: &LodSection,
    ) -> io::Result<()> {
        let path = region_path(world_dir, dimension, section.pos);
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

pub enum IoResponse {
    Loaded {
        dimension: String,
        section: LodSection,
    },
    Saved,
}

type SaveKey = (PathBuf, String, SectionPos);

#[derive(Default)]
struct IoQueueState {
    /// Coalesced pending saves (newest revision per `(world_dir, dimension, pos)` wins).
    pending_saves: HashMap<SaveKey, LodSection>,
    save_order: VecDeque<SaveKey>,
    pending_loads: VecDeque<(PathBuf, String, SectionPos)>,
    in_flight: usize,
    shutdown: bool,
    responses: Vec<IoResponse>,
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
}

impl LodIoWorker {
    pub fn spawn() -> Self {
        let shared = Arc::new(IoShared {
            state: Mutex::new(IoQueueState::default()),
            wake_worker: Condvar::new(),
            wake_flusher: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let handle = match thread::Builder::new()
            .name("dh-lod-io".into())
            .spawn(move || run_io_worker(worker_shared))
        {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("DH: failed to spawn dh-lod-io worker thread: {e}");
                None
            }
        };
        Self {
            shared,
            handle: Mutex::new(handle),
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

    /// Block until all currently queued saves and loads have finished and synced to disk.
    pub fn flush(&self) {
        let mut st = self.shared.state.lock().unwrap();
        if self.handle.lock().unwrap().is_none() {
            return;
        }
        while !st.pending_saves.is_empty() || !st.pending_loads.is_empty() || st.in_flight > 0 {
            st = self.shared.wake_flusher.wait(st).unwrap();
        }
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

fn run_io_worker(shared: Arc<IoShared>) {
    let mut cache = RegionCache::default();
    loop {
        enum Task {
            Save(PathBuf, String, LodSection),
            Load(PathBuf, String, SectionPos),
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
                if st.shutdown {
                    break Task::Exit;
                }
                st = shared.wake_worker.wait(st).unwrap();
            }
        };

        match task {
            Task::Exit => break,
            Task::Save(world_dir, dimension, section) => {
                let res = cache.save_section(&world_dir, &dimension, &section);
                if let Err(ref e) = res {
                    eprintln!(
                        "DH: failed to save section {:?} in {}: {e}",
                        section.pos, dimension
                    );
                }
                let mut st = shared.state.lock().unwrap();
                st.in_flight -= 1;
                if res.is_ok() {
                    st.responses.push(IoResponse::Saved);
                }
                if st.pending_saves.is_empty() && st.pending_loads.is_empty() && st.in_flight == 0 {
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
                if st.pending_saves.is_empty() && st.pending_loads.is_empty() && st.in_flight == 0 {
                    shared.wake_flusher.notify_all();
                }
            }
        }
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

    fn fresh_tmp_dir(label: &str) -> PathBuf {
        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "dh-storage-test-{}-{}-{seq}",
            label,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
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
        let tmp_dir = fresh_tmp_dir("region");

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

        let _ = fs::remove_dir_all(&tmp_dir);
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

        let tmp_dir = fresh_tmp_dir("worker");
        let pos = SectionPos::new(0, 2, -3);
        let worker = LodIoWorker::spawn();
        worker.request_save(
            tmp_dir.clone(),
            "minecraft:overworld".into(),
            sample_section(pos, 70),
        );
        worker.request_save(
            tmp_dir.clone(),
            "minecraft:overworld".into(),
            sample_section(pos, 95),
        );
        worker.flush();

        let mut reader = RegionCache::default();
        let loaded = reader
            .load_section(&tmp_dir, "minecraft:overworld", pos)
            .expect("flushed save present on disk");
        assert_eq!(loaded.get(0, 0).solid_top_y, 95);

        drop(worker);
        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
