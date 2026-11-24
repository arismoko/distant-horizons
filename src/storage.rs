//! Fast region-packed Zstd binary LOD storage (`DHREG001`) under
//! `<world_dir>/distant_horizons/<dim>/d<detail>/r.<rx>.<rz>.dhreg`.
//!
//! Architecture:
//! - Packs `16 × 16 = 256` LOD sections per region file, cutting filesystem metadata/inode
//!   overhead by up to 256× compared to one-file-per-section storage while avoiding SQLite's
//!   SQL/B-tree/FFI overhead.
//! - Each slot holds a CRC32-checked Zstd-compressed `DHLOD002` RLE section payload.
//! - Copy-on-write append updates: new/updated section payloads are appended at EOF and flushed
//!   before updating the 12-byte slot index entry, so an abrupt crash never corrupts existing
//!   sections. Regions compact atomically when dead space exceeds 2× live payload size.
//! - The background IO worker caches open region indices so hot reads are a single `seek` + `read`
//!   with zero directory lookups. Legacy `s.<x>.<z>.dhlod` files remain readable transparently.

use crate::lod::{ExtraSpan, LodColumn, LodSection, SECTION_CELLS, SectionPos};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

const MAGIC_V1: &[u8; 8] = b"DHLOD001";
const MAGIC_V2: &[u8; 8] = b"DHLOD002";
const REGION_MAGIC: &[u8; 8] = b"DHREG001";

/// `16 × 16` sections per `.dhreg` region file.
pub const REGION_SHIFT: i32 = 4;
pub const REGION_SIZE: usize = 1 << REGION_SHIFT;
pub const REGION_SLOTS: usize = REGION_SIZE * REGION_SIZE;
const SLOT_ENTRY_BYTES: usize = 12; // u64 offset + u32 payload_len (including 4-byte CRC32)
const REGION_HEADER_BYTES: usize = 8 + REGION_SLOTS * SLOT_ENTRY_BYTES;
const MAX_DECOMPRESSED_SECTION_BYTES: usize = 4 * 1024 * 1024;
const MAX_CACHED_REGIONS: usize = 32;

/// Encode a `64 × 64` LOD section into compact RLE binary format (`DHLOD002`, primary columns
/// plus optional extra vertical spans).
pub fn encode_section(section: &LodSection) -> Vec<u8> {
    let mut out = Vec::with_capacity(2048);
    out.extend_from_slice(MAGIC_V2);
    out.push(section.pos.detail);
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

    // Extra vertical spans tail (deterministic cell-index order).
    let mut keys: Vec<u16> = section
        .extra_spans
        .iter()
        .filter(|(_, spans)| !spans.is_empty())
        .map(|(&idx, _)| idx)
        .collect();
    keys.sort_unstable();
    out.extend_from_slice(&(keys.len() as u16).to_le_bytes());
    for cell_idx in keys {
        let spans = &section.extra_spans[&cell_idx];
        let count = spans.len().min(u8::MAX as usize) as u8;
        out.extend_from_slice(&cell_idx.to_le_bytes());
        out.push(count);
        for span in &spans[..count as usize] {
            out.extend_from_slice(&span.to_bytes());
        }
    }
    out
}

/// Decode a `DHLOD001` or `DHLOD002` binary LOD section payload.
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
    let x = i32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]);
    let z = i32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]);
    let pos = SectionPos::new(detail, x, z);

    let mut columns = Vec::with_capacity(SECTION_CELLS);
    let mut cursor = 17;
    while cursor + 18 <= bytes.len() && columns.len() < SECTION_CELLS {
        let run = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        let mut col_bytes = [0u8; 16];
        col_bytes.copy_from_slice(&bytes[cursor + 2..cursor + 18]);
        let col = LodColumn::from_bytes(col_bytes);
        cursor += 18;
        let remaining = SECTION_CELLS - columns.len();
        let count = run.min(remaining);
        columns.resize(columns.len() + count, col);
    }
    if columns.len() != SECTION_CELLS {
        return None;
    }

    let mut extra_spans = HashMap::new();
    if version >= 2 && cursor + 2 <= bytes.len() {
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

    Some(LodSection::from_columns_and_spans(
        pos,
        columns,
        extra_spans,
    ))
}

/// Compress an encoded `DHLOD002` payload with Zstd (fastest level) + leading CRC32 checksum.
pub fn compress_section_record(section: &LodSection) -> Vec<u8> {
    let raw = encode_section(section);
    let compressed = ruzstd::encoding::compress_to_vec(
        raw.as_slice(),
        ruzstd::encoding::CompressionLevel::Fastest,
    );
    let crc = crc32fast::hash(&compressed);
    let mut record = Vec::with_capacity(4 + compressed.len());
    record.extend_from_slice(&crc.to_le_bytes());
    record.extend_from_slice(&compressed);
    record
}

/// Verify CRC32, decompress Zstd, and decode a stored region record.
pub fn decompress_section_record(record: &[u8]) -> Option<LodSection> {
    if record.len() < 8 {
        return None;
    }
    let expected_crc = u32::from_le_bytes([record[0], record[1], record[2], record[3]]);
    let compressed = &record[4..];
    if crc32fast::hash(compressed) != expected_crc {
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

pub fn sanitize_dimension(dimension: &str) -> String {
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
        .join(sanitize_dimension(dimension))
        .join(format!("d{}", pos.detail))
        .join(format!("s.{}.{}.dhlod", pos.x, pos.z))
}

#[derive(Clone, Copy, Default)]
struct SlotEntry {
    offset: u64,
    len: u32,
}

struct OpenRegion {
    path: PathBuf,
    file: File,
    slots: [SlotEntry; REGION_SLOTS],
    file_len: u64,
    live_bytes: u64,
    last_used: u64,
}

impl OpenRegion {
    fn open_or_create(path: &Path, create: bool, tick: u64) -> io::Result<Option<Self>> {
        if !path.exists() {
            if !create {
                return Ok(None);
            }
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            let mut header = vec![0u8; REGION_HEADER_BYTES];
            header[..8].copy_from_slice(REGION_MAGIC);
            file.write_all(&header)?;
            file.flush()?;
            return Ok(Some(Self {
                path: path.to_path_buf(),
                file,
                slots: [SlotEntry::default(); REGION_SLOTS],
                file_len: REGION_HEADER_BYTES as u64,
                live_bytes: 0,
                last_used: tick,
            }));
        }

        let mut file = OpenOptions::new().read(true).write(create).open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < REGION_HEADER_BYTES as u64 {
            if !create {
                return Ok(None);
            }
            let mut header = vec![0u8; REGION_HEADER_BYTES];
            header[..8].copy_from_slice(REGION_MAGIC);
            file.seek(SeekFrom::Start(0))?;
            file.set_len(REGION_HEADER_BYTES as u64)?;
            file.write_all(&header)?;
            file.flush()?;
            return Ok(Some(Self {
                path: path.to_path_buf(),
                file,
                slots: [SlotEntry::default(); REGION_SLOTS],
                file_len: REGION_HEADER_BYTES as u64,
                live_bytes: 0,
                last_used: tick,
            }));
        }

        let mut header = vec![0u8; REGION_HEADER_BYTES];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header)?;
        if &header[..8] != REGION_MAGIC {
            return Ok(None);
        }
        let mut slots = [SlotEntry::default(); REGION_SLOTS];
        let mut live_bytes = 0u64;
        for (i, slot) in slots.iter_mut().enumerate() {
            let base = 8 + i * SLOT_ENTRY_BYTES;
            let offset = u64::from_le_bytes([
                header[base],
                header[base + 1],
                header[base + 2],
                header[base + 3],
                header[base + 4],
                header[base + 5],
                header[base + 6],
                header[base + 7],
            ]);
            let len = u32::from_le_bytes([
                header[base + 8],
                header[base + 9],
                header[base + 10],
                header[base + 11],
            ]);
            if len > 0 && offset >= REGION_HEADER_BYTES as u64 && offset + (len as u64) <= file_len
            {
                *slot = SlotEntry { offset, len };
                live_bytes += len as u64;
            }
        }

        Ok(Some(Self {
            path: path.to_path_buf(),
            file,
            slots,
            file_len,
            live_bytes,
            last_used: tick,
        }))
    }

    fn read_slot(&mut self, slot_idx: usize) -> Option<LodSection> {
        let entry = self.slots[slot_idx];
        if entry.len == 0 || entry.offset < REGION_HEADER_BYTES as u64 {
            return None;
        }
        let mut buf = vec![0u8; entry.len as usize];
        self.file.seek(SeekFrom::Start(entry.offset)).ok()?;
        self.file.read_exact(&mut buf).ok()?;
        decompress_section_record(&buf)
    }

    /// Copy-on-write append of `record` to EOF followed by slot-table update; triggers atomic
    /// compaction if dead space exceeds 2× live bytes (and > 64 KiB dead).
    fn write_slot(&mut self, slot_idx: usize, record: &[u8]) -> io::Result<()> {
        let new_offset = self.file_len;
        let new_len = record.len() as u32;

        // 1. Append payload at EOF so any crash leaves the previous slot intact.
        self.file.seek(SeekFrom::Start(new_offset))?;
        self.file.write_all(record)?;
        self.file.flush()?;

        // 2. Commit the 12-byte slot table entry in the header.
        let prev = self.slots[slot_idx];
        let header_pos = (8 + slot_idx * SLOT_ENTRY_BYTES) as u64;
        let mut entry_bytes = [0u8; SLOT_ENTRY_BYTES];
        entry_bytes[..8].copy_from_slice(&new_offset.to_le_bytes());
        entry_bytes[8..12].copy_from_slice(&new_len.to_le_bytes());
        self.file.seek(SeekFrom::Start(header_pos))?;
        self.file.write_all(&entry_bytes)?;
        self.file.flush()?;

        self.slots[slot_idx] = SlotEntry {
            offset: new_offset,
            len: new_len,
        };
        self.file_len = new_offset + (new_len as u64);
        self.live_bytes = self
            .live_bytes
            .saturating_sub(prev.len as u64)
            .saturating_add(new_len as u64);

        // 3. Compact atomically when stale overwritten payloads exceed 2× live data.
        let data_bytes = self.file_len.saturating_sub(REGION_HEADER_BYTES as u64);
        let dead_bytes = data_bytes.saturating_sub(self.live_bytes);
        if dead_bytes > 64 * 1024 && dead_bytes > self.live_bytes.saturating_mul(2) {
            self.compact()?;
        }
        Ok(())
    }

    fn compact(&mut self) -> io::Result<()> {
        let tmp_path = self.path.with_extension("dhreg.tmp");
        let mut new_slots = [SlotEntry::default(); REGION_SLOTS];
        let mut payload_buf = Vec::new();
        let mut cur_offset = REGION_HEADER_BYTES as u64;
        let mut new_live = 0u64;

        for (idx, entry) in self.slots.iter().enumerate() {
            if entry.len == 0 {
                continue;
            }
            let mut rec = vec![0u8; entry.len as usize];
            self.file.seek(SeekFrom::Start(entry.offset))?;
            if self.file.read_exact(&mut rec).is_ok() {
                new_slots[idx] = SlotEntry {
                    offset: cur_offset,
                    len: entry.len,
                };
                payload_buf.extend_from_slice(&rec);
                cur_offset += entry.len as u64;
                new_live += entry.len as u64;
            }
        }

        let mut header = vec![0u8; REGION_HEADER_BYTES];
        header[..8].copy_from_slice(REGION_MAGIC);
        for (idx, slot) in new_slots.iter().enumerate() {
            let base = 8 + idx * SLOT_ENTRY_BYTES;
            header[base..base + 8].copy_from_slice(&slot.offset.to_le_bytes());
            header[base + 8..base + 12].copy_from_slice(&slot.len.to_le_bytes());
        }

        {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&header)?;
            tmp.write_all(&payload_buf)?;
            tmp.flush()?;
        }
        fs::rename(&tmp_path, &self.path)?;

        self.file = OpenOptions::new().read(true).write(true).open(&self.path)?;
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
    fn get_region(&mut self, path: &Path, create: bool) -> io::Result<Option<&mut OpenRegion>> {
        self.tick = self.tick.wrapping_add(1);
        let now = self.tick;

        if !self.regions.contains_key(path) {
            if self.regions.len() >= MAX_CACHED_REGIONS
                && let Some(evict_key) = self
                    .regions
                    .iter()
                    .min_by_key(|(_, r)| r.last_used)
                    .map(|(k, _)| k.clone())
            {
                self.regions.remove(&evict_key);
            }

            // Always open existing region files read+write so later saves reuse the same handle.
            let open_rw = create || path.exists();
            let Some(reg) = OpenRegion::open_or_create(path, open_rw, now)? else {
                return Ok(None);
            };
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
        let record = compress_section_record(section);
        let Some(reg) = self.get_region(&path, true)? else {
            return Err(io::Error::other("failed to open region file"));
        };
        reg.write_slot(slot_idx, &record)
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
            && let Some(sec) = reg.read_slot(slot_idx)
        {
            return Some(sec);
        }
        // Fallback: seamlessly read legacy single-section `.dhlod` files if present.
        let legacy = legacy_section_path(world_dir, dimension, pos);
        let bytes = fs::read(legacy).ok()?;
        decode_section(&bytes)
    }
}

pub enum IoRequest {
    Save {
        world_dir: PathBuf,
        dimension: String,
        section: LodSection,
    },
    Load {
        world_dir: PathBuf,
        dimension: String,
        pos: SectionPos,
    },
}

pub enum IoResponse {
    Loaded {
        dimension: String,
        section: LodSection,
    },
    Saved,
}

/// Asynchronous disk IO worker for loading and saving LOD sections via region-packed `.dhreg` files.
pub struct LodIoWorker {
    tx: Sender<IoRequest>,
    rx: Mutex<Receiver<IoResponse>>,
}

impl LodIoWorker {
    pub fn spawn() -> Self {
        let (req_tx, req_rx) = mpsc::channel::<IoRequest>();
        let (res_tx, res_rx) = mpsc::channel::<IoResponse>();
        let _ = thread::Builder::new()
            .name("dh-lod-io".into())
            .spawn(move || {
                let mut cache = RegionCache::default();
                while let Ok(req) = req_rx.recv() {
                    match req {
                        IoRequest::Save {
                            world_dir,
                            dimension,
                            section,
                        } => {
                            if cache.save_section(&world_dir, &dimension, &section).is_ok() {
                                let _ = res_tx.send(IoResponse::Saved);
                            }
                        }
                        IoRequest::Load {
                            world_dir,
                            dimension,
                            pos,
                        } => {
                            if let Some(section) = cache.load_section(&world_dir, &dimension, pos) {
                                let _ = res_tx.send(IoResponse::Loaded { dimension, section });
                            }
                        }
                    }
                }
            });
        Self {
            tx: req_tx,
            rx: Mutex::new(res_rx),
        }
    }

    pub fn request_save(&self, world_dir: PathBuf, dimension: String, section: LodSection) {
        let _ = self.tx.send(IoRequest::Save {
            world_dir,
            dimension,
            section,
        });
    }

    pub fn request_load(&self, world_dir: PathBuf, dimension: String, pos: SectionPos) {
        let _ = self.tx.send(IoRequest::Load {
            world_dir,
            dimension,
            pos,
        });
    }

    pub fn drain_responses(&self) -> Vec<IoResponse> {
        let mut out = Vec::new();
        if let Ok(rx) = self.rx.lock() {
            while let Ok(res) = rx.try_recv() {
                out.push(res);
            }
        }
        out
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
    fn rle_section_roundtrip_is_exact_and_compact() {
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
    }

    #[test]
    fn region_store_packs_multiple_sections_with_zstd_and_crc() {
        let tmp_dir = std::env::temp_dir().join(format!("dh-region-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir).unwrap();

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

        // First two share r.0.0.dhreg; last two share r.-1.-1.dhreg (2 files total for 4 sections).
        assert!(region_path(&tmp_dir, "minecraft:overworld", positions[0]).exists());
        assert_eq!(
            region_path(&tmp_dir, "minecraft:overworld", positions[0]),
            region_path(&tmp_dir, "minecraft:overworld", positions[1])
        );
        assert_eq!(
            region_path(&tmp_dir, "minecraft:overworld", positions[2]),
            region_path(&tmp_dir, "minecraft:overworld", positions[3])
        );

        // Reload via fresh RegionCache (cold open from disk) and verify exact round-trip.
        let mut reader = RegionCache::default();
        for (i, &pos) in positions.iter().enumerate() {
            let loaded = reader
                .load_section(&tmp_dir, "minecraft:overworld", pos)
                .expect("section stored in region");
            assert_eq!(loaded.pos, pos);
            assert_eq!(loaded.get(0, 0).solid_top_y, 64 + i as i16);
        }

        // Overwrite an existing slot and verify the updated value wins.
        let updated = sample_section(positions[0], 120);
        cache
            .save_section(&tmp_dir, "minecraft:overworld", &updated)
            .unwrap();
        let mut reader2 = RegionCache::default();
        let loaded = reader2
            .load_section(&tmp_dir, "minecraft:overworld", positions[0])
            .unwrap();
        assert_eq!(loaded.get(0, 0).solid_top_y, 120);

        // Corrupt a CRC byte and confirm `decompress_section_record` rejects it safely.
        let mut rec = compress_section_record(&updated);
        rec[5] ^= 0xFF;
        assert!(decompress_section_record(&rec).is_none());

        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
