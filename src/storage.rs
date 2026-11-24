//! Pure-Rust binary LOD section persistence under `<world_dir>/distant_horizons/<dim>/`.
//!
//! Uses compact run-length encoding over the 4096-column section grid and atomic temp-file
//! replacement on a background IO worker thread so the game and server threads never block on disk.

use crate::lod::{ExtraSpan, LodColumn, LodSection, SECTION_CELLS, SectionPos};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

const MAGIC_V1: &[u8; 8] = b"DHLOD001";
const MAGIC_V2: &[u8; 8] = b"DHLOD002";

/// Encode a `64 x 64` LOD section into compact RLE binary format (`DHLOD002`, primary columns
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

/// Decode a `DHLOD001` or `DHLOD002` binary LOD section file.
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

pub fn section_path(world_dir: &Path, dimension: &str, pos: SectionPos) -> PathBuf {
    world_dir
        .join("distant_horizons")
        .join(sanitize_dimension(dimension))
        .join(format!("d{}", pos.detail))
        .join(format!("s.{}.{}.dhlod", pos.x, pos.z))
}

fn save_section_file(world_dir: &Path, dimension: &str, section: &LodSection) -> io::Result<()> {
    let path = section_path(world_dir, dimension, section.pos);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let encoded = encode_section(section);
    let tmp = path.with_extension("dhlod.tmp");
    {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(&encoded)?;
        file.flush()?;
    }
    fs::rename(&tmp, &path)?;
    Ok(())
}

fn load_section_file(world_dir: &Path, dimension: &str, pos: SectionPos) -> Option<LodSection> {
    let path = section_path(world_dir, dimension, pos);
    let bytes = fs::read(path).ok()?;
    decode_section(&bytes)
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

/// Asynchronous disk IO worker for loading and saving LOD sections.
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
                while let Ok(req) = req_rx.recv() {
                    match req {
                        IoRequest::Save {
                            world_dir,
                            dimension,
                            section,
                        } => {
                            if save_section_file(&world_dir, &dimension, &section).is_ok() {
                                let _ = res_tx.send(IoResponse::Saved);
                            }
                        }
                        IoRequest::Load {
                            world_dir,
                            dimension,
                            pos,
                        } => {
                            if let Some(section) = load_section_file(&world_dir, &dimension, pos) {
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

    #[test]
    fn rle_section_roundtrip_is_exact_and_compact() {
        let pos = SectionPos::new(1, -3, 7);
        let mut section = LodSection::new(pos);
        let grass = LodColumn::from_solid_and_fluid(
            68,
            60,
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
        let encoded = encode_section(&section);
        // 64 rows * 2 runs * 18 bytes + 17 header = ~2321 bytes (vs 65536 raw!)
        assert!(encoded.len() < 4096);
        let decoded = decode_section(&encoded).expect("valid encoded section");
        assert_eq!(decoded.pos, pos);
        assert_eq!(decoded.columns, section.columns);
        assert_eq!(decoded.populated, 4096);
        assert!(decoded.extra_spans.is_empty());

        // Verify extra vertical spans (overhangs/caverns) round-trip exactly in DHLOD002.
        let floor = ExtraSpan::new(36, 28, [96, 40, 40], 0, 12, false);
        section.set_cell_with_spans(5, 9, grass, vec![floor]);
        let encoded_v2 = encode_section(&section);
        let decoded_v2 = decode_section(&encoded_v2).expect("valid v2 section");
        assert_eq!(decoded_v2.extra_spans_at(5, 9), &[floor]);
    }
}
