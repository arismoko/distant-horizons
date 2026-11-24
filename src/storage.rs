//! Pure-Rust binary LOD section persistence under `<world_dir>/distant_horizons/<dim>/`.
//!
//! Uses compact run-length encoding over the 4096-column section grid and atomic temp-file
//! replacement on a background IO worker thread so the game and server threads never block on disk.

use crate::lod::{LodColumn, LodSection, SECTION_CELLS, SectionPos};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

const MAGIC: &[u8; 8] = b"DHLOD001";

/// Encode a `64 x 64` LOD section into compact RLE binary format.
pub fn encode_section(section: &LodSection) -> Vec<u8> {
    let mut out = Vec::with_capacity(2048);
    out.extend_from_slice(MAGIC);
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
    out
}

/// Decode a binary LOD section file.
pub fn decode_section(bytes: &[u8]) -> Option<LodSection> {
    if bytes.len() < 17 || &bytes[0..8] != MAGIC {
        return None;
    }
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
    Some(LodSection::from_columns(pos, columns))
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
    }
}
