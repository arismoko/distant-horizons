//! Active resource-pack block color and biome colormap sampler for Distant Sunrises.
//!
//! Reads `textures/block/*.png`, `textures/colormap/grass.png`, and
//! `textures/colormap/foliage.png` from Sunrise's active client resource-pack stack
//! (`host.resource(PackType::ClientResources, ...)`) on world entry and resource reload
//! (`client_resources_reloaded`), overriding the static vanilla fallback palette so resource
//! packs tint distant LOD terrain accurately.
//!
//! Uses a self-contained RFC 1950/1951 DEFLATE + PNG decoder so no extra crates are required.

use mod_api::{Identifier, PackType, ReloadContext};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Decoded RGBA8 image.
#[derive(Clone, Debug)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl DecodedImage {
    /// Sample pixel `(x, y)` clamped to image bounds.
    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let cx = x.min(self.width.saturating_sub(1)) as usize;
        let cy = y.min(self.height.saturating_sub(1)) as usize;
        let idx = (cy * self.width as usize + cx) * 4;
        [
            self.rgba[idx],
            self.rgba[idx + 1],
            self.rgba[idx + 2],
            self.rgba[idx + 3],
        ]
    }

    /// Compute the alpha-weighted average RGB color of all pixels with `alpha >= 32`
    /// (falling back to unweighted mean if all pixels are translucent).
    pub fn average_opaque_rgb(&self) -> Option<[u8; 3]> {
        if self.width == 0 || self.height == 0 || self.rgba.len() < 4 {
            return None;
        }
        let mut r_sum = 0u64;
        let mut g_sum = 0u64;
        let mut b_sum = 0u64;
        let mut w_sum = 0u64;

        for px in self.rgba.as_chunks::<4>().0 {
            let a = px[3] as u64;
            if a >= 32 {
                r_sum += px[0] as u64 * a;
                g_sum += px[1] as u64 * a;
                b_sum += px[2] as u64 * a;
                w_sum += a;
            }
        }
        if w_sum == 0 {
            let count = (self.rgba.len() / 4) as u64;
            for px in self.rgba.as_chunks::<4>().0 {
                r_sum += px[0] as u64;
                g_sum += px[1] as u64;
                b_sum += px[2] as u64;
            }
            return Some([
                (r_sum / count) as u8,
                (g_sum / count) as u8,
                (b_sum / count) as u8,
            ]);
        }
        Some([
            (r_sum / w_sum) as u8,
            (g_sum / w_sum) as u8,
            (b_sum / w_sum) as u8,
        ])
    }
}

/// Active resource-pack color overrides (thread-safe read from server/mesher threads).
#[derive(Default)]
pub struct ResourceColorPack {
    /// Block name without namespace (e.g. `"stone"`, `"oak_planks"`) -> average texture RGB.
    pub block_rgb: HashMap<String, [u8; 3]>,
    /// Active 256×256 grass colormap (`textures/colormap/grass.png`), if loaded.
    pub grass_colormap: Option<DecodedImage>,
    /// Active 256×256 foliage colormap (`textures/colormap/foliage.png`), if loaded.
    pub foliage_colormap: Option<DecodedImage>,
    pub loaded: bool,
}

static PACK_COLORS: OnceLock<RwLock<ResourceColorPack>> = OnceLock::new();

fn pack_store() -> &'static RwLock<ResourceColorPack> {
    PACK_COLORS.get_or_init(|| RwLock::new(ResourceColorPack::default()))
}

/// Look up an active resource-pack RGB override for a block name (stripped of `minecraft:`).
pub fn lookup_pack_block_rgb(name: &str) -> Option<[u8; 3]> {
    let guard = pack_store().read().ok()?;
    if !guard.loaded {
        return None;
    }
    guard.block_rgb.get(name).copied()
}

/// Sample vanilla's 256×256 grass/foliage colormap at `(temperature, downfall)` in `[0, 1]`.
pub fn sample_colormap(grass: bool, temperature: f32, downfall: f32) -> Option<[u8; 3]> {
    let guard = pack_store().read().ok()?;
    let img = if grass {
        guard.grass_colormap.as_ref()?
    } else {
        guard.foliage_colormap.as_ref()?
    };
    if img.width < 2 || img.height < 2 {
        return None;
    }
    let t = temperature.clamp(0.0, 1.0);
    let d = downfall.clamp(0.0, 1.0) * t;
    let x = ((1.0 - t) * (img.width - 1) as f32).round() as u32;
    let y = ((1.0 - d) * (img.height - 1) as f32).round() as u32;
    let px = img.pixel(x, y);
    Some([px[0], px[1], px[2]])
}

/// Number of block textures currently loaded from the active resource pack.
pub fn loaded_block_color_count() -> usize {
    pack_store().read().map(|g| g.block_rgb.len()).unwrap_or(0)
}

fn read_resource_bytes(host: &dyn ReloadContext, id: &Identifier) -> Option<Vec<u8>> {
    let mut reader = host.open_resource(PackType::ClientResources, id).ok()??;
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut bytes).ok()?;
    Some(bytes)
}

/// Reload block texture averages and grass/foliage colormaps from Sunrise's active client
/// resource-pack stack. Safe to call in `client_setup` and `client_resources_reloaded`.
pub fn reload_from_resources(host: &dyn ReloadContext) -> usize {
    let mut next = ResourceColorPack {
        loaded: true,
        ..ResourceColorPack::default()
    };

    if let Ok(id) = Identifier::parse("minecraft:textures/colormap/grass.png")
        && let Some(bytes) = read_resource_bytes(host, &id)
        && let Ok(img) = decode_png(&bytes)
    {
        next.grass_colormap = Some(img);
    }
    if let Ok(id) = Identifier::parse("minecraft:textures/colormap/foliage.png")
        && let Some(bytes) = read_resource_bytes(host, &id)
        && let Ok(img) = decode_png(&bytes)
    {
        next.foliage_colormap = Some(img);
    }

    let textures = host.list_resources(PackType::ClientResources, "textures/block", ".png");
    let mut raw_textures: HashMap<String, [u8; 3]> = HashMap::with_capacity(textures.len());
    for tex_id in textures {
        let Some(bytes) = read_resource_bytes(host, &tex_id) else {
            continue;
        };
        let Ok(img) = decode_png(&bytes) else {
            continue;
        };
        let Some(rgb) = img.average_opaque_rgb() else {
            continue;
        };
        let path = tex_id.path();
        let stem = path
            .strip_prefix("textures/block/")
            .unwrap_or(path)
            .strip_suffix(".png")
            .unwrap_or(path);
        if tex_id.namespace() == "minecraft" {
            raw_textures.insert(stem.to_owned(), rgb);
        } else {
            raw_textures.insert(format!("{}:{stem}", tex_id.namespace()), rgb);
        }
    }

    // Map top-face / primary texture stems to block names (preferring `*_top` over base name so
    // logs, grass, sandstone, pumpkins, hay, snowy blocks etc. match their aerial top appearance).
    for (stem, &rgb) in &raw_textures {
        next.block_rgb.insert(stem.clone(), rgb);
        if let Some(base) = stem.strip_suffix("_top") {
            next.block_rgb.insert(base.to_owned(), rgb);
        }
    }

    // Explicit vanilla top-texture aliases where texture stem differs from block ID.
    const ALIASES: &[(&str, &str)] = &[
        ("grass_block", "grass_block_top"),
        ("podzol", "podzol_top"),
        ("mycelium", "mycelium_top"),
        ("crimson_nylium", "crimson_nylium"),
        ("warped_nylium", "warped_nylium"),
        ("snow_block", "snow"),
        ("sandstone", "sandstone_top"),
        ("red_sandstone", "red_sandstone_top"),
        ("smooth_sandstone", "sandstone_top"),
        ("smooth_red_sandstone", "red_sandstone_top"),
        ("quartz_block", "quartz_block_top"),
        ("hay_block", "hay_block_top"),
        ("pumpkin", "pumpkin_top"),
        ("carved_pumpkin", "pumpkin_top"),
        ("jack_o_lantern", "pumpkin_top"),
        ("melon", "melon_top"),
        ("cactus", "cactus_top"),
        ("basalt", "basalt_top"),
        ("polished_basalt", "polished_basalt_top"),
        ("dirt_path", "dirt_path_top"),
        ("farmland", "farmland"),
        ("bone_block", "bone_block_top"),
        ("tnt", "tnt_top"),
        ("crafting_table", "crafting_table_top"),
        ("barrel", "barrel_top"),
    ];
    for &(block, tex) in ALIASES {
        if let Some(&rgb) = raw_textures.get(tex) {
            next.block_rgb.insert(block.to_owned(), rgb);
        }
    }

    let count = next.block_rgb.len();
    if let Ok(mut guard) = pack_store().write() {
        *guard = next;
    }
    count
}

// ============================================================================
// Self-contained PNG + RFC 1950/1951 DEFLATE decoder (zero external crates)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PngError {
    BadSignature,
    Truncated,
    UnsupportedFormat,
    BadDeflate,
    TooLarge,
}

const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
/// Guard against corrupted/bomb textures while allowing up to 4096×4096 resource-pack sheets.
const MAX_PNG_DIM: u32 = 4096;
const MAX_DECODED_BYTES: usize = 32 * 1024 * 1024;

/// Decode a standard non-interlaced 8-bit PNG (grayscale, RGB, indexed, gray+alpha, RGBA) into
/// RGBA8 pixels.
pub fn decode_png(data: &[u8]) -> Result<DecodedImage, PngError> {
    if data.len() < 33 || data[..8] != PNG_SIG {
        return Err(PngError::BadSignature);
    }

    let mut pos = 8usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut bit_depth = 0u8;
    let mut color_type = 0u8;
    let mut interlace = 0u8;
    let mut seen_ihdr = false;
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut trns: Vec<u8> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();

    while pos + 12 <= data.len() {
        let len =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let kind = &data[pos + 4..pos + 8];
        let body_start = pos + 8;
        let body_end = body_start.checked_add(len).ok_or(PngError::Truncated)?;
        if body_end + 4 > data.len() {
            return Err(PngError::Truncated);
        }
        let body = &data[body_start..body_end];
        pos = body_end + 4;

        match kind {
            b"IHDR" => {
                if body.len() != 13 {
                    return Err(PngError::Truncated);
                }
                width = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                height = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
                bit_depth = body[8];
                color_type = body[9];
                interlace = body[12];
                seen_ihdr = true;
            }
            b"PLTE" => {
                for &rgb in body.as_chunks::<3>().0 {
                    palette.push(rgb);
                }
            }
            b"tRNS" => {
                trns.extend_from_slice(body);
            }
            b"IDAT" => {
                idat.extend_from_slice(body);
            }
            b"IEND" => break,
            _ => {}
        }
    }

    if !seen_ihdr || width == 0 || height == 0 || width > MAX_PNG_DIM || height > MAX_PNG_DIM {
        return Err(PngError::UnsupportedFormat);
    }
    if interlace != 0
        || (bit_depth != 8
            && !(color_type == 3 && (bit_depth == 1 || bit_depth == 2 || bit_depth == 4)))
    {
        return Err(PngError::UnsupportedFormat);
    }

    let bpp_channels: usize = match color_type {
        0 => 1, // Grayscale
        2 => 3, // RGB
        3 => 1, // Indexed
        4 => 2, // Grayscale + Alpha
        6 => 4, // RGBA
        _ => return Err(PngError::UnsupportedFormat),
    };
    let row_bytes = if bit_depth == 8 {
        (width as usize) * bpp_channels
    } else {
        ((width as usize) * (bit_depth as usize)).div_ceil(8)
    };
    let filter_bpp = if bit_depth == 8 { bpp_channels } else { 1 };
    let expected_raw = (row_bytes + 1)
        .checked_mul(height as usize)
        .ok_or(PngError::TooLarge)?;
    if expected_raw > MAX_DECODED_BYTES {
        return Err(PngError::TooLarge);
    }

    let raw = inflate_zlib(&idat, expected_raw)?;
    if raw.len() < expected_raw {
        return Err(PngError::Truncated);
    }

    let mut prev_row = vec![0u8; row_bytes];
    let mut curr_row = vec![0u8; row_bytes];
    let mut rgba = Vec::with_capacity((width as usize) * (height as usize) * 4);

    for y in 0..height as usize {
        let src = &raw[y * (row_bytes + 1)..(y + 1) * (row_bytes + 1)];
        let filter = src[0];
        let filt = &src[1..];
        match filter {
            0 => curr_row.copy_from_slice(filt),
            1 => {
                for i in 0..row_bytes {
                    let left = if i >= filter_bpp {
                        curr_row[i - filter_bpp]
                    } else {
                        0
                    };
                    curr_row[i] = filt[i].wrapping_add(left);
                }
            }
            2 => {
                for i in 0..row_bytes {
                    curr_row[i] = filt[i].wrapping_add(prev_row[i]);
                }
            }
            3 => {
                for i in 0..row_bytes {
                    let left = if i >= filter_bpp {
                        curr_row[i - filter_bpp] as u16
                    } else {
                        0
                    };
                    let up = prev_row[i] as u16;
                    curr_row[i] = filt[i].wrapping_add(((left + up) / 2) as u8);
                }
            }
            4 => {
                for i in 0..row_bytes {
                    let a = if i >= filter_bpp {
                        curr_row[i - filter_bpp]
                    } else {
                        0
                    };
                    let b = prev_row[i];
                    let c = if i >= filter_bpp {
                        prev_row[i - filter_bpp]
                    } else {
                        0
                    };
                    curr_row[i] = filt[i].wrapping_add(paeth(a, b, c));
                }
            }
            _ => return Err(PngError::UnsupportedFormat),
        }

        match color_type {
            0 => {
                for &g in &curr_row {
                    rgba.extend_from_slice(&[g, g, g, 255]);
                }
            }
            2 => {
                for px in curr_row.as_chunks::<3>().0 {
                    rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
                }
            }
            3 => {
                for x in 0..width as usize {
                    let idx = match bit_depth {
                        8 => curr_row[x] as usize,
                        4 => {
                            let b = curr_row[x / 2];
                            (if x % 2 == 0 { b >> 4 } else { b & 0xF }) as usize
                        }
                        2 => {
                            let b = curr_row[x / 4];
                            ((b >> (6 - 2 * (x % 4))) & 0x3) as usize
                        }
                        1 => {
                            let b = curr_row[x / 8];
                            ((b >> (7 - (x % 8))) & 0x1) as usize
                        }
                        _ => 0,
                    };
                    let rgb = palette.get(idx).copied().unwrap_or([0, 0, 0]);
                    let a = trns.get(idx).copied().unwrap_or(255);
                    rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], a]);
                }
            }
            4 => {
                for px in curr_row.as_chunks::<2>().0 {
                    rgba.extend_from_slice(&[px[0], px[0], px[0], px[1]]);
                }
            }
            6 => {
                rgba.extend_from_slice(&curr_row);
            }
            _ => return Err(PngError::UnsupportedFormat),
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    Ok(DecodedImage {
        width,
        height,
        rgba,
    })
}

#[inline]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let ia = a as i32;
    let ib = b as i32;
    let ic = c as i32;
    let p = ia + ib - ic;
    let pa = (p - ia).abs();
    let pb = (p - ib).abs();
    let pc = (p - ic).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Inflate an RFC 1950 zlib stream containing RFC 1951 DEFLATE blocks.
fn inflate_zlib(data: &[u8], max_out: usize) -> Result<Vec<u8>, PngError> {
    if data.len() < 6 {
        return Err(PngError::BadDeflate);
    }
    let cmf = data[0];
    let flg = data[1];
    if (cmf & 0x0F) != 8 || !(((cmf as u16) << 8) | (flg as u16)).is_multiple_of(31) {
        return Err(PngError::BadDeflate);
    }
    let mut start = 2usize;
    if (flg & 0x20) != 0 {
        start += 4; // FDICT
    }
    if start >= data.len() {
        return Err(PngError::BadDeflate);
    }
    inflate_raw(&data[start..], max_out)
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u32,
    bit_cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    fn ensure(&mut self, n: u32) -> Result<(), PngError> {
        while self.bit_cnt < n {
            if self.byte_pos >= self.data.len() {
                return Err(PngError::BadDeflate);
            }
            self.bit_buf |= (self.data[self.byte_pos] as u32) << self.bit_cnt;
            self.byte_pos += 1;
            self.bit_cnt += 8;
        }
        Ok(())
    }

    fn read_bits(&mut self, n: u32) -> Result<u32, PngError> {
        if n == 0 {
            return Ok(0);
        }
        self.ensure(n)?;
        let val = self.bit_buf & ((1u32 << n) - 1);
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    fn align_byte(&mut self) {
        let drop = self.bit_cnt & 7;
        self.bit_buf >>= drop;
        self.bit_cnt -= drop;
    }

    fn read_aligned_u16(&mut self) -> Result<u16, PngError> {
        self.align_byte();
        let lo = self.read_bits(8)? as u16;
        let hi = self.read_bits(8)? as u16;
        Ok(lo | (hi << 8))
    }

    fn read_aligned_bytes(&mut self, len: usize, out: &mut Vec<u8>) -> Result<(), PngError> {
        self.align_byte();
        // Drain any buffered full bytes first.
        let mut remaining = len;
        while remaining > 0 && self.bit_cnt >= 8 {
            out.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf >>= 8;
            self.bit_cnt -= 8;
            remaining -= 1;
        }
        if self.byte_pos + remaining > self.data.len() {
            return Err(PngError::BadDeflate);
        }
        out.extend_from_slice(&self.data[self.byte_pos..self.byte_pos + remaining]);
        self.byte_pos += remaining;
        Ok(())
    }
}

struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn from_lengths(lengths: &[u8]) -> Result<Self, PngError> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if (l as usize) >= 16 {
                return Err(PngError::BadDeflate);
            }
            if l > 0 {
                counts[l as usize] += 1;
            }
        }
        let mut offsets = [0usize; 16];
        let mut total = 0usize;
        for bits in 1..16 {
            offsets[bits] = total;
            total += counts[bits] as usize;
        }
        let mut symbols = vec![0u16; total];
        for (sym, &l) in lengths.iter().enumerate() {
            if l > 0 {
                let idx = offsets[l as usize];
                symbols[idx] = sym as u16;
                offsets[l as usize] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16, PngError> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0usize;
        for len in 1..16 {
            code |= br.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                let pos = index + (code - first) as usize;
                return self.symbols.get(pos).copied().ok_or(PngError::BadDeflate);
            }
            index += count as usize;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(PngError::BadDeflate)
    }
}

fn inflate_raw(data: &[u8], max_out: usize) -> Result<Vec<u8>, PngError> {
    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u32; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u32; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];
    const CL_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    let mut br = BitReader::new(data);
    let mut out = Vec::with_capacity(max_out.min(65536));

    loop {
        let bfinal = br.read_bits(1)?;
        let btype = br.read_bits(2)?;
        match btype {
            0 => {
                let len = br.read_aligned_u16()? as usize;
                let nlen = br.read_aligned_u16()?;
                if (len as u16) != !nlen {
                    return Err(PngError::BadDeflate);
                }
                if out.len() + len > max_out {
                    return Err(PngError::TooLarge);
                }
                br.read_aligned_bytes(len, &mut out)?;
            }
            1 | 2 => {
                let (lit_tree, dist_tree) = if btype == 1 {
                    let mut lit_lens = [0u8; 288];
                    lit_lens[..144].fill(8);
                    lit_lens[144..256].fill(9);
                    lit_lens[256..280].fill(7);
                    lit_lens[280..288].fill(8);
                    let dist_lens = [5u8; 32];
                    (
                        Huffman::from_lengths(&lit_lens)?,
                        Huffman::from_lengths(&dist_lens)?,
                    )
                } else {
                    let hlit = br.read_bits(5)? as usize + 257;
                    let hdist = br.read_bits(5)? as usize + 1;
                    let hclen = br.read_bits(4)? as usize + 4;
                    if hlit > 286 || hdist > 30 {
                        return Err(PngError::BadDeflate);
                    }
                    let mut cl_lens = [0u8; 19];
                    for i in 0..hclen {
                        cl_lens[CL_ORDER[i]] = br.read_bits(3)? as u8;
                    }
                    let cl_tree = Huffman::from_lengths(&cl_lens)?;
                    let total = hlit + hdist;
                    let mut all_lens = Vec::with_capacity(total);
                    while all_lens.len() < total {
                        let sym = cl_tree.decode(&mut br)?;
                        match sym {
                            0..=15 => all_lens.push(sym as u8),
                            16 => {
                                let prev = *all_lens.last().ok_or(PngError::BadDeflate)?;
                                let rep = br.read_bits(2)? as usize + 3;
                                all_lens.resize((all_lens.len() + rep).min(total), prev);
                            }
                            17 => {
                                let rep = br.read_bits(3)? as usize + 3;
                                all_lens.resize((all_lens.len() + rep).min(total), 0);
                            }
                            18 => {
                                let rep = br.read_bits(7)? as usize + 11;
                                all_lens.resize((all_lens.len() + rep).min(total), 0);
                            }
                            _ => return Err(PngError::BadDeflate),
                        }
                    }
                    (
                        Huffman::from_lengths(&all_lens[..hlit])?,
                        Huffman::from_lengths(&all_lens[hlit..])?,
                    )
                };

                loop {
                    let sym = lit_tree.decode(&mut br)?;
                    match sym {
                        0..=255 => {
                            if out.len() >= max_out {
                                return Err(PngError::TooLarge);
                            }
                            out.push(sym as u8);
                        }
                        256 => break,
                        257..=285 => {
                            let lidx = (sym - 257) as usize;
                            let length =
                                LEN_BASE[lidx] as usize + br.read_bits(LEN_EXTRA[lidx])? as usize;
                            let dsym = dist_tree.decode(&mut br)? as usize;
                            if dsym >= 30 {
                                return Err(PngError::BadDeflate);
                            }
                            let dist =
                                DIST_BASE[dsym] as usize + br.read_bits(DIST_EXTRA[dsym])? as usize;
                            if dist == 0 || dist > out.len() {
                                return Err(PngError::BadDeflate);
                            }
                            if out.len() + length > max_out {
                                return Err(PngError::TooLarge);
                            }
                            for _ in 0..length {
                                let b = out[out.len() - dist];
                                out.push(b);
                            }
                        }
                        _ => return Err(PngError::BadDeflate),
                    }
                }
            }
            _ => return Err(PngError::BadDeflate),
        }
        if bfinal != 0 {
            break;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid RGBA8 PNG using uncompressed zlib stored blocks for deterministic tests.
    fn make_stored_rgba_png(width: u32, height: u32, pixels: &[[u8; 4]], filter: u8) -> Vec<u8> {
        let mut raw = Vec::new();
        for y in 0..height as usize {
            raw.push(filter);
            for x in 0..width as usize {
                let px = pixels[y * width as usize + x];
                match filter {
                    0 => raw.extend_from_slice(&px),
                    1 => {
                        let left = if x > 0 {
                            pixels[y * width as usize + x - 1]
                        } else {
                            [0; 4]
                        };
                        for c in 0..4 {
                            raw.push(px[c].wrapping_sub(left[c]));
                        }
                    }
                    _ => raw.extend_from_slice(&px),
                }
            }
        }
        // Wrap `raw` in a zlib stored block.
        let mut z = vec![0x78, 0x01];
        z.push(0x01); // BFINAL=1, BTYPE=00
        let len = raw.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(&raw);
        // Adler-32
        let mut s1 = 1u32;
        let mut s2 = 0u32;
        for &b in &raw {
            s1 = (s1 + b as u32) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        z.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());

        let mut png = PNG_SIG.to_vec();
        let push_chunk = |dst: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]| {
            dst.extend_from_slice(&(body.len() as u32).to_be_bytes());
            dst.extend_from_slice(kind);
            dst.extend_from_slice(body);
            dst.extend_from_slice(&0u32.to_be_bytes()); // CRC not checked by reader
        };
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA
        push_chunk(&mut png, b"IHDR", &ihdr);
        push_chunk(&mut png, b"IDAT", &z);
        push_chunk(&mut png, b"IEND", &[]);
        png
    }

    #[test]
    fn decode_png_handles_none_and_sub_filters_and_opaque_average() {
        let pixels = [
            [100, 150, 200, 255],
            [140, 170, 220, 255],
            [0, 0, 0, 0], // transparent pixel ignored by average_opaque_rgb
            [120, 160, 210, 255],
        ];
        for filter in [0u8, 1u8] {
            let png = make_stored_rgba_png(2, 2, &pixels, filter);
            let img = decode_png(&png).expect("decode synthetic PNG");
            assert_eq!(img.width, 2);
            assert_eq!(img.height, 2);
            assert_eq!(img.pixel(0, 0), [100, 150, 200, 255]);
            assert_eq!(img.pixel(1, 0), [140, 170, 220, 255]);
            assert_eq!(img.pixel(0, 1), [0, 0, 0, 0]);
            assert_eq!(img.average_opaque_rgb(), Some([120, 160, 210]));
        }
    }

    #[test]
    fn decode_png_rejects_bad_signature_and_truncated_data() {
        assert_eq!(
            decode_png(b"not a png").unwrap_err(),
            PngError::BadSignature
        );
        assert_eq!(decode_png(&PNG_SIG).unwrap_err(), PngError::BadSignature);
    }
}
