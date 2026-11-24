//! Vanilla 26.3 block and biome surface color palette for Distant Horizons LOD columns.

/// Iris / Distant Horizons standard material IDs (`StandardMacros.java:81-96`).
///
/// Exposed to translated Iris/OptiFine shader packs via `DH_BLOCK_*` preprocessor defines and
/// vertex attribute `dhMaterialId`.
pub const DH_BLOCK_UNKNOWN: u8 = 0;
pub const DH_BLOCK_LEAVES: u8 = 1;
pub const DH_BLOCK_STONE: u8 = 2;
pub const DH_BLOCK_WOOD: u8 = 3;
pub const DH_BLOCK_METAL: u8 = 4;
pub const DH_BLOCK_DIRT: u8 = 5;
pub const DH_BLOCK_LAVA: u8 = 6;
pub const DH_BLOCK_DEEPSLATE: u8 = 7;
pub const DH_BLOCK_SNOW: u8 = 8;
pub const DH_BLOCK_SAND: u8 = 9;
pub const DH_BLOCK_TERRACOTTA: u8 = 10;
pub const DH_BLOCK_NETHER_STONE: u8 = 11;
pub const DH_BLOCK_WATER: u8 = 12;
pub const DH_BLOCK_GRASS: u8 = 13;
pub const DH_BLOCK_AIR: u8 = 14;
pub const DH_BLOCK_ILLUMINATED: u8 = 15;

/// Classify a block ID into its 4-bit Iris/DH `dhMaterialId` (`0..=15`).
pub fn classify_dh_material(block_id: &str) -> u8 {
    let name = strip_mc_namespace(block_id);
    if matches!(name, "air" | "cave_air" | "void_air") {
        return DH_BLOCK_AIR;
    }
    if matches!(name, "water" | "bubble_column") {
        return DH_BLOCK_WATER;
    }
    if matches!(name, "lava" | "magma_block") {
        return DH_BLOCK_LAVA;
    }
    if matches!(
        name,
        "glowstone"
            | "sea_lantern"
            | "shroomlight"
            | "ochre_froglight"
            | "verdant_froglight"
            | "pearlescent_froglight"
            | "jack_o_lantern"
            | "redstone_lamp"
            | "lantern"
            | "soul_lantern"
            | "beacon"
            | "end_rod"
            | "crying_obsidian"
            | "respawn_anchor"
    ) || name.contains("torch")
        || name.contains("campfire")
        || name.contains("copper_bulb")
    {
        return DH_BLOCK_ILLUMINATED;
    }
    if matches!(
        name,
        "grass_block" | "moss_block" | "moss_carpet" | "mycelium" | "podzol" | "warped_nylium"
    ) || name.contains("grass")
    {
        return DH_BLOCK_GRASS;
    }
    if name.contains("leaves") || matches!(name, "azalea" | "flowering_azalea" | "mangrove_roots") {
        return DH_BLOCK_LEAVES;
    }
    if name.contains("snow") || matches!(name, "powder_snow") {
        return DH_BLOCK_SNOW;
    }
    if name.contains("sand") || matches!(name, "gravel" | "suspicious_sand" | "suspicious_gravel") {
        return DH_BLOCK_SAND;
    }
    if name.contains("terracotta") || name.contains("concrete") || name.contains("clay") {
        return DH_BLOCK_TERRACOTTA;
    }
    if name.contains("deepslate") || matches!(name, "tuff" | "blackstone" | "gilded_blackstone") {
        return DH_BLOCK_DEEPSLATE;
    }
    if matches!(
        name,
        "netherrack"
            | "crimson_nylium"
            | "nether_bricks"
            | "red_nether_bricks"
            | "cracked_nether_bricks"
            | "chiseled_nether_bricks"
            | "basalt"
            | "polished_basalt"
            | "smooth_basalt"
            | "soul_soil"
            | "nether_quartz_ore"
            | "nether_gold_ore"
            | "ancient_debris"
    ) {
        return DH_BLOCK_NETHER_STONE;
    }
    if matches!(
        name,
        "dirt"
            | "coarse_dirt"
            | "rooted_dirt"
            | "dirt_path"
            | "farmland"
            | "mud"
            | "packed_mud"
            | "muddy_mangrove_roots"
    ) {
        return DH_BLOCK_DIRT;
    }
    if name.contains("log")
        || name.contains("wood")
        || name.contains("planks")
        || name.contains("stem")
        || name.contains("hyphae")
        || name.contains("bamboo")
        || matches!(
            name,
            "bookshelf" | "crafting_table" | "barrel" | "chest" | "oak_fence" | "note_block"
        )
    {
        return DH_BLOCK_WOOD;
    }
    if name.contains("iron")
        || name.contains("gold")
        || name.contains("copper")
        || name.contains("netherite")
        || matches!(name, "anvil" | "chipped_anvil" | "damaged_anvil" | "chain")
    {
        return DH_BLOCK_METAL;
    }
    if name.contains("stone")
        || name.contains("cobble")
        || name.contains("granite")
        || name.contains("diorite")
        || name.contains("andesite")
        || name.contains("calcite")
        || name.contains("dripstone")
        || name.contains("prismarine")
        || name.contains("quartz")
        || name.contains("brick")
        || name.contains("ore")
        || matches!(name, "bedrock" | "obsidian" | "end_stone" | "purpur_block")
    {
        return DH_BLOCK_STONE;
    }
    DH_BLOCK_UNKNOWN
}

/// True for air and sparse non-solid plants/decorations that an LOD column should look through
/// to pick the terrain block underneath.
pub fn is_skippable_for_lod(block_id: &str) -> bool {
    let name = strip_mc_namespace(block_id);
    matches!(
        name,
        "air"
            | "cave_air"
            | "void_air"
            | "short_grass"
            | "tall_grass"
            | "fern"
            | "large_fern"
            | "dead_bush"
            | "bush"
            | "firefly_bush"
            | "short_dry_grass"
            | "tall_dry_grass"
            | "leaf_litter"
            | "wildflowers"
            | "pink_petals"
            | "dandelion"
            | "poppy"
            | "blue_orchid"
            | "allium"
            | "azure_bluet"
            | "red_tulip"
            | "orange_tulip"
            | "white_tulip"
            | "pink_tulip"
            | "oxeye_daisy"
            | "cornflower"
            | "lily_of_the_valley"
            | "wither_rose"
            | "sunflower"
            | "lilac"
            | "rose_bush"
            | "peony"
            | "vine"
            | "glow_lichen"
            | "seagrass"
            | "tall_seagrass"
            | "kelp"
            | "kelp_plant"
            | "sugar_cane"
            | "torch"
            | "wall_torch"
            | "redstone_torch"
            | "soul_torch"
            | "brown_mushroom"
            | "red_mushroom"
            | "crimson_roots"
            | "warped_roots"
            | "nether_sprouts"
            | "hanging_roots"
            | "cobweb"
            | "rail"
            | "powered_rail"
            | "detector_rail"
            | "activator_rail"
    )
}

/// True for water/translucent fluids rendered on the transparent LOD layer.
pub fn is_water_block(block_id: &str) -> bool {
    let name = strip_mc_namespace(block_id);
    matches!(name, "water" | "bubble_column")
}

/// Vanilla `(temperature, downfall)` pairs for colormap sampling (`textures/colormap/{grass,foliage}.png`).
fn biome_climate(name: &str) -> Option<(f32, f32)> {
    match name {
        "plains" | "sunflower_plains" | "beach" => Some((0.8, 0.4)),
        "river" => Some((0.5, 0.5)),
        "forest" | "flower_forest" | "dark_forest" => Some((0.7, 0.8)),
        "birch_forest" | "old_growth_birch_forest" => Some((0.6, 0.6)),
        "taiga" | "old_growth_pine_taiga" => Some((0.25, 0.8)),
        "old_growth_spruce_taiga" => Some((0.25, 0.8)),
        "snowy_taiga" => Some((-0.5, 0.4)),
        "snowy_plains" | "ice_spikes" | "snowy_slopes" | "frozen_peaks" | "jagged_peaks"
        | "grove" | "snowy_beach" | "frozen_river" | "frozen_ocean" | "deep_frozen_ocean" => {
            Some((0.0, 0.5))
        }
        "windswept_hills"
        | "windswept_gravelly_hills"
        | "windswept_forest"
        | "stony_shore"
        | "stony_peaks" => Some((0.2, 0.3)),
        "jungle" | "bamboo_jungle" => Some((0.95, 0.9)),
        "sparse_jungle" => Some((0.95, 0.8)),
        "savanna" | "savanna_plateau" | "windswept_savanna" | "desert" => Some((2.0, 0.0)),
        "meadow" => Some((0.5, 0.8)),
        "mushroom_fields" => Some((0.9, 1.0)),
        "ocean"
        | "deep_ocean"
        | "lukewarm_ocean"
        | "deep_lukewarm_ocean"
        | "warm_ocean"
        | "cold_ocean"
        | "deep_cold_ocean" => Some((0.5, 0.5)),
        _ => None,
    }
}

/// Grass tint `[r, g, b]` matching vanilla 26.3 biome climates (sampled from active resource-pack
/// `textures/colormap/grass.png` when loaded).
pub fn biome_grass_tint(biome_id: &str) -> [u8; 3] {
    let name = strip_mc_namespace(biome_id);
    if !matches!(
        name,
        "swamp"
            | "mangrove_swamp"
            | "badlands"
            | "eroded_badlands"
            | "wooded_badlands"
            | "dark_forest"
            | "cherry_grove"
            | "pale_garden"
    ) && let Some((temp, rain)) = biome_climate(name)
        && let Some(rgb) = crate::resource_colors::sample_colormap(true, temp, rain)
    {
        return rgb;
    }
    match name {
        "plains" | "sunflower_plains" | "river" | "beach" => [145, 189, 89],
        "forest" | "flower_forest" => [121, 192, 90],
        "birch_forest" | "old_growth_birch_forest" => [136, 187, 103],
        "dark_forest" => [80, 122, 50],
        "pale_garden" => [119, 130, 114],
        "taiga" | "old_growth_pine_taiga" => [134, 183, 131],
        "old_growth_spruce_taiga" => [134, 184, 127],
        "snowy_taiga" | "snowy_plains" | "ice_spikes" | "snowy_slopes" | "frozen_peaks"
        | "jagged_peaks" | "grove" | "snowy_beach" | "frozen_river" | "frozen_ocean"
        | "deep_frozen_ocean" => [128, 180, 151],
        "windswept_hills"
        | "windswept_gravelly_hills"
        | "windswept_forest"
        | "stony_shore"
        | "stony_peaks" => [138, 182, 137],
        "jungle" | "bamboo_jungle" => [89, 201, 60],
        "sparse_jungle" => [100, 199, 63],
        "savanna" | "savanna_plateau" | "windswept_savanna" | "desert" => [191, 183, 85],
        "badlands" | "eroded_badlands" | "wooded_badlands" => [144, 129, 77],
        "swamp" | "mangrove_swamp" => [106, 112, 57],
        "meadow" => [131, 187, 109],
        "cherry_grove" => [182, 219, 97],
        "mushroom_fields" => [85, 201, 63],
        "ocean"
        | "deep_ocean"
        | "lukewarm_ocean"
        | "deep_lukewarm_ocean"
        | "warm_ocean"
        | "cold_ocean"
        | "deep_cold_ocean" => [142, 185, 113],
        _ => [134, 186, 94],
    }
}

/// Foliage (leaves) tint `[r, g, b]` matching vanilla 26.3 biome climates (sampled from active
/// resource-pack `textures/colormap/foliage.png` when loaded).
pub fn biome_foliage_tint(biome_id: &str) -> [u8; 3] {
    let name = strip_mc_namespace(biome_id);
    if !matches!(
        name,
        "swamp"
            | "mangrove_swamp"
            | "badlands"
            | "eroded_badlands"
            | "wooded_badlands"
            | "cherry_grove"
            | "pale_garden"
    ) && let Some((temp, rain)) = biome_climate(name)
        && let Some(rgb) = crate::resource_colors::sample_colormap(false, temp, rain)
    {
        return rgb;
    }
    match name {
        "plains" | "sunflower_plains" => [119, 171, 47],
        "forest" | "flower_forest" => [89, 174, 48],
        "birch_forest" | "old_growth_birch_forest" => [107, 169, 65],
        "dark_forest" => [89, 174, 48],
        "pale_garden" => [135, 141, 118],
        "taiga" | "old_growth_pine_taiga" | "old_growth_spruce_taiga" => [104, 164, 100],
        "snowy_taiga" | "snowy_plains" | "grove" | "snowy_slopes" => [96, 161, 123],
        "jungle" | "bamboo_jungle" => [48, 187, 11],
        "sparse_jungle" => [62, 184, 15],
        "savanna" | "savanna_plateau" | "windswept_savanna" | "desert" => [174, 164, 42],
        "badlands" | "eroded_badlands" | "wooded_badlands" => [158, 129, 77],
        "swamp" => [106, 112, 57],
        "mangrove_swamp" => [141, 177, 39],
        "cherry_grove" => [182, 219, 97],
        "meadow" => [99, 169, 72],
        _ => [94, 168, 52],
    }
}

/// Biome water tint `[r, g, b]`, using Sunrise's biome water color when provided (`!= 0`).
pub fn biome_water_color(biome_id: &str, water_color_hint: u32) -> [u8; 3] {
    if water_color_hint != 0 {
        return [
            ((water_color_hint >> 16) & 0xFF) as u8,
            ((water_color_hint >> 8) & 0xFF) as u8,
            (water_color_hint & 0xFF) as u8,
        ];
    }
    let name = strip_mc_namespace(biome_id);
    match name {
        "warm_ocean" => [67, 213, 238],
        "lukewarm_ocean" | "deep_lukewarm_ocean" => [69, 173, 242],
        "cold_ocean" | "deep_cold_ocean" | "snowy_taiga" => [61, 87, 214],
        "frozen_ocean" | "deep_frozen_ocean" | "frozen_river" => [57, 56, 201],
        "swamp" => [97, 123, 100],
        "mangrove_swamp" => [58, 122, 106],
        "meadow" => [14, 78, 207],
        "cherry_grove" => [93, 183, 239],
        "pale_garden" => [118, 136, 157],
        _ => [63, 118, 228],
    }
}

fn multiply_rgb(a: [u8; 3], b: [u8; 3]) -> [u8; 3] {
    [
        ((a[0] as u16 * b[0] as u16) / 255) as u8,
        ((a[1] as u16 * b[1] as u16) / 255) as u8,
        ((a[2] as u16 * b[2] as u16) / 255) as u8,
    ]
}

/// Compute surface RGB for a block in a biome (with optional server biome water color hint and
/// active resource-pack texture overrides when loaded).
pub fn block_color(block_id: &str, biome_id: &str, water_color_hint: u32) -> [u8; 3] {
    let name = strip_mc_namespace(block_id);
    if matches!(name, "water" | "bubble_column") {
        return biome_water_color(biome_id, water_color_hint);
    }
    if let Some(pack_rgb) = crate::resource_colors::lookup_pack_block_rgb(name) {
        if name == "grass_block" {
            return multiply_rgb(pack_rgb, biome_grass_tint(biome_id));
        }
        if name.contains("leaves")
            && !matches!(
                name,
                "cherry_leaves" | "azalea_leaves" | "flowering_azalea_leaves" | "pale_oak_leaves"
            )
        {
            let tint = match name {
                "spruce_leaves" => [97, 153, 97],
                "birch_leaves" => [128, 167, 85],
                _ => biome_foliage_tint(biome_id),
            };
            return multiply_rgb(pack_rgb, tint);
        }
        return pack_rgb;
    }
    match name {
        "grass_block" => biome_grass_tint(biome_id),
        "dirt" | "coarse_dirt" | "rooted_dirt" | "farmland" => [134, 96, 67],
        "dirt_path" => [148, 121, 65],
        "podzol" => [122, 87, 57],
        "mycelium" => [111, 98, 101],
        "mud" | "muddy_mangrove_roots" => [60, 57, 61],
        "sand" => [219, 207, 163],
        "sandstone" | "cut_sandstone" | "smooth_sandstone" | "chiseled_sandstone" => {
            [216, 203, 155]
        }
        "red_sand" => [190, 102, 33],
        "red_sandstone" | "smooth_red_sandstone" | "cut_red_sandstone" => [181, 97, 31],
        "gravel" => [131, 127, 126],
        "clay" => [160, 166, 179],
        "stone" | "cobblestone" | "stone_bricks" | "mossy_cobblestone" | "smooth_stone" => {
            [125, 125, 125]
        }
        "deepslate" | "cobbled_deepslate" | "deepslate_bricks" | "deepslate_tiles" => [80, 80, 82],
        "granite" | "polished_granite" => [149, 103, 85],
        "diorite" | "polished_diorite" => [188, 188, 188],
        "andesite" | "polished_andesite" => [136, 136, 136],
        "tuff" => [108, 109, 102],
        "calcite" => [223, 224, 220],
        "dripstone_block" => [134, 107, 92],
        "terracotta" => [152, 94, 67],
        "white_terracotta" => [209, 178, 161],
        "orange_terracotta" => [161, 83, 37],
        "magenta_terracotta" => [149, 88, 108],
        "light_blue_terracotta" => [113, 108, 137],
        "yellow_terracotta" => [186, 133, 35],
        "lime_terracotta" => [103, 117, 52],
        "pink_terracotta" => [161, 78, 78],
        "gray_terracotta" => [57, 42, 35],
        "light_gray_terracotta" => [135, 106, 97],
        "cyan_terracotta" => [86, 91, 91],
        "purple_terracotta" => [118, 70, 86],
        "blue_terracotta" => [74, 59, 91],
        "brown_terracotta" => [77, 51, 35],
        "green_terracotta" => [76, 83, 42],
        "red_terracotta" => [143, 61, 46],
        "black_terracotta" => [37, 22, 16],
        "snow" | "snow_block" | "powder_snow" => [245, 251, 251],
        "ice" => [145, 183, 253],
        "packed_ice" => [141, 180, 250],
        "blue_ice" => [116, 167, 253],
        "water" | "bubble_column" => biome_water_color(biome_id, water_color_hint),
        "lava" => [212, 90, 18],
        "bedrock" => [86, 86, 86],
        "obsidian" | "crying_obsidian" => [20, 18, 30],
        "netherrack" => [97, 38, 38],
        "crimson_nylium" => [131, 31, 31],
        "warped_nylium" => [43, 114, 101],
        "soul_sand" => [81, 62, 50],
        "soul_soil" => [75, 57, 46],
        "basalt" | "smooth_basalt" | "polished_basalt" => [73, 72, 77],
        "blackstone" | "polished_blackstone" => [42, 35, 40],
        "magma_block" => [142, 63, 31],
        "glowstone" => [171, 131, 84],
        "nether_bricks" | "red_nether_bricks" => [44, 21, 26],
        "end_stone" | "end_stone_bricks" => [219, 222, 158],
        "purpur_block" | "purpur_pillar" => [169, 125, 169],
        "moss_block" | "moss_carpet" => [89, 109, 45],
        "sculk" => [12, 29, 36],
        // Foliage / leaves
        "oak_leaves" | "jungle_leaves" | "acacia_leaves" | "dark_oak_leaves"
        | "mangrove_leaves" => scale_rgb(biome_foliage_tint(biome_id), 0.88),
        "spruce_leaves" => [68, 106, 68],
        "birch_leaves" => [102, 135, 69],
        "cherry_leaves" => [235, 164, 188],
        "pale_oak_leaves" => [122, 128, 119],
        "azalea_leaves" | "flowering_azalea_leaves" => [90, 118, 43],
        "nether_wart_block" => [114, 2, 2],
        "warped_wart_block" => [22, 119, 121],
        // Logs & planks
        "oak_log" | "oak_wood" => [109, 85, 50],
        "oak_planks" => [162, 130, 78],
        "spruce_log" | "spruce_wood" => [58, 37, 16],
        "spruce_planks" => [114, 84, 48],
        "birch_log" | "birch_wood" => [216, 215, 210],
        "birch_planks" => [192, 175, 121],
        "jungle_log" | "jungle_wood" => [85, 67, 25],
        "jungle_planks" => [160, 115, 80],
        "acacia_log" | "acacia_wood" => [103, 96, 86],
        "acacia_planks" => [168, 90, 50],
        "dark_oak_log" | "dark_oak_wood" => [60, 46, 26],
        "dark_oak_planks" => [66, 43, 20],
        "mangrove_log" | "mangrove_wood" => [83, 66, 41],
        "cherry_log" | "cherry_wood" => [54, 33, 44],
        "pale_oak_log" | "pale_oak_wood" => [87, 77, 75],
        "cactus" => [85, 126, 42],
        "bamboo" | "bamboo_block" => [93, 144, 19],
        "pumpkin" | "carved_pumpkin" | "jack_o_lantern" => [198, 118, 24],
        "melon" => [114, 146, 30],
        "hay_block" => [166, 139, 12],
        "bricks" => [150, 97, 83],
        "coal_ore" | "iron_ore" | "copper_ore" | "gold_ore" | "diamond_ore" | "emerald_ore"
        | "lapis_ore" | "redstone_ore" => [120, 120, 120],
        _ => fallback_block_color(name, biome_id),
    }
}

fn fallback_block_color(name: &str, biome_id: &str) -> [u8; 3] {
    if name.contains("leaves") {
        return scale_rgb(biome_foliage_tint(biome_id), 0.88);
    }
    if name.contains("grass") {
        return biome_grass_tint(biome_id);
    }
    if name.contains("sand") {
        return [216, 203, 155];
    }
    if name.contains("snow") || name.contains("white_") {
        return [238, 244, 246];
    }
    if name.contains("ice") {
        return [141, 180, 250];
    }
    if name.contains("wood") || name.contains("log") || name.contains("planks") {
        return [130, 98, 60];
    }
    if name.contains("deepslate") || name.contains("black") {
        return [68, 68, 72];
    }
    if name.contains("terracotta") || name.contains("brick") {
        return [150, 92, 68];
    }
    if name.contains("copper") {
        return [165, 108, 84];
    }
    if name.contains("stone") || name.contains("ore") || name.contains("cobble") {
        return [125, 125, 125];
    }
    [128, 128, 128]
}

fn strip_mc_namespace(id: &str) -> &str {
    id.strip_prefix("minecraft:").unwrap_or(id)
}

fn scale_rgb(rgb: [u8; 3], factor: f32) -> [u8; 3] {
    [
        ((rgb[0] as f32) * factor).clamp(0.0, 255.0) as u8,
        ((rgb[1] as f32) * factor).clamp(0.0, 255.0) as u8,
        ((rgb[2] as f32) * factor).clamp(0.0, 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biome_and_block_colors_distinguish_surfaces() {
        let plains_grass = block_color("minecraft:grass_block", "minecraft:plains", 0);
        let desert_sand = block_color("minecraft:sand", "minecraft:desert", 0);
        let water_hint = block_color("minecraft:water", "minecraft:ocean", 0x3F76E4);
        assert_ne!(plains_grass, desert_sand);
        assert_eq!(water_hint, [0x3F, 0x76, 0xE4]);
        assert!(is_skippable_for_lod("minecraft:short_grass"));
        assert!(!is_skippable_for_lod("minecraft:oak_leaves"));
        assert!(is_water_block("minecraft:water"));
        assert_eq!(
            classify_dh_material("minecraft:grass_block"),
            DH_BLOCK_GRASS
        );
        assert_eq!(
            classify_dh_material("minecraft:oak_leaves"),
            DH_BLOCK_LEAVES
        );
        assert_eq!(classify_dh_material("minecraft:stone"), DH_BLOCK_STONE);
        assert_eq!(
            classify_dh_material("minecraft:deepslate"),
            DH_BLOCK_DEEPSLATE
        );
        assert_eq!(classify_dh_material("minecraft:water"), DH_BLOCK_WATER);
        assert_eq!(classify_dh_material("minecraft:lava"), DH_BLOCK_LAVA);
        assert_eq!(
            classify_dh_material("minecraft:glowstone"),
            DH_BLOCK_ILLUMINATED
        );
        assert_eq!(
            classify_dh_material("minecraft:netherrack"),
            DH_BLOCK_NETHER_STONE
        );
        assert_eq!(classify_dh_material("minecraft:snow_block"), DH_BLOCK_SNOW);
        assert_eq!(classify_dh_material("minecraft:sand"), DH_BLOCK_SAND);
        assert_eq!(
            classify_dh_material("minecraft:red_terracotta"),
            DH_BLOCK_TERRACOTTA
        );
        assert_eq!(classify_dh_material("minecraft:oak_log"), DH_BLOCK_WOOD);
        assert_eq!(classify_dh_material("minecraft:iron_block"), DH_BLOCK_METAL);
        assert_eq!(classify_dh_material("minecraft:dirt"), DH_BLOCK_DIRT);
        assert_eq!(classify_dh_material("minecraft:air"), DH_BLOCK_AIR);
    }
}
