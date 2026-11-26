use mod_api::{Host, Registrar, SettingKey, SettingSpec};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

static SETTING_KEYS: OnceLock<RwLock<HashMap<String, SettingKey>>> = OnceLock::new();

fn key_store() -> &'static RwLock<HashMap<String, SettingKey>> {
    SETTING_KEYS.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn setting_key(name: &str) -> Option<SettingKey> {
    key_store().read().ok()?.get(name).copied()
}

pub const MOD_ID: &str = "distantsunrises";

pub const KEY_ENABLED: &str = "enabled";
pub const KEY_RENDER_DISTANCE: &str = "render_distance_chunks";
pub const KEY_HORIZONTAL_QUALITY: &str = "horizontal_quality";
pub const KEY_VERTICAL_QUALITY: &str = "vertical_quality";
pub const KEY_DISTANT_GENERATION: &str = "distant_generation";
pub const KEY_DISTANT_GENERATION_MODE: &str = "distant_generation_mode";
pub const KEY_CAVE_CULLING: &str = "cave_culling_enabled";
pub const KEY_CAVE_CULLING_HEIGHT: &str = "cave_culling_height";
pub const KEY_FOG_ENABLED: &str = "fog_enabled";
pub const KEY_NOISE_ENABLED: &str = "noise_enabled";
pub const KEY_NOISE_INTENSITY: &str = "noise_intensity";
pub const KEY_OVERDRAW_PREVENTION: &str = "overdraw_prevention";
pub const KEY_FADE_ENABLED: &str = "fade_enabled";
pub const KEY_SHADING_ENABLED: &str = "shading_enabled";
pub const KEY_BRIGHTNESS: &str = "brightness_multiplier";
pub const KEY_SATURATION: &str = "saturation_multiplier";
pub const KEY_SSAO_ENABLED: &str = "ssao_enabled";
pub const KEY_SSAO_QUALITY: &str = "ssao_quality";
pub const KEY_SSAO_STRENGTH: &str = "ssao_strength";
pub const KEY_DISABLE_VANILLA_FOG: &str = "disable_vanilla_fog";
pub const KEY_FOG_FALLOFF: &str = "fog_falloff";
pub const KEY_FOG_SPHERICAL: &str = "fog_spherical";
pub const KEY_HEIGHT_FOG_ENABLED: &str = "height_fog_enabled";
pub const KEY_HEIGHT_FOG_BASE_HEIGHT: &str = "height_fog_base_height";
pub const KEY_HEIGHT_FOG_MIX_MODE: &str = "height_fog_mix_mode";
pub const KEY_CLOUDS_ENABLED: &str = "clouds_enabled";
pub const KEY_BEACON_BEAMS_ENABLED: &str = "beacon_beams_enabled";
pub const KEY_SHARPENING_AMOUNT: &str = "sharpening_amount";
pub const KEY_DEBUG_MODE: &str = "debug_mode";
pub const KEY_THREAD_PRESET: &str = "thread_preset";

/// Mirrors `EDhApiThreadPreset` and `ThreadPresetConfigEventHandler` (percent of logical
/// processors per preset; `CUSTOM` has no fixed value and is not offered).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ThreadPreset {
    MinimalImpact = 0,
    LowImpact = 1,
    #[default]
    Balanced = 2,
    Aggressive = 3,
    IPaidForTheWholeCpu = 4,
}

impl ThreadPreset {
    pub const ALL: [Self; 5] = [
        Self::MinimalImpact,
        Self::LowImpact,
        Self::Balanced,
        Self::Aggressive,
        Self::IPaidForTheWholeCpu,
    ];

    /// `getThreadCountByPercent(0.1 / 0.25 / 0.5 / 0.75 / 1.0)`.
    pub const fn percent(self) -> f64 {
        match self {
            Self::MinimalImpact => 0.1,
            Self::LowImpact => 0.25,
            Self::Balanced => 0.5,
            Self::Aggressive => 0.75,
            Self::IPaidForTheWholeCpu => 1.0,
        }
    }

    /// `ceil(total * percent)` clamped to `1..=total`.
    pub fn thread_count(self, total_logical_processors: usize) -> usize {
        let total = total_logical_processors.max(1);
        (((total as f64) * self.percent()).ceil() as usize).clamp(1, total)
    }

    /// `threadRunTime` column of the preset table: `MinimalImpact` 0.5, all others 1.0.
    pub const fn run_time_ratio(self) -> f64 {
        match self {
            Self::MinimalImpact => 0.5,
            _ => 1.0,
        }
    }

    pub fn from_index(index: u32) -> Self {
        Self::ALL.get(index as usize).copied().unwrap_or_default()
    }

    pub fn from_u8(v: u8) -> Self {
        Self::from_index(u32::from(v))
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().replace(['_', '-'], "").as_str() {
            "minimalimpact" | "minimal" => Some(Self::MinimalImpact),
            "lowimpact" | "low" => Some(Self::LowImpact),
            "balanced" => Some(Self::Balanced),
            "aggressive" => Some(Self::Aggressive),
            "ipaidforthewholecpu" | "whole" | "all" => Some(Self::IPaidForTheWholeCpu),
            _ => None,
        }
    }

    pub const fn label_key(self) -> &'static str {
        match self {
            Self::MinimalImpact => "distantsunrises.setting.thread_preset.minimal_impact",
            Self::LowImpact => "distantsunrises.setting.thread_preset.low_impact",
            Self::Balanced => "distantsunrises.setting.thread_preset.balanced",
            Self::Aggressive => "distantsunrises.setting.thread_preset.aggressive",
            Self::IPaidForTheWholeCpu => {
                "distantsunrises.setting.thread_preset.i_paid_for_the_whole_cpu"
            }
        }
    }
}

/// Debug visualization modes matching upstream `EDhApiDebugRendering`.
pub const DEBUG_OFF: u32 = 0;
pub const DEBUG_SHOW_DETAIL: u32 = 1;
pub const DEBUG_SHOW_GEN_MODE: u32 = 2;
pub const DEBUG_SHOW_LIGHTING: u32 = 3;
pub const DEBUG_SHOW_DETAIL_WIREFRAME: u32 = 4;

/// Runtime settings mirrored from Sunrise's mod settings UI (`Options > Mods > Distant Sunrises`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DhConfig {
    pub enabled: bool,
    pub render_distance_chunks: u32,
    /// 0 = Low, 1 = Medium, 2 = High.
    pub horizontal_quality: u32,
    /// 0 = Low (1 extra layer), 1 = Medium (2), 2 = High (4), 3 = Extreme (8).
    pub vertical_quality: u32,
    pub distant_generation: bool,
    /// 0 = Surface, 1 = Carvers, 2 = Features, 3 = Full Light.
    pub distant_generation_mode: u32,
    pub cave_culling_enabled: bool,
    pub cave_culling_height: i32,
    pub fog_enabled: bool,
    pub noise_enabled: bool,
    pub noise_intensity: f32,
    pub overdraw_prevention: f32,
    pub fade_enabled: bool,
    pub shading_enabled: bool,
    pub brightness_multiplier: f32,
    pub saturation_multiplier: f32,
    pub ssao_enabled: bool,
    /// 0 = Low (6 samples), 1 = Medium (12 samples), 2 = High (20 samples).
    pub ssao_quality: u32,
    pub ssao_strength: f32,
    pub disable_vanilla_fog: bool,
    /// 0 = Linear, 1 = Exponential, 2 = Exponential Squared.
    pub fog_falloff: u32,
    pub fog_spherical: bool,
    pub height_fog_enabled: bool,
    pub height_fog_base_height: f32,
    /// 0 = Basic (far only), 1 = Max, 2 = Addition, 3 = Multiply, 4 = Inverse Multiply.
    pub height_fog_mix_mode: u32,
    pub clouds_enabled: bool,
    pub beacon_beams_enabled: bool,
    /// FidelityFX Contrast-Adaptive Sharpening strength (0.0 = off, 1.0 = max).
    pub sharpening_amount: f32,
    /// `DEBUG_OFF..=DEBUG_SHOW_DETAIL_WIREFRAME`.
    pub debug_mode: u32,
    /// Worldgen/ingest thread preset (`EDhApiThreadPreset`).
    pub thread_preset: ThreadPreset,
}

impl Default for DhConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            render_distance_chunks: 128,
            horizontal_quality: 1,
            vertical_quality: 1,
            distant_generation: true,
            distant_generation_mode: 1,
            cave_culling_enabled: true,
            cave_culling_height: 60,
            fog_enabled: true,
            noise_enabled: true,
            noise_intensity: 0.12,
            overdraw_prevention: 0.9,
            fade_enabled: true,
            shading_enabled: true,
            brightness_multiplier: 1.0,
            saturation_multiplier: 1.0,
            ssao_enabled: true,
            ssao_quality: 1,
            ssao_strength: 0.45,
            disable_vanilla_fog: true,
            fog_falloff: 2,
            fog_spherical: false,
            height_fog_enabled: true,
            height_fog_base_height: 70.0,
            height_fog_mix_mode: 4,
            clouds_enabled: true,
            beacon_beams_enabled: true,
            sharpening_amount: 0.35,
            debug_mode: DEBUG_OFF,
            thread_preset: ThreadPreset::Balanced,
        }
    }
}

impl DhConfig {
    pub fn render_distance_blocks(self) -> f32 {
        (self.render_distance_chunks as f32) * 16.0
    }

    pub fn ssao_sample_count(self) -> u32 {
        if !self.ssao_enabled {
            return 0;
        }
        match self.ssao_quality {
            0 => 6,
            2 => 20,
            _ => 12,
        }
    }

    /// Maximum extra vertical spans retained per column beneath the primary surface.
    pub fn max_extra_spans(self) -> usize {
        match self.vertical_quality {
            0 => 1,
            1 => 2,
            2 => 4,
            _ => 8,
        }
    }

    /// Distance in blocks at which `detail` (0..=6) should subdivide to `detail - 1`.
    pub fn detail_split_distance_blocks(self, detail: u8) -> f32 {
        let quality_scale = match self.horizontal_quality {
            0 => 0.75,
            2 => 1.5,
            _ => 1.0,
        };
        let base_chunks = match detail {
            0 => 0.0,
            1 => 20.0,
            2 => 44.0,
            3 => 88.0,
            4 => 160.0,
            5 => 320.0,
            _ => 640.0,
        };
        base_chunks * 16.0 * quality_scale
    }

    pub fn register_settings(registrar: &mut dyn Registrar) {
        let specs = [
            SettingSpec::toggle(KEY_ENABLED, "distantsunrises.setting.enabled", true)
                .tooltip("distantsunrises.setting.enabled.tooltip"),
            SettingSpec::range(
                KEY_RENDER_DISTANCE,
                "distantsunrises.setting.render_distance_chunks",
                32.0,
                512.0,
                8.0,
                128.0,
            )
            .tooltip("distantsunrises.setting.render_distance_chunks.tooltip"),
            SettingSpec::choice(
                KEY_HORIZONTAL_QUALITY,
                "distantsunrises.setting.horizontal_quality",
                [
                    "distantsunrises.setting.quality.low",
                    "distantsunrises.setting.quality.medium",
                    "distantsunrises.setting.quality.high",
                ],
                1,
            )
            .tooltip("distantsunrises.setting.horizontal_quality.tooltip"),
            SettingSpec::choice(
                KEY_VERTICAL_QUALITY,
                "distantsunrises.setting.vertical_quality",
                [
                    "distantsunrises.setting.quality.low",
                    "distantsunrises.setting.quality.medium",
                    "distantsunrises.setting.quality.high",
                    "distantsunrises.setting.quality.extreme",
                ],
                1,
            )
            .tooltip("distantsunrises.setting.vertical_quality.tooltip"),
            SettingSpec::toggle(
                KEY_DISTANT_GENERATION,
                "distantsunrises.setting.distant_generation",
                true,
            )
            .tooltip("distantsunrises.setting.distant_generation.tooltip"),
            SettingSpec::choice(
                KEY_DISTANT_GENERATION_MODE,
                "distantsunrises.setting.distant_generation_mode",
                [
                    "distantsunrises.setting.gen_mode.surface",
                    "distantsunrises.setting.gen_mode.carvers",
                    "distantsunrises.setting.gen_mode.features",
                    "distantsunrises.setting.gen_mode.full",
                ],
                1,
            )
            .tooltip("distantsunrises.setting.distant_generation_mode.tooltip"),
            SettingSpec::toggle(
                KEY_CAVE_CULLING,
                "distantsunrises.setting.cave_culling_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.cave_culling_enabled.tooltip"),
            SettingSpec::range(
                KEY_CAVE_CULLING_HEIGHT,
                "distantsunrises.setting.cave_culling_height",
                -64.0,
                320.0,
                4.0,
                60.0,
            )
            .tooltip("distantsunrises.setting.cave_culling_height.tooltip"),
            SettingSpec::toggle(KEY_FOG_ENABLED, "distantsunrises.setting.fog_enabled", true)
                .tooltip("distantsunrises.setting.fog_enabled.tooltip"),
            SettingSpec::toggle(
                KEY_NOISE_ENABLED,
                "distantsunrises.setting.noise_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.noise_enabled.tooltip"),
            SettingSpec::range(
                KEY_NOISE_INTENSITY,
                "distantsunrises.setting.noise_intensity",
                0.0,
                0.5,
                0.02,
                0.12,
            )
            .tooltip("distantsunrises.setting.noise_intensity.tooltip"),
            SettingSpec::range(
                KEY_OVERDRAW_PREVENTION,
                "distantsunrises.setting.overdraw_prevention",
                0.0,
                1.0,
                0.05,
                0.9,
            )
            .tooltip("distantsunrises.setting.overdraw_prevention.tooltip"),
            SettingSpec::toggle(
                KEY_FADE_ENABLED,
                "distantsunrises.setting.fade_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.fade_enabled.tooltip"),
            SettingSpec::toggle(
                KEY_SHADING_ENABLED,
                "distantsunrises.setting.shading_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.shading_enabled.tooltip"),
            SettingSpec::range(
                KEY_BRIGHTNESS,
                "distantsunrises.setting.brightness_multiplier",
                0.5,
                2.0,
                0.05,
                1.0,
            )
            .tooltip("distantsunrises.setting.brightness_multiplier.tooltip"),
            SettingSpec::range(
                KEY_SATURATION,
                "distantsunrises.setting.saturation_multiplier",
                0.0,
                2.0,
                0.05,
                1.0,
            )
            .tooltip("distantsunrises.setting.saturation_multiplier.tooltip"),
            SettingSpec::toggle(
                KEY_SSAO_ENABLED,
                "distantsunrises.setting.ssao_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.ssao_enabled.tooltip"),
            SettingSpec::choice(
                KEY_SSAO_QUALITY,
                "distantsunrises.setting.ssao_quality",
                [
                    "distantsunrises.setting.quality.low",
                    "distantsunrises.setting.quality.medium",
                    "distantsunrises.setting.quality.high",
                ],
                1,
            )
            .tooltip("distantsunrises.setting.ssao_quality.tooltip"),
            SettingSpec::range(
                KEY_SSAO_STRENGTH,
                "distantsunrises.setting.ssao_strength",
                0.0,
                1.0,
                0.05,
                0.45,
            )
            .tooltip("distantsunrises.setting.ssao_strength.tooltip"),
            SettingSpec::toggle(
                KEY_DISABLE_VANILLA_FOG,
                "distantsunrises.setting.disable_vanilla_fog",
                true,
            )
            .tooltip("distantsunrises.setting.disable_vanilla_fog.tooltip"),
            SettingSpec::choice(
                KEY_FOG_FALLOFF,
                "distantsunrises.setting.fog_falloff",
                [
                    "distantsunrises.setting.fog_falloff.linear",
                    "distantsunrises.setting.fog_falloff.exponential",
                    "distantsunrises.setting.fog_falloff.exponential_squared",
                ],
                2,
            )
            .tooltip("distantsunrises.setting.fog_falloff.tooltip"),
            SettingSpec::toggle(
                KEY_FOG_SPHERICAL,
                "distantsunrises.setting.fog_spherical",
                false,
            )
            .tooltip("distantsunrises.setting.fog_spherical.tooltip"),
            SettingSpec::toggle(
                KEY_HEIGHT_FOG_ENABLED,
                "distantsunrises.setting.height_fog_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.height_fog_enabled.tooltip"),
            SettingSpec::range(
                KEY_HEIGHT_FOG_BASE_HEIGHT,
                "distantsunrises.setting.height_fog_base_height",
                -64.0,
                320.0,
                4.0,
                70.0,
            )
            .tooltip("distantsunrises.setting.height_fog_base_height.tooltip"),
            SettingSpec::choice(
                KEY_HEIGHT_FOG_MIX_MODE,
                "distantsunrises.setting.height_fog_mix_mode",
                [
                    "distantsunrises.setting.height_fog_mix.basic",
                    "distantsunrises.setting.height_fog_mix.max",
                    "distantsunrises.setting.height_fog_mix.addition",
                    "distantsunrises.setting.height_fog_mix.multiply",
                    "distantsunrises.setting.height_fog_mix.inverse_multiply",
                ],
                4,
            )
            .tooltip("distantsunrises.setting.height_fog_mix_mode.tooltip"),
            SettingSpec::toggle(
                KEY_CLOUDS_ENABLED,
                "distantsunrises.setting.clouds_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.clouds_enabled.tooltip"),
            SettingSpec::toggle(
                KEY_BEACON_BEAMS_ENABLED,
                "distantsunrises.setting.beacon_beams_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.beacon_beams_enabled.tooltip"),
            SettingSpec::range(
                KEY_SHARPENING_AMOUNT,
                "distantsunrises.setting.sharpening_amount",
                0.0,
                1.0,
                0.05,
                0.35,
            )
            .tooltip("distantsunrises.setting.sharpening_amount.tooltip"),
            SettingSpec::choice(
                KEY_DEBUG_MODE,
                "distantsunrises.setting.debug_mode",
                [
                    "distantsunrises.setting.debug.off",
                    "distantsunrises.setting.debug.detail",
                    "distantsunrises.setting.debug.gen_mode",
                    "distantsunrises.setting.debug.lighting",
                    "distantsunrises.setting.debug.wireframe",
                ],
                0,
            )
            .tooltip("distantsunrises.setting.debug_mode.tooltip"),
            SettingSpec::choice(
                KEY_THREAD_PRESET,
                "distantsunrises.setting.thread_preset",
                ThreadPreset::ALL.map(ThreadPreset::label_key),
                ThreadPreset::Balanced as usize,
            )
            .tooltip("distantsunrises.setting.thread_preset.tooltip"),
        ];
        let mut registered = HashMap::with_capacity(specs.len());
        for spec in specs {
            let name = spec.key.clone();
            if let Ok(key) = registrar.register_setting(spec) {
                registered.insert(name, key);
            }
        }
        if !registered.is_empty()
            && let Ok(mut guard) = key_store().write()
        {
            *guard = registered;
        }
    }

    pub fn read_from_host(host: &dyn Host) -> Self {
        let def = Self::default();
        let read_bool = |name: &str, fallback: bool| {
            setting_key(name)
                .and_then(|k| host.setting_bool(k).ok())
                .unwrap_or(fallback)
        };
        let read_choice = |name: &str, fallback: u32| {
            setting_key(name)
                .and_then(|k| host.setting_choice(k).ok())
                .unwrap_or(fallback)
        };
        let read_f64 = |name: &str| setting_key(name).and_then(|k| host.setting_f64(k).ok());
        Self {
            enabled: read_bool(KEY_ENABLED, def.enabled),
            render_distance_chunks: read_f64(KEY_RENDER_DISTANCE)
                .map(|v| (v.round() as u32).clamp(32, 512))
                .unwrap_or(def.render_distance_chunks),
            horizontal_quality: read_choice(KEY_HORIZONTAL_QUALITY, def.horizontal_quality),
            vertical_quality: read_choice(KEY_VERTICAL_QUALITY, def.vertical_quality),
            distant_generation: read_bool(KEY_DISTANT_GENERATION, def.distant_generation),
            distant_generation_mode: read_choice(
                KEY_DISTANT_GENERATION_MODE,
                def.distant_generation_mode,
            ),
            cave_culling_enabled: read_bool(KEY_CAVE_CULLING, def.cave_culling_enabled),
            cave_culling_height: read_f64(KEY_CAVE_CULLING_HEIGHT)
                .map(|v| (v.round() as i32).clamp(-64, 320))
                .unwrap_or(def.cave_culling_height),
            fog_enabled: read_bool(KEY_FOG_ENABLED, def.fog_enabled),
            noise_enabled: read_bool(KEY_NOISE_ENABLED, def.noise_enabled),
            noise_intensity: read_f64(KEY_NOISE_INTENSITY)
                .map(|v| (v as f32).clamp(0.0, 0.5))
                .unwrap_or(def.noise_intensity),
            overdraw_prevention: read_f64(KEY_OVERDRAW_PREVENTION)
                .map(|v| (v as f32).clamp(0.0, 1.0))
                .unwrap_or(def.overdraw_prevention),
            fade_enabled: read_bool(KEY_FADE_ENABLED, def.fade_enabled),
            shading_enabled: read_bool(KEY_SHADING_ENABLED, def.shading_enabled),
            brightness_multiplier: read_f64(KEY_BRIGHTNESS)
                .map(|v| (v as f32).clamp(0.5, 2.0))
                .unwrap_or(def.brightness_multiplier),
            saturation_multiplier: read_f64(KEY_SATURATION)
                .map(|v| (v as f32).clamp(0.0, 2.0))
                .unwrap_or(def.saturation_multiplier),
            ssao_enabled: read_bool(KEY_SSAO_ENABLED, def.ssao_enabled),
            ssao_quality: read_choice(KEY_SSAO_QUALITY, def.ssao_quality),
            ssao_strength: read_f64(KEY_SSAO_STRENGTH)
                .map(|v| (v as f32).clamp(0.0, 1.0))
                .unwrap_or(def.ssao_strength),
            disable_vanilla_fog: read_bool(KEY_DISABLE_VANILLA_FOG, def.disable_vanilla_fog),
            fog_falloff: read_choice(KEY_FOG_FALLOFF, def.fog_falloff),
            fog_spherical: read_bool(KEY_FOG_SPHERICAL, def.fog_spherical),
            height_fog_enabled: read_bool(KEY_HEIGHT_FOG_ENABLED, def.height_fog_enabled),
            height_fog_base_height: read_f64(KEY_HEIGHT_FOG_BASE_HEIGHT)
                .map(|v| (v as f32).clamp(-64.0, 320.0))
                .unwrap_or(def.height_fog_base_height),
            height_fog_mix_mode: read_choice(KEY_HEIGHT_FOG_MIX_MODE, def.height_fog_mix_mode),
            clouds_enabled: read_bool(KEY_CLOUDS_ENABLED, def.clouds_enabled),
            beacon_beams_enabled: read_bool(KEY_BEACON_BEAMS_ENABLED, def.beacon_beams_enabled),
            sharpening_amount: read_f64(KEY_SHARPENING_AMOUNT)
                .map(|v| (v as f32).clamp(0.0, 1.0))
                .unwrap_or(def.sharpening_amount),
            debug_mode: read_choice(KEY_DEBUG_MODE, def.debug_mode),
            thread_preset: ThreadPreset::from_index(read_choice(
                KEY_THREAD_PRESET,
                def.thread_preset as u32,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_preset_counts_match_dh_formula() {
        // DH javadoc table (4/8/16 cores) for the 0.1 and 1.0 rows, plus 50/25/75%.
        let cases = [
            (ThreadPreset::MinimalImpact, [1, 1, 2]),
            (ThreadPreset::LowImpact, [1, 2, 4]),
            (ThreadPreset::Balanced, [2, 4, 8]),
            (ThreadPreset::Aggressive, [3, 6, 12]),
            (ThreadPreset::IPaidForTheWholeCpu, [4, 8, 16]),
        ];
        for (preset, expected) in cases {
            for (total, want) in [4, 8, 16].into_iter().zip(expected) {
                assert_eq!(preset.thread_count(total), want, "{preset:?} on {total}");
            }
        }
        assert_eq!(ThreadPreset::MinimalImpact.thread_count(0), 1);
        assert_eq!(ThreadPreset::default(), ThreadPreset::Balanced);
        assert_eq!(
            ThreadPreset::parse("I_PAID_FOR_THE_WHOLE_CPU"),
            Some(ThreadPreset::IPaidForTheWholeCpu)
        );
    }
}
