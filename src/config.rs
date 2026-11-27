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
pub const KEY_DISTANT_GENERATION: &str = "distant_generation";
// A new key keeps persisted indices from the old four-choice setting from being reinterpreted.
pub const KEY_DISTANT_GENERATION_STAGE: &str = "distant_generation_stage";
pub const KEY_FOG_ENABLED: &str = "fog_enabled";
pub const KEY_OVERDRAW_PREVENTION: &str = "overdraw_prevention";
pub const KEY_SHADING_ENABLED: &str = "shading_enabled";
pub const KEY_SSAO_ENABLED: &str = "ssao_enabled";
pub const KEY_DISABLE_VANILLA_FOG: &str = "disable_vanilla_fog";
pub const KEY_FOG_SPHERICAL: &str = "fog_spherical";
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

/// Maximum extra vertical spans retained per column beneath the primary surface.
pub const DEFAULT_MAX_EXTRA_SPANS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistantGenerationMode {
    Surface,
    Features,
    Full,
}

impl DistantGenerationMode {
    pub const fn from_choice_index(index: u32) -> Self {
        match index {
            0 => Self::Surface,
            2 => Self::Full,
            _ => Self::Features,
        }
    }

    pub const fn feature_stage(self) -> Option<mod_api::GenerationStage> {
        match self {
            Self::Surface => None,
            Self::Features => Some(mod_api::GenerationStage::Features),
            Self::Full => Some(mod_api::GenerationStage::Light),
        }
    }
}

fn distant_generation_stage_spec() -> SettingSpec {
    SettingSpec::choice(
        KEY_DISTANT_GENERATION_STAGE,
        "distantsunrises.setting.distant_generation_stage",
        [
            "distantsunrises.setting.gen_mode.surface",
            "distantsunrises.setting.gen_mode.features",
            "distantsunrises.setting.gen_mode.full",
        ],
        1, // Features; legacy indices are not read under this key.
    )
    .tooltip("distantsunrises.setting.distant_generation_stage.tooltip")
}

/// Runtime settings mirrored from Sunrise's mod settings UI (`Options > Mods > Distant Sunrises`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DhConfig {
    pub enabled: bool,
    pub render_distance_chunks: u32,
    /// 0 = Low, 1 = Medium, 2 = High.
    pub horizontal_quality: u32,
    pub distant_generation: bool,
    pub distant_generation_mode: DistantGenerationMode,
    pub fog_enabled: bool,
    pub overdraw_prevention: f32,
    pub shading_enabled: bool,
    pub ssao_enabled: bool,
    pub disable_vanilla_fog: bool,
    pub fog_spherical: bool,
    /// Worldgen/ingest thread preset (`EDhApiThreadPreset`).
    pub thread_preset: ThreadPreset,
}

impl Default for DhConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            render_distance_chunks: 128,
            horizontal_quality: 1,
            distant_generation: true,
            distant_generation_mode: DistantGenerationMode::Features,
            fog_enabled: true,
            overdraw_prevention: 0.9,
            shading_enabled: true,
            ssao_enabled: true,
            disable_vanilla_fog: true,
            fog_spherical: false,
            thread_preset: ThreadPreset::Balanced,
        }
    }
}

impl DhConfig {
    pub fn render_distance_blocks(self) -> f32 {
        (self.render_distance_chunks as f32) * 16.0
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
            SettingSpec::toggle(
                KEY_DISTANT_GENERATION,
                "distantsunrises.setting.distant_generation",
                true,
            )
            .tooltip("distantsunrises.setting.distant_generation.tooltip"),
            distant_generation_stage_spec(),
            SettingSpec::toggle(KEY_FOG_ENABLED, "distantsunrises.setting.fog_enabled", true)
                .tooltip("distantsunrises.setting.fog_enabled.tooltip"),
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
                KEY_SHADING_ENABLED,
                "distantsunrises.setting.shading_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.shading_enabled.tooltip"),
            SettingSpec::toggle(
                KEY_SSAO_ENABLED,
                "distantsunrises.setting.ssao_enabled",
                true,
            )
            .tooltip("distantsunrises.setting.ssao_enabled.tooltip"),
            SettingSpec::toggle(
                KEY_DISABLE_VANILLA_FOG,
                "distantsunrises.setting.disable_vanilla_fog",
                true,
            )
            .tooltip("distantsunrises.setting.disable_vanilla_fog.tooltip"),
            SettingSpec::toggle(
                KEY_FOG_SPHERICAL,
                "distantsunrises.setting.fog_spherical",
                false,
            )
            .tooltip("distantsunrises.setting.fog_spherical.tooltip"),
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
            distant_generation: read_bool(KEY_DISTANT_GENERATION, def.distant_generation),
            distant_generation_mode: setting_key(KEY_DISTANT_GENERATION_STAGE)
                .and_then(|k| host.setting_choice(k).ok())
                .map(DistantGenerationMode::from_choice_index)
                .unwrap_or(def.distant_generation_mode),
            fog_enabled: read_bool(KEY_FOG_ENABLED, def.fog_enabled),
            overdraw_prevention: read_f64(KEY_OVERDRAW_PREVENTION)
                .map(|v| (v as f32).clamp(0.0, 1.0))
                .unwrap_or(def.overdraw_prevention),
            shading_enabled: read_bool(KEY_SHADING_ENABLED, def.shading_enabled),
            ssao_enabled: read_bool(KEY_SSAO_ENABLED, def.ssao_enabled),
            disable_vanilla_fog: read_bool(KEY_DISABLE_VANILLA_FOG, def.disable_vanilla_fog),
            fog_spherical: read_bool(KEY_FOG_SPHERICAL, def.fog_spherical),
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
    use mod_api::SettingValue;

    #[test]
    fn distant_generation_choice_indices() {
        use DistantGenerationMode::{Features, Full, Surface};
        for (index, expected) in [(0, Surface), (1, Features), (2, Full), (3, Features)] {
            assert_eq!(DistantGenerationMode::from_choice_index(index), expected);
        }
    }

    #[test]
    fn distant_generation_feature_stages() {
        use DistantGenerationMode::{Features, Full, Surface};
        use mod_api::GenerationStage;
        assert!(Surface.feature_stage().is_none());
        assert!(matches!(
            Features.feature_stage(),
            Some(GenerationStage::Features)
        ));
        assert!(matches!(Full.feature_stage(), Some(GenerationStage::Light)));
    }

    #[test]
    fn distant_generation_stage_reads_new_key_without_reusing_legacy_indices() {
        use DistantGenerationMode::{Features, Full, Surface};
        let spec = distant_generation_stage_spec();
        assert_eq!(spec.key, "distant_generation_stage");
        for (stored, expected_index, expected_mode) in [
            // An old settings folder has no value under the new key.
            (None, 1, Features),
            (Some("0"), 0, Surface),
            (Some("1"), 1, Features),
            (Some("2"), 2, Full),
            (Some("3"), 1, Features),
            (Some("invalid"), 1, Features),
        ] {
            let SettingValue::Choice(index) = spec.read(stored) else {
                panic!("generation stage must be a choice");
            };
            assert_eq!(index, expected_index, "stored value: {stored:?}");
            assert_eq!(
                DistantGenerationMode::from_choice_index(index as u32),
                expected_mode
            );
        }
    }

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
