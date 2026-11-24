use mod_api::{ClientHost, Registrar, SettingSpec};

pub const MOD_ID: &str = "distanthorizons";

pub const KEY_ENABLED: &str = "enabled";
pub const KEY_RENDER_DISTANCE: &str = "render_distance_chunks";
pub const KEY_HORIZONTAL_QUALITY: &str = "horizontal_quality";
pub const KEY_DISTANT_GENERATION: &str = "distant_generation";
pub const KEY_FOG_ENABLED: &str = "fog_enabled";
pub const KEY_NOISE_ENABLED: &str = "noise_enabled";
pub const KEY_NOISE_INTENSITY: &str = "noise_intensity";
pub const KEY_OVERDRAW_PREVENTION: &str = "overdraw_prevention";
pub const KEY_FADE_ENABLED: &str = "fade_enabled";
pub const KEY_SHADING_ENABLED: &str = "shading_enabled";

/// Runtime settings mirrored from Sunrise's mod settings UI (`Options > Mods > Distant Horizons`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DhConfig {
    pub enabled: bool,
    pub render_distance_chunks: u32,
    /// 0 = Low, 1 = Medium, 2 = High.
    pub horizontal_quality: u32,
    pub distant_generation: bool,
    pub fog_enabled: bool,
    pub noise_enabled: bool,
    pub noise_intensity: f32,
    pub overdraw_prevention: f32,
    pub fade_enabled: bool,
    pub shading_enabled: bool,
}

impl Default for DhConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            render_distance_chunks: 128,
            horizontal_quality: 1,
            distant_generation: true,
            fog_enabled: true,
            noise_enabled: true,
            noise_intensity: 0.12,
            overdraw_prevention: 0.35,
            fade_enabled: true,
            shading_enabled: true,
        }
    }
}

impl DhConfig {
    pub fn render_distance_blocks(self) -> f32 {
        (self.render_distance_chunks as f32) * 16.0
    }

    /// Distance in blocks at which `detail` (0..=4) should subdivide to `detail - 1`.
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
            _ => 160.0,
        };
        base_chunks * 16.0 * quality_scale
    }

    pub fn register_settings(registrar: &mut dyn Registrar) {
        let specs = [
            SettingSpec::toggle(KEY_ENABLED, "distanthorizons.setting.enabled", true)
                .tooltip("distanthorizons.setting.enabled.tooltip"),
            SettingSpec::range(
                KEY_RENDER_DISTANCE,
                "distanthorizons.setting.render_distance_chunks",
                32.0,
                256.0,
                8.0,
                128.0,
            )
            .tooltip("distanthorizons.setting.render_distance_chunks.tooltip"),
            SettingSpec::choice(
                KEY_HORIZONTAL_QUALITY,
                "distanthorizons.setting.horizontal_quality",
                [
                    "distanthorizons.setting.quality.low",
                    "distanthorizons.setting.quality.medium",
                    "distanthorizons.setting.quality.high",
                ],
                1,
            )
            .tooltip("distanthorizons.setting.horizontal_quality.tooltip"),
            SettingSpec::toggle(
                KEY_DISTANT_GENERATION,
                "distanthorizons.setting.distant_generation",
                true,
            )
            .tooltip("distanthorizons.setting.distant_generation.tooltip"),
            SettingSpec::toggle(KEY_FOG_ENABLED, "distanthorizons.setting.fog_enabled", true)
                .tooltip("distanthorizons.setting.fog_enabled.tooltip"),
            SettingSpec::toggle(
                KEY_NOISE_ENABLED,
                "distanthorizons.setting.noise_enabled",
                true,
            )
            .tooltip("distanthorizons.setting.noise_enabled.tooltip"),
            SettingSpec::range(
                KEY_NOISE_INTENSITY,
                "distanthorizons.setting.noise_intensity",
                0.0,
                0.5,
                0.02,
                0.12,
            )
            .tooltip("distanthorizons.setting.noise_intensity.tooltip"),
            SettingSpec::range(
                KEY_OVERDRAW_PREVENTION,
                "distanthorizons.setting.overdraw_prevention",
                0.0,
                1.0,
                0.05,
                0.35,
            )
            .tooltip("distanthorizons.setting.overdraw_prevention.tooltip"),
            SettingSpec::toggle(
                KEY_FADE_ENABLED,
                "distanthorizons.setting.fade_enabled",
                true,
            )
            .tooltip("distanthorizons.setting.fade_enabled.tooltip"),
            SettingSpec::toggle(
                KEY_SHADING_ENABLED,
                "distanthorizons.setting.shading_enabled",
                true,
            )
            .tooltip("distanthorizons.setting.shading_enabled.tooltip"),
        ];
        for spec in specs {
            let _ = registrar.register_setting(spec);
        }
    }

    pub fn read_from_client(host: &dyn ClientHost) -> Self {
        let def = Self::default();
        Self {
            enabled: host.setting_bool(KEY_ENABLED).unwrap_or(def.enabled),
            render_distance_chunks: host
                .setting_f64(KEY_RENDER_DISTANCE)
                .map(|v| (v.round() as u32).clamp(32, 256))
                .unwrap_or(def.render_distance_chunks),
            horizontal_quality: host
                .setting_choice(KEY_HORIZONTAL_QUALITY)
                .unwrap_or(def.horizontal_quality),
            distant_generation: host
                .setting_bool(KEY_DISTANT_GENERATION)
                .unwrap_or(def.distant_generation),
            fog_enabled: host
                .setting_bool(KEY_FOG_ENABLED)
                .unwrap_or(def.fog_enabled),
            noise_enabled: host
                .setting_bool(KEY_NOISE_ENABLED)
                .unwrap_or(def.noise_enabled),
            noise_intensity: host
                .setting_f64(KEY_NOISE_INTENSITY)
                .map(|v| (v as f32).clamp(0.0, 0.5))
                .unwrap_or(def.noise_intensity),
            overdraw_prevention: host
                .setting_f64(KEY_OVERDRAW_PREVENTION)
                .map(|v| (v as f32).clamp(0.0, 1.0))
                .unwrap_or(def.overdraw_prevention),
            fade_enabled: host
                .setting_bool(KEY_FADE_ENABLED)
                .unwrap_or(def.fade_enabled),
            shading_enabled: host
                .setting_bool(KEY_SHADING_ENABLED)
                .unwrap_or(def.shading_enabled),
        }
    }
}
