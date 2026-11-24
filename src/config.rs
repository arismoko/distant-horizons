use mod_api::{ClientHost, Registrar, SettingSpec};

pub const MOD_ID: &str = "distanthorizons";

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

/// Debug visualization modes matching upstream `EDhApiDebugRendering`.
pub const DEBUG_OFF: u32 = 0;
pub const DEBUG_SHOW_DETAIL: u32 = 1;
pub const DEBUG_SHOW_GEN_MODE: u32 = 2;
pub const DEBUG_SHOW_LIGHTING: u32 = 3;
pub const DEBUG_SHOW_DETAIL_WIREFRAME: u32 = 4;

/// Runtime settings mirrored from Sunrise's mod settings UI (`Options > Mods > Distant Horizons`).
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
            overdraw_prevention: 0.35,
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
            SettingSpec::toggle(KEY_ENABLED, "distanthorizons.setting.enabled", true)
                .tooltip("distanthorizons.setting.enabled.tooltip"),
            SettingSpec::range(
                KEY_RENDER_DISTANCE,
                "distanthorizons.setting.render_distance_chunks",
                32.0,
                512.0,
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
            SettingSpec::choice(
                KEY_VERTICAL_QUALITY,
                "distanthorizons.setting.vertical_quality",
                [
                    "distanthorizons.setting.quality.low",
                    "distanthorizons.setting.quality.medium",
                    "distanthorizons.setting.quality.high",
                    "distanthorizons.setting.quality.extreme",
                ],
                1,
            )
            .tooltip("distanthorizons.setting.vertical_quality.tooltip"),
            SettingSpec::toggle(
                KEY_DISTANT_GENERATION,
                "distanthorizons.setting.distant_generation",
                true,
            )
            .tooltip("distanthorizons.setting.distant_generation.tooltip"),
            SettingSpec::choice(
                KEY_DISTANT_GENERATION_MODE,
                "distanthorizons.setting.distant_generation_mode",
                [
                    "distanthorizons.setting.gen_mode.surface",
                    "distanthorizons.setting.gen_mode.carvers",
                    "distanthorizons.setting.gen_mode.features",
                    "distanthorizons.setting.gen_mode.full",
                ],
                1,
            )
            .tooltip("distanthorizons.setting.distant_generation_mode.tooltip"),
            SettingSpec::toggle(
                KEY_CAVE_CULLING,
                "distanthorizons.setting.cave_culling_enabled",
                true,
            )
            .tooltip("distanthorizons.setting.cave_culling_enabled.tooltip"),
            SettingSpec::range(
                KEY_CAVE_CULLING_HEIGHT,
                "distanthorizons.setting.cave_culling_height",
                -64.0,
                320.0,
                4.0,
                60.0,
            )
            .tooltip("distanthorizons.setting.cave_culling_height.tooltip"),
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
            SettingSpec::range(
                KEY_BRIGHTNESS,
                "distanthorizons.setting.brightness_multiplier",
                0.5,
                2.0,
                0.05,
                1.0,
            )
            .tooltip("distanthorizons.setting.brightness_multiplier.tooltip"),
            SettingSpec::range(
                KEY_SATURATION,
                "distanthorizons.setting.saturation_multiplier",
                0.0,
                2.0,
                0.05,
                1.0,
            )
            .tooltip("distanthorizons.setting.saturation_multiplier.tooltip"),
            SettingSpec::toggle(
                KEY_SSAO_ENABLED,
                "distanthorizons.setting.ssao_enabled",
                true,
            )
            .tooltip("distanthorizons.setting.ssao_enabled.tooltip"),
            SettingSpec::choice(
                KEY_SSAO_QUALITY,
                "distanthorizons.setting.ssao_quality",
                [
                    "distanthorizons.setting.quality.low",
                    "distanthorizons.setting.quality.medium",
                    "distanthorizons.setting.quality.high",
                ],
                1,
            )
            .tooltip("distanthorizons.setting.ssao_quality.tooltip"),
            SettingSpec::range(
                KEY_SSAO_STRENGTH,
                "distanthorizons.setting.ssao_strength",
                0.0,
                1.0,
                0.05,
                0.45,
            )
            .tooltip("distanthorizons.setting.ssao_strength.tooltip"),
            SettingSpec::toggle(
                KEY_DISABLE_VANILLA_FOG,
                "distanthorizons.setting.disable_vanilla_fog",
                true,
            )
            .tooltip("distanthorizons.setting.disable_vanilla_fog.tooltip"),
            SettingSpec::choice(
                KEY_FOG_FALLOFF,
                "distanthorizons.setting.fog_falloff",
                [
                    "distanthorizons.setting.fog_falloff.linear",
                    "distanthorizons.setting.fog_falloff.exponential",
                    "distanthorizons.setting.fog_falloff.exponential_squared",
                ],
                2,
            )
            .tooltip("distanthorizons.setting.fog_falloff.tooltip"),
            SettingSpec::toggle(
                KEY_FOG_SPHERICAL,
                "distanthorizons.setting.fog_spherical",
                false,
            )
            .tooltip("distanthorizons.setting.fog_spherical.tooltip"),
            SettingSpec::toggle(
                KEY_HEIGHT_FOG_ENABLED,
                "distanthorizons.setting.height_fog_enabled",
                true,
            )
            .tooltip("distanthorizons.setting.height_fog_enabled.tooltip"),
            SettingSpec::range(
                KEY_HEIGHT_FOG_BASE_HEIGHT,
                "distanthorizons.setting.height_fog_base_height",
                -64.0,
                320.0,
                4.0,
                70.0,
            )
            .tooltip("distanthorizons.setting.height_fog_base_height.tooltip"),
            SettingSpec::choice(
                KEY_HEIGHT_FOG_MIX_MODE,
                "distanthorizons.setting.height_fog_mix_mode",
                [
                    "distanthorizons.setting.height_fog_mix.basic",
                    "distanthorizons.setting.height_fog_mix.max",
                    "distanthorizons.setting.height_fog_mix.addition",
                    "distanthorizons.setting.height_fog_mix.multiply",
                    "distanthorizons.setting.height_fog_mix.inverse_multiply",
                ],
                4,
            )
            .tooltip("distanthorizons.setting.height_fog_mix_mode.tooltip"),
            SettingSpec::toggle(
                KEY_CLOUDS_ENABLED,
                "distanthorizons.setting.clouds_enabled",
                true,
            )
            .tooltip("distanthorizons.setting.clouds_enabled.tooltip"),
            SettingSpec::toggle(
                KEY_BEACON_BEAMS_ENABLED,
                "distanthorizons.setting.beacon_beams_enabled",
                true,
            )
            .tooltip("distanthorizons.setting.beacon_beams_enabled.tooltip"),
            SettingSpec::range(
                KEY_SHARPENING_AMOUNT,
                "distanthorizons.setting.sharpening_amount",
                0.0,
                1.0,
                0.05,
                0.35,
            )
            .tooltip("distanthorizons.setting.sharpening_amount.tooltip"),
            SettingSpec::choice(
                KEY_DEBUG_MODE,
                "distanthorizons.setting.debug_mode",
                [
                    "distanthorizons.setting.debug.off",
                    "distanthorizons.setting.debug.detail",
                    "distanthorizons.setting.debug.gen_mode",
                    "distanthorizons.setting.debug.lighting",
                    "distanthorizons.setting.debug.wireframe",
                ],
                0,
            )
            .tooltip("distanthorizons.setting.debug_mode.tooltip"),
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
                .map(|v| (v.round() as u32).clamp(32, 512))
                .unwrap_or(def.render_distance_chunks),
            horizontal_quality: host
                .setting_choice(KEY_HORIZONTAL_QUALITY)
                .unwrap_or(def.horizontal_quality),
            vertical_quality: host
                .setting_choice(KEY_VERTICAL_QUALITY)
                .unwrap_or(def.vertical_quality),
            distant_generation: host
                .setting_bool(KEY_DISTANT_GENERATION)
                .unwrap_or(def.distant_generation),
            distant_generation_mode: host
                .setting_choice(KEY_DISTANT_GENERATION_MODE)
                .unwrap_or(def.distant_generation_mode),
            cave_culling_enabled: host
                .setting_bool(KEY_CAVE_CULLING)
                .unwrap_or(def.cave_culling_enabled),
            cave_culling_height: host
                .setting_f64(KEY_CAVE_CULLING_HEIGHT)
                .map(|v| (v.round() as i32).clamp(-64, 320))
                .unwrap_or(def.cave_culling_height),
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
            brightness_multiplier: host
                .setting_f64(KEY_BRIGHTNESS)
                .map(|v| (v as f32).clamp(0.5, 2.0))
                .unwrap_or(def.brightness_multiplier),
            saturation_multiplier: host
                .setting_f64(KEY_SATURATION)
                .map(|v| (v as f32).clamp(0.0, 2.0))
                .unwrap_or(def.saturation_multiplier),
            ssao_enabled: host
                .setting_bool(KEY_SSAO_ENABLED)
                .unwrap_or(def.ssao_enabled),
            ssao_quality: host
                .setting_choice(KEY_SSAO_QUALITY)
                .unwrap_or(def.ssao_quality),
            ssao_strength: host
                .setting_f64(KEY_SSAO_STRENGTH)
                .map(|v| (v as f32).clamp(0.0, 1.0))
                .unwrap_or(def.ssao_strength),
            disable_vanilla_fog: host
                .setting_bool(KEY_DISABLE_VANILLA_FOG)
                .unwrap_or(def.disable_vanilla_fog),
            fog_falloff: host
                .setting_choice(KEY_FOG_FALLOFF)
                .unwrap_or(def.fog_falloff),
            fog_spherical: host
                .setting_bool(KEY_FOG_SPHERICAL)
                .unwrap_or(def.fog_spherical),
            height_fog_enabled: host
                .setting_bool(KEY_HEIGHT_FOG_ENABLED)
                .unwrap_or(def.height_fog_enabled),
            height_fog_base_height: host
                .setting_f64(KEY_HEIGHT_FOG_BASE_HEIGHT)
                .map(|v| (v as f32).clamp(-64.0, 320.0))
                .unwrap_or(def.height_fog_base_height),
            height_fog_mix_mode: host
                .setting_choice(KEY_HEIGHT_FOG_MIX_MODE)
                .unwrap_or(def.height_fog_mix_mode),
            clouds_enabled: host
                .setting_bool(KEY_CLOUDS_ENABLED)
                .unwrap_or(def.clouds_enabled),
            beacon_beams_enabled: host
                .setting_bool(KEY_BEACON_BEAMS_ENABLED)
                .unwrap_or(def.beacon_beams_enabled),
            sharpening_amount: host
                .setting_f64(KEY_SHARPENING_AMOUNT)
                .map(|v| (v as f32).clamp(0.0, 1.0))
                .unwrap_or(def.sharpening_amount),
            debug_mode: host
                .setting_choice(KEY_DEBUG_MODE)
                .unwrap_or(def.debug_mode),
        }
    }
}
