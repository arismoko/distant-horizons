//! Client commands (`/ds` / `/distantsunrises`) and Controls keybind handling.
//!
//! Uses Sunrise's `register_client_commands` / `command` hooks and `Registrar::register_key_mapping`
//! so every action works in single-player, LAN, and multiplayer without needing a server-side mod.

use crate::config::{
    DEBUG_OFF, DEBUG_SHOW_DETAIL, DEBUG_SHOW_DETAIL_WIREFRAME, DEBUG_SHOW_GEN_MODE,
    DEBUG_SHOW_LIGHTING, DhConfig, KEY_BEACON_BEAMS_ENABLED, KEY_CAVE_CULLING, KEY_CLOUDS_ENABLED,
    KEY_DEBUG_MODE, KEY_DISTANT_GENERATION, KEY_ENABLED, KEY_FADE_ENABLED, KEY_FOG_ENABLED,
    KEY_HEIGHT_FOG_ENABLED, KEY_HORIZONTAL_QUALITY, KEY_NOISE_ENABLED, KEY_RENDER_DISTANCE,
    KEY_SHARPENING_AMOUNT, KEY_SSAO_ENABLED, KEY_SSAO_QUALITY, KEY_VERTICAL_QUALITY, MOD_ID,
};
use crate::render_bridge::{runtime, stats};
use mod_api::{
    ClientHost, ClientSetupContext, Command, Flow, KeyConflictContext, KeyRegistration,
    ModKeyMappingRef, Registrar, Text,
};
use std::sync::atomic::Ordering;

pub const KEYBIND_TOGGLE_RENDERING: &str = "toggle_rendering";
pub const KEYBIND_CYCLE_QUALITY: &str = "cycle_quality";
pub const KEYBIND_CLEAR_CACHE: &str = "clear_cache";

/// Register Controls keybinds for Distant Sunrises.
pub fn register_key_mappings(registrar: &mut dyn Registrar) {
    registrar.register_key_mapping(
        KeyRegistration::new(KEYBIND_TOGGLE_RENDERING, "key.keyboard.unknown")
            .label("distantsunrises.key.toggle_rendering")
            .category("distantsunrises.key.category")
            .context(KeyConflictContext::InGame),
    );
    registrar.register_key_mapping(
        KeyRegistration::new(KEYBIND_CYCLE_QUALITY, "key.keyboard.unknown")
            .label("distantsunrises.key.cycle_quality")
            .category("distantsunrises.key.category")
            .context(KeyConflictContext::InGame),
    );
    registrar.register_key_mapping(
        KeyRegistration::new(KEYBIND_CLEAR_CACHE, "key.keyboard.unknown")
            .label("distantsunrises.key.clear_cache")
            .category("distantsunrises.key.category")
            .context(KeyConflictContext::InGame),
    );
}

/// Resolved client key mapping handles cached after `client_setup`.
#[derive(Default)]
pub struct DhKeybinds {
    pub toggle_rendering: Option<ModKeyMappingRef>,
    pub cycle_quality: Option<ModKeyMappingRef>,
    pub clear_cache: Option<ModKeyMappingRef>,
}

impl DhKeybinds {
    pub fn resolve(host: &dyn ClientSetupContext) -> Self {
        Self {
            toggle_rendering: host.mod_key_mapping(MOD_ID, KEYBIND_TOGGLE_RENDERING).ok(),
            cycle_quality: host.mod_key_mapping(MOD_ID, KEYBIND_CYCLE_QUALITY).ok(),
            clear_cache: host.mod_key_mapping(MOD_ID, KEYBIND_CLEAR_CACHE).ok(),
        }
    }

    pub fn poll(&self, host: &mut dyn ClientHost) {
        if let Some(m) = self.toggle_rendering
            && m.consume_click(host).unwrap_or(false)
        {
            let now_enabled = toggle_enabled(host);
            let key = if now_enabled {
                "distantsunrises.command.toggled_on"
            } else {
                "distantsunrises.command.toggled_off"
            };
            let _ = host.set_overlay_message(Text::translate(key, Vec::new()), false);
        }

        if let Some(m) = self.cycle_quality
            && m.consume_click(host).unwrap_or(false)
        {
            let next = cycle_horizontal_quality(host);
            let label_key = quality_label_key(next);
            let _ = host.set_overlay_message(
                Text::translate(
                    "distantsunrises.command.quality_set",
                    vec![Text::translate(label_key, Vec::new())],
                ),
                false,
            );
        }

        if let Some(m) = self.clear_cache
            && m.consume_click(host).unwrap_or(false)
        {
            let cleared = clear_active_dimension_cache();
            let _ = host.set_overlay_message(
                Text::translate(
                    "distantsunrises.command.cleared",
                    vec![Text::literal(cleared.to_string())],
                ),
                false,
            );
        }
    }
}

/// Register client command completions for `/ds` and `/distantsunrises`.
pub fn register_client_commands(commands: &mut Vec<Command>) {
    for prefix in ["ds", "distantsunrises"] {
        commands.push(Command::new(prefix));
        commands.push(Command::new(&format!("{prefix} status")));
        commands.push(Command::new(&format!("{prefix} toggle")));
        commands.push(Command::new(&format!("{prefix} radius <chunks:integer>")));
        commands.push(Command::new(&format!("{prefix} quality <preset:word>")));
        commands.push(Command::new(&format!("{prefix} preset <preset:word>")));
        commands.push(Command::new(&format!("{prefix} fog <mode:word>")));
        commands.push(Command::new(&format!("{prefix} heightfog <mode:word>")));
        commands.push(Command::new(&format!("{prefix} ssao <mode:word>")));
        commands.push(Command::new(&format!("{prefix} sharpen <mode:word>")));
        commands.push(Command::new(&format!("{prefix} clouds <mode:word>")));
        commands.push(Command::new(&format!("{prefix} beacons <mode:word>")));
        commands.push(Command::new(&format!("{prefix} debug <mode:word>")));
        commands.push(Command::new(&format!("{prefix} gen <mode:word>")));
        commands.push(Command::new(&format!("{prefix} cavecull <mode:word>")));
        commands.push(Command::new(&format!("{prefix} clear")));
        commands.push(Command::new(&format!("{prefix} remesh")));
        commands.push(Command::new(&format!("{prefix} help")));
    }
}

/// Execute `/ds ...` or `/distantsunrises ...` typed in chat.
pub fn handle_client_command(host: &mut dyn ClientHost, raw: &str) -> Flow {
    let trimmed = raw.trim().trim_start_matches('/');
    let mut parts = trimmed.split_whitespace();
    let Some(root) = parts.next() else {
        return Flow::Continue;
    };
    if !root.eq_ignore_ascii_case("ds") && !root.eq_ignore_ascii_case("distantsunrises") {
        return Flow::Continue;
    }

    let sub = parts.next().unwrap_or("status");
    let arg = parts.next();

    match sub.to_ascii_lowercase().as_str() {
        "status" | "stats" => {
            send_status(host);
        }
        "toggle" => {
            let enabled = toggle_enabled(host);
            let key = if enabled {
                "distantsunrises.command.toggled_on"
            } else {
                "distantsunrises.command.toggled_off"
            };
            let _ = host.chat_line(Text::translate(key, Vec::new()));
        }
        "radius" | "distance" => match arg.and_then(|s| s.parse::<u32>().ok()) {
            Some(chunks) => {
                let clamped = chunks.clamp(32, 512);
                host.set_setting(MOD_ID, KEY_RENDER_DISTANCE, &clamped.to_string());
                refresh_config(host);
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.radius_set",
                    vec![Text::literal(clamped.to_string())],
                ));
            }
            None => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_radius",
                    Vec::new(),
                ));
            }
        },
        "quality" => match arg.map(str::to_ascii_lowercase).as_deref() {
            Some("low") => {
                set_quality(host, 0);
            }
            Some("medium" | "med") => {
                set_quality(host, 1);
            }
            Some("high") => {
                set_quality(host, 2);
            }
            _ => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_quality",
                    Vec::new(),
                ));
            }
        },
        "preset" => match arg.map(str::to_ascii_lowercase).as_deref() {
            Some("low") => apply_preset(
                host,
                QualityPreset {
                    radius: 64,
                    h_qual: 0,
                    v_qual: 0,
                    ssao_q: 0,
                    ssao_on: false,
                    sharpen: 0.0,
                    label_key: "distantsunrises.setting.quality.low",
                },
            ),
            Some("medium" | "med") => apply_preset(
                host,
                QualityPreset {
                    radius: 128,
                    h_qual: 1,
                    v_qual: 1,
                    ssao_q: 1,
                    ssao_on: true,
                    sharpen: 0.25,
                    label_key: "distantsunrises.setting.quality.medium",
                },
            ),
            Some("high") => apply_preset(
                host,
                QualityPreset {
                    radius: 256,
                    h_qual: 2,
                    v_qual: 2,
                    ssao_q: 1,
                    ssao_on: true,
                    sharpen: 0.35,
                    label_key: "distantsunrises.setting.quality.high",
                },
            ),
            Some("extreme") => apply_preset(
                host,
                QualityPreset {
                    radius: 512,
                    h_qual: 2,
                    v_qual: 3,
                    ssao_q: 2,
                    ssao_on: true,
                    sharpen: 0.45,
                    label_key: "distantsunrises.setting.quality.extreme",
                },
            ),
            _ => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_preset",
                    Vec::new(),
                ));
            }
        },
        "ssao" => handle_bool_setting(host, KEY_SSAO_ENABLED, "ssao", arg),
        "sharpen" => match parse_toggle_arg(arg) {
            Some(v) => {
                host.set_setting(
                    MOD_ID,
                    KEY_SHARPENING_AMOUNT,
                    if v { "0.35" } else { "0.0" },
                );
                refresh_config(host);
                let _ = host.chat_line(Text::translate(
                    if v {
                        "distantsunrises.command.setting_on"
                    } else {
                        "distantsunrises.command.setting_off"
                    },
                    vec![Text::literal("sharpen")],
                ));
            }
            None => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_toggle",
                    vec![Text::literal("sharpen")],
                ));
            }
        },
        "heightfog" => handle_bool_setting(host, KEY_HEIGHT_FOG_ENABLED, "heightfog", arg),
        "clouds" => handle_bool_setting(host, KEY_CLOUDS_ENABLED, "clouds", arg),
        "beacons" => handle_bool_setting(host, KEY_BEACON_BEAMS_ENABLED, "beacons", arg),
        "debug" => {
            let parsed = match arg.map(str::to_ascii_lowercase).as_deref() {
                Some("off" | "none" | "0") => {
                    Some((DEBUG_OFF, "distantsunrises.setting.debug.off"))
                }
                Some("detail" | "lod" | "1") => {
                    Some((DEBUG_SHOW_DETAIL, "distantsunrises.setting.debug.detail"))
                }
                Some("gen" | "source" | "2") => Some((
                    DEBUG_SHOW_GEN_MODE,
                    "distantsunrises.setting.debug.gen_mode",
                )),
                Some("light" | "lighting" | "3") => Some((
                    DEBUG_SHOW_LIGHTING,
                    "distantsunrises.setting.debug.lighting",
                )),
                Some("wireframe" | "wire" | "4") => Some((
                    DEBUG_SHOW_DETAIL_WIREFRAME,
                    "distantsunrises.setting.debug.wireframe",
                )),
                _ => None,
            };
            match parsed {
                Some((mode, label_key)) => {
                    host.set_setting(MOD_ID, KEY_DEBUG_MODE, &mode.to_string());
                    refresh_config(host);
                    let _ = host.chat_line(Text::translate(
                        "distantsunrises.command.debug_set",
                        vec![Text::translate(label_key, Vec::new())],
                    ));
                }
                None => {
                    let _ = host.chat_line(Text::translate(
                        "distantsunrises.command.usage_debug",
                        Vec::new(),
                    ));
                }
            }
        }
        "fog" => match parse_toggle_arg(arg) {
            Some(v) => {
                host.set_setting(MOD_ID, KEY_FOG_ENABLED, if v { "true" } else { "false" });
                refresh_config(host);
                let _ = host.chat_line(Text::translate(
                    if v {
                        "distantsunrises.command.fog_on"
                    } else {
                        "distantsunrises.command.fog_off"
                    },
                    Vec::new(),
                ));
            }
            None => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_toggle",
                    vec![Text::literal("fog")],
                ));
            }
        },
        "fade" => match parse_toggle_arg(arg) {
            Some(v) => {
                host.set_setting(MOD_ID, KEY_FADE_ENABLED, if v { "true" } else { "false" });
                refresh_config(host);
                let _ = host.chat_line(Text::translate(
                    if v {
                        "distantsunrises.command.setting_on"
                    } else {
                        "distantsunrises.command.setting_off"
                    },
                    vec![Text::literal("fade")],
                ));
            }
            None => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_toggle",
                    vec![Text::literal("fade")],
                ));
            }
        },
        "noise" => match parse_toggle_arg(arg) {
            Some(v) => {
                host.set_setting(MOD_ID, KEY_NOISE_ENABLED, if v { "true" } else { "false" });
                refresh_config(host);
                let _ = host.chat_line(Text::translate(
                    if v {
                        "distantsunrises.command.setting_on"
                    } else {
                        "distantsunrises.command.setting_off"
                    },
                    vec![Text::literal("noise")],
                ));
            }
            None => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_toggle",
                    vec![Text::literal("noise")],
                ));
            }
        },
        "gen" => match parse_toggle_arg(arg) {
            Some(v) => {
                host.set_setting(
                    MOD_ID,
                    KEY_DISTANT_GENERATION,
                    if v { "true" } else { "false" },
                );
                refresh_config(host);
                let _ = host.chat_line(Text::translate(
                    if v {
                        "distantsunrises.command.setting_on"
                    } else {
                        "distantsunrises.command.setting_off"
                    },
                    vec![Text::literal("gen")],
                ));
            }
            None => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_toggle",
                    vec![Text::literal("gen")],
                ));
            }
        },
        "cavecull" => match parse_toggle_arg(arg) {
            Some(v) => {
                host.set_setting(MOD_ID, KEY_CAVE_CULLING, if v { "true" } else { "false" });
                refresh_config(host);
                let _ = host.chat_line(Text::translate(
                    if v {
                        "distantsunrises.command.setting_on"
                    } else {
                        "distantsunrises.command.setting_off"
                    },
                    vec![Text::literal("cavecull")],
                ));
            }
            None => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_toggle",
                    vec![Text::literal("cavecull")],
                ));
            }
        },
        "vquality" => match arg.and_then(|s| s.parse::<u32>().ok()) {
            Some(v) if v <= 3 => {
                host.set_setting(MOD_ID, KEY_VERTICAL_QUALITY, &v.to_string());
                refresh_config(host);
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.quality_set",
                    vec![Text::literal(v.to_string())],
                ));
            }
            _ => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_quality",
                    Vec::new(),
                ));
            }
        },
        "clear" => {
            let cleared = clear_active_dimension_cache();
            let _ = host.chat_line(Text::translate(
                "distantsunrises.command.cleared",
                vec![Text::literal(cleared.to_string())],
            ));
        }
        "remesh" => {
            let count = remesh_active_dimension();
            let _ = host.chat_line(Text::translate(
                "distantsunrises.command.remeshed",
                vec![Text::literal(count.to_string())],
            ));
        }
        _ => {
            let _ = host.chat_line(Text::translate("distantsunrises.command.help", Vec::new()));
        }
    }

    Flow::Cancel
}

fn parse_toggle_arg(arg: Option<&str>) -> Option<bool> {
    match arg.map(str::to_ascii_lowercase).as_deref() {
        Some("on" | "true" | "1" | "enable") => Some(true),
        Some("off" | "false" | "0" | "disable") => Some(false),
        _ => None,
    }
}

fn quality_label_key(q: u32) -> &'static str {
    match q {
        0 => "distantsunrises.setting.quality.low",
        2 => "distantsunrises.setting.quality.high",
        _ => "distantsunrises.setting.quality.medium",
    }
}

fn refresh_config(host: &dyn ClientHost) -> DhConfig {
    let cfg = DhConfig::read_from_client(host);
    let rt = runtime();
    rt.enabled_flag.store(cfg.enabled, Ordering::Relaxed);
    rt.inner.lock().unwrap().config = cfg;
    cfg
}

fn toggle_enabled(host: &dyn ClientHost) -> bool {
    let current = DhConfig::read_from_client(host).enabled;
    let next = !current;
    host.set_setting(MOD_ID, KEY_ENABLED, if next { "true" } else { "false" });
    refresh_config(host).enabled
}

fn cycle_horizontal_quality(host: &dyn ClientHost) -> u32 {
    let cur = DhConfig::read_from_client(host).horizontal_quality;
    let next = (cur + 1) % 3;
    host.set_setting(MOD_ID, KEY_HORIZONTAL_QUALITY, &next.to_string());
    refresh_config(host).horizontal_quality
}

fn set_quality(host: &mut dyn ClientHost, preset: u32) {
    host.set_setting(MOD_ID, KEY_HORIZONTAL_QUALITY, &preset.to_string());
    refresh_config(host);
    let _ = host.chat_line(Text::translate(
        "distantsunrises.command.quality_set",
        vec![Text::translate(quality_label_key(preset), Vec::new())],
    ));
}

fn handle_bool_setting(host: &mut dyn ClientHost, key: &str, label: &str, arg: Option<&str>) {
    match parse_toggle_arg(arg) {
        Some(v) => {
            host.set_setting(MOD_ID, key, if v { "true" } else { "false" });
            refresh_config(host);
            let _ = host.chat_line(Text::translate(
                if v {
                    "distantsunrises.command.setting_on"
                } else {
                    "distantsunrises.command.setting_off"
                },
                vec![Text::literal(label)],
            ));
        }
        None => {
            let _ = host.chat_line(Text::translate(
                "distantsunrises.command.usage_toggle",
                vec![Text::literal(label)],
            ));
        }
    }
}

struct QualityPreset {
    radius: u32,
    h_qual: u32,
    v_qual: u32,
    ssao_q: u32,
    ssao_on: bool,
    sharpen: f32,
    label_key: &'static str,
}

fn apply_preset(host: &mut dyn ClientHost, p: QualityPreset) {
    host.set_setting(MOD_ID, KEY_RENDER_DISTANCE, &p.radius.to_string());
    host.set_setting(MOD_ID, KEY_HORIZONTAL_QUALITY, &p.h_qual.to_string());
    host.set_setting(MOD_ID, KEY_VERTICAL_QUALITY, &p.v_qual.to_string());
    host.set_setting(
        MOD_ID,
        KEY_SSAO_ENABLED,
        if p.ssao_on { "true" } else { "false" },
    );
    host.set_setting(MOD_ID, KEY_SSAO_QUALITY, &p.ssao_q.to_string());
    host.set_setting(MOD_ID, KEY_SHARPENING_AMOUNT, &format!("{:.2}", p.sharpen));
    refresh_config(host);
    let _ = host.chat_line(Text::translate(
        "distantsunrises.command.preset_set",
        vec![Text::translate(p.label_key, Vec::new())],
    ));
}

fn send_status(host: &mut dyn ClientHost) {
    let s = stats();
    let (dim, cfg, rain, thunder) = {
        let st = runtime().inner.lock().unwrap();
        (
            st.active_dimension.clone(),
            st.config,
            st.rain_level,
            st.thunder_level,
        )
    };
    let state_key = if cfg.enabled {
        "distantsunrises.command.state_on"
    } else {
        "distantsunrises.command.state_off"
    };
    let _ = host.chat_line(Text::translate(
        "distantsunrises.command.status_line",
        vec![
            Text::translate(state_key, Vec::new()),
            Text::literal(dim),
            Text::literal(cfg.render_distance_chunks.to_string()),
            Text::literal(s.active_sections.to_string()),
            Text::literal(s.ready_meshes.to_string()),
            Text::literal(s.rendered_sections.to_string()),
            Text::literal(s.rendered_quads.to_string()),
            Text::literal(format!("{:.0}%/{:.0}%", rain * 100.0, thunder * 100.0)),
        ],
    ));
}

/// Drop cached sections and meshes for the active dimension so they rebuild cleanly.
pub fn clear_active_dimension_cache() -> usize {
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let dim = st.active_dimension.clone();
    let removed = st
        .dimensions
        .remove(&dim)
        .map(|d| d.sections.len())
        .unwrap_or(0);
    st.meshed_sections.remove(&dim);
    st.pending_gpu_uploads.retain(|m| m.dimension != dim);
    st.remote_requested.remove(&dim);
    st.gpu_reset_epoch = st.gpu_reset_epoch.wrapping_add(1);
    removed
}

/// Force re-meshing of all populated sections in the active dimension (useful after changing
/// visual settings such as saturation or brightness).
pub fn remesh_active_dimension() -> usize {
    let rt = runtime();
    let mut st = rt.inner.lock().unwrap();
    let dim = st.active_dimension.clone();
    let mut count = 0usize;
    if let Some(store) = st.dimensions.get(&dim) {
        for sec in store.sections.values() {
            if sec.populated > 0 {
                rt.mesher.submit(dim.clone(), sec.clone());
                count += 1;
            }
        }
    }
    st.gpu_reset_epoch = st.gpu_reset_epoch.wrapping_add(1);
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_parser_handles_expected_keywords() {
        assert_eq!(parse_toggle_arg(Some("on")), Some(true));
        assert_eq!(parse_toggle_arg(Some("TRUE")), Some(true));
        assert_eq!(parse_toggle_arg(Some("off")), Some(false));
        assert_eq!(parse_toggle_arg(Some("0")), Some(false));
        assert_eq!(parse_toggle_arg(Some("maybe")), None);
        assert_eq!(parse_toggle_arg(None), None);
    }

    #[test]
    fn client_command_list_covers_ds_and_distantsunrises_aliases() {
        let mut cmds = Vec::new();
        register_client_commands(&mut cmds);
        assert!(cmds.iter().any(|c| c.usage == "ds status"));
        assert!(cmds.iter().any(|c| c.usage == "ds radius <chunks:integer>"));
        assert!(cmds.iter().any(|c| c.usage == "distantsunrises clear"));
        assert!(cmds.iter().all(|c| c.level == 0));
    }
}
