//! Client commands (`/ds` / `/distantsunrises`) and Controls keybind handling.
//!
//! Uses Sunrise's `register_client_commands` / `command` hooks and `Registrar::register_key_mapping`
//! so every action works in single-player, LAN, and multiplayer without needing a server-side mod.

use crate::config::{
    DEBUG_OFF, DEBUG_SHOW_DETAIL, DEBUG_SHOW_DETAIL_WIREFRAME, DEBUG_SHOW_GEN_MODE,
    DEBUG_SHOW_LIGHTING, DhConfig, KEY_CAVE_CULLING, KEY_DEBUG_MODE, KEY_DISTANT_GENERATION,
    KEY_ENABLED, KEY_FADE_ENABLED, KEY_FOG_ENABLED, KEY_HEIGHT_FOG_ENABLED, KEY_HORIZONTAL_QUALITY,
    KEY_NOISE_ENABLED, KEY_RENDER_DISTANCE, KEY_SHARPENING_AMOUNT, KEY_SSAO_ENABLED,
    KEY_SSAO_QUALITY, KEY_THREAD_PRESET, KEY_VERTICAL_QUALITY, MOD_ID, ThreadPreset, setting_key,
};
use crate::render_bridge::{runtime, set_enabled, stats};
use mod_api::{
    ClientCommandRegistration, ClientHost, ClientSetupContext, Command, CommandArgumentValue,
    CommandInvocation, KeyConflictContext, KeyRegistration, ModKeyMappingRef, Registrar,
    SettingValue, Text,
};

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
            let message = clear_active_dimension_cache(host);
            let _ = host.set_overlay_message(message, false);
        }
    }
}

fn write_setting(host: &mut dyn ClientHost, key_name: &str, value: SettingValue) {
    if let Some(key) = setting_key(key_name) {
        let _ = host.set_setting(key, value);
    }
}

/// Register client command completions and executors for `/ds` and `/distantsunrises`.
pub fn register_client_commands(commands: &mut Vec<ClientCommandRegistration>) {
    let usages = [
        "",
        " status",
        " stats",
        " toggle",
        " radius <chunks:integer>",
        " distance <chunks:integer>",
        " quality <preset:word>",
        " vquality <level:integer>",
        " preset <preset:word>",
        " fog <mode:word>",
        " fade <mode:word>",
        " noise <mode:word>",
        " heightfog <mode:word>",
        " ssao <mode:word>",
        " sharpen <mode:word>",
        " debug <mode:word>",
        " gen <mode:word>",
        " threads <preset:word>",
        " cavecull <mode:word>",
        " clear",
        " remesh",
        " help",
    ];
    for prefix in ["ds", "distantsunrises"] {
        for suffix in usages {
            commands.push(Command::new(&format!("{prefix}{suffix}")).executes_client(
                |host, inv| {
                    handle_client_invocation(host, inv);
                    Some(1)
                },
            ));
        }
    }
}

fn handle_client_invocation(host: &mut dyn ClientHost, inv: &CommandInvocation) {
    let sub = inv.literals.get(1).map(String::as_str).unwrap_or("status");
    let arg_owned = inv.arguments.first().and_then(|a| match &a.value {
        CommandArgumentValue::String(s) => Some(s.clone()),
        CommandArgumentValue::Integer(n) => Some(n.to_string()),
        _ => None,
    });
    dispatch_client_command(host, sub, arg_owned.as_deref());
}

fn dispatch_client_command(host: &mut dyn ClientHost, sub: &str, arg: Option<&str>) {
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
                write_setting(
                    host,
                    KEY_RENDER_DISTANCE,
                    SettingValue::Range(f64::from(clamped)),
                );
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
                write_setting(
                    host,
                    KEY_SHARPENING_AMOUNT,
                    SettingValue::Range(if v { 0.35 } else { 0.0 }),
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
                    write_setting(host, KEY_DEBUG_MODE, SettingValue::Choice(mode as usize));
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
                write_setting(host, KEY_FOG_ENABLED, SettingValue::Toggle(v));
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
                write_setting(host, KEY_FADE_ENABLED, SettingValue::Toggle(v));
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
                write_setting(host, KEY_NOISE_ENABLED, SettingValue::Toggle(v));
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
                write_setting(host, KEY_DISTANT_GENERATION, SettingValue::Toggle(v));
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
                write_setting(host, KEY_CAVE_CULLING, SettingValue::Toggle(v));
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
                write_setting(host, KEY_VERTICAL_QUALITY, SettingValue::Choice(v as usize));
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
        "threads" => match arg.and_then(ThreadPreset::parse) {
            Some(preset) => {
                write_setting(
                    host,
                    KEY_THREAD_PRESET,
                    SettingValue::Choice(preset as usize),
                );
                refresh_config(host);
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.threads_set",
                    vec![Text::translate(preset.label_key(), Vec::new())],
                ));
            }
            None => {
                let _ = host.chat_line(Text::translate(
                    "distantsunrises.command.usage_threads",
                    Vec::new(),
                ));
            }
        },
        "clear" => {
            let message = clear_active_dimension_cache(host);
            let _ = host.chat_line(message);
        }
        "remesh" => {
            let count = crate::pipeline::invalidate_models();
            let _ = host.chat_line(Text::translate(
                "distantsunrises.command.remeshed",
                vec![Text::literal(count.to_string())],
            ));
        }
        _ => {
            let _ = host.chat_line(Text::translate("distantsunrises.command.help", Vec::new()));
        }
    }
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
    let cfg = DhConfig::read_from_host(host);
    let rt = runtime();
    set_enabled(cfg.enabled);
    rt.inner.lock().unwrap().config = cfg;
    rt.gen_pool.reconfigure(cfg.thread_preset);
    cfg
}

fn toggle_enabled(host: &mut dyn ClientHost) -> bool {
    let current = DhConfig::read_from_host(host).enabled;
    let next = !current;
    write_setting(host, KEY_ENABLED, SettingValue::Toggle(next));
    refresh_config(host).enabled
}

fn cycle_horizontal_quality(host: &mut dyn ClientHost) -> u32 {
    let cur = DhConfig::read_from_host(host).horizontal_quality;
    let next = (cur + 1) % 3;
    write_setting(
        host,
        KEY_HORIZONTAL_QUALITY,
        SettingValue::Choice(next as usize),
    );
    refresh_config(host).horizontal_quality
}

fn set_quality(host: &mut dyn ClientHost, preset: u32) {
    write_setting(
        host,
        KEY_HORIZONTAL_QUALITY,
        SettingValue::Choice(preset as usize),
    );
    refresh_config(host);
    let _ = host.chat_line(Text::translate(
        "distantsunrises.command.quality_set",
        vec![Text::translate(quality_label_key(preset), Vec::new())],
    ));
}

fn handle_bool_setting(host: &mut dyn ClientHost, key: &str, label: &str, arg: Option<&str>) {
    match parse_toggle_arg(arg) {
        Some(v) => {
            write_setting(host, key, SettingValue::Toggle(v));
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
    sharpen: f64,
    label_key: &'static str,
}

fn apply_preset(host: &mut dyn ClientHost, p: QualityPreset) {
    write_setting(
        host,
        KEY_RENDER_DISTANCE,
        SettingValue::Range(f64::from(p.radius)),
    );
    write_setting(
        host,
        KEY_HORIZONTAL_QUALITY,
        SettingValue::Choice(p.h_qual as usize),
    );
    write_setting(
        host,
        KEY_VERTICAL_QUALITY,
        SettingValue::Choice(p.v_qual as usize),
    );
    write_setting(host, KEY_SSAO_ENABLED, SettingValue::Toggle(p.ssao_on));
    write_setting(
        host,
        KEY_SSAO_QUALITY,
        SettingValue::Choice(p.ssao_q as usize),
    );
    write_setting(host, KEY_SHARPENING_AMOUNT, SettingValue::Range(p.sharpen));
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
            Text::literal(s.section_high_water.to_string()),
            Text::literal(s.resident_sections.to_string()),
            Text::literal(format!("{:.0}%/{:.0}%", rain * 100.0, thunder * 100.0)),
        ],
    ));
}

/// Refresh only live octree terrain; leave provider data and disk state alone. The legacy
/// `dimensions` and `remote_requested` runtime fields are deliberately untouched: they still
/// have active readers and writers (see the #47 commit body).
/// Local sessions report only the queued remesh count; remote sessions report evicted server
/// copies and the retained sections queued for remeshing.
fn clear_active_dimension_cache(host: &dyn ClientHost) -> Text {
    let Some(server) = host.client_session().client_server() else {
        return Text::translate(
            "distantsunrises.command.cleared_local",
            vec![Text::literal("0")],
        );
    };
    let dim = runtime().inner.lock().unwrap().active_dimension.clone();
    let (rebuild, dropped) = crate::pipeline::clear_client_cache(&dim, !server.integrated_local);
    if server.integrated_local {
        Text::translate(
            "distantsunrises.command.cleared_local",
            vec![Text::literal(rebuild.to_string())],
        )
    } else {
        Text::translate(
            "distantsunrises.command.cleared_remote",
            vec![
                Text::literal(dropped.to_string()),
                Text::literal(rebuild.to_string()),
            ],
        )
    }
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
        assert!(cmds.len() >= 12);
    }
}
