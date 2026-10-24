//! Pi's layered settings (`core/settings-manager.ts`): global `settings.json` under the agent
//! directory, then the project's `.pi/settings.json`; nested objects merge recursively, everything
//! else is replaced by the more specific layer. Unknown keys are preserved when writing.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Global,
    Project,
}

/// A settings file that could not be read or parsed (surfaced as a diagnostic, never fatal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsError {
    pub scope: Scope,
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error("Project is not trusted; refusing to write project settings")]
    ProjectNotTrusted,
    #[error("{0}: {1}")]
    Io(PathBuf, std::io::Error),
}

#[derive(Debug, Clone)]
pub struct Settings {
    global_path: PathBuf,
    project_path: PathBuf,
    project_trusted: bool,
    global: Map<String, Value>,
    project: Map<String, Value>,
    errors: Vec<SettingsError>,
}

/// Pi's `SETTINGS_DEFAULTS`, used when neither layer sets a value.
pub fn defaults() -> Value {
    json!({
        "transport": "auto",
        "steeringMode": "one-at-a-time",
        "followUpMode": "one-at-a-time",
        "compaction": {"enabled": true, "reserveTokens": 16384, "keepRecentTokens": 20000},
        "branchSummary": {"reserveTokens": 16384, "skipPrompt": false},
        "retry": {"enabled": true, "maxRetries": 3, "baseDelayMs": 2000,
                  "provider": {"maxRetryDelayMs": 60000}},
        "hideThinkingBlock": false,
        "quietStartup": false,
        "defaultProjectTrust": "ask",
        "collapseChangelog": false,
        "enableSkillCommands": true,
        "terminal": {"showImages": true, "imageWidthCells": 60, "clearOnShrink": false,
                     "showTerminalProgress": false},
        "images": {"autoResize": true, "blockImages": false},
        "doubleEscapeAction": "tree",
        "treeFilterMode": "default",
        "editorPaddingX": 0,
        "outputPad": 1,
        "autocompleteMaxVisible": 5,
        "markdown": {"codeBlockIndent": "  ", "mermaid": "streaming"},
        "codemode": {"mode": "on", "inlineBudget": 3000},
        "cacheWarming": "streaming",
        "tuiMode": "fullscreen",
        "fullscreenExitOutput": "transcript",
        "fullscreenScrollbar": "auto",
        "fullscreenCopyOnSelect": true,
        "fullscreenWheelScrollLines": "auto",
    })
}

/// Pi's `applyHttpProxySettings`: the `httpProxy` setting fills `HTTP_PROXY` / `HTTPS_PROXY`
/// only where they are not already set. Returns the variables to set.
pub fn proxy_env_updates(
    setting: Option<&str>,
    http_proxy_set: bool,
    https_proxy_set: bool,
) -> Vec<(&'static str, String)> {
    let Some(proxy) = setting.map(str::trim).filter(|p| !p.is_empty()) else {
        return Vec::new();
    };
    [
        ("HTTP_PROXY", http_proxy_set),
        ("HTTPS_PROXY", https_proxy_set),
    ]
    .into_iter()
    .filter(|(_, already)| !already)
    .map(|(name, _)| (name, proxy.to_owned()))
    .collect()
}

/// Settings editable as text, with their allowed values (a subset of Pi's settings schema).
const ENUM_SETTINGS: &[(&str, &[&str])] = &[
    ("doubleEscapeAction", &["tree", "fork", "none"]),
    ("tuiMode", &["fullscreen", "regular"]),
    ("steeringMode", &["all", "one-at-a-time"]),
    ("followUpMode", &["all", "one-at-a-time"]),
    ("defaultProjectTrust", &["ask", "always", "never"]),
    ("transport", &["auto", "sse", "websocket"]),
    (
        "treeFilterMode",
        &["default", "no-tools", "user-only", "labeled-only", "all"],
    ),
    ("codemode.mode", &["on", "only"]),
];
const NUMBER_SETTINGS: &[(&str, u64, u64)] = &[
    ("autocompleteMaxVisible", 1, 20),
    ("editorPaddingX", 0, 8),
    ("outputPad", 0, 8),
    ("codemode.inlineBudget", 0, 1_000_000),
    ("compaction.reserveTokens", 0, 10_000_000),
    ("compaction.keepRecentTokens", 0, 10_000_000),
    ("retry.maxRetries", 0, 100),
];

/// Allowed range for an integer setting, when the key is a numeric setting.
pub fn number_setting_range(key: &str) -> Option<(u64, u64)> {
    NUMBER_SETTINGS
        .iter()
        .find_map(|(candidate, minimum, maximum)| {
            (*candidate == key).then_some((*minimum, *maximum))
        })
}

/// Allowed choices for an enum-valued setting, when the key is an enum setting.
pub fn enum_setting_options(key: &str) -> Option<&'static [&'static str]> {
    ENUM_SETTINGS
        .iter()
        .find_map(|(candidate, choices)| (*candidate == key).then_some(*choices))
}

/// Parse and validate `text` for an enum or number setting.
pub fn parse_setting_value(key: &str, text: &str) -> Result<Value, String> {
    let text = text.trim();
    if key == "theme" {
        return (!text.is_empty())
            .then(|| json!(text))
            .ok_or_else(|| "theme must not be empty".to_owned());
    }
    if let Some((_, allowed)) = ENUM_SETTINGS.iter().find(|(k, _)| *k == key) {
        return allowed
            .contains(&text)
            .then(|| json!(text))
            .ok_or_else(|| format!("{key} must be one of: {}", allowed.join(", ")));
    }
    if let Some((min, max)) = number_setting_range(key) {
        return match text.parse::<u64>() {
            Ok(n) if (min..=max).contains(&n) => Ok(json!(n)),
            _ => Err(format!("{key} must be a whole number from {min} to {max}")),
        };
    }
    Err(format!("Unknown setting \"{key}\""))
}

/// Keys accepted by [`parse_setting_value`].
pub fn text_setting_keys() -> Vec<&'static str> {
    std::iter::once("theme")
        .chain(ENUM_SETTINGS.iter().map(|(k, _)| *k))
        .chain(NUMBER_SETTINGS.iter().map(|(k, _, _)| *k))
        .collect()
}

fn is_tool_modifier(entry: &Value) -> bool {
    entry
        .as_str()
        .is_some_and(|e| e.starts_with('+') || e.starts_with('-'))
}

fn deep_merge(base: &Map<String, Value>, overrides: &Map<String, Value>) -> Map<String, Value> {
    let mut result = base.clone();
    for (key, over) in overrides {
        if over.is_null() {
            continue;
        }
        let merged = match (base.get(key), over) {
            (Some(Value::Object(b)), Value::Object(o)) => Value::Object(deep_merge(b, o)),
            (Some(Value::Array(b)), Value::Array(o))
                if key == "defaultTools" && !o.is_empty() && o.iter().all(is_tool_modifier) =>
            {
                Value::Array(b.iter().chain(o).cloned().collect())
            }
            _ => over.clone(),
        };
        result.insert(key.clone(), merged);
    }
    result
}

/// Pi's `migrateSettings`: rewrite settings keys that changed shape between releases.
pub fn migrate(settings: &mut Map<String, Value>) {
    if !settings.contains_key("steeringMode")
        && let Some(value) = settings.remove("queueMode")
    {
        settings.insert("steeringMode".into(), value);
    }
    if !settings.contains_key("transport")
        && let Some(websockets) = settings.get("websockets").and_then(Value::as_bool)
    {
        settings.remove("websockets");
        settings.insert(
            "transport".into(),
            json!(if websockets { "websocket" } else { "sse" }),
        );
    }
    if let Some(Value::Object(skills)) = settings.get("skills").cloned() {
        if let Some(enabled) = skills.get("enableSkillCommands")
            && !settings.contains_key("enableSkillCommands")
        {
            settings.insert("enableSkillCommands".into(), enabled.clone());
        }
        match skills.get("customDirectories") {
            Some(Value::Array(dirs)) if !dirs.is_empty() => {
                settings.insert("skills".into(), Value::Array(dirs.clone()));
            }
            _ => {
                settings.remove("skills");
            }
        }
    }
    if let Some(Value::Object(retry)) = settings.get_mut("retry") {
        let provider_has_delay = retry
            .get("provider")
            .and_then(|p| p.get("maxRetryDelayMs"))
            .is_some_and(|v| !v.is_null());
        if let Some(delay) = retry.get("maxDelayMs").filter(|v| v.is_number()).cloned()
            && !provider_has_delay
        {
            let mut provider = match retry.get("provider") {
                Some(Value::Object(p)) => p.clone(),
                _ => Map::new(),
            };
            provider.insert("maxRetryDelayMs".into(), delay);
            retry.insert("provider".into(), Value::Object(provider));
        }
        retry.remove("maxDelayMs");
    }
}

fn load_layer(scope: Scope, path: &Path, errors: &mut Vec<SettingsError>) -> Map<String, Value> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Map::new(),
        Err(error) => {
            errors.push(SettingsError {
                scope,
                path: path.to_owned(),
                message: error.to_string(),
            });
            return Map::new();
        }
    };
    let contents = contents.strip_prefix('\u{feff}').unwrap_or(&contents);
    if contents.trim().is_empty() {
        return Map::new();
    }
    match serde_json::from_str::<Value>(contents) {
        Ok(Value::Object(mut map)) => {
            migrate(&mut map);
            map
        }
        Ok(_) => {
            errors.push(SettingsError {
                scope,
                path: path.to_owned(),
                message: "settings must be a JSON object".into(),
            });
            Map::new()
        }
        Err(error) => {
            errors.push(SettingsError {
                scope,
                path: path.to_owned(),
                message: error.to_string(),
            });
            Map::new()
        }
    }
}

impl Settings {
    /// Load layered settings only when the current project has an affirmative trust decision.
    /// Invalid trust data fails closed, so untrusted project settings cannot affect execution.
    pub fn load_for_project(agent_dir: &Path, cwd: &Path) -> Self {
        let project_trusted = crate::mcp::project_is_trusted(agent_dir, cwd).unwrap_or(false);
        Self::load(agent_dir, cwd, project_trusted)
    }

    /// Load both layers. An untrusted project contributes nothing and cannot be written.
    /// Prefer [`Self::load_for_project`] outside tests and trust-management code.
    pub fn load(agent_dir: &Path, cwd: &Path, project_trusted: bool) -> Self {
        let global_path = agent_dir.join("settings.json");
        let project_path = cwd.join(".pi/settings.json");
        let mut errors = Vec::new();
        let global = load_layer(Scope::Global, &global_path, &mut errors);
        let project = if project_trusted {
            load_layer(Scope::Project, &project_path, &mut errors)
        } else {
            Map::new()
        };
        Self {
            global_path,
            project_path,
            project_trusted,
            global,
            project,
            errors,
        }
    }

    /// The effective settings: global merged with project (no defaults applied).
    pub fn merged(&self) -> Value {
        Value::Object(deep_merge(&self.global, &self.project))
    }

    /// Look up a dotted path (`compaction.enabled`) in the effective settings, then the defaults.
    pub fn get(&self, path: &str) -> Option<Value> {
        let lookup = |root: &Value| {
            path.split('.')
                .try_fold(root, |value, key| value.get(key))
                .cloned()
        };
        lookup(&self.merged())
            .filter(|value| !value.is_null())
            .or_else(|| lookup(&defaults()))
    }

    /// Read the value that a setting editor should display in the selected layer.
    pub fn get_for_scope(&self, scope: Scope, path: &str) -> Option<Value> {
        let value = match scope {
            Scope::Global => self
                .get_from(path, &self.global)
                .or_else(|| self.get_from(path, defaults().as_object()?)),
            Scope::Project => self
                .get_from(path, &self.project)
                .or_else(|| self.get_from(path, &self.global))
                .or_else(|| self.get_from(path, defaults().as_object()?)),
        };
        value.filter(|value| !value.is_null())
    }

    fn get_from(&self, path: &str, layer: &Map<String, Value>) -> Option<Value> {
        let mut segments = path.split('.');
        let mut value = layer.get(segments.next()?)?;
        for segment in segments {
            value = value.get(segment)?;
        }
        Some(value.clone())
    }

    /// Whether project values from this settings snapshot were loaded and may be written.
    pub fn project_is_trusted(&self) -> bool {
        self.project_trusted
    }

    /// Drain the diagnostics collected while loading.
    pub fn drain_errors(&mut self) -> Vec<SettingsError> {
        std::mem::take(&mut self.errors)
    }

    /// Set a dotted path in one layer and persist it, keeping every other key in the file.
    pub fn set(&mut self, scope: Scope, path: &str, value: Value) -> Result<(), WriteError> {
        if scope == Scope::Project && !self.project_trusted {
            return Err(WriteError::ProjectNotTrusted);
        }
        let (layer, file) = match scope {
            Scope::Global => (&mut self.global, &self.global_path),
            Scope::Project => (&mut self.project, &self.project_path),
        };
        let mut keys = path.split('.').peekable();
        let mut cursor = &mut *layer;
        while let Some(key) = keys.next() {
            if keys.peek().is_none() {
                cursor.insert(key.to_owned(), value);
                break;
            }
            let entry = cursor
                .entry(key.to_owned())
                .or_insert_with(|| Value::Object(Map::new()));
            if !entry.is_object() {
                *entry = Value::Object(Map::new());
            }
            let Value::Object(next) = entry else {
                unreachable!("just made an object")
            };
            cursor = next;
        }
        write_atomically(file, &Value::Object(layer.clone()))
    }
}

fn write_atomically(path: &Path, value: &Value) -> Result<(), WriteError> {
    let io = |error| WriteError::Io(path.to_owned(), error);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    let temporary = path.with_extension("json.tmp");
    let body = serde_json::to_string_pretty(value).expect("settings serialize");
    std::fs::write(&temporary, body).map_err(io)?;
    std::fs::rename(&temporary, path).map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dirs() -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!("mu-settings-{}", uuid::Uuid::new_v4()));
        let (agent, cwd) = (root.join("agent"), root.join("project"));
        std::fs::create_dir_all(agent.clone()).unwrap();
        std::fs::create_dir_all(cwd.join(".pi")).unwrap();
        (agent, cwd)
    }

    #[test]
    fn project_settings_override_global_and_nested_objects_merge() {
        let (agent, cwd) = dirs();
        std::fs::write(
            agent.join("settings.json"),
            r#"{"compaction":{"enabled":false,"reserveTokens":1},"quietStartup":true}"#,
        )
        .unwrap();
        std::fs::write(
            cwd.join(".pi/settings.json"),
            r#"{"compaction":{"reserveTokens":9},"quietStartup":null}"#,
        )
        .unwrap();
        let settings = Settings::load(&agent, &cwd, true);
        assert_eq!(settings.get("compaction.enabled"), Some(json!(false)));
        assert_eq!(settings.get("compaction.reserveTokens"), Some(json!(9)));
        assert_eq!(
            settings.get("quietStartup"),
            Some(json!(true)),
            "null does not override"
        );
        assert_eq!(
            settings.get("compaction.keepRecentTokens"),
            Some(json!(20000)),
            "defaults fill gaps"
        );
        assert_eq!(settings.get("nope.nothing"), None);
    }

    #[test]
    fn default_tools_modifiers_append_and_plain_lists_replace() {
        let (agent, cwd) = dirs();
        std::fs::write(agent.join("settings.json"), r#"{"defaultTools":["read"]}"#).unwrap();
        std::fs::write(
            cwd.join(".pi/settings.json"),
            r#"{"defaultTools":["+grep"]}"#,
        )
        .unwrap();
        assert_eq!(
            Settings::load(&agent, &cwd, true).get("defaultTools"),
            Some(json!(["read", "+grep"]))
        );
        std::fs::write(
            cwd.join(".pi/settings.json"),
            r#"{"defaultTools":["bash"]}"#,
        )
        .unwrap();
        assert_eq!(
            Settings::load(&agent, &cwd, true).get("defaultTools"),
            Some(json!(["bash"]))
        );
    }

    #[test]
    fn numeric_ranges_are_exposed_for_settings_ui() {
        assert_eq!(
            number_setting_range("autocompleteMaxVisible"),
            Some((1, 20))
        );
        assert_eq!(number_setting_range("steeringMode"), None);
    }

    #[test]
    fn enum_choices_are_exposed_for_settings_ui() {
        assert_eq!(enum_setting_options("theme"), None);
        assert_eq!(
            enum_setting_options("tuiMode"),
            Some(["fullscreen", "regular"].as_slice())
        );
        assert_eq!(
            enum_setting_options("steeringMode"),
            Some(["all", "one-at-a-time"].as_slice())
        );
        assert_eq!(enum_setting_options("retry.maxRetries"), None);
    }

    #[test]
    fn enum_and_number_settings_are_validated() {
        assert_eq!(
            parse_setting_value("tuiMode", "regular").unwrap(),
            json!("regular")
        );
        assert_eq!(
            parse_setting_value("theme", "custom/path.json").unwrap(),
            json!("custom/path.json")
        );
        assert!(parse_setting_value("theme", " ").is_err());
        assert_eq!(
            parse_setting_value("doubleEscapeAction", " fork ").unwrap(),
            json!("fork")
        );
        assert_eq!(
            parse_setting_value("doubleEscapeAction", "x").unwrap_err(),
            "doubleEscapeAction must be one of: tree, fork, none"
        );
        assert_eq!(
            parse_setting_value("autocompleteMaxVisible", "7").unwrap(),
            json!(7)
        );
        assert!(parse_setting_value("autocompleteMaxVisible", "0").is_err());
        assert!(parse_setting_value("outputPad", "-1").is_err());
        assert_eq!(
            parse_setting_value("nope", "1").unwrap_err(),
            "Unknown setting \"nope\""
        );
        assert!(text_setting_keys().contains(&"codemode.mode"));
    }

    #[test]
    fn http_proxy_setting_fills_only_unset_variables() {
        assert_eq!(
            proxy_env_updates(Some(" http://p:1 "), false, false),
            [
                ("HTTP_PROXY", "http://p:1".to_owned()),
                ("HTTPS_PROXY", "http://p:1".to_owned())
            ]
        );
        assert_eq!(
            proxy_env_updates(Some("http://p:1"), true, false),
            [("HTTPS_PROXY", "http://p:1".to_owned())]
        );
        assert!(proxy_env_updates(Some("  "), false, false).is_empty());
        assert!(proxy_env_updates(None, false, false).is_empty());
    }

    #[test]
    fn migrates_legacy_keys_like_pi() {
        let mut map = serde_json::from_str::<Map<String, Value>>(
            r#"{"queueMode":"all","websockets":true,"skills":{"enableSkillCommands":false,"customDirectories":["a"]},"retry":{"maxDelayMs":5}}"#,
        ).unwrap();
        migrate(&mut map);
        assert_eq!(map["steeringMode"], "all");
        assert_eq!(map["transport"], "websocket");
        assert_eq!(map["enableSkillCommands"], false);
        assert_eq!(map["skills"], json!(["a"]));
        assert_eq!(map["retry"], json!({"provider":{"maxRetryDelayMs":5}}));
        assert!(!map.contains_key("queueMode") && !map.contains_key("websockets"));
    }

    #[test]
    fn bad_files_become_diagnostics_and_untrusted_projects_are_ignored() {
        let (agent, cwd) = dirs();
        std::fs::write(agent.join("settings.json"), "{not json").unwrap();
        std::fs::write(cwd.join(".pi/settings.json"), r#"{"quietStartup":true}"#).unwrap();
        let mut settings = Settings::load(&agent, &cwd, false);
        let errors = settings.drain_errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].scope, Scope::Global);
        assert!(settings.drain_errors().is_empty());
        assert_eq!(
            settings.get("quietStartup"),
            Some(json!(false)),
            "untrusted project ignored"
        );
        assert!(matches!(
            settings.set(Scope::Project, "quietStartup", json!(true)),
            Err(WriteError::ProjectNotTrusted)
        ));
    }

    #[test]
    fn set_writes_nested_paths_and_preserves_unknown_keys() {
        let (agent, cwd) = dirs();
        std::fs::write(agent.join("settings.json"), r#"{"custom":{"keep":1}}"#).unwrap();
        let mut settings = Settings::load(&agent, &cwd, true);
        settings
            .set(Scope::Global, "compaction.enabled", json!(false))
            .unwrap();
        settings
            .set(Scope::Global, "hideThinkingBlock", json!(true))
            .unwrap();
        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(agent.join("settings.json")).unwrap())
                .unwrap();
        assert_eq!(
            written,
            json!({"custom":{"keep":1},"compaction":{"enabled":false},"hideThinkingBlock":true})
        );
        assert_eq!(
            Settings::load(&agent, &cwd, true).get("compaction.enabled"),
            Some(json!(false))
        );
    }
}
