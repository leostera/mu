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
    /// Load both layers. An untrusted project contributes nothing and cannot be written.
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
            .filter(|v| !v.is_null())
            .or_else(|| lookup(&defaults()))
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
