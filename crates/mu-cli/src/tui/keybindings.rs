//! Configurable app keybindings (Pi's `core/keybindings.ts`): `keybindings.json` in the agent
//! directory maps action ids such as `app.model.cycleForward` to key ids such as `ctrl+p`.

use std::{collections::BTreeMap, path::Path};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde_json::Value;

/// A parsed key id: `ctrl+shift+p`, `shift+tab`, `escape`, `alt+up`, `f5`, …
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyId {
    label: String,
    code: KeyCode,
    modifiers: KeyModifiers,
}

impl KeyId {
    pub fn parse(text: &str) -> Option<Self> {
        let lower = text.trim().to_lowercase();
        let mut parts = lower.split('+').collect::<Vec<_>>();
        let key = parts.pop().filter(|k| !k.is_empty())?;
        let mut modifiers = KeyModifiers::NONE;
        for modifier in parts {
            modifiers |= match modifier {
                "ctrl" | "control" => KeyModifiers::CONTROL,
                "shift" => KeyModifiers::SHIFT,
                "alt" | "meta" | "option" => KeyModifiers::ALT,
                _ => return None,
            };
        }
        let code = match key {
            "escape" | "esc" => KeyCode::Esc,
            "enter" | "return" => KeyCode::Enter,
            "tab" => KeyCode::Tab,
            "space" => KeyCode::Char(' '),
            "backspace" => KeyCode::Backspace,
            "delete" => KeyCode::Delete,
            "up" => KeyCode::Up,
            "down" => KeyCode::Down,
            "left" => KeyCode::Left,
            "right" => KeyCode::Right,
            "home" => KeyCode::Home,
            "end" => KeyCode::End,
            "pageup" => KeyCode::PageUp,
            "pagedown" => KeyCode::PageDown,
            other => match other.strip_prefix('f').and_then(|n| n.parse::<u8>().ok()) {
                Some(n @ 1..=24) => KeyCode::F(n),
                _ => {
                    let mut chars = other.chars();
                    match (chars.next(), chars.next()) {
                        (Some(c), None) => KeyCode::Char(c),
                        _ => return None,
                    }
                }
            },
        };
        Some(Self {
            label: display_label(&lower),
            code,
            modifiers,
        })
    }

    /// Whether `event` is this key. Shift is implied by an uppercase letter, and Shift+Tab
    /// arrives as `BackTab`.
    pub fn matches(&self, event: &KeyEvent) -> bool {
        let mut modifiers = event.modifiers;
        let code = match event.code {
            KeyCode::BackTab => {
                modifiers |= KeyModifiers::SHIFT;
                KeyCode::Tab
            }
            KeyCode::Char(c) if c.is_uppercase() => {
                modifiers |= KeyModifiers::SHIFT;
                KeyCode::Char(c.to_ascii_lowercase())
            }
            other => other,
        };
        code == self.code && modifiers == self.modifiers
    }
}

/// `ctrl+shift+p` → `Ctrl+Shift+P`, `escape` → `Esc` (how `/hotkeys` shows a key).
fn display_label(lower: &str) -> String {
    lower
        .split('+')
        .map(|part| match part {
            "escape" | "esc" => "Esc".to_owned(),
            "pageup" => "PageUp".to_owned(),
            "pagedown" => "PageDown".to_owned(),
            part => {
                let mut chars = part.chars();
                chars
                    .next()
                    .map_or_else(String::new, |c| c.to_uppercase().chain(chars).collect())
            }
        })
        .collect::<Vec<_>>()
        .join("+")
}

/// The app actions Mu implements, with Pi's default keys.
const DEFAULTS: &[(&str, &[&str])] = &[
    ("app.thinking.cycle", &["shift+tab"]),
    ("app.model.cycleForward", &["ctrl+p"]),
    ("app.model.cycleBackward", &["shift+ctrl+p"]),
    ("app.model.select", &["ctrl+l"]),
    ("app.tools.expand", &["ctrl+o"]),
    ("app.thinking.toggle", &["ctrl+t"]),
    ("app.message.copy", &["ctrl+x"]),
    ("app.message.dequeue", &["alt+up"]),
];

/// Pi's pre-`app.`/`tui.` names, rewritten on load (`migrateKeybindingsConfig`).
const LEGACY_NAMES: &[(&str, &str)] = &[
    ("cycleThinkingLevel", "app.thinking.cycle"),
    ("cycleModelForward", "app.model.cycleForward"),
    ("cycleModelBackward", "app.model.cycleBackward"),
    ("selectModel", "app.model.select"),
    ("expandTools", "app.tools.expand"),
    ("toggleThinking", "app.thinking.toggle"),
    ("dequeue", "app.message.dequeue"),
];

#[derive(Debug, Clone)]
pub struct Keybindings {
    bindings: BTreeMap<String, Vec<KeyId>>,
}

impl Default for Keybindings {
    fn default() -> Self {
        Self::from_config(&Value::Null)
    }
}

impl Keybindings {
    /// Defaults overlaid with a `keybindings.json` object. Each value is a key id or a list of
    /// them; a legacy name is used only when the new name is absent, and entries with an
    /// unparsable key are ignored.
    pub fn from_config(config: &Value) -> Self {
        let mut bindings = DEFAULTS
            .iter()
            .map(|(action, keys)| {
                (
                    (*action).to_owned(),
                    keys.iter().filter_map(|k| KeyId::parse(k)).collect(),
                )
            })
            .collect::<BTreeMap<String, Vec<KeyId>>>();
        let Some(object) = config.as_object() else {
            return Self { bindings };
        };
        for (name, value) in object {
            let action = LEGACY_NAMES
                .iter()
                .find(|(legacy, _)| legacy == name)
                .map_or(name.as_str(), |(_, current)| *current);
            if action != name && object.contains_key(action) {
                continue;
            }
            let texts = match value {
                Value::String(text) => vec![text.as_str()],
                Value::Array(items) => items.iter().filter_map(Value::as_str).collect(),
                _ => continue,
            };
            let keys = texts
                .iter()
                .map(|t| KeyId::parse(t))
                .collect::<Option<Vec<_>>>();
            if let Some(keys) = keys {
                bindings.insert(action.to_owned(), keys);
            }
        }
        Self { bindings }
    }

    pub fn load(agent_dir: &Path) -> Self {
        let config = std::fs::read_to_string(agent_dir.join("keybindings.json"))
            .ok()
            .and_then(|text| {
                serde_json::from_str::<Value>(text.trim_start_matches('\u{feff}')).ok()
            })
            .unwrap_or(Value::Null);
        Self::from_config(&config)
    }

    /// Keys bound to `action`, as `/hotkeys` shows them: `Ctrl+P` or `Ctrl+P` / `F2`; `None` when
    /// the action is unbound.
    pub fn display(&self, action: &str) -> Option<String> {
        let keys = self.bindings.get(action).filter(|keys| !keys.is_empty())?;
        Some(
            keys.iter()
                .map(|k| k.label.as_str())
                .collect::<Vec<_>>()
                .join("` / `"),
        )
    }

    pub fn matches(&self, action: &str, event: &KeyEvent) -> bool {
        self.bindings
            .get(action)
            .is_some_and(|keys| keys.iter().any(|k| k.matches(event)))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    #[test]
    fn key_ids_parse_and_match_terminal_events() {
        let ctrl_p = KeyId::parse("ctrl+p").unwrap();
        assert!(ctrl_p.matches(&key(KeyCode::Char('p'), KeyModifiers::CONTROL)));
        assert!(!ctrl_p.matches(&key(KeyCode::Char('p'), KeyModifiers::NONE)));
        let shift_ctrl_p = KeyId::parse("shift+ctrl+p").unwrap();
        assert!(shift_ctrl_p.matches(&key(KeyCode::Char('P'), KeyModifiers::CONTROL)));
        assert!(shift_ctrl_p.matches(&key(
            KeyCode::Char('p'),
            KeyModifiers::CONTROL | KeyModifiers::SHIFT
        )));
        assert!(
            KeyId::parse("shift+tab")
                .unwrap()
                .matches(&key(KeyCode::BackTab, KeyModifiers::SHIFT))
        );
        assert!(
            KeyId::parse("alt+up")
                .unwrap()
                .matches(&key(KeyCode::Up, KeyModifiers::ALT))
        );
        assert!(
            KeyId::parse("f5")
                .unwrap()
                .matches(&key(KeyCode::F(5), KeyModifiers::NONE))
        );
        assert_eq!(KeyId::parse("hyper+x"), None);
        assert_eq!(KeyId::parse("ctrl+"), None);
        assert_eq!(KeyId::parse("ctrl+ab"), None);
    }

    #[test]
    fn display_labels_show_user_bindings() {
        let bindings = Keybindings::from_config(
            &json!({"app.model.select": ["ctrl+k", "f2"], "app.tools.expand": []}),
        );
        assert_eq!(
            bindings.display("app.model.select").unwrap(),
            "Ctrl+K` / `F2"
        );
        assert_eq!(
            Keybindings::default()
                .display("app.thinking.cycle")
                .unwrap(),
            "Shift+Tab"
        );
        assert_eq!(
            Keybindings::default()
                .display("app.model.cycleBackward")
                .unwrap(),
            "Shift+Ctrl+P"
        );
        assert_eq!(
            bindings.display("app.tools.expand"),
            None,
            "an empty list unbinds the action"
        );
    }

    #[test]
    fn user_config_overrides_defaults_and_migrates_legacy_names() {
        let defaults = Keybindings::default();
        let ctrl_l = key(KeyCode::Char('l'), KeyModifiers::CONTROL);
        assert!(defaults.matches("app.model.select", &ctrl_l));
        let custom = Keybindings::from_config(&json!({
            "app.model.select": ["ctrl+k", "f2"],
            "cycleThinkingLevel": "ctrl+y",
            "expandTools": "ctrl+e",
            "app.tools.expand": "ctrl+u",
            "app.message.copy": "bogus+key",
        }));
        assert!(
            !custom.matches("app.model.select", &ctrl_l),
            "user keys replace the defaults"
        );
        assert!(custom.matches("app.model.select", &key(KeyCode::F(2), KeyModifiers::NONE)));
        assert!(custom.matches(
            "app.thinking.cycle",
            &key(KeyCode::Char('y'), KeyModifiers::CONTROL)
        ));
        assert!(
            custom.matches(
                "app.tools.expand",
                &key(KeyCode::Char('u'), KeyModifiers::CONTROL)
            ),
            "new name wins over legacy"
        );
        assert!(!custom.matches(
            "app.tools.expand",
            &key(KeyCode::Char('e'), KeyModifiers::CONTROL)
        ));
        assert!(
            custom.matches(
                "app.message.copy",
                &key(KeyCode::Char('x'), KeyModifiers::CONTROL)
            ),
            "unparsable entries are ignored"
        );
    }
}
