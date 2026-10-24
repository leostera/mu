//! Which tools a session exposes: default loadouts, `--tools` allowlists and modifiers.

use std::collections::HashSet;

use serde_json::Value;

use super::AgentError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ToolSelection {
    Allowlist(Vec<String>),
    Modifiers(Vec<ToolModifier>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ToolModifier {
    Enable(String),
    Disable(String),
}

impl ToolSelection {
    pub(super) fn parse(entries: Vec<String>) -> Result<Self, AgentError> {
        let contains_modifier = entries
            .iter()
            .any(|entry| entry.starts_with('+') || entry.starts_with('-'));
        if !contains_modifier {
            let mut patterns = Vec::new();
            for entry in entries {
                let loadout: &[&str] = match entry.as_str() {
                    "coding" => &["read", "bash", "edit", "write"],
                    "read-only" => &["read", "grep", "find", "ls"],
                    "all" => &["*"],
                    _ => &[],
                };
                if loadout.is_empty() {
                    patterns.push(entry);
                } else {
                    patterns.extend(loadout.iter().map(|name| (*name).to_owned()));
                }
            }
            let mut seen = HashSet::new();
            patterns.retain(|pattern| seen.insert(pattern.clone()));
            return Ok(Self::Allowlist(patterns));
        }
        if entries
            .iter()
            .any(|entry| !entry.starts_with('+') && !entry.starts_with('-'))
        {
            return Err(AgentError::Configuration(
                "tool names cannot be mixed with +name or -name entries".into(),
            ));
        }
        entries
            .into_iter()
            .map(|entry| {
                let name = &entry[1..];
                if name.is_empty() {
                    return Err(AgentError::Configuration(
                        "tool modifiers require a tool name".into(),
                    ));
                }
                if name.contains('*') {
                    return Err(AgentError::Configuration(
                        "+name and -name entries take exact tool names, not patterns".into(),
                    ));
                }
                Ok(if entry.starts_with('+') {
                    ToolModifier::Enable(name.to_owned())
                } else {
                    ToolModifier::Disable(name.to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Self::Modifiers)
    }
}

pub(super) const DEFAULT_TOOL_NAMES: &[&str] = &["read", "bash", "edit", "write"];

pub(super) fn merge_default_tools(
    inherited: &mut Option<Vec<String>>,
    settings: &Value,
) -> Result<(), AgentError> {
    let Some(value) = settings.get("defaultTools") else {
        return Ok(());
    };
    let entries = value
        .as_array()
        .ok_or_else(|| {
            AgentError::Configuration("defaultTools must be an array of strings".into())
        })?
        .iter()
        .map(|entry| {
            entry.as_str().map(str::to_owned).ok_or_else(|| {
                AgentError::Configuration("defaultTools must be an array of strings".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let modifiers_only = entries
        .iter()
        .all(|entry| entry.starts_with('+') || entry.starts_with('-'));
    if modifiers_only {
        inherited.get_or_insert_with(Vec::new).extend(entries);
    } else {
        *inherited = Some(entries);
    }
    Ok(())
}

pub(super) fn resolve_default_tools(entries: Option<Vec<String>>) -> Vec<String> {
    let entries = entries.unwrap_or_else(|| {
        DEFAULT_TOOL_NAMES
            .iter()
            .map(|name| (*name).into())
            .collect()
    });
    let plain = entries
        .iter()
        .filter(|entry| !entry.starts_with('+') && !entry.starts_with('-'))
        .cloned()
        .collect::<Vec<_>>();
    let mut tools = if !plain.is_empty() || entries.is_empty() {
        plain
    } else {
        DEFAULT_TOOL_NAMES
            .iter()
            .map(|name| (*name).into())
            .collect()
    };
    for entry in entries
        .iter()
        .filter(|entry| entry.starts_with('+') || entry.starts_with('-'))
    {
        let name = &entry[1..];
        if name.is_empty() {
            continue;
        }
        if entry.starts_with('+') && !tools.contains(&name.to_owned()) {
            tools.push(name.to_owned());
        } else if entry.starts_with('-') {
            tools.retain(|tool| tool != name);
        }
    }
    let mut ordered = Vec::new();
    for tool in tools {
        if !ordered.contains(&tool) {
            ordered.push(tool);
        }
    }
    ordered
}

pub(super) fn apply_tool_selection(defaults: &[String], selection: &ToolSelection) -> Vec<String> {
    match selection {
        ToolSelection::Allowlist(patterns) => {
            let definitions = crate::tools::definitions();
            let mut tools = Vec::new();
            for pattern in patterns {
                for definition in &definitions {
                    let Some(name) = definition.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    if tool_name_matches(pattern, name) && !tools.iter().any(|tool| tool == name) {
                        tools.push(name.to_owned());
                    }
                }
            }
            tools
        }
        ToolSelection::Modifiers(modifiers) => {
            let mut tools = defaults.to_vec();
            for modifier in modifiers {
                match modifier {
                    ToolModifier::Enable(name) if !tools.contains(name) => {
                        tools.push(name.clone());
                    }
                    ToolModifier::Disable(name) => tools.retain(|tool| tool != name),
                    ToolModifier::Enable(_) => {}
                }
            }
            tools
        }
    }
}

pub(super) fn filter_tool_definitions(
    definitions: Vec<Value>,
    selection: &ToolSelection,
) -> Vec<Value> {
    match selection {
        ToolSelection::Allowlist(patterns) => definitions
            .into_iter()
            .filter(|definition| {
                definition
                    .get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|name| {
                        patterns
                            .iter()
                            .any(|pattern| tool_name_matches(pattern, name))
                    })
            })
            .collect(),
        ToolSelection::Modifiers(modifiers) => definitions
            .into_iter()
            .filter(|definition| {
                let name = definition.get("name").and_then(Value::as_str);
                !modifiers.iter().any(|modifier| {
                    matches!(modifier, ToolModifier::Disable(disabled) if name == Some(disabled))
                })
            })
            .collect(),
    }
}

pub(super) fn tool_name_matches(pattern: &str, name: &str) -> bool {
    globset::Glob::new(pattern)
        .ok()
        .is_some_and(|glob| glob.compile_matcher().is_match(name))
}
