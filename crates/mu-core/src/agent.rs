use std::{
    collections::{HashMap, HashSet, VecDeque},
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex as StdMutex},
};

use futures_util::{StreamExt, future::join_all};
use mu_session::{
    AssistantToolResponse, AssistantTurnResponse, MessageCost, MessageUsage, SessionDocument,
    ToolCallRecord,
};
use reqwest::{Client, Url, header};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::OnceCell;

const ORIGIN: &str = "https://opencode.cloudflare.dev";
const MCP_STARTUP_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("configuration: {0}")]
    Configuration(String),
    #[error("provider: {0}")]
    Provider(String),
    #[error("session: {0}")]
    Session(String),
    #[error("I/O: {0}")]
    Io(#[from] io::Error),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ToolSelection {
    Allowlist(Vec<String>),
    Modifiers(Vec<ToolModifier>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ToolModifier {
    Enable(String),
    Disable(String),
}

impl ToolSelection {
    fn parse(entries: Vec<String>) -> Result<Self, AgentError> {
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionInfo {
    pub id: String,
    pub name: Option<String>,
    pub preview: Option<String>,
    pub path: PathBuf,
    pub timestamp: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionTranscriptRole {
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionTranscriptTool {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
    pub output: Option<String>,
    pub is_error: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionTranscriptMessage {
    pub role: SessionTranscriptRole,
    pub text: String,
    pub thinking: String,
    pub tools: Vec<SessionTranscriptTool>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionOptions {
    pub path: Option<PathBuf>,
    pub directory: Option<PathBuf>,
    pub id: Option<String>,
    pub id_prefix: Option<String>,
    pub continue_latest: bool,
    pub no_session: bool,
    pub fork_from: Option<PathBuf>,
    pub name: Option<String>,
    pub tools: Option<Vec<String>>,
    pub no_tools: bool,
    pub no_builtin_tools: bool,
    pub skill_paths: Vec<PathBuf>,
    pub no_skills: bool,
}

struct Config {
    agent_dir: PathBuf,
    cwd: PathBuf,
    provider: String,
    model: String,
    model_choices: Vec<String>,
    hide_thinking_block: bool,
    retry: RetryPolicy,
    enabled_tools: Vec<String>,
}
struct Provider {
    transport: ProviderTransport,
    model: String,
    instructions: String,
    supports_reasoning: bool,
}

enum ProviderTransport {
    OpenCode {
        http: Client,
        endpoint: Url,
        token: String,
    },
    Demo,
    #[cfg(test)]
    Scripted(Arc<StdMutex<VecDeque<Result<Reply, String>>>>),
}
struct FunctionCall {
    call_id: String,
    name: String,
    arguments: Value,
}

struct ToolExecutionResult {
    output: String,
    is_error: bool,
    store_writes: Option<Value>,
    details: Option<Value>,
}

impl ToolExecutionResult {
    fn text(output: impl Into<String>, is_error: bool) -> Self {
        Self {
            output: output.into(),
            is_error,
            store_writes: None,
            details: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptBehavior {
    Steer,
    FollowUp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RetryPolicy {
    enabled: bool,
    max_retries: u32,
    base_delay_ms: u64,
    max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 2_000,
            max_delay_ms: 60_000,
        }
    }
}

#[derive(Clone)]
pub struct AgentPromptQueue {
    pending: Arc<StdMutex<PendingPrompts>>,
}

#[derive(Default)]
struct PendingPrompts {
    steering: VecDeque<String>,
    follow_up: VecDeque<String>,
}

impl AgentPromptQueue {
    pub fn enqueue(&self, prompt: String, behavior: PromptBehavior) {
        if prompt.trim().is_empty() {
            return;
        }
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match behavior {
            PromptBehavior::Steer => pending.steering.push_back(prompt),
            PromptBehavior::FollowUp => pending.follow_up.push_back(prompt),
        }
    }

    pub fn take_next(&self) -> Option<(String, PromptBehavior)> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        pending
            .steering
            .pop_front()
            .map(|prompt| (prompt, PromptBehavior::Steer))
            .or_else(|| {
                pending
                    .follow_up
                    .pop_front()
                    .map(|prompt| (prompt, PromptBehavior::FollowUp))
            })
    }

    fn take_steering(&self) -> Option<String> {
        self.pending
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .steering
            .pop_front()
    }

    pub fn clear(&self) {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        pending.steering.clear();
        pending.follow_up.clear();
    }
}

impl Default for AgentPromptQueue {
    fn default() -> Self {
        Self {
            pending: Arc::new(StdMutex::new(PendingPrompts::default())),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ToolStarted {
        call_id: String,
        name: String,
        arguments: Value,
    },
    ToolFinished {
        call_id: String,
        output: String,
        is_error: bool,
    },
    QueuedPromptStarted {
        prompt: String,
        behavior: PromptBehavior,
    },
    RetryScheduled {
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
        error: String,
    },
    RetryAttemptStarted {
        attempt: u32,
    },
    SkillCommandWarning(String),
}
enum TerminalResponse {
    Completed,
    Incomplete { reason: Option<String> },
}

#[derive(Debug)]
enum SkillCommandError {
    ReadFile { skill: String, error: String },
}

impl std::fmt::Display for SkillCommandError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadFile { skill, error } => {
                write!(formatter, "Could not read skill '{skill}': {error}")
            }
        }
    }
}

fn response_failed_error(event: &Value) -> AgentError {
    let error = event.pointer("/response/error");
    let message = if let Some(error) = error {
        format!(
            "{}: {}",
            error
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("no message")
        )
    } else if let Some(reason) = event
        .pointer("/response/incomplete_details/reason")
        .and_then(Value::as_str)
    {
        format!("incomplete: {reason}")
    } else {
        "Unknown error (no error details in response)".into()
    };
    AgentError::Provider(message)
}

fn response_error_event(event: &Value) -> AgentError {
    let code = event.get("code").and_then(Value::as_str);
    let message = event.get("message").and_then(Value::as_str);
    AgentError::Provider(match (code, message) {
        (Some(code), Some(message)) => format!("Error Code {code}: {message}"),
        _ => "Unknown error".into(),
    })
}

impl TerminalResponse {
    fn error_message(&self) -> Option<String> {
        match self {
            Self::Completed => None,
            Self::Incomplete {
                reason: Some(reason),
            } if reason == "max_output_tokens" => None,
            Self::Incomplete {
                reason: Some(reason),
            } => Some(format!("Response incomplete: {reason}")),
            Self::Incomplete { reason: None } => {
                Some("Response incomplete without a provider reason".into())
            }
        }
    }

    fn incomplete_reason(&self) -> Option<&str> {
        match self {
            Self::Completed => None,
            Self::Incomplete { reason } => reason.as_deref(),
        }
    }
}

struct Reply {
    text: String,
    thinking: String,
    thinking_signature: Option<String>,
    output_items: Vec<Value>,
    tool_calls: Vec<FunctionCall>,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    reasoning: Option<u64>,
    total_tokens: u64,
    incomplete_reason: Option<String>,
}

fn expand_skill_prompt(
    prompt: String,
    skills: &crate::skills::SkillCatalog,
    on_event: &mut impl FnMut(AgentEvent),
) -> String {
    match expand_skill_command(prompt.clone(), skills) {
        Ok(expanded) => expanded,
        Err(error) => {
            on_event(AgentEvent::SkillCommandWarning(error.to_string()));
            prompt
        }
    }
}

fn expand_skill_command(
    prompt: String,
    skills: &crate::skills::SkillCatalog,
) -> Result<String, SkillCommandError> {
    let Some(command) = prompt.strip_prefix("/skill:") else {
        return Ok(prompt);
    };
    let (name, arguments) = command
        .split_once(' ')
        .map_or((command, ""), |(name, arguments)| (name, arguments.trim()));
    let Some(skill) = skills.skills.iter().find(|skill| skill.name == name) else {
        return Ok(prompt);
    };
    let content =
        std::fs::read_to_string(&skill.file_path).map_err(|error| SkillCommandError::ReadFile {
            skill: name.to_owned(),
            error: error.to_string(),
        })?;
    let body = crate::skills::strip_frontmatter(&content);
    let base_dir = skill.file_path.parent().unwrap_or_else(|| Path::new(""));
    let block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path.display(),
        base_dir.display(),
        body,
    );
    Ok(if arguments.is_empty() {
        block
    } else {
        format!("{block}\n\n{arguments}")
    })
}

fn configured_setting<T>(
    agent_dir: &Path,
    cwd: &Path,
    key: &str,
    parse: impl Fn(&Value) -> Option<T>,
) -> Option<T> {
    let mut setting = None;
    for path in [
        agent_dir.join("settings.json"),
        cwd.join(".pi/settings.json"),
    ] {
        let Ok(contents) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(settings) = serde_json::from_str::<Value>(&contents) else {
            continue;
        };
        if let Some(value) = settings.get(key).and_then(&parse) {
            setting = Some(value);
        }
    }
    setting
}

fn shell_setting(agent_dir: &Path, cwd: &Path, key: &str) -> Option<String> {
    configured_setting(agent_dir, cwd, key, |value| {
        value.as_str().map(str::to_owned)
    })
    .filter(|value| !value.is_empty())
}

fn skill_commands_enabled(agent_dir: &Path, cwd: &Path) -> bool {
    configured_setting(agent_dir, cwd, "enableSkillCommands", Value::as_bool).unwrap_or(true)
}

fn configured_bash_shell_path(agent_dir: &Path, cwd: &Path) -> Option<String> {
    let configured = shell_setting(agent_dir, cwd, "shellPath")?;
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    let path = match home {
        Some(home) if configured == "~" => home,
        Some(home) if configured.starts_with("~/") || configured.starts_with("~\\") => {
            home.join(&configured[2..])
        }
        _ => PathBuf::from(configured),
    };
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    Some(path.to_string_lossy().into_owned())
}

fn config() -> Result<Config, AgentError> {
    let cwd = std::env::current_dir()?;
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| AgentError::Configuration("HOME is not set".into()))?;
    let agent_dir = std::env::var_os("PI_CODING_AGENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".pi/agent"));
    let mut provider = None;
    let mut model = None;
    let mut hide_thinking_block = false;
    let mut retry = RetryPolicy::default();
    let mut default_tool_entries = None;
    for path in [
        agent_dir.join("settings.json"),
        cwd.join(".pi/settings.json"),
    ] {
        if !path.exists() {
            continue;
        }
        let value: Value = serde_json::from_str(
            &std::fs::read_to_string(&path)
                .map_err(|e| AgentError::Configuration(format!("{}: {e}", path.display())))?,
        )
        .map_err(|e| AgentError::Configuration(format!("{}: {e}", path.display())))?;
        if let Some(v) = value.get("defaultProvider").and_then(Value::as_str) {
            provider = Some(v.to_owned());
        }
        if let Some(v) = value.get("defaultModel").and_then(Value::as_str) {
            model = Some(v.to_owned());
        }
        if let Some(v) = value.get("hideThinkingBlock").and_then(Value::as_bool) {
            hide_thinking_block = v;
        }
        update_retry_policy(&mut retry, &value);
        merge_default_tools(&mut default_tool_entries, &value)?;
    }
    let provider = provider
        .ok_or_else(|| AgentError::Configuration("Pi settings have no defaultProvider".into()))?;
    let model = model
        .ok_or_else(|| AgentError::Configuration("Pi settings have no defaultModel".into()))?;
    let model_choices = configured_models(&agent_dir, &provider, &model)?;
    Ok(Config {
        agent_dir,
        cwd,
        provider,
        model,
        model_choices,
        hide_thinking_block,
        retry,
        enabled_tools: resolve_default_tools(default_tool_entries),
    })
}

const DEFAULT_TOOL_NAMES: &[&str] = &["read", "bash", "edit", "write"];

fn merge_default_tools(
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

fn resolve_default_tools(entries: Option<Vec<String>>) -> Vec<String> {
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

fn apply_tool_selection(defaults: &[String], selection: &ToolSelection) -> Vec<String> {
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

fn tool_prompt_snippet(name: &str) -> Option<&'static str> {
    match name {
        "read" => Some("Read file contents"),
        "bash" => Some("Execute bash commands (ls, grep, find, etc.)"),
        "edit" => Some(
            "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
        ),
        "write" => Some("Create or overwrite files"),
        "grep" => Some("Search file contents for patterns (respects .gitignore)"),
        "find" => Some("Find files by glob pattern (respects .gitignore)"),
        "ls" => Some("List directory contents"),
        "powershell" => Some("Execute PowerShell commands"),
        _ => None,
    }
}

fn tool_prompt_guidelines(name: &str) -> &'static [&'static str] {
    match name {
        "read" => &["Use read to examine files instead of cat or sed."],
        "edit" => &[
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
        ],
        "write" => &["Use write only for new files or complete rewrites."],
        "bash" | "powershell" => {
            &["You can inspect PI_* environment variables for current model and session details."]
        }
        _ => &[],
    }
}

fn instructions_for_tools(base: &str, tools: &[Value]) -> String {
    let selected = tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    let snippets = selected
        .iter()
        .filter_map(|name| tool_prompt_snippet(name).map(|snippet| format!("- {name}: {snippet}")))
        .collect::<Vec<_>>();
    let mut guidelines = Vec::new();
    let mut add_guideline = |guideline: &str| {
        if !guidelines.iter().any(|existing| existing == guideline) {
            guidelines.push(guideline.to_owned());
        }
    };
    let has_shell = selected
        .iter()
        .any(|name| matches!(*name, "bash" | "powershell"));
    let has_search_or_list = selected
        .iter()
        .any(|name| matches!(*name, "grep" | "find" | "ls"));
    if has_shell && !has_search_or_list {
        if selected.contains(&"bash") && selected.contains(&"powershell") {
            add_guideline(
                "Use bash or PowerShell for file operations like listing, searching, and finding files",
            );
        } else if selected.contains(&"powershell") {
            add_guideline(
                "Use PowerShell for file operations like listing, searching, and finding files",
            );
        } else {
            add_guideline("Use bash for file operations like ls, rg, find");
        }
    }
    for name in &selected {
        for guideline in tool_prompt_guidelines(name) {
            add_guideline(guideline);
        }
    }
    add_guideline("Be concise in your responses");
    add_guideline("Show file paths clearly when working with files");

    let tool_section = if snippets.is_empty() {
        "(none)".to_owned()
    } else {
        snippets.join("\n")
    };
    let rules = guidelines
        .iter()
        .map(|guideline| format!("- {guideline}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{base}\n\n<tools>\n{tool_section}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>\n<rules>\n{rules}\n</rules>"
    )
}

fn mark_strict_builtin_tool(mut definition: Value) -> Value {
    let name = definition
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if matches!(name, "read" | "bash" | "edit" | "write") {
        if let Some(parameters) = definition.get_mut("parameters") {
            make_strict_schema(parameters);
        }
        definition["strict"] = Value::Bool(true);
    }
    definition
}

fn make_strict_schema(schema: &mut Value) {
    if let Some(object) = schema.as_object_mut() {
        if let Some(items) = object.get_mut("items") {
            make_strict_schema(items);
        }
        if let Some(variants) = object.get_mut("anyOf").and_then(Value::as_array_mut) {
            for variant in variants {
                make_strict_schema(variant);
            }
        }
        let previously_required = object
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect::<HashSet<_>>();
        if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
            let property_names = properties.keys().cloned().collect::<Vec<_>>();
            for (name, property) in properties {
                make_strict_schema(property);
                if !previously_required.contains(name) && !schema_allows_null(property) {
                    *property = json!({"anyOf":[property.clone(),{"type":"null"}]});
                }
            }
            object.insert("required".into(), json!(property_names));
            object.insert("additionalProperties".into(), Value::Bool(false));
        }
    }
}

fn schema_allows_null(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("null")
        || schema
            .get("type")
            .and_then(Value::as_array)
            .is_some_and(|types| types.iter().any(|kind| kind.as_str() == Some("null")))
        || schema.get("const") == Some(&Value::Null)
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.contains(&Value::Null))
        || schema
            .get("anyOf")
            .and_then(Value::as_array)
            .is_some_and(|variants| variants.iter().any(schema_allows_null))
}

fn filter_tool_definitions(definitions: Vec<Value>, selection: &ToolSelection) -> Vec<Value> {
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

fn tool_name_matches(pattern: &str, name: &str) -> bool {
    globset::Glob::new(pattern)
        .ok()
        .is_some_and(|glob| glob.compile_matcher().is_match(name))
}

fn update_retry_policy(retry: &mut RetryPolicy, settings: &Value) {
    let Some(settings) = settings.get("retry").and_then(Value::as_object) else {
        return;
    };
    if let Some(enabled) = settings.get("enabled").and_then(Value::as_bool) {
        retry.enabled = enabled;
    }
    if let Some(max_retries) = settings.get("maxRetries").and_then(Value::as_u64) {
        retry.max_retries = max_retries.min(u32::MAX as u64) as u32;
    }
    if let Some(base_delay_ms) = settings.get("baseDelayMs").and_then(Value::as_u64) {
        retry.base_delay_ms = base_delay_ms;
    }
    if let Some(max_delay_ms) = settings
        .get("maxAgentDelayMs")
        .or_else(|| settings.get("maxDelayMs"))
        .and_then(Value::as_u64)
    {
        retry.max_delay_ms = max_delay_ms;
    }
}

fn configured_models(
    agent_dir: &Path,
    provider: &str,
    default_model: &str,
) -> Result<Vec<String>, AgentError> {
    let path = agent_dir.join("models.json");
    let mut models = if path.exists() {
        let value: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).map_err(|error| {
                AgentError::Configuration(format!("{}: {error}", path.display()))
            })?)
            .map_err(|error| AgentError::Configuration(format!("{}: {error}", path.display())))?;
        value
            .get("providers")
            .and_then(|providers| providers.get(provider))
            .and_then(|provider| provider.get("models"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|model| model.get("id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect()
    } else {
        Vec::new()
    };
    if !models.iter().any(|model| model == default_model) {
        models.insert(0, default_model.to_owned());
    }
    Ok(models)
}

async fn token(provider: &str) -> Result<String, AgentError> {
    let env = if provider == "opencode-cf" {
        "OPENCODE_CF_TOKEN".into()
    } else {
        format!("{}_TOKEN", provider.replace('-', "_").to_ascii_uppercase())
    };
    if let Ok(value) = std::env::var(&env) {
        if !value.trim().is_empty() {
            return Ok(value.trim().to_owned());
        }
    }
    let bin = std::env::var_os("CLOUDFLARED_PATH").unwrap_or_else(|| "cloudflared".into());
    let output = tokio::process::Command::new(bin)
        .args(["access", "token", "-app=https://opencode.cloudflare.dev"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .map_err(|e| AgentError::Provider(format!("could not run cloudflared: {e}")))?;
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() || value.is_empty() || value.to_lowercase().contains("login") {
        return Err(AgentError::Provider(format!(
            "no Cloudflare Access token; run `cloudflared access login -app={ORIGIN}` or set {env}"
        )));
    }
    Ok(value)
}

fn trusted_url(value: &str) -> Result<Url, AgentError> {
    let url =
        Url::parse(value).map_err(|e| AgentError::Provider(format!("invalid gateway URL: {e}")))?;
    let host = url.host_str().unwrap_or_default();
    if url.scheme() != "https"
        || !(host == "opencode.cloudflare.dev" || host.ends_with(".opencode.cloudflare.dev"))
    {
        return Err(AgentError::Provider(format!(
            "refusing untrusted gateway URL `{value}`"
        )));
    }
    Ok(url)
}

async fn connect(config: &Config, skill_prompt: String) -> Result<Provider, AgentError> {
    if !config.provider.starts_with("opencode-cf") {
        return Err(AgentError::Configuration(format!(
            "Pi provider `{}` is unsupported (expected opencode-cf)",
            config.provider
        )));
    }
    let token = token(&config.provider).await?;
    let http = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| AgentError::Provider(format!("HTTP client: {e}")))?;
    let well: Value = http
        .get(format!("{ORIGIN}/.well-known/opencode"))
        .send()
        .await
        .map_err(|e| AgentError::Provider(format!("gateway discovery failed: {e}")))?
        .error_for_status()
        .map_err(|e| AgentError::Provider(format!("gateway discovery failed: {e}")))?
        .json()
        .await
        .map_err(|e| AgentError::Provider(format!("invalid gateway discovery: {e}")))?;
    let remote = well
        .get("remote_config")
        .ok_or_else(|| AgentError::Provider("discovery has no remote_config".into()))?;
    let config_url = remote
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Provider("discovery has no config URL".into()))?;
    let mut request = http.get(trusted_url(config_url)?);
    if let Some(headers) = remote.get("headers").and_then(Value::as_object) {
        for (name, value) in headers {
            if let Some(value) = value.as_str() {
                let value = value
                    .replace("{env:TOKEN}", &token)
                    .replace("${TOKEN}", &token);
                request = request.header(name, value);
            }
        }
    }
    let gateway: Value = request
        .send()
        .await
        .map_err(|e| AgentError::Provider(format!("gateway config fetch failed: {e}")))?
        .error_for_status()
        .map_err(|e| AgentError::Provider(format!("gateway config fetch failed: {e}")))?
        .json()
        .await
        .map_err(|e| AgentError::Provider(format!("invalid gateway config: {e}")))?;
    let providers = gateway
        .get("provider")
        .and_then(Value::as_object)
        .ok_or_else(|| AgentError::Provider("config has no provider catalog".into()))?;
    let mut endpoint = None;
    let mut supports_reasoning = false;
    for (id, p) in providers {
        let Some(m) = p.pointer(&format!("/models/{}", config.model)) else {
            continue;
        };
        let base = m
            .pointer("/provider/api")
            .and_then(Value::as_str)
            .or_else(|| p.pointer("/options/baseURL").and_then(Value::as_str))
            .or_else(|| p.pointer("/options/baseUrl").and_then(Value::as_str))
            .or_else(|| p.get("api").and_then(Value::as_str))
            .ok_or_else(|| AgentError::Provider(format!("provider `{id}` has no API URL")))?;
        if id != "openai"
            && !Url::parse(base)
                .map(|u| u.path().contains("/openai"))
                .unwrap_or(false)
        {
            return Err(AgentError::Provider(format!(
                "model uses unsupported provider `{id}`"
            )));
        }
        let mut url = trusted_url(base)?;
        url.set_path(&format!("{}/responses", url.path().trim_end_matches('/')));
        supports_reasoning = m.get("reasoning").and_then(Value::as_bool).unwrap_or(false);
        endpoint = Some(url);
        break;
    }
    let endpoint = endpoint.ok_or_else(|| {
        AgentError::Provider(format!(
            "model `{}` not found in gateway catalog",
            config.model
        ))
    })?;
    Ok(Provider {
        transport: ProviderTransport::OpenCode {
            http,
            endpoint,
            token,
        },
        model: config.model.clone(),
        instructions: format!(
            "You are Mu, a helpful coding assistant. Be clear, practical, and concise.{skill_prompt}"
        ),
        supports_reasoning,
    })
}

impl Provider {
    async fn complete(
        &self,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        #[cfg(test)]
        if let ProviderTransport::Scripted(replies) = &self.transport {
            let reply = replies
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .pop_front()
                .ok_or_else(|| AgentError::Provider("scripted provider exhausted".into()))?
                .map_err(AgentError::Provider)?;
            for character in reply.thinking.chars() {
                on_event(AgentEvent::ThinkingDelta(character.to_string()));
            }
            for character in reply.text.chars() {
                on_event(AgentEvent::TextDelta(character.to_string()));
            }
            return Ok(reply);
        }
        if matches!(&self.transport, ProviderTransport::Demo) {
            let prompt = history
                .last()
                .and_then(|item| item.pointer("/content/0/text"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let text = format!("Demo backend received: {prompt}");
            for character in text.chars() {
                on_event(AgentEvent::TextDelta(character.to_string()));
            }
            return Ok(Reply {
                text,
                thinking: String::new(),
                thinking_signature: None,
                output_items: Vec::new(),
                tool_calls: Vec::new(),
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                reasoning: None,
                total_tokens: 0,
                incomplete_reason: None,
            });
        }
        let ProviderTransport::OpenCode {
            http,
            endpoint,
            token,
        } = &self.transport
        else {
            unreachable!("demo backend returned above")
        };
        let input = history;
        let instructions = instructions_for_tools(&self.instructions, tools);
        let mut request_body = json!({"model":self.model,"instructions":instructions,"input":input,"tools":tools,"stream":true,"store":false});
        if self.supports_reasoning {
            request_body["reasoning"] = json!({"effort":"medium","summary":"auto"});
            request_body["include"] = json!(["reasoning.encrypted_content"]);
        }
        let response = http
            .post(endpoint.clone())
            .bearer_auth(token)
            .header("cf-access-token", token)
            .header("x-requested-with", "xmlhttprequest")
            .header(header::ACCEPT, "text/event-stream")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(AgentError::Provider(format!(
                "model request returned HTTP {}",
                response.status()
            )));
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        let mut text = String::new();
        let mut thinking = String::new();
        let mut thinking_signature = None;
        let mut output_items = Vec::new();
        let mut tool_calls = Vec::new();
        let mut terminal_response = None;
        let (mut input_tokens, mut output_tokens) = (0, 0);
        let (mut cache_read, mut cache_write, mut reasoning, mut total_tokens) = (0, 0, None, 0);
        while let Some(chunk) = stream.next().await {
            bytes.extend_from_slice(
                &chunk.map_err(|e| AgentError::Provider(format!("model stream failed: {e}")))?,
            );
            while let Some(pos) = bytes.windows(2).position(|w| w == b"\n\n") {
                let frame: Vec<_> = bytes.drain(..pos + 2).collect();
                let frame = String::from_utf8_lossy(&frame);
                let Some(data) = frame.lines().find_map(|line| line.strip_prefix("data:")) else {
                    continue;
                };
                if let Ok(event) = serde_json::from_str::<Value>(data.trim()) {
                    match event.get("type").and_then(Value::as_str).unwrap_or("") {
                        "response.output_text.delta" | "response.refusal.delta" => {
                            if let Some(s) = event.get("delta").and_then(Value::as_str) {
                                text.push_str(s);
                                on_event(AgentEvent::TextDelta(s.to_owned()));
                            }
                        }
                        "response.reasoning_summary_text.delta"
                        | "response.reasoning_text.delta" => {
                            if let Some(s) = event.get("delta").and_then(Value::as_str) {
                                thinking.push_str(s);
                                on_event(AgentEvent::ThinkingDelta(s.to_owned()));
                            }
                        }
                        "response.output_item.done" => {
                            if event.pointer("/item/type").and_then(Value::as_str)
                                == Some("reasoning")
                            {
                                let item = event.get("item").unwrap_or(&Value::Null);
                                thinking_signature = Some(item.to_string());
                                let final_thinking = response_reasoning(item);
                                if !final_thinking.is_empty() {
                                    if thinking.is_empty() {
                                        thinking = final_thinking.clone();
                                        on_event(AgentEvent::ThinkingDelta(final_thinking));
                                    } else if let Some(suffix) =
                                        final_thinking.strip_prefix(&thinking)
                                    {
                                        if !suffix.is_empty() {
                                            thinking.push_str(suffix);
                                            on_event(AgentEvent::ThinkingDelta(suffix.to_owned()));
                                        }
                                    }
                                }
                            }
                        }
                        "response.completed" | "response.incomplete" => {
                            let incomplete_reason = event
                                .pointer("/response/incomplete_details/reason")
                                .and_then(Value::as_str)
                                .map(str::to_owned);
                            terminal_response = Some(
                                if event.get("type").and_then(Value::as_str)
                                    == Some("response.completed")
                                {
                                    TerminalResponse::Completed
                                } else {
                                    TerminalResponse::Incomplete {
                                        reason: incomplete_reason.clone(),
                                    }
                                },
                            );
                            if let Some(items) =
                                event.pointer("/response/output").and_then(Value::as_array)
                            {
                                output_items = items.clone();
                                if let Some(delta) =
                                    merge_final_response_text(&mut text, &output_items)
                                {
                                    on_event(AgentEvent::TextDelta(delta));
                                }
                            }
                            if let Some(output) =
                                event.pointer("/response/output").and_then(Value::as_array)
                            {
                                if let Some(item) = output.iter().find(|item| {
                                    item.get("type").and_then(Value::as_str) == Some("reasoning")
                                }) {
                                    if thinking_signature.is_none()
                                        || item.get("encrypted_content").is_some()
                                    {
                                        thinking_signature = Some(item.to_string());
                                    }
                                    if thinking.is_empty() {
                                        thinking = response_reasoning(item);
                                        if !thinking.is_empty() {
                                            on_event(AgentEvent::ThinkingDelta(thinking.clone()));
                                        }
                                    }
                                }
                                for item in output.iter().filter(|item| {
                                    item.get("type").and_then(Value::as_str)
                                        == Some("function_call")
                                }) {
                                    let (Some(call_id), Some(name), Some(raw_arguments)) = (
                                        item.get("call_id").and_then(Value::as_str),
                                        item.get("name").and_then(Value::as_str),
                                        item.get("arguments").and_then(Value::as_str),
                                    ) else {
                                        continue;
                                    };
                                    let arguments = serde_json::from_str(raw_arguments)
                                        .unwrap_or_else(|_| {
                                            if incomplete_reason.as_deref()
                                                == Some("max_output_tokens")
                                            {
                                                Value::String(raw_arguments.to_owned())
                                            } else {
                                                Value::Null
                                            }
                                        });
                                    if !arguments.is_null() {
                                        tool_calls.push(FunctionCall {
                                            call_id: call_id.to_owned(),
                                            name: name.to_owned(),
                                            arguments,
                                        });
                                    }
                                }
                            }
                            let usage = event.pointer("/response/usage");
                            let raw_input = usage
                                .and_then(|value| value.get("input_tokens"))
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            cache_read = usage
                                .and_then(|value| {
                                    value.pointer("/input_tokens_details/cached_tokens")
                                })
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            cache_write = usage
                                .and_then(|value| {
                                    value.pointer("/input_tokens_details/cache_write_tokens")
                                })
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            input_tokens = raw_input.saturating_sub(cache_read + cache_write);
                            output_tokens = usage
                                .and_then(|value| value.get("output_tokens"))
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            reasoning = usage
                                .and_then(|value| {
                                    value.pointer("/output_tokens_details/reasoning_tokens")
                                })
                                .and_then(Value::as_u64);
                            total_tokens = usage
                                .and_then(|value| value.get("total_tokens"))
                                .and_then(Value::as_u64)
                                .unwrap_or(raw_input + output_tokens);
                        }
                        "response.failed" => return Err(response_failed_error(&event)),
                        "error" => return Err(response_error_event(&event)),
                        _ => {}
                    }
                }
            }
        }
        let terminal_response = terminal_response.ok_or_else(|| {
            AgentError::Provider(
                "OpenAI Responses stream ended before a terminal response event".into(),
            )
        })?;
        let incomplete_reason = terminal_response.incomplete_reason().map(str::to_owned);
        if let Some(message) = terminal_response.error_message() {
            return Err(AgentError::Provider(message));
        }
        Ok(Reply {
            text,
            thinking,
            thinking_signature,
            output_items,
            tool_calls,
            input: input_tokens,
            output: output_tokens,
            cache_read,
            cache_write,
            reasoning,
            total_tokens,
            incomplete_reason,
        })
    }
}

fn response_message_text(output_items: &[Value]) -> String {
    output_items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter_map(|block| match block.get("type").and_then(Value::as_str) {
            Some("output_text") => block.get("text").and_then(Value::as_str),
            Some("refusal") => block.get("refusal").and_then(Value::as_str),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn merge_final_response_text(streamed_text: &mut String, output_items: &[Value]) -> Option<String> {
    let final_text = response_message_text(output_items);
    if final_text.is_empty() || final_text == *streamed_text {
        return None;
    }
    if let Some(delta) = final_text.strip_prefix(streamed_text.as_str()) {
        streamed_text.push_str(delta);
        return Some(delta.to_owned());
    }
    *streamed_text = final_text;
    None
}

fn response_reasoning(item: &Value) -> String {
    let parts = |key: &str| {
        item.get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    let summary = parts("summary");
    if summary.is_empty() {
        parts("content")
    } else {
        summary
    }
}

fn api_name(transport: &ProviderTransport) -> &'static str {
    if matches!(transport, ProviderTransport::Demo) {
        "mu-demo"
    } else {
        "openai-responses"
    }
}

fn session_active_mcp_tools(session: &SessionDocument) -> HashSet<String> {
    let mut active = HashSet::new();
    for entry in session.active_branch_entries() {
        if entry.entry_type.as_str() != "custom"
            || entry
                .extra_fields()
                .get("customType")
                .and_then(Value::as_str)
                != Some("mcp-tool-loadout")
        {
            continue;
        }
        active = entry
            .extra_fields()
            .get("data")
            .and_then(|data| data.get("active"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
    }
    active
}

fn session_codemode_store(session: &SessionDocument) -> Value {
    let mut store = serde_json::Map::new();
    for entry in session.active_branch_entries() {
        if entry.entry_type.as_str() != "custom"
            || entry
                .extra_fields()
                .get("customType")
                .and_then(Value::as_str)
                != Some("codemode-store")
        {
            continue;
        }
        let Some(data) = entry.extra_fields().get("data").and_then(Value::as_object) else {
            continue;
        };
        if let Some(deleted) = data.get("delete").and_then(Value::as_array) {
            for key in deleted.iter().filter_map(Value::as_str) {
                store.remove(key);
            }
        }
        if let Some(values) = data.get("set").and_then(Value::as_object) {
            store.extend(values.clone());
        }
    }
    Value::Object(store)
}

fn load_session(path: &Path) -> Result<(SessionDocument, Vec<Value>), AgentError> {
    let contents =
        std::fs::read_to_string(path).map_err(|error| AgentError::Session(error.to_string()))?;
    let session = SessionDocument::parse_jsonl(&contents)
        .map_err(|error| AgentError::Session(error.to_string()))?;
    let history = session_history(&session);
    Ok((session, history))
}

const SESSION_HTML_STYLE: &str = "body{margin:0;background:#17191f;color:#e5e7eb;font:16px/1.6 system-ui,sans-serif}main{max-width:900px;margin:3rem auto;padding:0 1.25rem}header{border-bottom:1px solid #454955;margin-bottom:2rem}article{padding:1rem 1.25rem;margin:1rem 0;border-radius:12px;background:#22252d}article.user{background:#292d38}h1,h2{line-height:1.2}h2{font-size:1rem;color:#aeb8ce}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:.5rem 0}details{margin:.75rem 0;padding:.75rem;background:#191b21;border-radius:8px}.error{color:#ff8a8a}.meta{color:#9ca3af;font-size:.9rem}";

fn append_transcript_message_html(html: &mut String, message: SessionTranscriptMessage) {
    let (role, class) = match message.role {
        SessionTranscriptRole::User => ("User", "user"),
        SessionTranscriptRole::Assistant => ("Assistant", "assistant"),
    };
    html.push_str(&format!("<article class=\"{class}\"><h2>{role}</h2>"));
    if !message.thinking.is_empty() {
        html.push_str(&format!(
            "<details><summary>Thinking</summary><pre>{}</pre></details>",
            escape_html(&message.thinking)
        ));
    }
    if !message.text.is_empty() {
        html.push_str(&format!("<pre>{}</pre>", escape_html(&message.text)));
    }
    for tool in message.tools {
        append_tool_html(html, tool);
    }
    html.push_str("</article>");
}

fn append_tool_html(html: &mut String, tool: SessionTranscriptTool) {
    let arguments = serde_json::to_string_pretty(&tool.arguments)
        .unwrap_or_else(|_| tool.arguments.to_string());
    let error_class = if tool.is_error {
        " class=\"error\""
    } else {
        ""
    };
    html.push_str(&format!(
        "<details{error_class}><summary>Tool: {}</summary><pre>{}</pre>",
        escape_html(&tool.name),
        escape_html(&arguments)
    ));
    if let Some(output) = tool.output {
        html.push_str(&format!("<pre>{}</pre>", escape_html(&output)));
    }
    html.push_str("</details>");
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn session_transcript(session: &SessionDocument) -> Vec<SessionTranscriptMessage> {
    let mut transcript: Vec<SessionTranscriptMessage> = Vec::new();
    for entry in session.active_branch_entries() {
        let Some(message) = entry
            .extra_fields()
            .get("message")
            .and_then(Value::as_object)
        else {
            continue;
        };
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let blocks = message
            .get("content")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        match role {
            "user" => {
                let text = blocks
                    .iter()
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    transcript.push(SessionTranscriptMessage {
                        role: SessionTranscriptRole::User,
                        text,
                        thinking: String::new(),
                        tools: Vec::new(),
                    });
                }
            }
            "assistant" => {
                let mut text = Vec::new();
                let mut thinking = Vec::new();
                let mut tools = Vec::new();
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(value) = block.get("text").and_then(Value::as_str) {
                                text.push(value);
                            }
                        }
                        Some("thinking") => {
                            if let Some(value) = block.get("thinking").and_then(Value::as_str) {
                                thinking.push(value);
                            }
                        }
                        Some("toolCall") => {
                            let (Some(call_id), Some(name)) = (
                                block.get("id").and_then(Value::as_str),
                                block.get("name").and_then(Value::as_str),
                            ) else {
                                continue;
                            };
                            tools.push(SessionTranscriptTool {
                                call_id: call_id.to_owned(),
                                name: name.to_owned(),
                                arguments: block.get("arguments").cloned().unwrap_or(Value::Null),
                                output: None,
                                is_error: false,
                            });
                        }
                        _ => {}
                    }
                }
                let text = text.join("\n");
                let thinking = thinking.join("\n");
                if !text.is_empty() || !thinking.is_empty() || !tools.is_empty() {
                    transcript.push(SessionTranscriptMessage {
                        role: SessionTranscriptRole::Assistant,
                        text,
                        thinking,
                        tools,
                    });
                }
            }
            "toolResult" => {
                let Some(call_id) = message.get("toolCallId").and_then(Value::as_str) else {
                    continue;
                };
                let output = blocks
                    .iter()
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n");
                if let Some(tool) = transcript
                    .iter_mut()
                    .rev()
                    .flat_map(|message| message.tools.iter_mut())
                    .find(|tool| tool.call_id == call_id)
                {
                    tool.output = Some(output);
                    tool.is_error = message
                        .get("isError")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                }
            }
            _ => {}
        }
    }
    transcript
}

fn session_preview(session: &SessionDocument) -> Option<String> {
    let content = session
        .active_branch_entries()
        .into_iter()
        .rev()
        .filter(|entry| entry.entry_type.as_str() == "message")
        .find_map(|entry| {
            let message = entry.extra_fields().get("message")?;
            (message.get("role")?.as_str()? == "user")
                .then(|| message.get("content"))
                .flatten()
        })?;
    let text = match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| {
                let kind = block.get("type")?.as_str()?;
                matches!(kind, "text" | "input_text" | "output_text")
                    .then(|| block.get("text")?.as_str())
                    .flatten()
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    };
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return None;
    }
    let mut chars = normalized.chars();
    let preview = chars.by_ref().take(160).collect::<String>();
    Some(if chars.next().is_some() {
        format!("{preview}…")
    } else {
        preview
    })
}

fn session_history(session: &SessionDocument) -> Vec<Value> {
    let mut history = Vec::new();
    for entry in session.active_branch_entries() {
        let Some(message) = entry
            .extra_fields()
            .get("message")
            .and_then(Value::as_object)
        else {
            continue;
        };
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let content = message.get("content").and_then(Value::as_array);
        match role {
            "user" => {
                let blocks = content
                    .into_iter()
                    .flatten()
                    .filter_map(|block| {
                        let text = block.get("text")?.as_str()?;
                        Some(json!({"type":"input_text","text":text}))
                    })
                    .collect::<Vec<_>>();
                if !blocks.is_empty() {
                    history.push(json!({"role":"user","content":blocks}));
                }
            }
            "assistant" => {
                let mut text_blocks = Vec::new();
                let mut tool_calls = Vec::new();
                for block in content.into_iter().flatten() {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(text) = block.get("text").and_then(Value::as_str) {
                                text_blocks.push(json!({"type":"output_text","text":text}));
                            }
                        }
                        Some("toolCall") => {
                            let Some(call_id) = block.get("id").and_then(Value::as_str) else {
                                continue;
                            };
                            let Some(name) = block.get("name").and_then(Value::as_str) else {
                                continue;
                            };
                            let arguments = block.get("arguments").cloned().unwrap_or(Value::Null);
                            let arguments =
                                arguments.as_str().map(str::to_owned).unwrap_or_else(|| {
                                    serde_json::to_string(&arguments)
                                        .unwrap_or_else(|_| "{}".into())
                                });
                            tool_calls.push(json!({
                                "type":"function_call",
                                "call_id":call_id,
                                "name":name,
                                "arguments":arguments,
                            }));
                        }
                        _ => {}
                    }
                }
                if !text_blocks.is_empty() {
                    history.push(json!({"role":"assistant","content":text_blocks}));
                }
                history.extend(tool_calls);
            }
            "toolResult" => {
                let text = content
                    .into_iter()
                    .flatten()
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n");
                if let Some(call_id) = message.get("toolCallId").and_then(Value::as_str) {
                    history.push(json!({
                        "type":"function_call_output",
                        "call_id":call_id,
                        "output":text,
                    }));
                }
            }
            _ => {}
        }
    }
    history
}

fn session_directory(agent_dir: &Path, cwd: &Path, override_dir: Option<&Path>) -> PathBuf {
    if let Some(directory) = override_dir {
        return directory.to_owned();
    }
    let cwd = cwd
        .to_string_lossy()
        .trim_start_matches(['/', '\\'])
        .replace(['/', '\\', ':'], "-");
    agent_dir.join("sessions").join(format!("--{cwd}--"))
}

fn resolved_path(path: &Path) -> PathBuf {
    crate::path_utils::canonicalize_path(path)
}

fn session_header(path: &Path) -> Option<Value> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let mut first_line = String::new();
    reader.read_line(&mut first_line).ok()?;
    let header = serde_json::from_str::<Value>(&first_line).ok()?;
    (header.get("type").and_then(Value::as_str) == Some("session")).then_some(header)
}

fn find_session_id_path(
    agent_dir: &Path,
    cwd: &Path,
    override_dir: Option<&Path>,
    id: &str,
    allow_prefix: bool,
) -> Option<PathBuf> {
    let directory = session_directory(agent_dir, cwd, override_dir);
    let expected_cwd = resolved_path(cwd);
    let mut sessions = std::fs::read_dir(directory)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file()
                || path
                    .extension()
                    .is_none_or(|extension| extension != "jsonl")
            {
                return None;
            }
            Some((entry.metadata().ok()?.modified().ok()?, path))
        })
        .collect::<Vec<_>>();
    sessions.sort_by_key(|session| std::cmp::Reverse(session.0));
    sessions.into_iter().map(|(_, path)| path).find(|path| {
        let Some(header) = session_header(path) else {
            return false;
        };
        header
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|session_id| {
                if allow_prefix {
                    session_id.starts_with(id)
                } else {
                    session_id == id
                }
            })
            && header
                .get("cwd")
                .and_then(Value::as_str)
                .is_some_and(|session_cwd| resolved_path(Path::new(session_cwd)) == expected_cwd)
    })
}

fn normalize_session_name(name: Option<String>) -> Result<Option<String>, AgentError> {
    name.map(|name| {
        let normalized = name
            .split(['\r', '\n'])
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned();
        if normalized.is_empty() {
            Err(AgentError::Session(
                "--name requires a non-empty value".into(),
            ))
        } else {
            Ok(normalized)
        }
    })
    .transpose()
}

fn validate_session_id(id: &str) -> Result<(), AgentError> {
    let valid_edge = |character: char| character.is_ascii_alphanumeric();
    let characters = id.chars().collect::<Vec<_>>();
    if characters.is_empty()
        || !valid_edge(characters[0])
        || !valid_edge(*characters.last().expect("non-empty id"))
        || characters.iter().any(|character| {
            !character.is_ascii_alphanumeric() && !matches!(character, '.' | '_' | '-')
        })
    {
        return Err(AgentError::Session(
            "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', and start and end with an alphanumeric character".into(),
        ));
    }
    Ok(())
}

fn list_session_paths(
    agent_dir: &Path,
    cwd: &Path,
    override_dir: Option<&Path>,
) -> Vec<(std::time::SystemTime, PathBuf, Value)> {
    let directory = session_directory(agent_dir, cwd, override_dir);
    let expected_cwd = resolved_path(cwd);
    let mut sessions = std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file()
                || path
                    .extension()
                    .is_none_or(|extension| extension != "jsonl")
            {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            let header = session_header(&path)?;
            let session_cwd = Path::new(header.get("cwd")?.as_str()?);
            (resolved_path(session_cwd) == expected_cwd).then_some((modified, path, header))
        })
        .collect::<Vec<_>>();
    sessions.sort_by_key(|session| std::cmp::Reverse(session.0));
    sessions
}

fn latest_session_path(
    agent_dir: &Path,
    cwd: &Path,
    override_dir: Option<&Path>,
) -> Option<PathBuf> {
    list_session_paths(agent_dir, cwd, override_dir)
        .into_iter()
        .next()
        .map(|(_, path, _)| path)
}

fn session_path(c: &Config, override_dir: Option<&Path>) -> PathBuf {
    new_session_path(&c.agent_dir, &c.cwd, override_dir)
}

fn new_session_path(agent_dir: &Path, cwd: &Path, override_dir: Option<&Path>) -> PathBuf {
    let dir = session_directory(agent_dir, cwd, override_dir);
    let timestamp = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
        .replace([':', '.'], "-");
    dir.join(format!("{timestamp}_{}.jsonl", uuid::Uuid::new_v4()))
}

fn resolve_session_options(
    config: &Config,
    options: SessionOptions,
) -> Result<(PathBuf, SessionDocument, Vec<Value>), AgentError> {
    if let Some(id) = options.id.as_deref() {
        validate_session_id(id)?;
    }
    let SessionOptions {
        path,
        directory,
        id,
        id_prefix,
        continue_latest,
        no_session,
        fork_from,
        name,
        tools: _,
        no_tools: _,
        no_builtin_tools: _,
        skill_paths: _,
        no_skills: _,
    } = options;
    let name = normalize_session_name(name)?;
    if let Some(source_path) = fork_from {
        let (source, _) = load_session(&source_path)?;
        let id = id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        validate_session_id(&id)?;
        if find_session_id_path(
            &config.agent_dir,
            &config.cwd,
            directory.as_deref(),
            &id,
            false,
        )
        .is_some()
        {
            return Err(AgentError::Session(format!(
                "Session already exists with id '{id}'"
            )));
        }
        let parent_session = source_path.to_string_lossy().into_owned();
        let mut session = source
            .fork_with_id(
                config.cwd.to_string_lossy().into_owned(),
                id,
                parent_session,
            )
            .ok_or_else(|| AgentError::Session("source session has no header".into()))?;
        if let Some(name) = name {
            session.append_session_name(&name);
        }
        let path = session_path(config, directory.as_deref());
        if !no_session {
            write_session(&session, &path)?;
        }
        let history = session_history(&session);
        return Ok((path, session, history));
    }
    let requested_path = path
        .or_else(|| {
            continue_latest.then(|| {
                latest_session_path(&config.agent_dir, &config.cwd, directory.as_deref())
            })?
        })
        .or_else(|| {
            id.as_deref().and_then(|id| {
                find_session_id_path(
                    &config.agent_dir,
                    &config.cwd,
                    directory.as_deref(),
                    id,
                    false,
                )
            })
        })
        .or_else(|| {
            id_prefix.as_deref().and_then(|prefix| {
                find_session_id_path(
                    &config.agent_dir,
                    &config.cwd,
                    directory.as_deref(),
                    prefix,
                    true,
                )
            })
        });
    if id_prefix.is_some() && requested_path.is_none() {
        return Err(AgentError::Session(format!(
            "No session found matching '{}' in this working directory",
            id_prefix.as_deref().unwrap_or_default()
        )));
    }
    let (path, mut session, history) = match requested_path {
        Some(path) => {
            let (session, history) = load_session(&path)?;
            (path, session, history)
        }
        None => {
            let session = match id {
                Some(id) => {
                    SessionDocument::new_with_id(config.cwd.to_string_lossy().into_owned(), id)
                }
                None => SessionDocument::new(config.cwd.to_string_lossy().into_owned()),
            };
            (
                session_path(config, directory.as_deref()),
                session,
                Vec::new(),
            )
        }
    };
    if let Some(name) = name {
        session.append_session_name(&name);
        if !no_session {
            write_session(&session, &path)?;
        }
    }
    Ok((path, session, history))
}

async fn execute_in_parallel<I, F>(futures: I) -> Vec<F::Output>
where
    I: IntoIterator<Item = F>,
    F: std::future::Future,
{
    join_all(futures).await
}

async fn execute_independent_tool(
    mcp: &crate::mcp::McpRegistry,
    cwd: &Path,
    shell_environment: &HashMap<String, String>,
    call: &FunctionCall,
) -> Result<ToolExecutionResult, String> {
    match mcp.call(&call.name, &call.arguments).await {
        Some(Ok(result)) => Ok(ToolExecutionResult {
            output: result.content,
            is_error: result.is_error,
            store_writes: None,
            details: result.structured_content,
        }),
        Some(Err(error)) => Err(error),
        None => crate::tools::execute_with_environment(
            cwd,
            &call.name,
            &call.arguments,
            shell_environment,
        )
        .await
        .map(|result| ToolExecutionResult {
            output: result.content,
            is_error: result.is_error,
            store_writes: None,
            details: result.details.or(result.structured_content),
        })
        .map_err(|error| error.to_string()),
    }
}

fn retryable_provider_error(error: &AgentError) -> bool {
    let AgentError::Provider(message) = error else {
        return false;
    };
    let message = message.to_ascii_lowercase();
    if [
        "gousagelimiterror",
        "freeusagelimiterror",
        "monthly usage limit reached",
        "available balance",
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
        "subscription_sharing_usage_limit_exceeded",
        "usage limit",
    ]
    .iter()
    .any(|phrase| message.contains(phrase))
    {
        return false;
    }
    [
        "429",
        "500",
        "502",
        "503",
        "504",
        "520",
        "524",
        "rate limit",
        "rate-limit",
        "ratelimit",
        "too many requests",
        "overloaded",
        "server_busy",
        "servers are currently busy",
        "currently experiencing high demand",
        "model is at capacity",
        "service unavailable",
        "service is unavailable",
        "temporarily unavailable",
        "server error",
        "internal error",
        "network",
        "connection",
        "fetch failed",
        "eai_again",
        "enotfound",
        "socket hang up",
        "timeout",
        "timed out",
        "upstream",
        "stream failed",
        "stream was interrupted",
        "provider returned error",
        "provider returned an error",
        "request buffer limit while retrying upstream",
        "unable to forward upstream",
        "upstream request failed",
        "upstream_stream_read_error",
        "http/2 stream failed",
        "other side closed",
        "getaddrinfo",
        "socket connection was closed",
        "ended without",
        "stream ended before message_stop",
        "terminal response event",
        "http2 request did not get a response",
        "pending stream has been canceled",
        "retry delay",
        "you can retry your request",
        "try your request again",
        "please retry your request",
        "resourceexhausted",
        "subscription_sharing_usage_unavailable",
        "subscription_sharing_user_unavailable",
        "terminated",
    ]
    .iter()
    .any(|phrase| message.contains(phrase))
}

fn retry_delay_ms(policy: RetryPolicy, attempt: u32) -> u64 {
    let exponent = attempt.saturating_sub(1).min(63);
    let multiplier = 1_u64.checked_shl(exponent).unwrap_or(u64::MAX);
    policy
        .base_delay_ms
        .saturating_mul(multiplier)
        .min(policy.max_delay_ms)
}

fn usage(r: &Reply) -> MessageUsage {
    MessageUsage {
        input: r.input,
        output: r.output,
        cache_read: r.cache_read,
        cache_write: r.cache_write,
        reasoning: r.reasoning,
        total_tokens: r.total_tokens,
        cost: MessageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

fn write_session(session: &SessionDocument, path: &Path) -> Result<(), AgentError> {
    session
        .write_atomic(path)
        .map_err(|error| AgentError::Session(error.to_string()))
}

pub struct MuAgent {
    provider: Provider,
    provider_id: String,
    model_choices: Vec<String>,
    session: SessionDocument,
    history: Vec<Value>,
    session_path: PathBuf,
    persist_session: bool,
    agent_dir: PathBuf,
    cwd: PathBuf,
    skills: crate::skills::SkillCatalog,
    hide_thinking_block: bool,
    retry: RetryPolicy,
    enabled_tools: Vec<String>,
    no_tools: bool,
    tool_selection: Option<ToolSelection>,
    mcp: crate::mcp::McpRegistry,
    mcp_startup: Option<Arc<OnceCell<crate::mcp::McpRegistry>>>,
    active_mcp_tools: HashSet<String>,
    prompt_queue: AgentPromptQueue,
}

impl MuAgent {
    pub async fn connect() -> Result<Self, AgentError> {
        Self::connect_with_session(None).await
    }

    pub async fn connect_with_session(session_file: Option<PathBuf>) -> Result<Self, AgentError> {
        Self::connect_with_options(SessionOptions {
            path: session_file,
            ..SessionOptions::default()
        })
        .await
    }

    pub async fn connect_with_options(options: SessionOptions) -> Result<Self, AgentError> {
        let mut config = config()?;
        let tool_selection = options
            .tools
            .clone()
            .map(ToolSelection::parse)
            .transpose()?;
        if let Some(selection) = &tool_selection {
            config.enabled_tools = apply_tool_selection(&config.enabled_tools, selection);
        }
        if options.no_builtin_tools || options.no_tools {
            config.enabled_tools.clear();
        }
        let no_tools = options.no_tools;
        let persist_session = !options.no_session;
        let skills = crate::skills::SkillCatalog::discover_with_paths(
            &config.cwd,
            &config.agent_dir,
            &options.skill_paths,
            !options.no_skills,
        );
        let (session_path, session, history) = resolve_session_options(&config, options)?;
        let mcp_startup = Arc::new(OnceCell::new());
        let background_mcp = Arc::clone(&mcp_startup);
        let mcp_agent_dir = config.agent_dir.clone();
        let mcp_cwd = config.cwd.clone();
        let mcp_startup_task = tokio::spawn(async move {
            background_mcp
                .get_or_init(|| async move {
                    crate::mcp::McpRegistry::connect(&mcp_agent_dir, &mcp_cwd).await
                })
                .await;
        });
        let provider = match connect(&config, skills.system_prompt()).await {
            Ok(provider) => provider,
            Err(error) => {
                mcp_startup_task.abort();
                return Err(error);
            }
        };
        drop(mcp_startup_task);
        let mcp = crate::mcp::McpRegistry::default();
        let active_mcp_tools = session_active_mcp_tools(&session);
        Ok(Self {
            provider,
            provider_id: config.provider,
            model_choices: config.model_choices,
            session,
            history,
            session_path,
            persist_session,
            agent_dir: config.agent_dir.clone(),
            cwd: config.cwd.clone(),
            skills,
            hide_thinking_block: config.hide_thinking_block,
            retry: config.retry,
            enabled_tools: config.enabled_tools,
            no_tools,
            tool_selection,
            mcp,
            mcp_startup: Some(mcp_startup),
            active_mcp_tools,
            prompt_queue: AgentPromptQueue::default(),
        })
    }

    /// Create a deterministic no-network agent for terminal previews and tests.
    pub fn demo() -> Result<Self, AgentError> {
        let cwd = std::env::current_dir()?;
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or_else(|| AgentError::Configuration("HOME is not set".into()))?;
        let config = Config {
            agent_dir: std::env::var_os("PI_CODING_AGENT_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join(".pi/agent")),
            cwd,
            provider: "mu-demo".into(),
            model: "mu-demo".into(),
            model_choices: vec!["mu-demo".into(), "mu-demo-alt".into()],
            hide_thinking_block: false,
            retry: RetryPolicy::default(),
            enabled_tools: resolve_default_tools(None),
        };
        Self::demo_from_config(config)
    }

    fn demo_from_config(config: Config) -> Result<Self, AgentError> {
        let path = session_path(&config, None);
        let session = SessionDocument::new(config.cwd.to_string_lossy().into_owned());
        let active_mcp_tools = session_active_mcp_tools(&session);
        Ok(Self {
            provider: Provider {
                transport: ProviderTransport::Demo,
                model: config.model.clone(),
                instructions: "You are Mu, a deterministic demo assistant.".into(),
                supports_reasoning: false,
            },
            provider_id: config.provider,
            model_choices: config.model_choices,
            session,
            history: Vec::new(),
            session_path: path,
            persist_session: true,
            agent_dir: config.agent_dir.clone(),
            cwd: config.cwd.clone(),
            skills: crate::skills::SkillCatalog::discover(&config.cwd, &config.agent_dir),
            hide_thinking_block: config.hide_thinking_block,
            retry: config.retry,
            enabled_tools: config.enabled_tools,
            no_tools: false,
            tool_selection: None,
            mcp: crate::mcp::McpRegistry::default(),
            mcp_startup: None,
            active_mcp_tools,
            prompt_queue: AgentPromptQueue::default(),
        })
    }

    pub fn model(&self) -> &str {
        &self.provider.model
    }

    pub fn provider(&self) -> &str {
        &self.provider_id
    }

    pub fn mcp_tools(&self) -> Vec<String> {
        self.mcp
            .codemode_definitions()
            .into_iter()
            .filter_map(|definition| definition.get("name")?.as_str().map(str::to_owned))
            .collect()
    }

    pub fn mcp_status(&self) -> String {
        self.mcp_startup
            .as_ref()
            .and_then(|startup| startup.get())
            .map_or_else(
                || {
                    if self.mcp_startup.is_some() {
                        "MCP servers are connecting in the background.".to_owned()
                    } else {
                        self.mcp.status_text()
                    }
                },
                crate::mcp::McpRegistry::status_text,
            )
    }

    async fn ensure_mcp_ready(&mut self) {
        let Some(startup) = self.mcp_startup.clone() else {
            return;
        };
        let agent_dir = self.agent_dir.clone();
        let cwd = self.cwd.clone();
        let connected = tokio::time::timeout(
            MCP_STARTUP_WAIT,
            startup.get_or_init(|| async move {
                crate::mcp::McpRegistry::connect(&agent_dir, &cwd).await
            }),
        )
        .await;
        if let Ok(registry) = connected {
            self.mcp = registry.clone();
            self.mcp_startup = None;
        }
    }

    pub fn prompt_queue(&self) -> AgentPromptQueue {
        self.prompt_queue.clone()
    }

    pub fn model_choices(&self) -> &[String] {
        &self.model_choices
    }

    fn save_session(&self) -> Result<(), AgentError> {
        if self.persist_session {
            write_session(&self.session, &self.session_path)?;
        }
        Ok(())
    }

    pub fn select_model(&mut self, model: &str) -> Result<bool, AgentError> {
        if !self.model_choices.iter().any(|choice| choice == model) {
            return Ok(false);
        }
        if self.provider.model == model {
            return Ok(true);
        }
        self.provider.model = model.to_owned();
        self.session.append_model_change(&self.provider_id, model);
        self.save_session()?;
        Ok(true)
    }

    pub fn session_path(&self) -> &Path {
        &self.session_path
    }

    pub fn session_transcript(&self) -> Vec<SessionTranscriptMessage> {
        session_transcript(&self.session)
    }

    pub fn session_info(&self) -> SessionInfo {
        let header = self.session.header();
        SessionInfo {
            id: header
                .map(|header| header.id.as_str().to_owned())
                .unwrap_or_default(),
            name: self.session.session_name().map(str::to_owned),
            preview: session_preview(&self.session),
            path: self.session_path.clone(),
            timestamp: header
                .map(|header| header.timestamp.as_str().to_owned())
                .unwrap_or_default(),
        }
    }

    pub fn name_session(&mut self, name: &str) -> Result<(), AgentError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(AgentError::Session("Session name must not be empty".into()));
        }
        let mut session = self.session.clone();
        session.append_session_name(name);
        if self.persist_session {
            write_session(&session, &self.session_path)?;
        }
        self.session = session;
        Ok(())
    }

    /// Export the current active branch as Pi JSONL or a standalone HTML transcript.
    pub fn export_session(&self, path: &Path) -> Result<(), AgentError> {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("jsonl") => write_session(&self.session, path),
            Some("html") => std::fs::write(path, self.render_session_html())
                .map_err(|error| AgentError::Session(error.to_string())),
            _ => Err(AgentError::Session(
                "Session export path must end in .jsonl or .html".into(),
            )),
        }
    }

    fn render_session_html(&self) -> String {
        let info = self.session_info();
        let title = info.name.as_deref().unwrap_or("Mu session");
        let mut html = format!(
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{}</title><style>{SESSION_HTML_STYLE}</style></head><body><main><header><h1>{}</h1><p class=\"meta\">Session {} · {}</p></header>",
            escape_html(title),
            escape_html(title),
            escape_html(&info.id),
            escape_html(&info.timestamp),
        );
        for message in self.session_transcript() {
            append_transcript_message_html(&mut html, message);
        }
        html.push_str("</main></body></html>");
        html
    }

    pub fn import_session(&mut self, path: &Path) -> Result<(), AgentError> {
        let (session, history) = load_session(path)?;
        self.session = session;
        self.history = history;
        self.session_path = path.to_owned();
        self.persist_session = true;
        self.active_mcp_tools = session_active_mcp_tools(&self.session);
        self.prompt_queue.clear();
        Ok(())
    }

    pub fn clone_session(&mut self) -> Result<(), AgentError> {
        if !self.persist_session {
            return Err(AgentError::Session(
                "Cannot clone a session that is not persisted to disk".into(),
            ));
        }
        let new_id = uuid::Uuid::new_v4().to_string();
        let new_session = self
            .session
            .fork_with_id(
                self.cwd.to_string_lossy().into_owned(),
                new_id,
                self.session_path.to_string_lossy().into_owned(),
            )
            .ok_or_else(|| AgentError::Session("Current session has no header".into()))?;
        let new_path = new_session_path(&self.agent_dir, &self.cwd, None);
        write_session(&new_session, &new_path)?;

        self.session = new_session;
        self.history = session_history(&self.session);
        self.session_path = new_path;
        self.active_mcp_tools = session_active_mcp_tools(&self.session);
        self.prompt_queue.clear();
        Ok(())
    }

    pub fn start_new_session(&mut self) -> Result<(), AgentError> {
        let parent = self.session_path.parent().ok_or_else(|| {
            AgentError::Session("current session path has no parent directory".into())
        })?;
        let timestamp = chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
            .replace([':', '.'], "-");
        let next_path = parent.join(format!("{timestamp}_{}.jsonl", uuid::Uuid::new_v4()));
        let next_session = SessionDocument::new(self.cwd.to_string_lossy().into_owned());
        if self.persist_session {
            write_session(&next_session, &next_path)?;
        }

        self.session = next_session;
        self.history.clear();
        self.session_path = next_path;
        self.active_mcp_tools.clear();
        self.prompt_queue.clear();
        Ok(())
    }

    /// Skills discovered from Pi's global and project skill directories.
    pub fn skills(&self) -> &crate::skills::SkillCatalog {
        &self.skills
    }

    fn expand_prompt(&self, prompt: String, on_event: &mut impl FnMut(AgentEvent)) -> String {
        if !skill_commands_enabled(&self.agent_dir, &self.cwd) {
            return prompt;
        }
        expand_skill_prompt(prompt, &self.skills, on_event)
    }

    pub fn hide_thinking_block(&self) -> bool {
        self.hide_thinking_block
    }

    /// Execute one of Mu's Pi-compatible built-in tools in the project directory.
    pub async fn execute_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<crate::tools::ToolResult, crate::tools::ToolError> {
        if !self.enabled_tools.iter().any(|enabled| enabled == name) {
            return Err(crate::tools::ToolError::Arguments(format!(
                "tool `{name}` is not enabled"
            )));
        }
        crate::tools::execute_with_environment(
            &self.cwd,
            name,
            arguments,
            &self.shell_environment(),
        )
        .await
    }

    fn shell_environment(&self) -> HashMap<String, String> {
        let mut environment = HashMap::new();
        if let Some(header) = self.session.header() {
            environment.insert("PI_SESSION_ID".into(), header.id.as_str().to_owned());
        }
        if self.persist_session {
            environment.insert(
                "PI_SESSION_FILE".into(),
                self.session_path.to_string_lossy().into_owned(),
            );
        }
        environment.insert("PI_PROVIDER".into(), self.provider_id.clone());
        environment.insert("PI_MODEL".into(), self.provider.model.clone());
        if let Some(prefix) = shell_setting(&self.agent_dir, &self.cwd, "shellCommandPrefix") {
            environment.insert(crate::tools::SHELL_COMMAND_PREFIX_ENV.into(), prefix);
        }
        if let Some(path) = configured_bash_shell_path(&self.agent_dir, &self.cwd) {
            environment.insert(crate::tools::BASH_SHELL_PATH_ENV.into(), path);
        }
        environment
    }

    fn record_queued_prompt(
        &mut self,
        prompt: String,
        behavior: PromptBehavior,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<(), AgentError> {
        let prompt = self.expand_prompt(prompt, on_event);
        self.session.append_user_text(prompt.clone());
        self.save_session()?;
        self.history.push(json!({
            "role":"user",
            "content":[{"type":"input_text","text":prompt}],
        }));
        on_event(AgentEvent::QueuedPromptStarted { prompt, behavior });
        Ok(())
    }

    fn codemode_store(&self) -> Value {
        session_codemode_store(&self.session)
    }

    fn refresh_mcp_if_ready(&mut self) {
        let Some(registry) = self
            .mcp_startup
            .as_ref()
            .and_then(|startup| startup.get())
            .cloned()
        else {
            return;
        };
        self.mcp = registry;
        self.mcp_startup = None;
    }

    async fn complete_with_retry(
        &mut self,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let mut attempt = 0;
        loop {
            self.refresh_mcp_if_ready();
            match self
                .provider
                .complete(
                    &self.history,
                    &self.tool_definitions(&self.active_mcp_tools),
                    on_event,
                )
                .await
            {
                Ok(reply) => return Ok(reply),
                Err(error)
                    if self.retry.enabled
                        && attempt < self.retry.max_retries
                        && retryable_provider_error(&error) =>
                {
                    attempt += 1;
                    let delay_ms = retry_delay_ms(self.retry, attempt);
                    on_event(AgentEvent::RetryScheduled {
                        attempt,
                        max_attempts: self.retry.max_retries,
                        delay_ms,
                        error: error.to_string(),
                    });
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    on_event(AgentEvent::RetryAttemptStarted { attempt });
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn tool_definitions(&self, active_mcp_tools: &HashSet<String>) -> Vec<Value> {
        if self.no_tools {
            return Vec::new();
        }
        let all_definitions = crate::tools::definitions();
        let mut definitions = self
            .enabled_tools
            .iter()
            .filter_map(|enabled| {
                all_definitions
                    .iter()
                    .find(|definition| {
                        definition.get("name").and_then(Value::as_str) == Some(enabled)
                    })
                    .cloned()
            })
            .map(mark_strict_builtin_tool)
            .collect::<Vec<_>>();
        definitions.extend(self.mcp.definitions_for_model(active_mcp_tools));
        if self.mcp.codemode_enabled() {
            definitions.push(crate::codemode::definition(&self.mcp));
        }
        if self.mcp.has_deferred_tools() {
            definitions.push(crate::tool_search::definition());
        }
        if let Some(selection) = &self.tool_selection {
            definitions = filter_tool_definitions(definitions, selection);
        }
        definitions
    }

    pub async fn submit(
        &mut self,
        prompt: String,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<(), AgentError> {
        let prompt = self.expand_prompt(prompt, on_event);
        self.session.append_user_text(prompt.clone());
        self.save_session()?;
        self.history
            .push(json!({"role":"user","content":[{"type":"input_text","text":prompt}]}));
        self.ensure_mcp_ready().await;
        if let Some(steering) = self.prompt_queue.take_steering() {
            self.record_queued_prompt(steering, PromptBehavior::Steer, on_event)?;
        }
        let mut reply = self.complete_with_retry(on_event).await?;
        loop {
            if reply.tool_calls.is_empty() {
                break;
            }
            let api = api_name(&self.provider.transport);
            let truncated_tool_arguments =
                reply.incomplete_reason.as_deref() == Some("max_output_tokens");
            let calls = std::mem::take(&mut reply.tool_calls);
            let records = calls
                .iter()
                .map(|call| ToolCallRecord {
                    id: call.call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect::<Vec<_>>();
            self.session.append_assistant_tool_calls(
                api,
                &self.provider_id,
                &self.provider.model,
                AssistantToolResponse {
                    thinking: reply.thinking.clone(),
                    thinking_signature: reply.thinking_signature.clone(),
                    text: reply.text.clone(),
                    calls: records,
                    usage: usage(&reply),
                },
            );
            self.save_session()?;

            // Preserve the provider's full assistant output (including reasoning
            // signatures and all sibling calls) for stateless Responses replay.
            self.history.extend(reply.output_items.iter().cloned());
            // The Responses API expects all assistant calls before their outputs.
            for call in &calls {
                on_event(AgentEvent::ToolStarted {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
            }
            let mut parallel_results = if !self.no_tools
                && !truncated_tool_arguments
                && calls
                    .iter()
                    .all(|call| !matches!(call.name.as_str(), "tool_search" | "codemode"))
            {
                let shell_environment = self.shell_environment();
                Some(
                    execute_in_parallel(calls.iter().map(|call| {
                        execute_independent_tool(&self.mcp, &self.cwd, &shell_environment, call)
                    }))
                    .await
                    .into_iter(),
                )
            } else {
                None
            };
            for call in calls {
                let result = if self.no_tools {
                    Err(format!("Tool {} is not available", call.name))
                } else if truncated_tool_arguments {
                    Err(format!(
                        "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
                        call.name
                    ))
                } else if let Some(results) = parallel_results.as_mut() {
                    results.next().expect("one result per tool call")
                } else if call.name == "tool_search" {
                    let query = call
                        .arguments
                        .get("query")
                        .and_then(Value::as_str)
                        .filter(|query| !query.trim().is_empty());
                    match query {
                        Some(query) => {
                            let limit = call
                                .arguments
                                .get("limit")
                                .and_then(Value::as_u64)
                                .unwrap_or(8)
                                .clamp(1, 30) as usize;
                            let matches =
                                self.mcp.search_tools(query, limit, &self.active_mcp_tools);
                            for tool in &matches {
                                if let Some(name) = tool.get("name").and_then(Value::as_str) {
                                    self.active_mcp_tools.insert(name.to_owned());
                                }
                            }
                            let output = if matches.is_empty() {
                                "No matching tools found.".to_owned()
                            } else {
                                let loaded = matches
                                    .iter()
                                    .map(|tool| {
                                        format!(
                                            "- {}: {}",
                                            tool.get("name")
                                                .and_then(Value::as_str)
                                                .unwrap_or_default(),
                                            tool.get("description")
                                                .and_then(Value::as_str)
                                                .unwrap_or_default()
                                                .lines()
                                                .next()
                                                .unwrap_or_default()
                                        )
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                format!(
                                    "Loaded {} tool{}. They are available from your next call:\n{loaded}",
                                    matches.len(),
                                    if matches.len() == 1 { "" } else { "s" }
                                )
                            };
                            if !matches.is_empty() {
                                let mut active =
                                    self.active_mcp_tools.iter().cloned().collect::<Vec<_>>();
                                active.sort();
                                self.session.append_custom_entry(
                                    "mcp-tool-loadout",
                                    json!({"active":active}),
                                );
                                self.save_session()?;
                            }
                            Ok(ToolExecutionResult::text(output, false))
                        }
                        None => Err("tool_search requires a non-empty query".into()),
                    }
                } else if call.name == "codemode" {
                    let store = self.codemode_store();
                    crate::codemode::execute(
                        &self.cwd,
                        &self.mcp,
                        &store,
                        &call.arguments,
                        &self.shell_environment(),
                    )
                    .await
                    .map(|(output, is_error, store_writes)| ToolExecutionResult {
                        output,
                        is_error,
                        store_writes,
                        details: None,
                    })
                    .map_err(|error| error.to_string())
                } else {
                    match self.mcp.call(&call.name, &call.arguments).await {
                        Some(Ok(result)) => Ok(ToolExecutionResult {
                            output: result.content,
                            is_error: result.is_error,
                            store_writes: None,
                            details: result.structured_content,
                        }),
                        Some(Err(error)) => Err(error),
                        None => self
                            .execute_tool(&call.name, &call.arguments)
                            .await
                            .map(|result| ToolExecutionResult {
                                output: result.content,
                                is_error: result.is_error,
                                store_writes: None,
                                details: result.details.or(result.structured_content),
                            })
                            .map_err(|error| error.to_string()),
                    }
                };
                let result = match result {
                    Ok(result) => result,
                    Err(error) => ToolExecutionResult::text(error, true),
                };
                if let Some(store_writes) = result.store_writes {
                    self.session
                        .append_custom_entry("codemode-store", store_writes);
                    self.save_session()?;
                }
                on_event(AgentEvent::ToolFinished {
                    call_id: call.call_id.clone(),
                    output: result.output.clone(),
                    is_error: result.is_error,
                });
                self.history.push(json!({
                    "type":"function_call_output", "call_id":call.call_id, "output":result.output,
                }));
                self.session.append_tool_result_with_details(
                    &call.call_id,
                    &call.name,
                    result.output,
                    result.is_error,
                    result.details,
                );
                self.save_session()?;
            }
            if let Some(steering) = self.prompt_queue.take_steering() {
                self.record_queued_prompt(steering, PromptBehavior::Steer, on_event)?;
            }
            reply = self.complete_with_retry(on_event).await?;
        }
        let api = api_name(&self.provider.transport);
        self.session.append_assistant_response(
            api,
            &self.provider_id,
            &self.provider.model,
            AssistantTurnResponse {
                thinking: reply.thinking.clone(),
                thinking_signature: reply.thinking_signature.clone(),
                text: reply.text.clone(),
                usage: usage(&reply),
            },
        );
        self.save_session()?;
        if reply.output_items.is_empty() {
            self.history.push(
                json!({"role":"assistant","content":[{"type":"output_text","text":reply.text}]}),
            );
        } else {
            self.history.extend(reply.output_items);
        }
        if let Some((prompt, behavior)) = self.prompt_queue.take_next() {
            on_event(AgentEvent::QueuedPromptStarted {
                prompt: prompt.clone(),
                behavior,
            });
            return Box::pin(self.submit(prompt, on_event)).await;
        }
        Ok(())
    }
}

pub fn list_sessions_for_current_cwd(
    session_dir: Option<&Path>,
) -> Result<Vec<SessionInfo>, AgentError> {
    let config = config()?;
    Ok(
        list_session_paths(&config.agent_dir, &config.cwd, session_dir)
            .into_iter()
            .filter_map(|(_, path, header)| {
                let contents = std::fs::read_to_string(&path).ok()?;
                let document = SessionDocument::parse_jsonl(&contents).ok()?;
                Some(SessionInfo {
                    id: header.get("id")?.as_str()?.to_owned(),
                    name: document.session_name().map(str::to_owned),
                    preview: session_preview(&document),
                    path,
                    timestamp: header.get("timestamp")?.as_str()?.to_owned(),
                })
            })
            .collect(),
    )
}

pub fn latest_session_for_current_cwd(
    session_dir: Option<&Path>,
) -> Result<Option<PathBuf>, AgentError> {
    let config = config()?;
    Ok(latest_session_path(
        &config.agent_dir,
        &config.cwd,
        session_dir,
    ))
}

pub async fn run_prompt(prompt: String) -> Result<(), AgentError> {
    run_prompt_with_session(prompt, None).await
}

pub async fn run_prompt_with_session(
    prompt: String,
    session_file: Option<PathBuf>,
) -> Result<(), AgentError> {
    run_prompt_with_options(
        prompt,
        SessionOptions {
            path: session_file,
            ..SessionOptions::default()
        },
    )
    .await
}

pub async fn run_prompt_with_options(
    prompt: String,
    session_options: SessionOptions,
) -> Result<(), AgentError> {
    let mut agent = MuAgent::connect_with_options(session_options).await?;
    agent
        .submit(prompt, &mut |event| {
            match event {
                AgentEvent::TextDelta(text) => print!("{text}"),
                AgentEvent::RetryScheduled {
                    attempt,
                    max_attempts,
                    delay_ms,
                    error,
                } => eprintln!(
                    "Retry {attempt}/{max_attempts} in {:.1}s: {error} (interrupt to cancel)",
                    delay_ms as f64 / 1000.0
                ),
                _ => {}
            }
            let _ = io::stdout().flush();
        })
        .await?;
    println!();
    Ok(())
}

pub async fn run_cli() -> Result<(), AgentError> {
    let mut agent = MuAgent::connect().await?;
    println!(
        "Mu — {} via {} (session {})",
        agent.model(),
        agent.provider(),
        agent.session_path().display()
    );
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        return agent
            .submit(args.join(" "), &mut |event| {
                if let AgentEvent::TextDelta(text) = event {
                    print!("{text}")
                }
            })
            .await;
    }
    let mut line = String::new();
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let prompt = line.trim();
        if prompt == "/exit" || prompt == "/quit" {
            break;
        }
        if prompt.is_empty() {
            continue;
        }
        agent
            .submit(prompt.to_owned(), &mut |event| {
                if let AgentEvent::TextDelta(text) = event {
                    print!("{text}");
                }
                let _ = io::stdout().flush();
            })
            .await?;
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        AgentPromptQueue, FunctionCall, PromptBehavior, Reply, TerminalResponse, configured_models,
        execute_in_parallel, find_session_id_path, latest_session_path, load_session,
        merge_final_response_text, resolve_session_options, response_message_text,
        session_active_mcp_tools, session_codemode_store, session_history, session_preview,
        session_transcript, trusted_url, validate_session_id,
    };
    use super::{Config, SessionOptions};
    use mu_session::{
        AssistantToolResponse, AssistantTurnResponse, MessageUsage, SessionDocument, ToolCallRecord,
    };
    use serde_json::Value;
    use std::path::Path;

    #[test]
    fn session_transcript_reconstructs_text_thinking_and_completed_tools() {
        let mut session = SessionDocument::new("/tmp/project");
        session.append_user_text("Run the command");
        session.append_assistant_tool_calls(
            "openai-responses",
            "provider",
            "model",
            AssistantToolResponse {
                thinking: "Checking first".into(),
                thinking_signature: None,
                text: "I will inspect this.".into(),
                calls: vec![ToolCallRecord {
                    id: "call-1".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({"command":"pwd"}),
                }],
                usage: MessageUsage::default(),
            },
        );
        session.append_tool_result("call-1", "bash", "success: /tmp/project".into(), true);
        session.append_assistant_response(
            "openai-responses",
            "provider",
            "model",
            AssistantTurnResponse {
                thinking: String::new(),
                thinking_signature: None,
                text: "Done.".into(),
                usage: MessageUsage::default(),
            },
        );

        let transcript = session_transcript(&session);
        assert_eq!(transcript.len(), 3);
        assert_eq!(transcript[0].role, super::SessionTranscriptRole::User);
        assert_eq!(transcript[0].text, "Run the command");
        assert_eq!(transcript[1].role, super::SessionTranscriptRole::Assistant);
        assert_eq!(transcript[1].thinking, "Checking first");
        assert_eq!(transcript[1].text, "I will inspect this.");
        assert_eq!(transcript[1].tools[0].name, "bash");
        assert_eq!(
            transcript[1].tools[0].output.as_deref(),
            Some("success: /tmp/project")
        );
        assert!(transcript[1].tools[0].is_error);
        assert_eq!(transcript[2].text, "Done.");
    }

    #[test]
    fn session_preview_uses_the_latest_user_text_and_compacts_whitespace() {
        let mut session = SessionDocument::new("/tmp/project");
        session.append_user_text("earlier prompt");
        session.append_user_text("  latest\n prompt   text  ");
        assert_eq!(
            session_preview(&session).as_deref(),
            Some("latest prompt text")
        );

        let long_prompt = "x".repeat(200);
        session.append_user_text(&long_prompt);
        assert_eq!(
            session_preview(&session).unwrap(),
            format!("{}…", "x".repeat(160))
        );
    }

    #[test]
    fn project_shell_command_prefix_overrides_agent_settings() {
        let root = std::env::temp_dir().join(format!("mu-shell-prefix-{}", uuid::Uuid::new_v4()));
        let agent_dir = root.join("agent");
        let project_dir = root.join("project/.pi");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::create_dir_all(&project_dir).unwrap();
        std::fs::write(
            agent_dir.join("settings.json"),
            r#"{"shellCommandPrefix":"global setup","enableSkillCommands":true}"#,
        )
        .unwrap();
        std::fs::write(
            project_dir.join("settings.json"),
            r#"{"shellCommandPrefix":"project setup","shellPath":"bin/bash","enableSkillCommands":false}"#,
        )
        .unwrap();
        assert_eq!(
            super::shell_setting(&agent_dir, &root.join("project"), "shellCommandPrefix")
                .as_deref(),
            Some("project setup")
        );
        assert_eq!(
            super::configured_bash_shell_path(&agent_dir, &root.join("project")).as_deref(),
            Some(root.join("project/bin/bash").to_string_lossy().as_ref())
        );
        assert!(!super::skill_commands_enabled(
            &agent_dir,
            &root.join("project")
        ));
        assert!(super::skill_commands_enabled(
            &root.join("missing-agent"),
            &root
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cloning_session_copies_the_active_branch_to_a_new_file() {
        let root = std::env::temp_dir().join(format!("mu-session-clone-{}", uuid::Uuid::new_v4()));
        let cwd = root.join("project");
        let agent_dir = root.join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        let config = super::Config {
            agent_dir,
            cwd: cwd.clone(),
            provider: "demo".into(),
            model: "demo".into(),
            model_choices: vec!["demo".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        };
        let mut agent = super::MuAgent::demo_from_config(config).unwrap();
        agent
            .session
            .append_user_text("keep <script>alert('x')</script>");
        agent.session.append_assistant_turn(
            "responses",
            "demo",
            "demo",
            "thinking <private>",
            "assistant says <hello>",
            mu_session::MessageUsage::default(),
        );
        agent.session.append_tool_call(
            "call-1",
            "read",
            serde_json::json!({"path":"a&b.txt"}),
            "responses",
            "demo",
            "demo",
        );
        agent.session.append_tool_result(
            "call-1",
            "read",
            "<script>tool output</script>".into(),
            true,
        );
        agent.save_session().unwrap();
        let original_path = agent.session_path().to_owned();
        let original_id = agent.session.header().unwrap().id.as_str().to_owned();
        let entry_count = agent.session.active_branch_entries().len();

        agent.clone_session().unwrap();

        let cloned_header = agent.session.header().unwrap();
        assert_ne!(cloned_header.id.as_str(), original_id);
        assert_eq!(
            cloned_header.parent_session.as_deref(),
            Some(original_path.to_string_lossy().as_ref())
        );
        assert_eq!(agent.session.active_branch_entries().len(), entry_count);
        assert!(agent.session_path().is_file());
        assert_ne!(agent.session_path(), original_path);

        let clone_id = cloned_header.id.as_str().to_owned();
        let html_path = root.join("export.html");
        agent.export_session(&html_path).unwrap();
        let html = std::fs::read_to_string(html_path).unwrap();
        assert!(html.contains("keep &lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;"));
        assert!(html.contains("thinking &lt;private&gt;"));
        assert!(html.contains("assistant says &lt;hello&gt;"));
        assert!(html.contains("<details class=\"error\"><summary>Tool: read"));
        assert!(html.contains("a&amp;b.txt"));
        assert!(html.contains("&lt;script&gt;tool output&lt;/script&gt;"));
        assert!(!html.contains("<script>alert('x')</script>"));
        assert!(!html.contains("<script>tool output</script>"));
        assert!(agent.export_session(&root.join("export.txt")).is_err());

        let export_path = root.join("export.jsonl");
        agent.export_session(&export_path).unwrap();
        agent.start_new_session().unwrap();
        agent.import_session(&export_path).unwrap();
        assert_eq!(agent.session.header().unwrap().id.as_str(), clone_id);
        assert_eq!(agent.session.active_branch_entries().len(), entry_count);
        assert_eq!(agent.session_path(), export_path);
        agent.name_session("  Refactor session  ").unwrap();
        assert_eq!(agent.session.session_name(), Some("Refactor session"));
        let (saved_session, _) = load_session(&export_path).unwrap();
        assert_eq!(saved_session.session_name(), Some("Refactor session"));
        assert!(agent.name_session("  ").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn skill_commands_expand_frontmatter_body_and_arguments() {
        let root = std::env::temp_dir().join(format!("mu-skill-command-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let file_path = root.join("review/SKILL.md");
        std::fs::create_dir_all(file_path.parent().unwrap()).unwrap();
        std::fs::write(
            &file_path,
            "---\nname: review\ndescription: review code\n---\nReview this file carefully.\n",
        )
        .unwrap();
        let skills = crate::skills::SkillCatalog {
            skills: vec![crate::skills::Skill {
                name: "review".into(),
                description: "review code".into(),
                file_path: file_path.clone(),
                disable_model_invocation: true,
            }],
            diagnostics: Vec::new(),
        };

        let expanded =
            super::expand_skill_command("/skill:review  check edge cases ".into(), &skills)
                .unwrap();
        assert_eq!(
            expanded,
            format!(
                "<skill name=\"review\" location=\"{}\">\nReferences are relative to {}.\n\nReview this file carefully.\n</skill>\n\ncheck edge cases",
                file_path.display(),
                file_path.parent().unwrap().display(),
            )
        );
        assert_eq!(
            super::expand_skill_command("/skill:missing argument".into(), &skills).unwrap(),
            "/skill:missing argument"
        );
        assert_eq!(
            super::expand_skill_command("/skill:review ".into(), &skills).unwrap(),
            expanded.strip_suffix("\n\ncheck edge cases").unwrap(),
            "a space with no remaining argument still expands the skill"
        );
        for prompt in ["/skill:review\targument", "/skill:review\u{00a0}argument"] {
            assert_eq!(
                super::expand_skill_command(prompt.into(), &skills).unwrap(),
                prompt,
                "Pi uses a literal ASCII space as the skill-argument delimiter"
            );
        }
        let mut warning = None;
        let unchanged =
            super::expand_skill_prompt("/skill:missing argument".into(), &skills, &mut |event| {
                warning = Some(event)
            });
        assert_eq!(unchanged, "/skill:missing argument");
        assert!(warning.is_none());
        std::fs::remove_file(&file_path).unwrap();
        let mut warning = None;
        assert_eq!(
            super::expand_skill_prompt("/skill:review".into(), &skills, &mut |event| {
                warning = Some(event)
            }),
            "/skill:review"
        );
        assert!(matches!(
            warning,
            Some(super::AgentEvent::SkillCommandWarning(message))
                if message.contains("Could not read skill 'review'")
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incomplete_responses_preserve_only_the_output_token_limit_as_a_retryable_result() {
        assert_eq!(TerminalResponse::Completed.error_message(), None);
        assert_eq!(
            TerminalResponse::Incomplete {
                reason: Some("max_output_tokens".into())
            }
            .error_message(),
            None
        );
        assert_eq!(
            TerminalResponse::Incomplete {
                reason: Some("content_filter".into())
            }
            .error_message()
            .as_deref(),
            Some("Response incomplete: content_filter")
        );
        assert_eq!(
            TerminalResponse::Incomplete { reason: None }
                .error_message()
                .as_deref(),
            Some("Response incomplete without a provider reason")
        );
    }

    #[test]
    fn responses_failure_events_preserve_provider_error_details() {
        assert_eq!(
            super::response_failed_error(&serde_json::json!({
                "response":{"error":{"code":"server_error","message":"unavailable"}}
            }))
            .to_string(),
            "provider: server_error: unavailable"
        );
        assert_eq!(
            super::response_failed_error(&serde_json::json!({
                "response":{"incomplete_details":{"reason":"content_filter"}}
            }))
            .to_string(),
            "provider: incomplete: content_filter"
        );
        assert_eq!(
            super::response_failed_error(&serde_json::json!({"response":{}})).to_string(),
            "provider: Unknown error (no error details in response)"
        );
        assert_eq!(
            super::response_error_event(&serde_json::json!({
                "code":"rate_limit_exceeded","message":"slow down"
            }))
            .to_string(),
            "provider: Error Code rate_limit_exceeded: slow down"
        );
        assert_eq!(
            super::response_error_event(&serde_json::json!({})).to_string(),
            "provider: Unknown error"
        );
    }

    #[test]
    fn responses_final_message_recovers_text_and_refusals_missing_from_deltas() {
        let output = vec![serde_json::json!({
            "type":"message",
            "content":[
                {"type":"output_text","text":"partial "},
                {"type":"refusal","refusal":"answer"}
            ]
        })];
        assert_eq!(response_message_text(&output), "partial answer");

        let mut streamed = String::new();
        assert_eq!(
            merge_final_response_text(&mut streamed, &output).as_deref(),
            Some("partial answer")
        );
        assert_eq!(streamed, "partial answer");
        assert_eq!(merge_final_response_text(&mut streamed, &output), None);
    }

    #[test]
    fn responses_final_message_appends_only_the_unstreamed_suffix() {
        let output = vec![serde_json::json!({
            "type":"message",
            "content":[{"type":"refusal","refusal":"Refused."}]
        })];
        let mut streamed = "Ref".to_owned();
        assert_eq!(
            merge_final_response_text(&mut streamed, &output).as_deref(),
            Some("used.")
        );
        assert_eq!(streamed, "Refused.");
    }

    #[test]
    fn latest_session_for_a_working_directory_uses_mtime_and_validates_session_cwd() {
        let root = std::env::temp_dir().join(format!("mu-session-list-{}", uuid::Uuid::new_v4()));
        let session_dir = root.join("sessions/--tmp-project--");
        std::fs::create_dir_all(&session_dir).unwrap();
        let header = |cwd: &str| {
            format!(
                "{}\n",
                serde_json::json!({
                    "type":"session",
                    "id":"session-id",
                    "timestamp":"2025-01-01T00:00:00.000Z",
                    "cwd":cwd
                })
            )
        };
        let old_name = session_dir.join("2024-01-01_old.jsonl");
        let lexically_newer_name = session_dir.join("2025-01-01_new.jsonl");
        let other_cwd_name = session_dir.join("2026-01-01-other.jsonl");
        std::fs::write(&old_name, header("/tmp/project")).unwrap();
        std::fs::write(&lexically_newer_name, header("/tmp/project")).unwrap();
        std::fs::write(&other_cwd_name, header("/tmp/other-project")).unwrap();
        std::fs::write(session_dir.join("invalid.jsonl"), "not json").unwrap();
        std::fs::create_dir(session_dir.join("2027-01-01-directory.jsonl")).unwrap();
        let set_mtime =
            |path: &Path, seconds| {
                std::fs::File::open(path)
                    .unwrap()
                    .set_times(std::fs::FileTimes::new().set_modified(
                        std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds),
                    ))
                    .unwrap();
            };
        set_mtime(&old_name, 300);
        set_mtime(&lexically_newer_name, 200);
        set_mtime(&other_cwd_name, 400);

        assert_eq!(
            latest_session_path(&root, Path::new("/tmp/project"), None),
            Some(old_name)
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_session_id_lookup_honors_custom_directory_and_project_cwd() {
        let root = std::env::temp_dir().join(format!("mu-session-id-{}", uuid::Uuid::new_v4()));
        let session_dir = root.join("custom-sessions");
        std::fs::create_dir_all(&session_dir).unwrap();
        for (id, cwd, file_name) in [
            ("wanted-id", "/tmp/project", "project.jsonl"),
            ("wanted-id-prefix", "/tmp/project", "prefix.jsonl"),
            ("foreign-id", "/tmp/other-project", "foreign.jsonl"),
        ] {
            SessionDocument::new_with_id(cwd, id)
                .write_atomic(&session_dir.join(file_name))
                .unwrap();
        }

        assert_eq!(
            find_session_id_path(
                &root,
                Path::new("/tmp/project"),
                Some(&session_dir),
                "wanted-id",
                false
            ),
            Some(session_dir.join("project.jsonl"))
        );
        assert!(
            find_session_id_path(
                &root,
                Path::new("/tmp/project"),
                Some(&session_dir),
                "wanted",
                false
            )
            .is_none()
        );
        assert!(validate_session_id("Session_1.test-2").is_ok());
        assert!(validate_session_id("-invalid").is_err());
        assert!(validate_session_id("invalid id").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn session_id_options_open_prefixes_or_create_the_requested_id_in_custom_directory() {
        let root =
            std::env::temp_dir().join(format!("mu-session-options-{}", uuid::Uuid::new_v4()));
        let cwd = root.join("project");
        let session_dir = root.join("session-files");
        std::fs::create_dir_all(&cwd).unwrap();
        let config = Config {
            agent_dir: root.join("agent"),
            cwd: cwd.clone(),
            provider: "test-provider".into(),
            model: "test-model".into(),
            model_choices: vec!["test-model".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        };

        let (new_path, new_session, _) = resolve_session_options(
            &config,
            SessionOptions {
                directory: Some(session_dir.clone()),
                id: Some("requested-id".into()),
                name: Some("  Feature\nname  ".into()),
                ..SessionOptions::default()
            },
        )
        .unwrap();
        assert_eq!(new_path.parent(), Some(session_dir.as_path()));
        assert_eq!(new_session.header().unwrap().id.as_str(), "requested-id");
        assert_eq!(new_session.session_name(), Some("Feature name"));
        new_session.write_atomic(&new_path).unwrap();

        let (opened_path, opened_session, _) = resolve_session_options(
            &config,
            SessionOptions {
                directory: Some(session_dir.clone()),
                id_prefix: Some("requested".into()),
                ..SessionOptions::default()
            },
        )
        .unwrap();
        assert_eq!(opened_path, new_path);
        assert_eq!(opened_session.header().unwrap().id.as_str(), "requested-id");
        let (continued_path, _, _) = resolve_session_options(
            &config,
            SessionOptions {
                directory: Some(session_dir.clone()),
                continue_latest: true,
                ..SessionOptions::default()
            },
        )
        .unwrap();
        assert_eq!(continued_path, new_path);
        assert!(
            resolve_session_options(
                &config,
                SessionOptions {
                    directory: Some(session_dir),
                    id_prefix: Some("missing".into()),
                    ..SessionOptions::default()
                },
            )
            .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn forking_a_session_creates_a_new_header_and_preserves_source_history() {
        let root = std::env::temp_dir().join(format!("mu-session-fork-{}", uuid::Uuid::new_v4()));
        let source_path = root.join("source.jsonl");
        let target_cwd = root.join("target");
        let session_dir = root.join("forks");
        std::fs::create_dir_all(&target_cwd).unwrap();
        let mut source = SessionDocument::new_with_id("/source", "source-id");
        source.append_user_text("forked context");
        source.append_custom_entry("custom-state", serde_json::json!({"value":7}));
        source.write_atomic(&source_path).unwrap();
        let config = Config {
            agent_dir: root.join("agent"),
            cwd: target_cwd.clone(),
            provider: "test-provider".into(),
            model: "test-model".into(),
            model_choices: vec!["test-model".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        };

        let (fork_path, fork, history) = resolve_session_options(
            &config,
            SessionOptions {
                directory: Some(session_dir.clone()),
                id: Some("fork-id".into()),
                fork_from: Some(source_path.clone()),
                ..SessionOptions::default()
            },
        )
        .unwrap();
        let header = fork.header().unwrap();
        assert_eq!(header.id.as_str(), "fork-id");
        assert_eq!(header.cwd, target_cwd.to_string_lossy());
        assert_eq!(
            header.parent_session.as_deref(),
            Some(source_path.to_string_lossy().as_ref())
        );
        assert_eq!(history[0]["content"][0]["text"], "forked context");
        assert!(fork_path.exists());
        assert_eq!(source.header().unwrap().id.as_str(), "source-id");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loading_a_session_file_restores_its_document_and_model_history() {
        let root = std::env::temp_dir().join(format!("mu-session-load-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("session.jsonl");
        let mut session = SessionDocument::new("/tmp/project");
        session.append_user_text("remember this");
        session.append_assistant_turn(
            "openai-responses",
            "test-provider",
            "test-model",
            "",
            "I will.",
            MessageUsage::default(),
        );
        session.append_custom_entry(
            "mcp-tool-loadout",
            serde_json::json!({"active":["mcp__test__echo"]}),
        );
        session.append_custom_entry("codemode-store", serde_json::json!({"set":{"answer":42}}));
        session.write_atomic(&path).unwrap();

        let (loaded, history) = load_session(&path).unwrap();
        assert_eq!(loaded.header().unwrap().cwd, "/tmp/project");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0]["content"][0]["text"], "remember this");
        assert_eq!(history[1]["content"][0]["text"], "I will.");
        assert!(session_active_mcp_tools(&loaded).contains("mcp__test__echo"));
        assert_eq!(session_codemode_store(&loaded)["answer"], 42);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resumed_history_reconstructs_user_assistant_tool_call_and_tool_result() {
        let mut session = SessionDocument::new("/tmp/project");
        session.append_user_text("hello");
        session.append_assistant_tool_calls(
            "openai-responses",
            "test-provider",
            "test-model",
            AssistantToolResponse {
                thinking: String::new(),
                thinking_signature: None,
                text: String::new(),
                calls: vec![ToolCallRecord {
                    id: "call-1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path":"file.txt"}),
                }],
                usage: MessageUsage::default(),
            },
        );
        session.append_tool_result("call-1", "read", "file contents".into(), false);
        session.append_assistant_turn(
            "openai-responses",
            "test-provider",
            "test-model",
            "",
            "done",
            MessageUsage::default(),
        );

        let history = session_history(&session);
        assert_eq!(history.len(), 4);
        assert_eq!(history[0]["role"], "user");
        assert_eq!(history[0]["content"][0]["text"], "hello");
        assert_eq!(history[1]["type"], "function_call");
        assert_eq!(history[1]["call_id"], "call-1");
        assert_eq!(history[1]["arguments"], r#"{"path":"file.txt"}"#);
        assert_eq!(history[2]["type"], "function_call_output");
        assert_eq!(history[2]["output"], "file contents");
        assert_eq!(history[3]["content"][0]["text"], "done");
    }

    #[tokio::test]
    async fn ephemeral_sessions_update_memory_without_creating_a_session_file() {
        let root =
            std::env::temp_dir().join(format!("mu-ephemeral-session-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let config = super::Config {
            agent_dir: root.join("agent"),
            cwd: root.clone(),
            provider: "mu-demo".into(),
            model: "mu-demo".into(),
            model_choices: vec!["mu-demo".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        };
        let mut agent = super::MuAgent::demo_from_config(config).unwrap();
        agent.persist_session = false;
        let path = agent.session_path().to_owned();

        agent.submit("hello".into(), &mut |_| {}).await.unwrap();
        assert!(!path.exists());
        assert_eq!(agent.session.active_branch_entries().len(), 2);
        agent.start_new_session().unwrap();
        assert!(!agent.session_path().exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn default_tools_merge_and_resolve_like_pi_settings() {
        let mut entries = None;
        assert_eq!(
            super::resolve_default_tools(entries.clone()),
            ["read", "bash", "edit", "write"]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        super::merge_default_tools(
            &mut entries,
            &serde_json::json!({"defaultTools":["read", "bash", "+grep"]}),
        )
        .unwrap();
        super::merge_default_tools(
            &mut entries,
            &serde_json::json!({"defaultTools":["-bash", "+find"]}),
        )
        .unwrap();
        assert_eq!(
            super::resolve_default_tools(entries),
            ["read", "grep", "find"]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );

        let mut empty = None;
        super::merge_default_tools(&mut empty, &serde_json::json!({"defaultTools":[]})).unwrap();
        assert!(super::resolve_default_tools(empty).is_empty());
        assert!(
            super::merge_default_tools(&mut None, &serde_json::json!({"defaultTools":["read", 4]}))
                .is_err()
        );

        let root = std::env::temp_dir().join(format!("mu-tool-loadout-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let mut agent = super::MuAgent::demo_from_config(super::Config {
            agent_dir: root.join("agent"),
            cwd: root.clone(),
            provider: "mu-demo".into(),
            model: "mu-demo".into(),
            model_choices: vec!["mu-demo".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: vec!["grep".to_owned()],
        })
        .unwrap();
        let shell_environment = agent.shell_environment();
        assert_eq!(
            shell_environment["PI_SESSION_ID"],
            agent.session.header().unwrap().id.as_str()
        );
        assert_eq!(shell_environment["PI_PROVIDER"], "mu-demo");
        assert_eq!(shell_environment["PI_MODEL"], "mu-demo");
        assert_eq!(
            shell_environment["PI_SESSION_FILE"],
            agent.session_path().to_string_lossy()
        );
        let exposed = agent
            .tool_definitions(&HashSet::new())
            .into_iter()
            .filter_map(|definition| {
                definition
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect::<HashSet<_>>();
        assert_eq!(exposed, ["grep".to_owned()].into_iter().collect());
        agent.enabled_tools = super::resolve_default_tools(None);
        let strict_definitions = agent.tool_definitions(&HashSet::new());
        for definition in &strict_definitions {
            if matches!(
                definition["name"].as_str(),
                Some("read" | "bash" | "edit" | "write")
            ) {
                assert_eq!(definition["strict"], true);
            }
        }
        let read = strict_definitions
            .iter()
            .find(|definition| definition["name"] == "read")
            .unwrap();
        let required = read["parameters"]["required"].as_array().unwrap();
        assert_eq!(required.len(), 3);
        assert!(
            read["parameters"]["properties"]["offset"]["anyOf"]
                .as_array()
                .unwrap()
                .iter()
                .any(|variant| variant["type"] == "null")
        );
        agent.no_tools = true;
        assert!(agent.tool_definitions(&HashSet::new()).is_empty());
        std::fs::remove_dir_all(root).unwrap();

        let defaults = super::resolve_default_tools(None);
        assert_eq!(
            super::apply_tool_selection(
                &defaults,
                &super::ToolSelection::parse(vec!["-bash".into(), "+grep".into()]).unwrap(),
            ),
            ["read", "edit", "write", "grep"]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            super::apply_tool_selection(
                &defaults,
                &super::ToolSelection::parse(vec!["read".into(), "gr*".into()]).unwrap(),
            ),
            ["read", "grep"]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        assert!(super::ToolSelection::parse(vec!["read".into(), "+grep".into()]).is_err());
        assert_eq!(
            super::apply_tool_selection(
                &defaults,
                &super::ToolSelection::parse(vec!["coding".into()]).unwrap(),
            ),
            ["read", "bash", "edit", "write"]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            super::apply_tool_selection(
                &defaults,
                &super::ToolSelection::parse(vec!["read-only".into()]).unwrap(),
            ),
            ["read", "grep", "find", "ls"]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            super::apply_tool_selection(
                &defaults,
                &super::ToolSelection::parse(vec!["all".into()]).unwrap(),
            )
            .len(),
            8
        );
        let definitions = vec![
            serde_json::json!({"name":"read"}),
            serde_json::json!({"name":"mcp__docs__search"}),
            serde_json::json!({"name":"tool_search"}),
        ];
        let allowlist =
            super::ToolSelection::parse(vec!["read".into(), "mcp__docs__*".into()]).unwrap();
        let allowlisted = super::filter_tool_definitions(definitions.clone(), &allowlist);
        assert_eq!(allowlisted.len(), 2);
        let all = super::ToolSelection::parse(vec!["all".into()]).unwrap();
        assert_eq!(
            super::filter_tool_definitions(definitions.clone(), &all).len(),
            definitions.len()
        );
        let modifiers = super::ToolSelection::parse(vec!["-tool_search".into()]).unwrap();
        let modified = super::filter_tool_definitions(definitions, &modifiers);
        assert_eq!(modified.len(), 2);
    }

    #[test]
    fn builtin_prompt_contributions_follow_the_selected_tools() {
        let read_and_edit = vec![
            serde_json::json!({"name":"read"}),
            serde_json::json!({"name":"edit"}),
        ];
        let instructions = super::instructions_for_tools("base instructions", &read_and_edit);
        assert!(instructions.contains("- read: Read file contents"));
        assert!(instructions.contains("- edit: Make precise file edits"));
        assert!(instructions.contains("Use read to examine files instead of cat or sed."));
        assert!(
            instructions
                .contains("Use edit for precise changes (edits[].oldText must match exactly)")
        );
        assert!(instructions.contains("Show file paths clearly when working with files"));

        let shell_only = super::instructions_for_tools(
            "base instructions",
            &[serde_json::json!({"name":"bash"})],
        );
        assert!(shell_only.contains("Use bash for file operations like ls, rg, find"));
        assert!(shell_only.contains("PI_* environment variables"));
        let shell_with_search = super::instructions_for_tools(
            "base instructions",
            &[
                serde_json::json!({"name":"bash"}),
                serde_json::json!({"name":"grep"}),
            ],
        );
        assert!(!shell_with_search.contains("Use bash for file operations like ls, rg, find"));
    }

    #[test]
    fn strict_sampling_builtins_use_required_nullable_optional_parameters() {
        let definitions = crate::tools::definitions()
            .into_iter()
            .map(super::mark_strict_builtin_tool)
            .collect::<Vec<_>>();
        let read = definitions
            .iter()
            .find(|definition| definition["name"] == "read")
            .unwrap();
        assert_eq!(read["strict"], true);
        assert_eq!(read["parameters"]["required"].as_array().unwrap().len(), 3);
        assert!(
            read["parameters"]["properties"]["offset"]["anyOf"]
                .as_array()
                .unwrap()
                .iter()
                .any(|schema| schema["type"] == "null")
        );

        let edit = definitions
            .iter()
            .find(|definition| definition["name"] == "edit")
            .unwrap();
        let item_schema = &edit["parameters"]["properties"]["edits"]["items"];
        assert_eq!(item_schema["required"].as_array().unwrap().len(), 2);
        assert_eq!(item_schema["additionalProperties"], false);
        let grep = definitions
            .iter()
            .find(|definition| definition["name"] == "grep")
            .unwrap();
        assert!(grep.get("strict").is_none());
        assert!(super::schema_allows_null(
            &serde_json::json!({"type":["string","null"]})
        ));
        assert!(super::schema_allows_null(
            &serde_json::json!({"const":null})
        ));
        assert!(super::schema_allows_null(
            &serde_json::json!({"enum":["value",null]})
        ));
        assert!(super::schema_allows_null(
            &serde_json::json!({"anyOf":[{"type":"null"}]})
        ));
        assert_eq!(
            super::ToolSelection::parse(vec!["+gr*".into()])
                .unwrap_err()
                .to_string(),
            "configuration: +name and -name entries take exact tool names, not patterns"
        );
    }

    #[tokio::test]
    async fn scripted_provider_runs_a_complete_streaming_tool_turn_offline() {
        let root = std::env::temp_dir().join(format!("mu-scripted-agent-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create temporary project");
        let config = super::Config {
            agent_dir: root.join("agent"),
            cwd: root.clone(),
            provider: "test-provider".into(),
            model: "test-model".into(),
            model_choices: vec!["test-model".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        };
        let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
        let replies =
            std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from([
                Ok(Reply {
                    text: String::new(),
                    thinking: String::new(),
                    thinking_signature: None,
                    output_items: vec![serde_json::json!({
                        "type":"function_call",
                        "call_id":"call-write",
                        "name":"write",
                        "arguments":"{\"path\":\"result.txt\",\"content\":\"saved\"}"
                    })],
                    tool_calls: vec![FunctionCall {
                        call_id: "call-write".into(),
                        name: "write".into(),
                        arguments: serde_json::json!({"path":"result.txt","content":"saved"}),
                    }],
                    input: 0,
                    output: 0,
                    cache_read: 0,
                    cache_write: 0,
                    reasoning: None,
                    total_tokens: 0,
                    incomplete_reason: None,
                }),
                Ok(Reply {
                    text: "saved file".into(),
                    thinking: "verify result".into(),
                    thinking_signature: None,
                    output_items: Vec::new(),
                    tool_calls: Vec::new(),
                    input: 0,
                    output: 0,
                    cache_read: 0,
                    cache_write: 0,
                    reasoning: None,
                    total_tokens: 0,
                    incomplete_reason: None,
                }),
            ])));
        agent.provider.transport = super::ProviderTransport::Scripted(replies);
        let mut events = Vec::new();

        agent
            .submit("create the result file".into(), &mut |event| {
                events.push(event)
            })
            .await
            .expect("complete scripted agent turn");

        assert_eq!(
            std::fs::read_to_string(root.join("result.txt")).expect("tool output file"),
            "saved"
        );
        assert!(events.contains(&super::AgentEvent::ToolStarted {
            call_id: "call-write".into(),
            name: "write".into(),
            arguments: serde_json::json!({"path":"result.txt","content":"saved"}),
        }));
        assert!(events.contains(&super::AgentEvent::ToolFinished {
            call_id: "call-write".into(),
            output: "Successfully wrote to result.txt".into(),
            is_error: false,
        }));
        assert!(events.contains(&super::AgentEvent::ThinkingDelta("v".into())));
        assert!(events.contains(&super::AgentEvent::TextDelta("s".into())));
        assert_eq!(agent.history[1]["type"], "function_call");
        assert_eq!(agent.history[2]["type"], "function_call_output");
        assert_eq!(
            agent.history[2]["output"],
            "Successfully wrote to result.txt"
        );
        let session = std::fs::read_to_string(agent.session_path()).expect("read session");
        let parsed = SessionDocument::parse_jsonl(&session).expect("parse session");
        assert!(parsed.entries().any(|entry| {
            entry
                .extra_fields()
                .get("message")
                .and_then(Value::as_object)
                .and_then(|message| message.get("role"))
                .and_then(Value::as_str)
                == Some("toolResult")
        }));
        std::fs::remove_dir_all(root).expect("remove temporary project");
    }

    #[tokio::test]
    async fn output_token_limited_tool_arguments_are_reported_but_never_executed() {
        let root = std::env::temp_dir().join(format!("mu-truncated-tool-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let config = super::Config {
            agent_dir: root.join("agent"),
            cwd: root.clone(),
            provider: "test-provider".into(),
            model: "test-model".into(),
            model_choices: vec!["test-model".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        };
        let mut agent = super::MuAgent::demo_from_config(config).unwrap();
        let arguments = serde_json::json!({"path":"truncated.txt","content":"must not be written"});
        let replies = std::collections::VecDeque::from([
            Ok(Reply {
                text: String::new(),
                thinking: String::new(),
                thinking_signature: None,
                output_items: vec![serde_json::json!({
                    "type":"function_call",
                    "call_id":"truncated-call",
                    "name":"write",
                    "arguments":arguments.to_string()
                })],
                tool_calls: vec![FunctionCall {
                    call_id: "truncated-call".into(),
                    name: "write".into(),
                    arguments,
                }],
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                reasoning: None,
                total_tokens: 0,
                incomplete_reason: Some("max_output_tokens".into()),
            }),
            Ok(Reply {
                text: "I will retry with a complete tool call".into(),
                thinking: String::new(),
                thinking_signature: None,
                output_items: Vec::new(),
                tool_calls: Vec::new(),
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                reasoning: None,
                total_tokens: 0,
                incomplete_reason: None,
            }),
        ]);
        agent.provider.transport =
            super::ProviderTransport::Scripted(std::sync::Arc::new(std::sync::Mutex::new(replies)));
        let mut events = Vec::new();

        agent
            .submit("write a file".into(), &mut |event| events.push(event))
            .await
            .unwrap();

        assert!(!root.join("truncated.txt").exists());
        assert!(events.contains(&super::AgentEvent::ToolFinished {
            call_id: "truncated-call".into(),
            output: "Tool call \"write\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.".into(),
            is_error: true,
        }));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn scripted_provider_can_complete_more_than_eight_tool_rounds() {
        let root =
            std::env::temp_dir().join(format!("mu-many-tool-rounds-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create temporary project");
        let config = super::Config {
            agent_dir: root.join("agent"),
            cwd: root.clone(),
            provider: "test-provider".into(),
            model: "test-model".into(),
            model_choices: vec!["test-model".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        };
        let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
        let mut scripted = std::collections::VecDeque::new();
        for round in 0..9 {
            let call_id = format!("call-{round}");
            let arguments = serde_json::json!({"command":"true"});
            scripted.push_back(Ok(Reply {
                text: String::new(),
                thinking: String::new(),
                thinking_signature: None,
                output_items: vec![serde_json::json!({
                    "type":"function_call",
                    "call_id":call_id,
                    "name":"bash",
                    "arguments":arguments.to_string()
                })],
                tool_calls: vec![FunctionCall {
                    call_id,
                    name: "bash".into(),
                    arguments,
                }],
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                reasoning: None,
                total_tokens: 0,
                incomplete_reason: None,
            }));
        }
        scripted.push_back(Ok(Reply {
            text: "all nine rounds completed".into(),
            thinking: String::new(),
            thinking_signature: None,
            output_items: Vec::new(),
            tool_calls: Vec::new(),
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            reasoning: None,
            total_tokens: 0,
            incomplete_reason: None,
        }));
        agent.provider.transport = super::ProviderTransport::Scripted(std::sync::Arc::new(
            std::sync::Mutex::new(scripted),
        ));

        agent
            .submit("run nine independent commands".into(), &mut |_| {})
            .await
            .expect("agent loop should have no arbitrary tool-round limit");

        assert_eq!(agent.history.len(), 20);
        assert_eq!(
            agent.history.last().unwrap()["content"][0]["text"],
            "all nine rounds completed"
        );
        std::fs::remove_dir_all(root).expect("remove temporary project");
    }

    #[tokio::test]
    async fn scripted_provider_retries_transient_failures_with_bounded_attempt_events() {
        let root = std::env::temp_dir().join(format!("mu-provider-retry-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create temporary project");
        let config = super::Config {
            agent_dir: root.join("agent"),
            cwd: root.clone(),
            provider: "test-provider".into(),
            model: "test-model".into(),
            model_choices: vec!["test-model".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy {
                enabled: true,
                max_retries: 2,
                base_delay_ms: 0,
                max_delay_ms: 0,
            },
            enabled_tools: super::resolve_default_tools(None),
        };
        let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
        agent.provider.transport = super::ProviderTransport::Scripted(std::sync::Arc::new(
            std::sync::Mutex::new(std::collections::VecDeque::from([
                Err("model request returned HTTP 503".into()),
                Err("upstream connection timed out".into()),
                Ok(Reply {
                    text: "recovered".into(),
                    thinking: String::new(),
                    thinking_signature: None,
                    output_items: Vec::new(),
                    tool_calls: Vec::new(),
                    input: 0,
                    output: 0,
                    cache_read: 0,
                    cache_write: 0,
                    reasoning: None,
                    total_tokens: 0,
                    incomplete_reason: None,
                }),
            ])),
        ));
        let mut events = Vec::new();

        agent
            .submit("retry this request".into(), &mut |event| events.push(event))
            .await
            .expect("retry eventually succeeds");

        let scheduled = events
            .iter()
            .filter_map(|event| match event {
                super::AgentEvent::RetryScheduled {
                    attempt,
                    max_attempts,
                    delay_ms,
                    error,
                } => Some((*attempt, *max_attempts, *delay_ms, error.as_str())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            scheduled,
            [
                (1, 2, 0, "provider: model request returned HTTP 503"),
                (2, 2, 0, "provider: upstream connection timed out")
            ]
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, super::AgentEvent::RetryAttemptStarted { .. }))
                .count(),
            2
        );
        assert!(events.contains(&super::AgentEvent::TextDelta("r".into())));
        assert_eq!(agent.history[0]["role"], "user");
        assert_eq!(agent.history[1]["role"], "assistant");
        std::fs::remove_dir_all(root).expect("remove temporary project");
    }

    #[tokio::test]
    async fn cancelling_during_retry_backoff_does_not_start_another_attempt() {
        let root = std::env::temp_dir().join(format!("mu-retry-cancel-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create temporary project");
        let config = super::Config {
            agent_dir: root.join("agent"),
            cwd: root.clone(),
            provider: "test-provider".into(),
            model: "test-model".into(),
            model_choices: vec!["test-model".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy {
                enabled: true,
                max_retries: 1,
                base_delay_ms: 60_000,
                max_delay_ms: 60_000,
            },
            enabled_tools: super::resolve_default_tools(None),
        };
        let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
        agent.provider.transport = super::ProviderTransport::Scripted(std::sync::Arc::new(
            std::sync::Mutex::new(std::collections::VecDeque::from([
                Err("HTTP 503 temporary failure".into()),
                Ok(Reply {
                    text: "must not run".into(),
                    thinking: String::new(),
                    thinking_signature: None,
                    output_items: Vec::new(),
                    tool_calls: Vec::new(),
                    input: 0,
                    output: 0,
                    cache_read: 0,
                    cache_write: 0,
                    reasoning: None,
                    total_tokens: 0,
                    incomplete_reason: None,
                }),
            ])),
        ));
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            agent
                .submit("cancel during backoff".into(), &mut |event| {
                    let _ = event_tx.send(event);
                })
                .await
        });
        let mut scheduled = false;
        while let Some(event) = event_rx.recv().await {
            if matches!(event, super::AgentEvent::RetryScheduled { .. }) {
                scheduled = true;
                break;
            }
        }
        assert!(scheduled);

        task.abort();

        assert!(task.await.unwrap_err().is_cancelled());
        std::fs::remove_dir_all(root).expect("remove temporary project");
    }

    #[test]
    fn retry_policy_excludes_quota_errors_and_caps_exponential_backoff() {
        let mut configured = super::RetryPolicy::default();
        super::update_retry_policy(
            &mut configured,
            &serde_json::json!({"retry":{"enabled":false,"maxRetries":1,"baseDelayMs":50,"maxAgentDelayMs":200}}),
        );
        assert_eq!(
            configured,
            super::RetryPolicy {
                enabled: false,
                max_retries: 1,
                base_delay_ms: 50,
                max_delay_ms: 200,
            }
        );
        super::update_retry_policy(
            &mut configured,
            &serde_json::json!({"retry":{"maxDelayMs":300}}),
        );
        assert_eq!(configured.max_delay_ms, 300);
        assert!(super::retryable_provider_error(
            &super::AgentError::Provider("HTTP 503 service unavailable".into())
        ));
        assert!(!super::retryable_provider_error(
            &super::AgentError::Provider("insufficient_quota: billing limit".into())
        ));
        let policy = super::RetryPolicy {
            enabled: true,
            max_retries: 4,
            base_delay_ms: 2_000,
            max_delay_ms: 5_000,
        };
        assert_eq!(super::retry_delay_ms(policy, 1), 2_000);
        assert_eq!(super::retry_delay_ms(policy, 2), 4_000);
        assert_eq!(super::retry_delay_ms(policy, 3), 5_000);
    }

    #[tokio::test]
    async fn scripted_provider_failure_persists_the_user_turn_without_an_assistant_turn() {
        let root = std::env::temp_dir().join(format!("mu-provider-error-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("create temporary project");
        let config = super::Config {
            agent_dir: root.join("agent"),
            cwd: root.clone(),
            provider: "test-provider".into(),
            model: "test-model".into(),
            model_choices: vec!["test-model".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        };
        let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
        agent.provider.transport = super::ProviderTransport::Scripted(std::sync::Arc::new(
            std::sync::Mutex::new(std::collections::VecDeque::from([Err(
                "invalid request: malformed payload".into(),
            )])),
        ));
        let mut events = Vec::new();

        let error = agent
            .submit("request that fails".into(), &mut |event| events.push(event))
            .await
            .expect_err("scripted provider failure should propagate");

        assert_eq!(
            error.to_string(),
            "provider: invalid request: malformed payload"
        );
        assert!(events.is_empty());
        assert_eq!(agent.history.len(), 1);
        assert_eq!(agent.history[0]["role"], "user");
        let session = std::fs::read_to_string(agent.session_path()).expect("read session");
        let parsed = SessionDocument::parse_jsonl(&session).expect("parse session");
        let messages = parsed
            .active_branch_entries()
            .into_iter()
            .filter(|entry| entry.entry_type.as_str() == "message")
            .filter_map(|entry| entry.extra_fields().get("message"))
            .collect::<Vec<_>>();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        std::fs::remove_dir_all(root).expect("remove temporary project");
    }

    #[tokio::test]
    async fn starting_a_new_session_clears_context_and_preserves_the_old_file() {
        let root = std::env::temp_dir().join(format!("mu-new-session-{}", uuid::Uuid::new_v4()));
        let agent_dir = root.join("agent");
        std::fs::create_dir_all(&root).expect("create temporary project");
        let config = super::Config {
            agent_dir,
            cwd: root.clone(),
            provider: "mu-demo".into(),
            model: "mu-demo".into(),
            model_choices: vec!["mu-demo".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        };
        let mut agent = super::MuAgent::demo_from_config(config).expect("demo agent");
        agent
            .submit("old conversation".into(), &mut |_| {})
            .await
            .expect("write old session");
        let old_path = agent.session_path().to_owned();
        agent.active_mcp_tools.insert("mcp__test__echo".into());
        agent
            .prompt_queue
            .enqueue("pending prompt".into(), PromptBehavior::FollowUp);

        agent.start_new_session().expect("start new session");

        assert_ne!(agent.session_path(), old_path);
        assert!(agent.history.is_empty());
        assert!(agent.active_mcp_tools.is_empty());
        assert_eq!(agent.prompt_queue.take_next(), None);
        let old_session = std::fs::read_to_string(&old_path).expect("preserve old session file");
        assert!(
            super::SessionDocument::parse_jsonl(&old_session)
                .expect("parse old session")
                .entries()
                .count()
                > 0
        );
        let new_session = std::fs::read_to_string(agent.session_path()).expect("new session file");
        assert_eq!(
            super::SessionDocument::parse_jsonl(&new_session)
                .expect("parse new session")
                .entries()
                .count(),
            0
        );
        std::fs::remove_dir_all(root).expect("remove temporary project");
    }

    #[test]
    fn prompt_queue_prioritizes_fifo_steering_before_follow_ups() {
        let queue = AgentPromptQueue::default();
        queue.enqueue("later".into(), PromptBehavior::FollowUp);
        queue.enqueue("steer one".into(), PromptBehavior::Steer);
        queue.enqueue("steer two".into(), PromptBehavior::Steer);

        assert_eq!(queue.take_steering().as_deref(), Some("steer one"));
        assert_eq!(queue.take_steering().as_deref(), Some("steer two"));
        assert_eq!(queue.take_steering(), None);
        assert_eq!(
            queue.take_next(),
            Some(("later".into(), PromptBehavior::FollowUp))
        );
        assert_eq!(queue.take_next(), None);
    }

    #[tokio::test]
    async fn independent_tool_calls_run_concurrently_and_keep_call_order() {
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(2));
        let futures = (0..2).map(|value| {
            let barrier = std::sync::Arc::clone(&barrier);
            async move {
                barrier.wait().await;
                value
            }
        });
        assert_eq!(execute_in_parallel(futures).await, [0, 1]);
    }

    #[tokio::test]
    async fn submit_consumes_steering_then_follow_up_prompts_as_new_user_turns() {
        let mut agent = super::MuAgent::demo().expect("demo agent");
        let queue = agent.prompt_queue();
        let mut inserted_steering = false;
        let mut inserted_follow_up = false;
        let mut queued_started = Vec::new();

        agent
            .submit("initial request".into(), &mut |event| match event {
                super::AgentEvent::TextDelta(_) if !inserted_steering => {
                    inserted_steering = true;
                    queue.enqueue("steer the current work".into(), PromptBehavior::Steer);
                }
                super::AgentEvent::QueuedPromptStarted {
                    behavior: PromptBehavior::Steer,
                    ..
                } if !inserted_follow_up => {
                    inserted_follow_up = true;
                    queued_started.push(PromptBehavior::Steer);
                    queue.enqueue("after that, summarize".into(), PromptBehavior::FollowUp);
                }
                super::AgentEvent::QueuedPromptStarted { behavior, .. } => {
                    queued_started.push(behavior);
                }
                _ => {}
            })
            .await
            .expect("complete queued turns");

        assert_eq!(
            queued_started,
            [PromptBehavior::Steer, PromptBehavior::FollowUp]
        );
        let user_text = agent
            .history
            .iter()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("user"))
            .filter_map(|message| message.pointer("/content/0/text").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert_eq!(
            user_text,
            [
                "initial request",
                "steer the current work",
                "after that, summarize",
            ]
        );
    }

    #[test]
    fn restores_codemode_store_and_tool_loadout_from_the_active_session_branch() {
        let input = [
            r#"{"type":"session","version":3,"id":"session","timestamp":"t","cwd":"/tmp"}"#,
            r#"{"type":"custom","id":"root","parentId":null,"timestamp":"t","customType":"codemode-store","data":{"set":{"count":1,"old":true},"delete":[]}}"#,
            r#"{"type":"custom","id":"sibling","parentId":"root","timestamp":"t","customType":"codemode-store","data":{"set":{"sibling":true},"delete":[]}}"#,
            r#"{"type":"custom","id":"leaf","parentId":"root","timestamp":"t","customType":"codemode-store","data":{"set":{"count":2},"delete":["old"]}}"#,
            r#"{"type":"custom","id":"loadout","parentId":"leaf","timestamp":"t","customType":"mcp-tool-loadout","data":{"active":["mcp__test__echo"]}}"#,
        ]
        .join("\n");
        let session = SessionDocument::parse_jsonl(&input).expect("parse session branch");
        assert_eq!(
            session_codemode_store(&session),
            serde_json::json!({"count":2})
        );
        assert_eq!(
            session_active_mcp_tools(&session),
            ["mcp__test__echo".to_owned()].into()
        );
    }

    #[test]
    fn reads_final_openai_responses_reasoning_when_no_reasoning_delta_arrived() {
        let item = serde_json::json!({
            "type":"reasoning",
            "summary":[{"type":"summary_text","text":"Plan the change."},{"type":"summary_text","text":"Then implement it."}],
            "content":[{"type":"reasoning_text","text":"private detail"}]
        });
        assert_eq!(
            super::response_reasoning(&item),
            "Plan the change.\n\nThen implement it."
        );
    }

    #[test]
    fn allows_access_gateway_and_rejects_untrusted_hosts() {
        assert!(trusted_url("https://gateway.opencode.cloudflare.dev/openai").is_ok());
        assert!(trusted_url("https://attacker.example/openai").is_err());
    }

    #[test]
    fn reads_custom_model_choices_and_keeps_the_default_model_available() {
        let dir = std::env::temp_dir().join(format!("mu-model-config-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create model config directory");
        std::fs::write(
            dir.join("models.json"),
            r#"{"providers":{"oracle":{"models":[{"id":"alpha"},{"id":"beta"}]}}}"#,
        )
        .expect("write model config");

        let models = configured_models(&dir, "oracle", "beta").expect("read model choices");
        assert_eq!(models, ["alpha", "beta"]);
        let includes_unlisted_default =
            configured_models(&dir, "oracle", "gamma").expect("include configured default");
        assert_eq!(includes_unlisted_default, ["gamma", "alpha", "beta"]);
        std::fs::remove_dir_all(dir).expect("remove model config directory");

        let absent = configured_models(Path::new("/no/such/mu/config"), "oracle", "default")
            .expect("fall back to default model");
        assert_eq!(absent, ["default"]);
    }
}
