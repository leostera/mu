use std::{
    collections::{HashMap, HashSet, VecDeque},
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex as StdMutex},
};

use futures_util::StreamExt;
use mu_session::{MessageCost, MessageUsage, SessionDocument};
use reqwest::{Client, Url, header};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::OnceCell;

mod anthropic;

/// `(mime type, base64 data)` of an `input_image` part whose URL is a `data:` URL.
fn data_url_parts(part: &Value) -> Option<(String, String)> {
    let url = part.get("image_url")?.as_str()?;
    let (header, data) = url.strip_prefix("data:")?.split_once(";base64,")?;
    Some((header.to_owned(), data.to_owned()))
}

/// A tool output as plain text: strings pass through; image-bearing outputs keep their text parts
/// and note each omitted image (providers without tool-result image support).
fn output_text(item: &Value) -> String {
    match &item["output"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part["type"].as_str() {
                Some("input_image") => {
                    "[image omitted: this provider does not accept images in tool results]"
                        .to_owned()
                }
                _ => part["text"].as_str().unwrap_or_default().to_owned(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}
mod bedrock;
mod completions;
mod google;
mod runtime;
mod sigv4;

use crate::agent_loop::AgentMessage;
use tokio_util::sync::CancellationToken;

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
    /// A compaction entry; the text is its summary and the value the tokens before compaction.
    CompactionSummary(u64),
    /// A branch summary entry left when navigating the tree.
    BranchSummary,
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
    /// Use the deterministic offline backend instead of the configured provider.
    pub demo: bool,
    /// `--system-prompt`: replaces the default prompt.
    pub system_prompt: Option<String>,
    /// `--append-system-prompt`: each entry is literal text or a file path whose contents are used.
    pub append_system_prompt: Vec<String>,
    /// `--no-context-files`: skip `AGENTS.md`/`CLAUDE.md` discovery.
    pub no_context_files: bool,
    /// `--exclude-tools`: names or `*` patterns to disable, applied after selection.
    pub exclude_tools: Vec<String>,
    /// `--thinking`: requested reasoning level.
    pub thinking: ThinkingSetting,
    /// `--provider`: overrides `defaultProvider`.
    pub provider: Option<String>,
    /// `--model`: model id or `provider/id`, overrides `defaultModel`.
    pub model: Option<String>,
    /// `--no-mcp`: do not connect MCP servers.
    pub no_mcp: bool,
    /// `--prompt-template`: extra template files or directories.
    pub prompt_templates: Vec<PathBuf>,
    /// `--no-prompt-templates`: skip default template directories.
    pub no_prompt_templates: bool,
    /// `--api-key`: runtime API key for the selected provider (never persisted).
    pub api_key: Option<String>,
    /// `--approve` (`Some(true)`) / `--no-approve` (`Some(false)`): project trust for this run.
    pub project_trust: Option<bool>,
    /// `--theme`: theme name for this run (overrides the `theme` setting).
    pub theme: Option<String>,
    /// `--models`: comma-separated patterns scoping Ctrl+P cycling (overrides `enabledModels`).
    pub models: Vec<String>,
    /// `--verbose`: print the resolved startup configuration to stderr (print mode only).
    pub verbose: bool,
    /// Interactive runs continue in the current directory when the session's stored cwd is gone
    /// (Pi asks first; print and RPC runs refuse).
    pub continue_in_current_cwd: bool,
}

/// Pi's thinking levels (`off` … `max`); Mu's default is `medium`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ThinkingSetting {
    Off,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    Xhigh,
    Max,
}

impl std::str::FromStr for ThinkingSetting {
    type Err = String;

    fn from_str(level: &str) -> Result<Self, String> {
        Ok(match level {
            "off" => Self::Off,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::Xhigh,
            "max" => Self::Max,
            other => {
                return Err(format!(
                    "Invalid thinking level \"{other}\". Valid values: off, minimal, low, medium, high, xhigh, max"
                ));
            }
        })
    }
}

impl ThinkingSetting {
    /// The next level in Pi's cycle (`off → minimal → … → max → off`).
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Minimal,
            Self::Minimal => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High => Self::Xhigh,
            Self::Xhigh => Self::Max,
            Self::Max => Self::Off,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }

    /// OpenAI Responses `reasoning.effort`, or `None` when thinking is off.
    fn effort(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Minimal => Some("minimal"),
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
            Self::Xhigh | Self::Max => Some("xhigh"),
        }
    }
}

/// Resolved prompt customizations.
#[derive(Clone, Debug, Default)]
struct PromptOptions {
    system_prompt: Option<String>,
    appended: Vec<String>,
    context: String,
    thinking: ThinkingSetting,
    api_key: Option<String>,
    theme: Option<String>,
    models: Vec<String>,
}

impl PromptOptions {
    fn from_session_options(options: &SessionOptions) -> Self {
        Self {
            system_prompt: options.system_prompt.as_deref().map(read_prompt_input),
            appended: options
                .append_system_prompt
                .iter()
                .map(|text| read_prompt_input(text))
                .collect(),
            context: String::new(),
            thinking: options.thinking,
            api_key: options.api_key.clone().filter(|key| !key.is_empty()),
            theme: options.theme.clone(),
            models: options.models.clone(),
        }
    }

    /// Load `AGENTS.md`-style files unless disabled.
    fn with_context_files(mut self, cwd: &Path, agent_dir: &Path, disabled: bool) -> Self {
        if !disabled {
            let files = crate::context_files::load_project_context_files(cwd, agent_dir);
            if !files.is_empty() {
                self.context = crate::context_files::render_project_context(&files);
            }
        }
        self
    }

    fn appendix(&self) -> String {
        let mut sections = self.appended.clone();
        if !self.context.is_empty() {
            sections.push(self.context.clone());
        }
        sections.join("\n\n")
    }
}

/// Pi accepts either literal text or a path to a file whose contents are the prompt.
fn read_prompt_input(input: &str) -> String {
    let path = Path::new(input);
    if path.is_file() {
        if let Ok(contents) = std::fs::read_to_string(path) {
            return contents;
        }
    }
    input.to_owned()
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
#[derive(Clone)]
struct Provider {
    transport: ProviderTransport,
    model: String,
    instructions: String,
    /// A custom `--system-prompt` replaces the default prompt, including the tool section.
    describes_tools: bool,
    /// `--append-system-prompt` text placed after everything else.
    appendix: String,
    supports_reasoning: bool,
    /// The model accepts image input (`input` includes `image` in `models.json`; default yes).
    supports_images: bool,
    /// Requested reasoning effort (`--thinking`); `None` means off.
    reasoning_effort: Option<&'static str>,
    supports_strict_sampling: bool,
}

/// How Bedrock requests authenticate.
#[derive(Clone)]
enum BedrockAuth {
    /// A Bedrock API key sent as a bearer token.
    Bearer(String),
    /// AWS SigV4 signing with environment credentials.
    Sigv4 {
        credentials: sigv4::AwsCredentials,
        region: String,
    },
}

/// `bedrock-runtime.<region>.amazonaws.com` → `<region>`.
fn bedrock_region_from_host(host: &str) -> Option<String> {
    let rest = host.strip_prefix("bedrock-runtime.")?;
    rest.strip_suffix(".amazonaws.com").map(str::to_owned)
}

/// How a Gemini request authenticates: Generative AI takes an API key header, Vertex AI an OAuth
/// access token.
#[derive(Clone)]
enum GoogleAuth {
    ApiKey(String),
    Bearer(String),
}

#[derive(Clone)]
enum ProviderTransport {
    OpenCode {
        http: Client,
        endpoint: Url,
        token: String,
        gateway_headers: bool,
    },
    /// Bedrock Converse (`api: "bedrock-converse-stream"`), bearer-token auth only.
    Bedrock {
        http: Client,
        endpoint: Url,
        auth: BedrockAuth,
    },
    /// Anthropic Messages API (`api: "anthropic-messages"` in `models.json`).
    Anthropic {
        http: Client,
        endpoint: Url,
        key: String,
    },
    /// OpenAI Chat Completions (`api: "openai-completions"`).
    Completions {
        http: Client,
        endpoint: Url,
        key: String,
    },
    /// Google Generative AI (`api: "google-generative-ai"`); the endpoint includes the model.
    Google {
        http: Client,
        endpoint: Url,
        auth: GoogleAuth,
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

/// How many queued messages the loop injects at a queue drain point (Pi's `QueueMode`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum QueueMode {
    All,
    #[default]
    OneAtATime,
}

impl QueueMode {
    fn from_setting(value: &str) -> Option<Self> {
        match value {
            "all" => Some(Self::All),
            "one-at-a-time" => Some(Self::OneAtATime),
            _ => None,
        }
    }

    fn drain(self, queue: &mut VecDeque<String>) -> Vec<String> {
        match self {
            Self::All => queue.drain(..).collect(),
            Self::OneAtATime => queue.pop_front().into_iter().collect(),
        }
    }
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

    /// Drain queued follow-ups: every one for `All`, only the oldest for `OneAtATime`.
    fn take_follow_ups(&self, mode: QueueMode) -> Vec<String> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        mode.drain(&mut pending.follow_up)
    }

    /// Drain queued steering messages according to `mode`.
    fn take_steering_batch(&self, mode: QueueMode) -> Vec<String> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        mode.drain(&mut pending.steering)
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
        /// Structured tool details (truncation info, diffs) for renderers.
        details: Option<Value>,
        duration_ms: Option<u64>,
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
    /// The context passed the compaction threshold; a summary is being generated.
    AutoCompactionStarted,
    /// Automatic compaction finished: the summary now in the session and the tokens it replaced.
    AutoCompactionFinished {
        summary: String,
        tokens_before: u64,
    },
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

fn format_mcp_tool_summary(definition: &Value) -> Option<String> {
    let name = definition.get("name")?.as_str()?;
    let description = definition
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .lines()
        .next()
        .unwrap_or_default();
    Some(if description.is_empty() {
        name.to_owned()
    } else {
        format!("{name}: {description}")
    })
}

fn response_error_event(event: &Value) -> AgentError {
    // Pi interpolates absent JavaScript event fields as the literal `undefined`.
    let code = event
        .get("code")
        .and_then(Value::as_str)
        .unwrap_or("undefined");
    let message = event
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("undefined");
    AgentError::Provider(format!("Error Code {code}: {message}"))
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

fn configured_queue_mode(agent_dir: &Path, cwd: &Path, key: &str) -> QueueMode {
    configured_setting(agent_dir, cwd, key, |value| {
        value.as_str().and_then(QueueMode::from_setting)
    })
    .unwrap_or_default()
}

fn configured_setting<T>(
    agent_dir: &Path,
    cwd: &Path,
    key: &str,
    parse: impl Fn(&Value) -> Option<T>,
) -> Option<T> {
    let settings = crate::settings::Settings::load(agent_dir, cwd, true);
    settings.merged().get(key).and_then(parse)
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
    config_for(false)
}

/// Pi's `formatNoApiKeyFoundMessage` (`core/auth-guidance.ts`).
fn no_api_key_message(provider: &str) -> String {
    format!(
        "No API key found for {provider}.\n\nUse /login to log into a provider via OAuth or API key, or set apiKey for it in models.json (a literal, $ENV_VAR, or !command)."
    )
}

/// Pi's `formatNoModelSelectedMessage` (`core/auth-guidance.ts`), with the cause first.
fn no_model_selected_message(cause: &str) -> String {
    format!(
        "{cause}. No model selected.\n\nUse /login to log into a provider via OAuth or API key, or add one to models.json.\n\nThen use /model to select a model."
    )
}

/// Read Pi settings; with `demo`, a missing provider or model falls back to the demo backend.
fn config_for(demo: bool) -> Result<Config, AgentError> {
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
    let provider = match provider {
        Some(provider) => provider,
        None if demo => "mu-demo".to_owned(),
        None => {
            return Err(AgentError::Configuration(no_model_selected_message(
                "Pi settings have no defaultProvider",
            )));
        }
    };
    let model = match model {
        Some(model) => model,
        None if demo => "mu-demo".to_owned(),
        None => {
            return Err(AgentError::Configuration(no_model_selected_message(
                "Pi settings have no defaultModel",
            )));
        }
    };
    let mut model_choices = configured_models(&agent_dir, &provider, &model)?;
    if demo && provider == "mu-demo" && model_choices.len() == 1 {
        model_choices.push("mu-demo-alt".into());
    }
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

/// Apply `--provider` / `--model` (`provider/id` allowed) over the settings defaults.
/// Resolve `enabledModels` patterns to model ids, in pattern order without duplicates.
///
/// A pattern matches a model's id or `provider/id` case-insensitively and may use `*`/`?` globs;
/// an optional `:<thinking>` suffix is ignored. With no patterns, or none matching, every
/// choice is returned.
fn scope_models(patterns: &[String], choices: &[String], provider: &str) -> Vec<String> {
    let mut scoped: Vec<String> = Vec::new();
    for pattern in patterns {
        let base = pattern
            .rsplit_once(':')
            .filter(|(_, level)| level.parse::<ThinkingSetting>().is_ok())
            .map_or(pattern.as_str(), |(base, _)| base);
        let Ok(glob) = globset::GlobBuilder::new(&base.to_lowercase()).build() else {
            continue;
        };
        let matcher = glob.compile_matcher();
        for choice in choices {
            let id = choice.to_lowercase();
            if (matcher.is_match(&id)
                || matcher.is_match(format!("{}/{id}", provider.to_lowercase())))
                && !scoped.contains(choice)
            {
                scoped.push(choice.clone());
            }
        }
    }
    if scoped.is_empty() {
        choices.to_vec()
    } else {
        scoped
    }
}

fn apply_model_override(
    config: &mut Config,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<(), AgentError> {
    if let Some(provider) = provider {
        config.provider = provider.to_owned();
    }
    if let Some(model) = model {
        match model.split_once('/') {
            Some((provider, id)) if provider_is_known(config, provider) => {
                config.provider = provider.to_owned();
                config.model = id.to_owned();
            }
            _ => config.model = model.to_owned(),
        }
    }
    if provider.is_some() || model.is_some() {
        config.model_choices =
            configured_models(&config.agent_dir, &config.provider, &config.model)?;
    }
    Ok(())
}

/// A prefix before `/` names a provider only if it is `opencode-cf*` or listed in `models.json`.
fn provider_is_known(config: &Config, name: &str) -> bool {
    name.starts_with("opencode-cf")
        || std::fs::read_to_string(config.agent_dir.join("models.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .is_some_and(|json| json["providers"].get(name).is_some())
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

/// A `models.json` model accepts images unless it lists `input` without `image`.
fn model_accepts_images(model: &Value) -> bool {
    model
        .get("input")
        .and_then(Value::as_array)
        .is_none_or(|input| input.iter().any(|kind| kind == "image"))
}

fn model_supports_strict_sampling(model: &Value) -> bool {
    model
        .pointer("/compat/supportsStrictMode")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn mark_strict_builtin_tool(mut definition: Value) -> Value {
    let name = definition
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(name, "read" | "bash" | "edit" | "write") {
        return definition;
    }
    // Pi's constrained sampling is "prefer": an unsupported schema keeps the tool non-strict.
    if let Some(strict) = definition
        .get("parameters")
        .and_then(|parameters| strict_json_schema(parameters).ok())
    {
        definition["parameters"] = strict;
        definition["strict"] = Value::Bool(true);
    }
    definition
}

/// Keywords provider strict modes reject (Pi's `UNSUPPORTED_STRICT_SCHEMA_KEYS`).
const UNSUPPORTED_STRICT_SCHEMA_KEYS: [&str; 16] = [
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

fn is_structured_schema(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    let types: Vec<&str> = match object.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    types.contains(&"object")
        || types.contains(&"array")
        || object.contains_key("properties")
        || object.contains_key("items")
}

/// Convert a tool schema to the strict subset (Pi's `makeStrictJsonSchema`): every property
/// required, optional ones made nullable, `additionalProperties: false`; unsupported shapes
/// are errors.
fn strict_json_schema(schema: &Value) -> Result<Value, String> {
    let mut clone = schema.clone();
    if !clone.is_object() {
        return Err("root schema must have type object".into());
    }
    strictify_node(&mut clone)?;
    if clone.get("type").and_then(Value::as_str) != Some("object") {
        return Err("root schema must have type object".into());
    }
    Ok(clone)
}

fn strictify_node(schema: &mut Value) -> Result<(), String> {
    let Some(object) = schema.as_object_mut() else {
        return Err("boolean schemas are unsupported".into());
    };
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS {
        if object.contains_key(key) {
            return Err(format!("{key} schemas are unsupported"));
        }
    }
    if let Some(any_of) = object.get_mut("anyOf") {
        let variants = any_of
            .as_array_mut()
            .filter(|variants| !variants.is_empty())
            .ok_or("anyOf must contain at least one schema")?;
        for variant in variants {
            if is_structured_schema(variant) {
                return Err("object and array unions are unsupported".into());
            }
            strictify_node(variant)?;
        }
    }
    if let Some(items) = object.get_mut("items") {
        if items.is_array() {
            return Err("tuple schemas are unsupported".into());
        }
        strictify_node(items)?;
    }
    let is_object = object.get("type").and_then(Value::as_str) == Some("object");
    if object.contains_key("properties") && !is_object {
        return Err("properties require type object".into());
    }
    if !is_object {
        return Ok(());
    }
    if object
        .get("additionalProperties")
        .is_some_and(|value| *value != Value::Bool(false))
    {
        return Err("schema-valued or true additionalProperties is unsupported".into());
    }
    let required = match object.get("required") {
        None => HashSet::new(),
        Some(Value::Array(keys)) => keys
            .iter()
            .map(|key| key.as_str().map(str::to_owned))
            .collect::<Option<HashSet<_>>>()
            .ok_or("object required must be a string array")?,
        Some(_) => return Err("object required must be a string array".into()),
    };
    let mut properties = match object.remove("properties") {
        None => serde_json::Map::new(),
        Some(Value::Object(properties)) => properties,
        Some(_) => return Err("object properties must be a schema map".into()),
    };
    if required.iter().any(|key| !properties.contains_key(key)) {
        return Err("required contains an unknown property".into());
    }
    let names = properties.keys().cloned().collect::<Vec<_>>();
    for (name, property) in &mut properties {
        strictify_node(property)?;
        if !required.contains(name) && !schema_allows_null(property) {
            *property = json!({"anyOf":[property.clone(),{"type":"null"}]});
        }
    }
    let object = schema.as_object_mut().expect("checked above");
    object.insert("properties".into(), Value::Object(properties));
    object.insert("required".into(), json!(names));
    object.insert("additionalProperties".into(), Value::Bool(false));
    Ok(())
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

/// Refresh a stored Anthropic OAuth credential that has expired and save the new tokens.
async fn refresh_expired_oauth(config: &Config) -> Result<(), AgentError> {
    let path = config.agent_dir.join("auth.json");
    let Some((_, refresh, expires)) = crate::auth::oauth_access(&path, &config.provider) else {
        return Ok(());
    };
    let client_id = crate::auth::load(&path)
        .ok()
        .and_then(|mut all| all.remove(&config.provider))
        .and_then(|credential| match credential {
            crate::auth::Credential::Oauth { extra, .. } => extra
                .get("clientId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            crate::auth::Credential::ApiKey { .. } => None,
        });
    if !crate::oauth::is_expired(expires) {
        return Ok(());
    }
    let refreshed = match config.provider.as_str() {
        "anthropic" => crate::oauth::anthropic::refresh_token(&refresh).await,
        "xai" => crate::oauth::xai::refresh_token(&refresh).await,
        "openai-codex" => {
            crate::oauth::openai_codex::refresh_token(&refresh, crate::oauth::openai_codex::TOKEN_URL)
                .await
        }
        "github-copilot" => {
            let enterprise = crate::auth::load(&path)
                .ok()
                .and_then(|mut all| all.remove(&config.provider))
                .and_then(|credential| match credential {
                    crate::auth::Credential::Oauth { extra, .. } => extra
                        .get("enterpriseUrl")
                        .and_then(Value::as_str)
                        .and_then(crate::oauth::github_copilot::normalize_domain),
                    crate::auth::Credential::ApiKey { .. } => None,
                });
            crate::oauth::github_copilot::copilot_token(&refresh, enterprise.as_deref()).await
        }
        "meta" => {
            crate::oauth::meta::mint_api_key(&refresh, crate::oauth::meta::API_KEY_MINT_URL).await
        }
        "openai-chatgpt" => match client_id {
            Some(client_id) => {
                crate::oauth::openai_chatgpt::refresh_token(
                    &refresh,
                    &client_id,
                    crate::oauth::openai_chatgpt::TOKEN_URL,
                )
                .await
            }
            None => Err(
                "Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT"
                    .to_owned(),
            ),
        },
        "kimi-coding" => {
            let host = std::env::var("KIMI_CODE_OAUTH_HOST")
                .or_else(|_| std::env::var("KIMI_OAUTH_HOST"))
                .unwrap_or_else(|_| crate::oauth::kimi::DEFAULT_HOST.to_owned());
            crate::oauth::kimi::refresh_token(host.trim_end_matches('/'), &refresh).await
        }
        _ => return Ok(()),
    };
    let credential = refreshed.map_err(|error| {
        AgentError::Provider(format!(
            "OAuth refresh failed ({error}); run `mu auth login --provider {}`",
            config.provider
        ))
    })?;
    crate::auth::set_credential(&path, &config.provider, credential)
        .map_err(|error| AgentError::Configuration(error.to_string()))
}

async fn connect(
    config: &Config,
    skill_prompt: String,
    prompt: &PromptOptions,
) -> Result<Provider, AgentError> {
    if !config.provider.starts_with("opencode-cf") {
        refresh_expired_oauth(config).await?;
        return connect_custom_provider(config, skill_prompt, prompt);
    }
    let token = match &prompt.api_key {
        Some(key) => key.clone(),
        None => match crate::auth::api_key(&config.agent_dir.join("auth.json"), &config.provider) {
            Some(key) => key,
            None => token(&config.provider).await?,
        },
    };
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
    let mut supports_strict_sampling = false;
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
        supports_strict_sampling = model_supports_strict_sampling(m);
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
            gateway_headers: true,
        },
        model: config.model.clone(),
        instructions: match &prompt.system_prompt {
            Some(custom) => format!("{custom}{skill_prompt}"),
            None => format!(
                "You are Mu, a helpful coding assistant. Be clear, practical, and concise.{skill_prompt}"
            ),
        },
        describes_tools: prompt.system_prompt.is_none(),
        appendix: prompt.appendix(),
        supports_reasoning,
        supports_images: true,
        reasoning_effort: prompt.thinking.effort(),
        supports_strict_sampling,
    })
}

impl Provider {
    async fn complete_google(
        &self,
        http: &Client,
        endpoint: &Url,
        auth: &GoogleAuth,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let mut system = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            system = format!("{system}\n\n{}", self.appendix);
        }
        let request = http.post(endpoint.clone());
        let request = match auth {
            GoogleAuth::Bearer(token) => request.bearer_auth(token),
            GoogleAuth::ApiKey(key) => request.header("x-goog-api-key", key),
        };
        let response = request
            .json(&{
                let mut body = google::request_body(&system, history, tools);
                if self.supports_reasoning {
                    body["generationConfig"] = json!({"thinkingConfig": google::thinking_config(&self.model, self.reasoning_effort)});
                }
                body
            })
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(http_failure(response).await);
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        let mut decoder = google::Decoder::default();
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
                    decoder
                        .feed(&event, on_event)
                        .map_err(AgentError::Provider)?;
                }
            }
        }
        if !decoder.finished() {
            return Err(AgentError::Provider(
                "Google stream ended before a finish reason".into(),
            ));
        }
        Ok(decoder.into_reply())
    }

    async fn complete_completions(
        &self,
        http: &Client,
        endpoint: &Url,
        key: &str,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let mut system = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            system = format!("{system}\n\n{}", self.appendix);
        }
        let effort = self.reasoning_effort.filter(|_| self.supports_reasoning);
        let body = completions::request_body(&self.model, &system, history, tools, effort);
        let response = http
            .post(endpoint.clone())
            .bearer_auth(key)
            .header(header::ACCEPT, "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(http_failure(response).await);
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        let mut decoder = completions::Decoder::default();
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
                if data.trim() == "[DONE]" {
                    decoder.feed_done();
                } else if let Ok(event) = serde_json::from_str::<Value>(data.trim()) {
                    decoder
                        .feed(&event, on_event)
                        .map_err(AgentError::Provider)?;
                }
            }
        }
        if !decoder.finished() {
            return Err(AgentError::Provider(
                "Completions stream ended before a finish reason".into(),
            ));
        }
        Ok(decoder.into_reply())
    }

    async fn complete_bedrock(
        &self,
        http: &Client,
        endpoint: &Url,
        auth: &BedrockAuth,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let mut system = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            system = format!("{system}\n\n{}", self.appendix);
        }
        let mut request_json = bedrock::request_body(&system, history, tools);
        if self.supports_reasoning
            && let Some((fields, budget)) =
                bedrock::thinking_fields(&self.model, self.reasoning_effort)
        {
            request_json["additionalModelRequestFields"] = fields;
            // Claude requires max tokens above the thinking budget.
            if budget > 0 {
                request_json["inferenceConfig"]["maxTokens"] = json!(budget + 16_384);
            }
        }
        let body = serde_json::to_vec(&request_json).expect("request serializes");
        let mut request = http
            .post(endpoint.clone())
            .header(header::CONTENT_TYPE, "application/json");
        match auth {
            BedrockAuth::Bearer(key) => request = request.bearer_auth(key),
            BedrockAuth::Sigv4 {
                credentials,
                region,
            } => {
                let host = endpoint.host_str().unwrap_or_default().to_owned();
                let amz_date = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
                let signed = sigv4::sign(
                    "POST",
                    endpoint.path(),
                    endpoint.query().unwrap_or_default(),
                    &[
                        ("host".into(), host),
                        ("content-type".into(), "application/json".into()),
                    ],
                    &body,
                    credentials,
                    region,
                    "bedrock",
                    &amz_date,
                );
                request = request
                    .header("x-amz-date", signed.amz_date)
                    .header(header::AUTHORIZATION, signed.authorization);
                if let Some(token) = &credentials.session_token {
                    request = request.header("x-amz-security-token", token);
                }
            }
        }
        let response = request
            .body(body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            let body: Value = response.json().await.unwrap_or(Value::Null);
            let detail = body["message"].as_str().unwrap_or("no details");
            return Err(AgentError::Provider(format!(
                "model request returned HTTP {status}: {detail}"
            )));
        }
        // `converse-stream` answers with AWS's binary event stream, decoded as it arrives.
        let mut stream = response.bytes_stream();
        let mut decoder = bedrock::EventStreamDecoder::default();
        let mut accumulator = bedrock::StreamAccumulator::default();
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| AgentError::Provider(format!("model stream failed: {e}")))?;
            for message in decoder.push(&chunk).map_err(AgentError::Provider)? {
                accumulator
                    .feed(&message, on_event)
                    .map_err(AgentError::Provider)?;
            }
        }
        Ok(accumulator.into_reply())
    }

    async fn complete_anthropic(
        &self,
        http: &Client,
        endpoint: &Url,
        key: &str,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let mut system = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            system = format!("{system}\n\n{}", self.appendix);
        }
        let budget = self
            .reasoning_effort
            .filter(|_| self.supports_reasoning)
            .map(|effort| match effort {
                "minimal" => 1_024,
                "low" => 2_048,
                "medium" => 8_192,
                "high" => 16_384,
                _ => 32_768,
            });
        let oauth = anthropic::is_oauth_token(key);
        if oauth {
            system = format!("{}\n\n{system}", anthropic::CLAUDE_CODE_IDENTITY);
        }
        let (history, wire_tools) = if oauth {
            anthropic::rename_for_claude_code(history, tools)
        } else {
            (history.to_vec(), tools.to_vec())
        };
        let mut body = anthropic::request_body(&self.model, &system, &history, &wire_tools, budget);
        if let Some(budget) = budget {
            body["max_tokens"] = json!(budget + 16_384);
        }
        let request = http.post(endpoint.clone());
        let request = if oauth {
            // Pi: OAuth tokens use Bearer auth plus Claude Code identity headers.
            request
                .bearer_auth(key)
                .header("anthropic-beta", "claude-code-20250219,oauth-2025-04-20")
                .header("user-agent", "claude-cli/2.1.280")
                .header("x-app", "cli")
        } else {
            request.header("x-api-key", key)
        };
        let response = request
            .header("anthropic-version", "2023-06-01")
            .header(header::ACCEPT, "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(http_failure(response).await);
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        let mut decoder = anthropic::Decoder::default();
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
                    decoder
                        .feed(&event, on_event)
                        .map_err(AgentError::Provider)?;
                }
            }
        }
        if !decoder.finished() {
            return Err(AgentError::Provider(
                "Anthropic stream ended before message_stop".into(),
            ));
        }
        let mut reply = decoder.into_reply();
        if oauth {
            for call in &mut reply.tool_calls {
                call.name = anthropic::from_claude_code_name(&call.name, tools);
            }
        }
        Ok(reply)
    }

    async fn complete(
        &self,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let downgraded;
        let history = if self.supports_images {
            history
        } else {
            downgraded = runtime::downgrade_images(history);
            &downgraded
        };
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
        if let ProviderTransport::Google {
            http,
            endpoint,
            auth,
        } = &self.transport
        {
            return self
                .complete_google(http, endpoint, auth, history, tools, on_event)
                .await;
        }
        if let ProviderTransport::Completions {
            http,
            endpoint,
            key,
        } = &self.transport
        {
            return self
                .complete_completions(http, endpoint, key, history, tools, on_event)
                .await;
        }
        if let ProviderTransport::Bedrock {
            http,
            endpoint,
            auth,
        } = &self.transport
        {
            return self
                .complete_bedrock(http, endpoint, auth, history, tools, on_event)
                .await;
        }
        if let ProviderTransport::Anthropic {
            http,
            endpoint,
            key,
        } = &self.transport
        {
            return self
                .complete_anthropic(http, endpoint, key, history, tools, on_event)
                .await;
        }
        let ProviderTransport::OpenCode {
            http,
            endpoint,
            token,
            gateway_headers,
        } = &self.transport
        else {
            unreachable!("demo backend returned above")
        };
        let input = history;
        let mut instructions = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            instructions = format!("{instructions}\n\n{}", self.appendix);
        }
        let mut request_body = json!({"model":self.model,"instructions":instructions,"input":input,"tools":tools,"stream":true,"store":false});
        if let Some(effort) = self.reasoning_effort.filter(|_| self.supports_reasoning) {
            request_body["reasoning"] = json!({"effort":effort,"summary":"auto"});
            request_body["include"] = json!(["reasoning.encrypted_content"]);
        }
        let mut request = http.post(endpoint.clone()).bearer_auth(token);
        if *gateway_headers {
            // Cloudflare Access headers must never be sent to third-party providers.
            request = request
                .header("cf-access-token", token)
                .header("x-requested-with", "xmlhttprequest");
        }
        let response = request
            .header(header::ACCEPT, "text/event-stream")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(http_failure(response).await);
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

/// A provider declared in `models.json` with `api: "openai-responses"`, `baseUrl` and `apiKey`
/// (an environment variable name, else the literal key).
fn connect_custom_provider(
    config: &Config,
    skill_prompt: String,
    prompt: &PromptOptions,
) -> Result<Provider, AgentError> {
    let unsupported = || {
        AgentError::Configuration(format!(
            "Provider `{}` is not configured: add it to models.json with api \"openai-responses\", baseUrl and apiKey",
            config.provider
        ))
    };
    let models: Value = std::fs::read_to_string(config.agent_dir.join("models.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .ok_or_else(unsupported)?;
    let entry = models["providers"]
        .get(&config.provider)
        .ok_or_else(unsupported)?;
    match entry.get("api").and_then(Value::as_str) {
        Some(
            "openai-responses"
            | "anthropic-messages"
            | "openai-completions"
            | "google-generative-ai"
            | "google-vertex"
            | "bedrock-converse-stream",
        ) => {}
        other => {
            return Err(AgentError::Configuration(format!(
                "Provider `{}` uses unsupported api `{}` (supported: openai-responses, openai-completions, anthropic-messages, google-generative-ai, google-vertex, bedrock-converse-stream)",
                config.provider,
                other.unwrap_or("none")
            )));
        }
    }
    let base = match (
        entry.get("baseUrl").and_then(Value::as_str),
        entry.get("api").and_then(Value::as_str),
    ) {
        (Some(base), _) => base.to_owned(),
        (None, Some("google-vertex")) => vertex_base_url(entry)?,
        (None, _) => return Err(unsupported()),
    };
    let base = base.as_str();
    let mut endpoint = Url::parse(base)
        .map_err(|error| AgentError::Configuration(format!("invalid baseUrl: {error}")))?;
    let anthropic = entry.get("api").and_then(Value::as_str) == Some("anthropic-messages");
    let completions = entry.get("api").and_then(Value::as_str) == Some("openai-completions");
    let vertex = entry.get("api").and_then(Value::as_str) == Some("google-vertex");
    let google = vertex || entry.get("api").and_then(Value::as_str) == Some("google-generative-ai");
    let bedrock = entry.get("api").and_then(Value::as_str) == Some("bedrock-converse-stream");
    let suffix = if bedrock {
        format!("model/{}/converse-stream", config.model)
    } else if anthropic {
        "messages".to_owned()
    } else if completions {
        "chat/completions".to_owned()
    } else if google {
        format!("models/{}:streamGenerateContent", config.model)
    } else {
        "responses".to_owned()
    };
    endpoint.set_path(&format!(
        "{}/{suffix}",
        endpoint.path().trim_end_matches('/')
    ));
    if google {
        endpoint.set_query(Some("alt=sse"));
    }
    // Bedrock may omit `apiKey` and sign with AWS credentials from the environment instead.
    let key = match entry.get("apiKey").and_then(Value::as_str) {
        Some(key) => key,
        None if bedrock => "",
        None if anthropic => "",
        None => return Err(unsupported()),
    };
    let token = prompt
        .api_key
        .clone()
        .or_else(|| crate::auth::api_key(&config.agent_dir.join("auth.json"), &config.provider))
        .or_else(|| {
            crate::auth::oauth_access(&config.agent_dir.join("auth.json"), &config.provider)
                .map(|(access, _, _)| access)
        })
        .or_else(|| crate::config_value::resolve(key, &HashMap::new()))
        .filter(|v| !v.is_empty())
        .unwrap_or_default();
    if token.is_empty() && !bedrock {
        return Err(AgentError::Configuration(no_api_key_message(
            &config.provider,
        )));
    }
    let bedrock_auth = if !bedrock {
        None
    } else if !token.is_empty() {
        Some(BedrockAuth::Bearer(token.clone()))
    } else {
        let credentials = sigv4::AwsCredentials::from_env().ok_or_else(|| {
            AgentError::Configuration(
                "Bedrock needs an apiKey (Bedrock API key) or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
                    .into(),
            )
        })?;
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .ok()
            .or_else(|| bedrock_region_from_host(endpoint.host_str().unwrap_or_default()))
            .ok_or_else(|| {
                AgentError::Configuration("Bedrock region unknown: set AWS_REGION".into())
            })?;
        Some(BedrockAuth::Sigv4 {
            credentials,
            region,
        })
    };
    let http = provider_http_client(entry.get("headers"))?;
    let model = configured_model_entry(&config.agent_dir, &config.provider, &config.model);
    let flag = |name: &str| {
        model
            .as_ref()
            .and_then(|m| m.get(name))
            .and_then(Value::as_bool)
    };
    Ok(Provider {
        transport: if bedrock {
            ProviderTransport::Bedrock {
                http: http.clone(),
                endpoint,
                auth: bedrock_auth.expect("bedrock auth resolved above"),
            }
        } else if anthropic {
            ProviderTransport::Anthropic {
                http: http.clone(),
                endpoint,
                key: token,
            }
        } else if google {
            ProviderTransport::Google {
                http: http.clone(),
                endpoint,
                auth: if vertex {
                    GoogleAuth::Bearer(token)
                } else {
                    GoogleAuth::ApiKey(token)
                },
            }
        } else if completions {
            ProviderTransport::Completions {
                http: http.clone(),
                endpoint,
                key: token,
            }
        } else {
            ProviderTransport::OpenCode {
                http: http.clone(),
                endpoint,
                token,
                gateway_headers: false,
            }
        },
        model: config.model.clone(),
        instructions: match &prompt.system_prompt {
            Some(custom) => format!("{custom}{skill_prompt}"),
            None => format!(
                "You are Mu, a helpful coding assistant. Be clear, practical, and concise.{skill_prompt}"
            ),
        },
        describes_tools: prompt.system_prompt.is_none(),
        appendix: prompt.appendix(),
        supports_reasoning: flag("reasoning").unwrap_or(false),
        supports_images: model.as_ref().is_none_or(model_accepts_images),
        reasoning_effort: prompt.thinking.effort(),
        supports_strict_sampling: model.as_ref().is_some_and(model_supports_strict_sampling),
    })
}

/// An HTTP client sending the provider's `headers` from `models.json` on every request; values
/// resolve like other config values (`$VAR`, `!command`) and unresolved ones are skipped.
fn provider_http_client(headers: Option<&Value>) -> Result<Client, AgentError> {
    let mut defaults = header::HeaderMap::new();
    for (name, value) in headers.and_then(Value::as_object).into_iter().flatten() {
        let Some(resolved) = value
            .as_str()
            .and_then(|v| crate::config_value::resolve(v, &HashMap::new()))
        else {
            continue;
        };
        let invalid =
            |what: &str| AgentError::Configuration(format!("invalid header {what} `{name}`"));
        defaults.insert(
            header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| invalid("name"))?,
            header::HeaderValue::from_str(&resolved).map_err(|_| invalid("value"))?,
        );
    }
    Client::builder()
        .default_headers(defaults)
        .build()
        .map_err(|error| AgentError::Configuration(format!("cannot build HTTP client: {error}")))
}

/// Vertex AI's regional endpoint from `project` and `location` (default `us-central1`) when
/// `models.json` gives no `baseUrl`.
fn vertex_base_url(entry: &Value) -> Result<String, AgentError> {
    let project = entry
        .get("project")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
        .filter(|p| !p.is_empty())
        .ok_or_else(|| {
            AgentError::Configuration(
                "Vertex AI needs `baseUrl`, or `project` (or GOOGLE_CLOUD_PROJECT) in models.json"
                    .into(),
            )
        })?;
    let location = entry
        .get("location")
        .and_then(Value::as_str)
        .unwrap_or("us-central1");
    let host = if location == "global" {
        "aiplatform.googleapis.com".to_owned()
    } else {
        format!("{location}-aiplatform.googleapis.com")
    };
    Ok(format!(
        "https://{host}/v1/projects/{project}/locations/{location}/publishers/google"
    ))
}

fn api_name(transport: &ProviderTransport) -> &'static str {
    if matches!(transport, ProviderTransport::Demo) {
        "mu-demo"
    } else if matches!(
        transport,
        ProviderTransport::Google {
            auth: GoogleAuth::Bearer(_),
            ..
        }
    ) {
        "google-vertex"
    } else if matches!(transport, ProviderTransport::Google { .. }) {
        "google-generative-ai"
    } else if matches!(transport, ProviderTransport::Completions { .. }) {
        "openai-completions"
    } else if matches!(transport, ProviderTransport::Bedrock { .. }) {
        "bedrock-converse-stream"
    } else if matches!(transport, ProviderTransport::Anthropic { .. }) {
        "anthropic-messages"
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

/// Pi's `formatMissingSessionCwdError`, when the cwd stored in the session file is gone.
fn missing_session_cwd_error(
    session: &SessionDocument,
    file: &Path,
    fallback: &Path,
) -> Option<String> {
    let stored = session.header().map(|header| header.cwd.clone())?;
    if stored.is_empty() || Path::new(&stored).exists() {
        return None;
    }
    Some(format!(
        "Stored session working directory does not exist: {stored}\nSession file: {}\nCurrent working directory: {}",
        file.display(),
        fallback.display()
    ))
}

fn load_session(path: &Path) -> Result<(SessionDocument, Vec<AgentMessage>), AgentError> {
    let contents =
        std::fs::read_to_string(path).map_err(|error| AgentError::Session(error.to_string()))?;
    let session = SessionDocument::parse_jsonl(&contents)
        .map_err(|error| AgentError::Session(error.to_string()))?;
    let history = runtime::session_messages(&session);
    Ok((session, history))
}

fn render_session_html(session: &SessionDocument) -> String {
    let header = session.header();
    let id = header.map(|h| h.id.as_str().to_owned()).unwrap_or_default();
    let timestamp = header
        .map(|h| h.timestamp.as_str().to_owned())
        .unwrap_or_default();
    let title = session.session_name().unwrap_or("Mu session");
    let mut html = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{}</title><style>{SESSION_HTML_STYLE}</style></head><body><main><header><h1>{}</h1><p class=\"meta\">Session {} · {}</p></header>",
        escape_html(title),
        escape_html(title),
        escape_html(&id),
        escape_html(&timestamp),
    );
    for message in session_transcript(session) {
        append_transcript_message_html(&mut html, message);
    }
    html.push_str("</main></body></html>");
    html
}

/// `--export <session.jsonl> [output.html]`: render a session file without connecting a provider.
///
/// The default output is `mu-session-<name>.html` in the working directory (Pi's naming scheme).
pub fn export_session_file(input: &Path, output: Option<&Path>) -> Result<PathBuf, AgentError> {
    let (session, _) = load_session(input)?;
    let output = match output {
        Some(path) => path.to_owned(),
        None => {
            let stem = input
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            std::env::current_dir()?.join(format!("mu-session-{stem}.html"))
        }
    };
    std::fs::write(&output, render_session_html(&session))
        .map_err(|error| AgentError::Session(error.to_string()))?;
    Ok(output)
}

const SESSION_HTML_STYLE: &str = "body{margin:0;background:#17191f;color:#e5e7eb;font:16px/1.6 system-ui,sans-serif}main{max-width:900px;margin:3rem auto;padding:0 1.25rem}header{border-bottom:1px solid #454955;margin-bottom:2rem}article{padding:1rem 1.25rem;margin:1rem 0;border-radius:12px;background:#22252d}article.user{background:#292d38}h1,h2{line-height:1.2}h2{font-size:1rem;color:#aeb8ce}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:.5rem 0}details{margin:.75rem 0;padding:.75rem;background:#191b21;border-radius:8px}.error{color:#ff8a8a}.meta{color:#9ca3af;font-size:.9rem}";

fn append_transcript_message_html(html: &mut String, message: SessionTranscriptMessage) {
    let (role, class) = match message.role {
        SessionTranscriptRole::User => ("User", "user"),
        SessionTranscriptRole::Assistant => ("Assistant", "assistant"),
        SessionTranscriptRole::CompactionSummary(_) => ("Compaction summary", "assistant"),
        SessionTranscriptRole::BranchSummary => ("Branch summary", "assistant"),
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
        let summary_role = match entry.entry_type.as_str() {
            "compaction" => Some(SessionTranscriptRole::CompactionSummary(
                entry
                    .extra_fields()
                    .get("tokensBefore")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            )),
            "branch_summary" => Some(SessionTranscriptRole::BranchSummary),
            _ => None,
        };
        if let Some(role) = summary_role {
            let summary = entry
                .extra_fields()
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !summary.is_empty() {
                transcript.push(SessionTranscriptMessage {
                    role,
                    text: summary.to_owned(),
                    thinking: String::new(),
                    tools: Vec::new(),
                });
            }
            continue;
        }
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
        let blocks_owned = message_blocks(message);
        let blocks = blocks_owned.as_slice();
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

/// The active branch as Pi-style JSON entries (`type`, `id`, and the entry's own fields).
pub(crate) fn branch_entry_values(session: &SessionDocument) -> Vec<Value> {
    session
        .active_branch_entries()
        .into_iter()
        .map(|entry| {
            let mut value = serde_json::Map::new();
            value.insert("type".into(), json!(entry.entry_type.as_str()));
            if let Some(id) = &entry.id {
                value.insert("id".into(), json!(id.as_str()));
            }
            value.extend(
                entry
                    .extra_fields()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone())),
            );
            Value::Object(value)
        })
        .collect()
}

/// Message content as blocks; Pi stores plain-text user messages as a bare string.
pub(crate) fn message_blocks(message: &serde_json::Map<String, Value>) -> Vec<Value> {
    match message.get("content") {
        Some(Value::Array(blocks)) => blocks.clone(),
        Some(Value::String(text)) => vec![json!({"type":"text","text":text})],
        _ => Vec::new(),
    }
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

/// `~` and `~/…` expand to the home directory (Pi's `expandTildePath`).
fn expand_tilde(path: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    match (path, home) {
        ("~", Some(home)) => home,
        (rest, Some(home)) if rest.starts_with("~/") => home.join(&rest[2..]),
        _ => PathBuf::from(path),
    }
}

/// Pi's session-directory precedence: `--session-dir`, then `PI_CODING_AGENT_SESSION_DIR`, then
/// the `sessionDir` setting.
fn configured_session_dir(
    flag: Option<&Path>,
    env: Option<&str>,
    setting: Option<&str>,
) -> Option<PathBuf> {
    flag.map(Path::to_owned)
        .or_else(|| env.filter(|v| !v.is_empty()).map(expand_tilde))
        .or_else(|| setting.filter(|v| !v.is_empty()).map(expand_tilde))
}

fn session_directory(agent_dir: &Path, cwd: &Path, override_dir: Option<&Path>) -> PathBuf {
    let setting = crate::settings::Settings::load(agent_dir, cwd, true).get("sessionDir");
    let env = std::env::var("PI_CODING_AGENT_SESSION_DIR").ok();
    if let Some(directory) = configured_session_dir(
        override_dir,
        env.as_deref(),
        setting.as_ref().and_then(Value::as_str),
    ) {
        return directory;
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
) -> Result<(PathBuf, SessionDocument, Vec<AgentMessage>), AgentError> {
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
        demo: _,
        system_prompt: _,
        append_system_prompt: _,
        no_context_files: _,
        exclude_tools: _,
        thinking: _,
        provider: _,
        model: _,
        no_mcp: _,
        prompt_templates: _,
        no_prompt_templates: _,
        api_key: _,
        project_trust: _,
        theme: _,
        models: _,
        verbose: _,
        continue_in_current_cwd,
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
        let history = runtime::session_messages(&session);
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
            if let Some(message) = missing_session_cwd_error(&session, &path, &config.cwd) {
                if continue_in_current_cwd {
                    eprintln!("mu: {message}\nmu: continuing in the current working directory");
                } else {
                    return Err(AgentError::Session(message));
                }
            }
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

/// `model request returned HTTP <status>` plus the start of the provider's error body, which
/// carries quota and billing reasons the retry policy needs to see.
async fn http_failure(response: reqwest::Response) -> AgentError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    AgentError::Provider(http_failure_message(status, &body))
}

fn http_failure_message(status: reqwest::StatusCode, body: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        return format!("model request returned HTTP {status}");
    }
    let shown = body.chars().take(500).collect::<String>();
    let ellipsis = if body.chars().count() > 500 {
        "…"
    } else {
        ""
    };
    format!("model request returned HTTP {status}: {shown}{ellipsis}")
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

/// One row of `/tree`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeRow {
    pub id: String,
    pub depth: usize,
    pub label: String,
    /// On the path from the root to the current tip.
    pub on_branch: bool,
    pub is_leaf: bool,
}

/// Result of [`MuAgent::compact`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionOutcome {
    pub summary: String,
    pub tokens_before: u64,
}

/// Plain-text rendering of messages for the summarizer.
/// Token, cost, and context totals shown in Pi's footer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FooterStats {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub cost: f64,
    /// Tokens in the most recent successful assistant response (the context size estimate).
    pub context_tokens: Option<u64>,
    pub context_window: u64,
}

fn session_footer_stats(session: &SessionDocument) -> FooterStats {
    let mut stats = FooterStats::default();
    for entry in session.entries() {
        let Some(message) = entry.extra_fields().get("message") else {
            continue;
        };
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(usage) = message.get("usage") else {
            continue;
        };
        let field = |name: &str| usage.get(name).and_then(Value::as_u64).unwrap_or(0);
        stats.input += field("input");
        stats.output += field("output");
        stats.cache_read += field("cacheRead");
        stats.cache_write += field("cacheWrite");
        stats.cost += usage
            .pointer("/cost/total")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let failed = matches!(
            message.get("stopReason").and_then(Value::as_str),
            Some("error" | "aborted")
        );
        if !failed {
            let total = field("totalTokens");
            stats.context_tokens = Some(if total > 0 {
                total
            } else {
                field("input") + field("output") + field("cacheRead") + field("cacheWrite")
            });
        }
    }
    stats
}

/// A model's entry in `models.json` for `provider`.
fn configured_model_entry(agent_dir: &Path, provider: &str, model: &str) -> Option<Value> {
    let contents = std::fs::read_to_string(agent_dir.join("models.json")).ok()?;
    let value: Value = serde_json::from_str(&contents).ok()?;
    value
        .get("providers")?
        .get(provider)?
        .get("models")?
        .as_array()?
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model))
        .cloned()
}

/// Everything both startup paths derive from the configuration and session options.
struct Prepared {
    config: Config,
    tool_selection: Option<ToolSelection>,
    no_tools: bool,
    persist_session: bool,
    skills: crate::skills::SkillCatalog,
    session_path: PathBuf,
    session: SessionDocument,
    transcript: Vec<AgentMessage>,
    prompt: PromptOptions,
    excluded_tools: Vec<String>,
    templates: Vec<crate::prompt_templates::PromptTemplate>,
    resources: ResourceOptions,
}

/// Flags that decide which skills and prompt templates are discovered (kept for `/reload`).
#[derive(Clone, Debug, Default)]
struct ResourceOptions {
    skill_paths: Vec<PathBuf>,
    no_skills: bool,
    template_paths: Vec<PathBuf>,
    no_templates: bool,
}

impl Prepared {
    fn new(mut config: Config, options: SessionOptions) -> Result<Self, AgentError> {
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
        let excluded_tools = options.exclude_tools.clone();
        let resources = ResourceOptions {
            skill_paths: options.skill_paths.clone(),
            no_skills: options.no_skills,
            template_paths: options.prompt_templates.clone(),
            no_templates: options.no_prompt_templates,
        };
        let templates = crate::prompt_templates::load_prompt_templates(
            &config.cwd,
            &config.agent_dir,
            &options.prompt_templates,
            !options.no_prompt_templates,
        );
        let prompt = PromptOptions::from_session_options(&options).with_context_files(
            &config.cwd,
            &config.agent_dir,
            options.no_context_files,
        );
        let skills = crate::skills::SkillCatalog::discover_with_paths(
            &config.cwd,
            &config.agent_dir,
            &options.skill_paths,
            !options.no_skills,
        );
        let (session_path, session, transcript) = resolve_session_options(&config, options)?;
        Ok(Self {
            config,
            tool_selection,
            no_tools,
            persist_session,
            skills,
            session_path,
            session,
            transcript,
            prompt,
            excluded_tools,
            templates,
            resources,
        })
    }

    fn into_agent(
        self,
        provider: Provider,
        mcp_startup: Option<Arc<OnceCell<crate::mcp::McpRegistry>>>,
    ) -> MuAgent {
        let active_mcp_tools = session_active_mcp_tools(&self.session);
        MuAgent {
            provider,
            provider_id: self.config.provider,
            model_choices: self.config.model_choices,
            session: self.session,
            transcript: self.transcript,
            session_path: self.session_path,
            persist_session: self.persist_session,
            agent_dir: self.config.agent_dir,
            cwd: self.config.cwd,
            skills: self.skills,
            hide_thinking_block: self.config.hide_thinking_block,
            retry: self.config.retry,
            enabled_tools: self.config.enabled_tools,
            no_tools: self.no_tools,
            tool_selection: self.tool_selection,
            excluded_tools: self.excluded_tools,
            templates: self.templates,
            resources: self.resources,
            thinking: self.prompt.thinking,
            no_mcp: false,
            theme_override: self.prompt.theme,
            model_scope_override: self.prompt.models,
            mcp: crate::mcp::McpRegistry::default(),
            mcp_startup,
            active_mcp_tools,
            prompt_queue: AgentPromptQueue::default(),
            event_tap: None,
        }
    }
}

pub struct MuAgent {
    provider: Provider,
    provider_id: String,
    model_choices: Vec<String>,
    session: SessionDocument,
    transcript: Vec<AgentMessage>,
    session_path: PathBuf,
    persist_session: bool,
    agent_dir: PathBuf,
    cwd: PathBuf,
    skills: crate::skills::SkillCatalog,
    templates: Vec<crate::prompt_templates::PromptTemplate>,
    resources: ResourceOptions,
    hide_thinking_block: bool,
    retry: RetryPolicy,
    enabled_tools: Vec<String>,
    no_tools: bool,
    tool_selection: Option<ToolSelection>,
    excluded_tools: Vec<String>,
    thinking: ThinkingSetting,
    no_mcp: bool,
    theme_override: Option<String>,
    model_scope_override: Vec<String>,
    mcp: crate::mcp::McpRegistry,
    mcp_startup: Option<Arc<OnceCell<crate::mcp::McpRegistry>>>,
    active_mcp_tools: HashSet<String>,
    prompt_queue: AgentPromptQueue,
    event_tap: Option<EventTap>,
}

/// Observer of raw agent-loop events (used by `--mode json`).
pub type EventTap = Arc<dyn Fn(&crate::agent_loop::AgentEvent) + Send + Sync>;

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
        if options.demo {
            return Self::open_demo(options);
        }
        let mut config = config_for(false)?;
        apply_model_override(
            &mut config,
            options.provider.as_deref(),
            options.model.as_deref(),
        )?;
        let no_mcp = options.no_mcp;
        crate::mcp::set_project_trust_override(options.project_trust);
        let prepared = Prepared::new(config, options)?;
        let mcp_startup = Arc::new(OnceCell::new());
        let background_mcp = Arc::clone(&mcp_startup);
        let mcp_agent_dir = prepared.config.agent_dir.clone();
        let mcp_cwd = prepared.config.cwd.clone();
        let mcp_startup_task = tokio::spawn(async move {
            background_mcp
                .get_or_init(|| async move {
                    if no_mcp {
                        crate::mcp::McpRegistry::default()
                    } else {
                        crate::mcp::McpRegistry::connect(&mcp_agent_dir, &mcp_cwd).await
                    }
                })
                .await;
        });
        let provider = match connect(
            &prepared.config,
            prepared.skills.system_prompt(),
            &prepared.prompt,
        )
        .await
        {
            Ok(provider) => provider,
            Err(error) => {
                mcp_startup_task.abort();
                return Err(error);
            }
        };
        drop(mcp_startup_task);
        let mut agent = prepared.into_agent(provider, Some(mcp_startup));
        agent.no_mcp = no_mcp;
        Ok(agent)
    }

    /// Create a deterministic no-network agent for terminal previews and tests.
    pub fn demo() -> Result<Self, AgentError> {
        Self::open_demo(SessionOptions {
            demo: true,
            ..SessionOptions::default()
        })
    }

    fn open_demo(options: SessionOptions) -> Result<Self, AgentError> {
        let config = config_for(true)?;
        let prepared = Prepared::new(config, options)?;
        let provider = Provider {
            transport: ProviderTransport::Demo,
            model: prepared.config.model.clone(),
            instructions: "You are Mu, a deterministic demo assistant.".into(),
            describes_tools: true,
            appendix: String::new(),
            supports_reasoning: false,
            supports_images: true,
            reasoning_effort: None,
            supports_strict_sampling: true,
        };
        Ok(prepared.into_agent(provider, None))
    }

    #[cfg(test)]
    fn demo_from_config(config: Config) -> Result<Self, AgentError> {
        let path = session_path(&config, None);
        let session = SessionDocument::new(config.cwd.to_string_lossy().into_owned());
        let active_mcp_tools = session_active_mcp_tools(&session);
        Ok(Self {
            provider: Provider {
                transport: ProviderTransport::Demo,
                model: config.model.clone(),
                instructions: "You are Mu, a deterministic demo assistant.".into(),
                describes_tools: true,
                appendix: String::new(),
                supports_reasoning: false,
                supports_images: true,
                reasoning_effort: None,
                supports_strict_sampling: true,
            },
            provider_id: config.provider,
            model_choices: config.model_choices,
            session,
            transcript: Vec::new(),
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
            excluded_tools: Vec::new(),
            templates: Vec::new(),
            resources: ResourceOptions::default(),
            thinking: ThinkingSetting::default(),
            no_mcp: false,
            theme_override: None,
            model_scope_override: Vec::new(),
            mcp: crate::mcp::McpRegistry::default(),
            mcp_startup: None,
            active_mcp_tools,
            prompt_queue: AgentPromptQueue::default(),
            event_tap: None,
        })
    }

    /// Observe every raw agent-loop event as it happens.
    pub fn set_event_tap(&mut self, tap: EventTap) {
        self.event_tap = Some(tap);
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
            .iter()
            .filter_map(format_mcp_tool_summary)
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
        let no_mcp = self.no_mcp;
        let connected = tokio::time::timeout(
            MCP_STARTUP_WAIT,
            startup.get_or_init(|| async move {
                if no_mcp {
                    crate::mcp::McpRegistry::default()
                } else {
                    crate::mcp::McpRegistry::connect(&agent_dir, &cwd).await
                }
            }),
        )
        .await;
        if let Ok(registry) = connected {
            self.mcp = registry.clone();
            self.mcp_startup = None;
        }
    }

    /// `/mcp reload`: drop the connected servers and connect again from the current config
    /// (Pi's MCP `reload`). Previously loaded deferred tools are forgotten, as the tools may differ.
    pub async fn reload_mcp(&mut self) -> String {
        self.mcp_startup = None;
        self.active_mcp_tools.clear();
        self.mcp = if self.no_mcp {
            crate::mcp::McpRegistry::default()
        } else {
            crate::mcp::McpRegistry::connect(&self.agent_dir, &self.cwd).await
        };
        self.mcp.status_text()
    }

    pub fn prompt_queue(&self) -> AgentPromptQueue {
        self.prompt_queue.clone()
    }

    /// Models Ctrl+P cycles through: `enabledModels` patterns applied to the choices.
    pub fn cycle_models(&self) -> Vec<String> {
        let patterns = if self.model_scope_override.is_empty() {
            configured_setting(&self.agent_dir, &self.cwd, "enabledModels", |value| {
                value.as_array().map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
            })
            .unwrap_or_default()
        } else {
            self.model_scope_override.clone()
        };
        scope_models(&patterns, &self.model_choices, &self.provider_id)
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

    /// Set the reasoning level for later requests (`/thinking`).
    pub fn set_thinking(&mut self, level: ThinkingSetting) {
        self.provider.reasoning_effort = level.effort();
        self.thinking = level;
    }

    pub fn thinking(&self) -> ThinkingSetting {
        self.thinking
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

    /// Text of the newest assistant message on the active branch (empty when none).
    pub fn last_assistant_text(&self) -> String {
        branch_entry_values(&self.session)
            .iter()
            .rev()
            .filter_map(|entry| entry.get("message")?.as_object().cloned())
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
            .map(|message| {
                message_blocks(&message)
                    .iter()
                    .filter_map(|block| block.get("text")?.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .find(|text| !text.trim().is_empty())
            .unwrap_or_default()
    }

    /// Text block for `/session`: identity plus message counts and token/cost totals.
    pub fn session_summary(&self) -> String {
        let info = self.session_info();
        let (mut user, mut assistant, mut tool_results, mut tool_calls) = (0, 0, 0, 0);
        for entry in self.session.entries() {
            let Some(message) = entry.extra_fields().get("message") else {
                continue;
            };
            match message["role"].as_str() {
                Some("user") => user += 1,
                Some("assistant") => {
                    assistant += 1;
                    tool_calls += message["content"].as_array().map_or(0, |blocks| {
                        blocks.iter().filter(|b| b["type"] == "toolCall").count()
                    });
                }
                Some("toolResult") => tool_results += 1,
                _ => {}
            }
        }
        let stats = self.footer_stats();
        format!(
            "Session: {}\nName: {}\nFile: {}\nCreated: {}\nMessages: {user} user, {assistant} assistant, {tool_calls} tool calls, {tool_results} tool results\nTokens: {} input, {} output, {} cache read, {} cache write\nCost: ${:.3}",
            info.id,
            info.name.as_deref().unwrap_or("(unnamed)"),
            info.path.display(),
            info.timestamp,
            stats.input,
            stats.output,
            stats.cache_read,
            stats.cache_write,
            stats.cost,
        )
    }

    /// Resolved startup configuration, one `key: value` per line (`--verbose`).
    pub fn startup_summary(&self) -> String {
        format!(
            "provider: {}\nmodel: {}\nthinking: {}\nsession: {}\ncwd: {}\ntools: {}\nskills: {}\nprompt templates: {}",
            self.provider_id,
            self.provider.model,
            format!("{:?}", self.thinking).to_lowercase(),
            self.session_path.display(),
            self.cwd.display(),
            if self.no_tools {
                "(none)".to_owned()
            } else {
                self.enabled_tools.join(", ")
            },
            self.skills.skills.len(),
            self.templates.len(),
        )
    }

    /// Display name for a model id (`name` in `models.json`, else the id).
    pub fn model_display_name(&self, model: &str) -> String {
        configured_model_entry(&self.agent_dir, &self.provider_id, model)
            .and_then(|entry| entry.get("name").and_then(Value::as_str).map(str::to_owned))
            .unwrap_or_else(|| model.to_owned())
    }

    /// Usage totals and context window for the footer.
    pub fn footer_stats(&self) -> FooterStats {
        let mut stats = session_footer_stats(&self.session);
        stats.context_window =
            configured_model_entry(&self.agent_dir, &self.provider_id, &self.provider.model)
                .and_then(|entry| entry.get("contextWindow").and_then(Value::as_u64))
                .unwrap_or(0);
        stats
    }

    /// Save a project-trust decision for the working directory (`/trust`).
    pub fn set_project_trust(&self, trusted: bool) -> Result<(), AgentError> {
        crate::mcp::save_project_trust(&self.agent_dir, &self.cwd, trusted)
            .map_err(AgentError::Configuration)
    }

    /// Store an API key for `provider` in `auth.json` (0600, atomic).
    pub fn login_api_key(&self, provider: &str, key: &str) -> Result<(), AgentError> {
        if provider.trim().is_empty() || key.trim().is_empty() {
            return Err(AgentError::Configuration(
                "provider and API key must not be empty".into(),
            ));
        }
        crate::auth::set_api_key(
            &self.agent_dir.join("auth.json"),
            provider.trim(),
            key.trim(),
        )
        .map_err(|error| AgentError::Configuration(error.to_string()))
    }

    /// Remove a provider's stored credential from `auth.json`; returns whether one existed.
    pub fn logout(&self, provider: &str) -> Result<bool, AgentError> {
        crate::auth::remove(&self.agent_dir.join("auth.json"), provider)
            .map_err(|error| AgentError::Configuration(error.to_string()))
    }

    /// Effective layered settings, read fresh from disk.
    pub fn settings(&self) -> crate::settings::Settings {
        crate::settings::Settings::load(&self.agent_dir, &self.cwd, true)
    }

    /// Flip a boolean setting in the global `settings.json`; returns the new value.
    pub fn toggle_setting(&self, key: &str) -> Result<bool, AgentError> {
        let mut settings = self.settings();
        let next = !settings.get(key).and_then(|v| v.as_bool()).unwrap_or(false);
        settings
            .set(crate::settings::Scope::Global, key, Value::Bool(next))
            .map_err(|error| AgentError::Configuration(error.to_string()))?;
        Ok(next)
    }

    /// Set an enum or number setting from text (`/settings <key> <value>`), validated like Pi's
    /// schema for the keys Mu exposes. Returns the stored value.
    pub fn set_setting_text(&self, key: &str, text: &str) -> Result<Value, AgentError> {
        let value =
            crate::settings::parse_setting_value(key, text).map_err(AgentError::Configuration)?;
        self.settings()
            .set(crate::settings::Scope::Global, key, value.clone())
            .map_err(|error| AgentError::Configuration(error.to_string()))?;
        Ok(value)
    }

    /// The `codemode.inlineBudget` setting (Pi default 3000 estimated tokens).
    pub(crate) fn codemode_inline_budget(&self) -> Option<usize> {
        self.settings()
            .get("codemode.inlineBudget")
            .and_then(|value| value.as_u64())
            .map(|budget| budget as usize)
    }

    /// `codemode.mode` is `only` (Pi's default is `on`).
    pub(crate) fn codemode_only(&self) -> bool {
        self.settings()
            .get("codemode.mode")
            .as_ref()
            .and_then(Value::as_str)
            == Some("only")
    }

    pub fn agent_dir(&self) -> &Path {
        &self.agent_dir
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn session_name(&self) -> Option<&str> {
        self.session.session_name()
    }

    /// `compaction.{enabled,reserveTokens,keepRecentTokens}` from settings (project overrides agent).
    pub fn compaction_settings(&self) -> crate::compaction::CompactionSettings {
        let mut settings = crate::compaction::CompactionSettings::default();
        let read = |key: &str| {
            configured_setting(&self.agent_dir, &self.cwd, "compaction", |value| {
                value.get(key).cloned()
            })
        };
        if let Some(enabled) = read("enabled").and_then(|v| v.as_bool()) {
            settings.enabled = enabled;
        }
        if let Some(tokens) = read("reserveTokens").and_then(|v| v.as_u64()) {
            settings.reserve_tokens = tokens;
        }
        if let Some(tokens) = read("keepRecentTokens").and_then(|v| v.as_u64()) {
            settings.keep_recent_tokens = tokens;
        }
        settings
    }

    /// Whether the last response left the context past the auto-compaction threshold.
    pub fn needs_auto_compaction(&self) -> bool {
        let stats = self.footer_stats();
        match stats.context_tokens {
            Some(tokens) if stats.context_window > 0 => crate::compaction::should_compact(
                tokens,
                stats.context_window,
                &self.compaction_settings(),
            ),
            _ => false,
        }
    }

    /// Summarize older history, append a `compaction` entry, and rebuild the transcript.
    ///
    /// Returns `Ok(None)` when there is nothing before the cut point to compact.
    pub async fn compact(&mut self) -> Result<Option<CompactionOutcome>, AgentError> {
        self.compact_with(None).await
    }

    /// Compact with optional custom focus instructions (`/compact <instructions>`).
    pub async fn compact_with(
        &mut self,
        custom_instructions: Option<&str>,
    ) -> Result<Option<CompactionOutcome>, AgentError> {
        let settings = self.compaction_settings();
        let branch = branch_entry_values(&self.session);
        // Pi summarizes the projected context: a previous compaction's summary is updated with
        // the entries that followed it rather than re-summarizing from the start.
        let path = crate::compaction::build_context_entries(&branch);
        let previous_summary = path
            .first()
            .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
            .and_then(|entry| entry.get("summary")?.as_str().map(str::to_owned));
        let boundary_start = usize::from(previous_summary.is_some());
        let cut = crate::compaction::find_cut_point(
            &path,
            boundary_start,
            path.len(),
            settings.keep_recent_tokens,
        );
        let Some(first_kept_id) = path
            .get(cut.first_kept_entry_index)
            .and_then(|entry| entry.get("id")?.as_str().map(str::to_owned))
        else {
            return Ok(None);
        };
        // A cut inside a turn summarizes the earlier turns and the turn's prefix separately.
        let (history_end, prefix) = match (cut.is_split_turn, cut.turn_start_index) {
            (true, Some(turn_start)) => (
                turn_start.max(boundary_start),
                &path[turn_start.max(boundary_start)..cut.first_kept_entry_index],
            ),
            _ => (cut.first_kept_entry_index.max(boundary_start), &path[0..0]),
        };
        let summarized = &path[boundary_start..history_end];
        let has_messages = |entries: &[Value]| entries.iter().any(|e| e.get("message").is_some());
        if !has_messages(summarized) && !has_messages(prefix) {
            return Ok(None);
        }
        let tokens_before = path
            .iter()
            .filter_map(|entry| entry.get("message"))
            .map(crate::compaction::estimate_tokens)
            .sum();
        let summary = if has_messages(prefix) {
            let history_text = if has_messages(summarized) {
                self.summarize(
                    &crate::compaction::serialize_conversation(summarized),
                    previous_summary.as_deref(),
                    custom_instructions,
                )
                .await?
            } else {
                previous_summary
                    .clone()
                    .unwrap_or_else(|| "No prior history.".to_owned())
            };
            let prefix_text = self
                .summarize_text(format!(
                    "# Conversation\n{}\n\n# Instructions\n{}",
                    crate::compaction::serialize_conversation(prefix),
                    crate::compaction::TURN_PREFIX_SUMMARIZATION_PROMPT
                ))
                .await?;
            format!("{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_text}")
        } else {
            self.summarize(
                &crate::compaction::serialize_conversation(summarized),
                previous_summary.as_deref(),
                custom_instructions,
            )
            .await?
        };
        let mut file_ops = crate::compaction::FileOperations::default();
        if let Some(previous) = &previous_summary {
            file_ops.add_previous_summary(previous);
        }
        file_ops.add_entries(summarized);
        file_ops.add_entries(prefix);
        let summary = format!("{summary}{}", file_ops.format());
        self.session
            .append_compaction(&summary, &first_kept_id, tokens_before);
        self.save_session()?;
        self.transcript = runtime::session_messages(&self.session);
        Ok(Some(CompactionOutcome {
            summary,
            tokens_before,
        }))
    }

    /// One summarization call over `conversation` (Pi's `generateSummary`).
    async fn summarize(
        &mut self,
        conversation: &str,
        previous_summary: Option<&str>,
        custom_instructions: Option<&str>,
    ) -> Result<String, AgentError> {
        let previous = previous_summary
            .map(|summary| format!("<previous-summary>\n{summary}\n</previous-summary>\n\n"))
            .unwrap_or_default();
        self.summarize_text(format!(
            "<conversation>\n{conversation}\n</conversation>\n\n{previous}{}",
            crate::compaction::summarization_prompt_for(
                custom_instructions,
                previous_summary.is_some()
            )
        ))
        .await
    }

    async fn summarize_text(&mut self, prompt: String) -> Result<String, AgentError> {
        let history = vec![json!({
            "role":"user",
            "content":[{"type":"input_text","text":prompt}],
        })];
        let reply = self.provider.complete(&history, &[], &mut |_| {}).await?;
        if reply.incomplete_reason.is_some() || reply.text.trim().is_empty() {
            return Err(AgentError::Provider(
                "Summarization did not complete; the session was not changed".into(),
            ));
        }
        Ok(reply.text)
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
        render_session_html(&self.session)
    }

    pub fn import_session(&mut self, path: &Path) -> Result<(), AgentError> {
        let (session, history) = load_session(path)?;
        if let Some(message) = missing_session_cwd_error(&session, path, &self.cwd) {
            return Err(AgentError::Session(message));
        }
        self.session = session;
        self.transcript = history;
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
        self.transcript = runtime::session_messages(&self.session);
        self.session_path = new_path;
        self.active_mcp_tools = session_active_mcp_tools(&self.session);
        self.prompt_queue.clear();
        Ok(())
    }

    /// Every message entry as a tree row (`depth` = distance from the root), oldest first.
    pub fn tree_rows(&self) -> Vec<TreeRow> {
        let active: HashSet<String> = self
            .session
            .active_branch_entries()
            .iter()
            .filter_map(|entry| entry.id.as_ref().map(|id| id.as_str().to_owned()))
            .collect();
        let leaf = self.session.leaf_id().map(|id| id.as_str().to_owned());
        let mut depths: HashMap<String, usize> = HashMap::new();
        let mut rows = Vec::new();
        for entry in self.session.entries() {
            let Some(id) = entry.id.as_ref().map(|id| id.as_str().to_owned()) else {
                continue;
            };
            let depth = match &entry.parent_id {
                mu_session::ParentReference::Entry(parent) => {
                    depths.get(parent.as_str()).map_or(0, |d| d + 1)
                }
                _ => 0,
            };
            depths.insert(id.clone(), depth);
            let Some(message) = entry
                .extra_fields()
                .get("message")
                .and_then(Value::as_object)
            else {
                continue;
            };
            let role = message.get("role").and_then(Value::as_str).unwrap_or("?");
            let text = message_blocks(message)
                .iter()
                .filter_map(|block| block.get("text")?.as_str().map(str::to_owned))
                .collect::<Vec<_>>()
                .join(" ");
            let summary = text.split_whitespace().collect::<Vec<_>>().join(" ");
            rows.push(TreeRow {
                on_branch: active.contains(&id),
                is_leaf: leaf.as_deref() == Some(id.as_str()),
                label: format!("{role}: {}", summary.chars().take(60).collect::<String>()),
                id,
                depth,
            });
        }
        rows
    }

    /// Move the session tip to `entry_id`; the next message branches from there (Pi's `/tree`).
    pub fn navigate_to(&mut self, entry_id: &str) -> Result<(), AgentError> {
        if !self.session.set_leaf(entry_id) {
            return Err(AgentError::Session("No such entry".into()));
        }
        self.transcript = runtime::session_messages(&self.session);
        self.prompt_queue.clear();
        Ok(())
    }

    /// Switch to `entry_id` and leave a summary of the branch being abandoned (Pi's
    /// `navigateTree` with `summarize`). With nothing to summarize it just navigates.
    pub async fn navigate_with_summary(
        &mut self,
        entry_id: &str,
        custom_instructions: Option<&str>,
    ) -> Result<Option<String>, AgentError> {
        let old_leaf = self.session.leaf_id().map(|id| id.as_str().to_owned());
        let old_branch = branch_entry_values(&self.session);
        self.navigate_to(entry_id)?;
        let kept = branch_entry_values(&self.session)
            .iter()
            .filter_map(|e| e.get("id")?.as_str().map(str::to_owned))
            .collect::<std::collections::HashSet<_>>();
        let abandoned = old_branch
            .into_iter()
            .filter(|e| {
                e.get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| !kept.contains(id))
            })
            .collect::<Vec<_>>();
        let Some(from_id) =
            old_leaf.filter(|_| abandoned.iter().any(|e| e.get("message").is_some()))
        else {
            return Ok(None);
        };
        let instructions = match custom_instructions.map(str::trim).filter(|c| !c.is_empty()) {
            Some(custom) => format!(
                "{}\n\nAdditional focus: {custom}",
                crate::compaction::BRANCH_SUMMARY_PROMPT
            ),
            None => crate::compaction::BRANCH_SUMMARY_PROMPT.to_owned(),
        };
        let prompt = format!(
            "<conversation>\n{}\n</conversation>\n\n{instructions}",
            crate::compaction::serialize_conversation(&abandoned)
        );
        let text = match self.summarize_text(prompt).await {
            Ok(text) => text,
            Err(error) => return Err(error),
        };
        let mut file_ops = crate::compaction::FileOperations::default();
        file_ops.add_entries(&abandoned);
        let summary = format!(
            "{}{text}{}",
            crate::compaction::BRANCH_SUMMARY_PREAMBLE,
            file_ops.format()
        );
        self.session.append_branch_summary(&summary, &from_id);
        self.save_session()?;
        self.transcript = runtime::session_messages(&self.session);
        Ok(Some(summary))
    }

    /// User messages on the active branch, oldest first, as `(entry id, text)`.
    pub fn user_messages(&self) -> Vec<(String, String)> {
        branch_entry_values(&self.session)
            .into_iter()
            .filter_map(|entry| {
                let id = entry.get("id")?.as_str()?.to_owned();
                let message = entry.get("message")?.as_object()?;
                if message.get("role")?.as_str()? != "user" {
                    return None;
                }
                let text = message_blocks(message)
                    .iter()
                    .filter_map(|b| b.get("text")?.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
                    .join("\n");
                (!text.is_empty()).then_some((id, text))
            })
            .collect()
    }

    /// Fork into a new session holding everything before the user message `entry_id`.
    /// Returns that message's text so it can be edited and resubmitted (Pi's `/fork`).
    pub fn fork_before_user_message(&mut self, entry_id: &str) -> Result<String, AgentError> {
        let text = self
            .user_messages()
            .into_iter()
            .find(|(id, _)| id == entry_id)
            .map(|(_, text)| text)
            .ok_or_else(|| AgentError::Session("No such user message".into()))?;
        let truncated = self
            .session
            .truncated_before(entry_id)
            .ok_or_else(|| AgentError::Session("No such user message".into()))?;
        let fork = truncated
            .fork_with_id(
                self.cwd.to_string_lossy().into_owned(),
                uuid::Uuid::new_v4().to_string(),
                self.session_path.to_string_lossy().into_owned(),
            )
            .ok_or_else(|| AgentError::Session("Current session has no header".into()))?;
        let path = new_session_path(&self.agent_dir, &self.cwd, None);
        if self.persist_session {
            write_session(&fork, &path)?;
        }
        self.session = fork;
        self.transcript = runtime::session_messages(&self.session);
        self.session_path = path;
        self.active_mcp_tools = session_active_mcp_tools(&self.session);
        self.prompt_queue.clear();
        Ok(text)
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
        self.transcript.clear();
        self.session_path = next_path;
        self.active_mcp_tools.clear();
        self.prompt_queue.clear();
        Ok(())
    }

    /// Re-discover skills and prompt templates from disk (`/reload`); returns a one-line summary.
    ///
    /// Skill descriptions already in the system prompt stay until the next session start.
    pub fn reload_resources(&mut self) -> String {
        let resources = self.resources.clone();
        self.skills = crate::skills::SkillCatalog::discover_with_paths(
            &self.cwd,
            &self.agent_dir,
            &resources.skill_paths,
            !resources.no_skills,
        );
        self.templates = crate::prompt_templates::load_prompt_templates(
            &self.cwd,
            &self.agent_dir,
            &resources.template_paths,
            !resources.no_templates,
        );
        format!(
            "Reloaded: {} skills, {} prompt templates",
            self.skills.skills.len(),
            self.templates.len()
        )
    }

    /// Prompt templates, then `skill:<name>` commands: `(name, description, argument hint)`.
    pub fn dynamic_slash_commands(&self) -> Vec<(String, String, Option<String>)> {
        let templates = self.templates.iter().map(|template| {
            (
                template.name.clone(),
                template.description.clone(),
                template.argument_hint.clone(),
            )
        });
        let skills = skill_commands_enabled(&self.agent_dir, &self.cwd)
            .then(|| self.skills.skills.iter())
            .into_iter()
            .flatten()
            .map(|skill| {
                (
                    format!("skill:{}", skill.name),
                    skill.description.clone(),
                    None,
                )
            });
        templates.chain(skills).collect()
    }

    /// Skills discovered from Pi's global and project skill directories.
    pub fn skills(&self) -> &crate::skills::SkillCatalog {
        &self.skills
    }

    fn expand_prompt(&self, prompt: String, on_event: &mut impl FnMut(AgentEvent)) -> String {
        let prompt = crate::prompt_templates::expand_prompt_template(&prompt, &self.templates);
        if !skill_commands_enabled(&self.agent_dir, &self.cwd) {
            return prompt;
        }
        expand_skill_prompt(prompt, &self.skills, on_event)
    }

    /// The `theme` setting (project settings override agent settings), if any.
    pub fn theme_name(&self) -> Option<String> {
        if let Some(name) = &self.theme_override {
            return Some(name.clone());
        }
        configured_setting(&self.agent_dir, &self.cwd, "theme", |value| {
            value.as_str().map(str::to_owned)
        })
    }

    pub fn hide_thinking_block(&self) -> bool {
        self.hide_thinking_block
    }

    /// Run a built-in tool directly in the project directory (the TUI's `!command`).
    pub async fn run_local_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<crate::tools::ToolResult, crate::tools::ToolError> {
        crate::tools::execute_with_environment(
            &self.cwd,
            name,
            arguments,
            &self.shell_environment(),
        )
        .await
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

    fn codemode_store(&self) -> Value {
        session_codemode_store(&self.session)
    }

    /// Provider-facing tool definitions for the current selection.
    #[cfg(test)]
    fn tool_definitions(&self, active_mcp_tools: &HashSet<String>) -> Vec<Value> {
        let plan = runtime::ToolPlan {
            enabled_tools: self.enabled_tools.clone(),
            no_tools: self.no_tools,
            selection: self.tool_selection.clone(),
            excluded: self.excluded_tools.clone(),
            codemode_inline_budget: self.codemode_inline_budget(),
            codemode_only: self.codemode_only(),
        };
        plan.definitions(&self.mcp, active_mcp_tools)
            .iter()
            .filter_map(runtime::declaration_from_definition)
            .map(|declaration| {
                runtime::request_tool(&declaration, self.provider.supports_strict_sampling)
            })
            .collect()
    }

    /// Run one user turn; queued steering and follow-up prompts continue it.
    pub async fn submit(
        &mut self,
        prompt: String,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<(), AgentError> {
        self.submit_with_cancel(prompt, on_event, &CancellationToken::new())
            .await
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
        vec![prompt],
        Vec::new(),
        SessionOptions {
            path: session_file,
            ..SessionOptions::default()
        },
    )
    .await
}

/// Run one prompt and print every agent event as a JSON line (Pi's `--mode json`).
pub async fn run_prompt_json(
    prompts: Vec<String>,
    images: Vec<(String, String)>,
    session_options: SessionOptions,
) -> Result<(), AgentError> {
    let mut agent = MuAgent::connect_with_options(session_options).await?;
    agent.set_event_tap(Arc::new(|event| {
        println!("{}", crate::agent_loop::event_to_json(event));
    }));
    let mut images = Some(images);
    for prompt in prompts {
        agent
            .submit_with_images(
                prompt,
                images.take().unwrap_or_default(),
                &mut |_| {},
                &CancellationToken::new(),
            )
            .await?;
    }
    Ok(())
}

pub async fn run_prompt_with_options(
    prompts: Vec<String>,
    images: Vec<(String, String)>,
    session_options: SessionOptions,
) -> Result<(), AgentError> {
    let verbose = session_options.verbose;
    let mut agent = MuAgent::connect_with_options(session_options).await?;
    if verbose {
        eprintln!("{}", agent.startup_summary());
    }
    let mut images = Some(images);
    for prompt in prompts {
        agent
            .submit_with_images(prompt, images.take().unwrap_or_default(), &mut |event| {
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
            }, &CancellationToken::new())
            .await?;
    }
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

    use super::runtime::{responses_input, session_messages};
    use super::{
        AgentPromptQueue, FunctionCall, PromptBehavior, QueueMode, Reply, TerminalResponse,
        configured_models, find_session_id_path, latest_session_path, load_session,
        merge_final_response_text, resolve_session_options, response_message_text,
        session_active_mcp_tools, session_codemode_store, session_preview, session_transcript,
        trusted_url, validate_session_id,
    };
    use super::{Config, SessionOptions, ThinkingSetting};
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
    fn mcp_tool_summaries_show_first_description_line() {
        assert_eq!(
            super::format_mcp_tool_summary(&serde_json::json!({
                "name":"mcp__fs__read",
                "description":"Read a file.\nReturns the content."
            }))
            .as_deref(),
            Some("mcp__fs__read: Read a file.")
        );
        assert_eq!(
            super::format_mcp_tool_summary(&serde_json::json!({"name":"mcp__fs__empty"}))
                .as_deref(),
            Some("mcp__fs__empty")
        );
        assert_eq!(
            super::format_mcp_tool_summary(&serde_json::json!({"description":"missing name"})),
            None
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
            "provider: Error Code undefined: undefined"
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
        let history = responses_input(&history);
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
        let history = responses_input(&history);
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

        let history = responses_input(&session_messages(&session));
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
        agent.provider.supports_strict_sampling = false;
        let unsupported_definitions = agent.tool_definitions(&HashSet::new());
        let unsupported_read = unsupported_definitions
            .iter()
            .find(|definition| definition["name"] == "read")
            .unwrap();
        assert!(unsupported_read.get("strict").is_none());
        assert_eq!(
            unsupported_read["parameters"]["required"],
            serde_json::json!(["path"])
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
    fn model_strict_sampling_capability_defaults_to_unsupported_and_honors_opt_in() {
        assert!(!super::model_supports_strict_sampling(&serde_json::json!(
            {}
        )));
        assert!(super::model_supports_strict_sampling(&serde_json::json!({
            "compat":{"supportsStrictMode":true}
        })));
        assert!(!super::model_supports_strict_sampling(&serde_json::json!({
            "compat":{"supportsStrictMode":false}
        })));
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
        assert!(
            events.contains(&super::AgentEvent::ToolFinished {
                call_id: "call-write".into(),
                output: "Successfully wrote to result.txt".into(),
                is_error: false,
                details: events
                    .iter()
                    .find_map(|event| match event {
                        super::AgentEvent::ToolFinished { details, .. } => Some(details.clone()),
                        _ => None,
                    })
                    .flatten(),
                duration_ms: events.iter().find_map(|event| match event {
                    super::AgentEvent::ToolFinished { duration_ms, .. } => *duration_ms,
                    _ => None,
                }),
            })
        );
        assert!(events.contains(&super::AgentEvent::ThinkingDelta("v".into())));
        assert!(events.contains(&super::AgentEvent::TextDelta("s".into())));
        assert_eq!(
            responses_input(&agent.transcript)[1]["type"],
            "function_call"
        );
        assert_eq!(
            responses_input(&agent.transcript)[2]["type"],
            "function_call_output"
        );
        assert_eq!(
            responses_input(&agent.transcript)[2]["output"],
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
            details: None,
            duration_ms: None,
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

        assert_eq!(responses_input(&agent.transcript).len(), 20);
        assert_eq!(
            responses_input(&agent.transcript).last().unwrap()["content"][0]["text"],
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
        assert_eq!(responses_input(&agent.transcript)[0]["role"], "user");
        assert_eq!(responses_input(&agent.transcript)[1]["role"], "assistant");
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
        assert_eq!(responses_input(&agent.transcript).len(), 1);
        assert_eq!(responses_input(&agent.transcript)[0]["role"], "user");
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

    fn scripted_reply(text: &str, calls: Vec<(&str, &str, Value)>) -> Reply {
        Reply {
            text: text.into(),
            thinking: String::new(),
            thinking_signature: None,
            output_items: Vec::new(),
            tool_calls: calls
                .into_iter()
                .map(|(call_id, name, arguments)| FunctionCall {
                    call_id: call_id.into(),
                    name: name.into(),
                    arguments,
                })
                .collect(),
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            reasoning: None,
            total_tokens: 0,
            incomplete_reason: None,
        }
    }

    fn scripted_agent(root: &Path, replies: Vec<Result<Reply, String>>) -> super::MuAgent {
        std::fs::create_dir_all(root).expect("create temporary project");
        let mut agent = super::MuAgent::demo_from_config(super::Config {
            agent_dir: root.join("agent"),
            cwd: root.to_owned(),
            provider: "test-provider".into(),
            model: "test-model".into(),
            model_choices: vec!["test-model".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: super::resolve_default_tools(None),
        })
        .expect("demo agent");
        agent.provider.transport = super::ProviderTransport::Scripted(std::sync::Arc::new(
            std::sync::Mutex::new(replies.into()),
        ));
        agent
    }

    #[test]
    fn prompt_inputs_read_files_or_fall_back_to_literal_text() {
        let dir = std::env::temp_dir().join(format!("mu-prompt-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("extra.md");
        std::fs::write(&file, "from file").unwrap();
        let options = super::PromptOptions::from_session_options(&SessionOptions {
            system_prompt: Some("literal prompt".into()),
            append_system_prompt: vec![file.to_string_lossy().into_owned(), "inline".into()],
            ..SessionOptions::default()
        });
        assert_eq!(options.system_prompt.as_deref(), Some("literal prompt"));
        assert_eq!(options.appendix(), "from file\n\ninline");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn excluded_tools_remove_exact_and_wildcard_matches_after_selection() {
        let root = std::env::temp_dir().join(format!("mu-exclude-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(&root, vec![]);
        agent.enabled_tools = ["read", "bash", "grep", "find"].map(String::from).to_vec();
        agent.excluded_tools = vec!["bash".into(), "gr*".into()];
        let names = agent
            .tool_definitions(&HashSet::new())
            .iter()
            .filter_map(|d| d["name"].as_str().map(str::to_owned))
            .collect::<Vec<_>>();
        assert_eq!(names, ["read", "find"]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn thinking_levels_parse_like_pi_and_map_to_effort() {
        assert_eq!(ThinkingSetting::Max.next(), ThinkingSetting::Off);
        assert_eq!(ThinkingSetting::Medium.next().name(), "high");
        assert_eq!("high".parse::<ThinkingSetting>(), Ok(ThinkingSetting::High));
        assert_eq!(ThinkingSetting::Off.effort(), None);
        assert_eq!(ThinkingSetting::Max.effort(), Some("xhigh"));
        assert_eq!(ThinkingSetting::default().effort(), Some("medium"));
        assert_eq!(
            "loud".parse::<ThinkingSetting>().unwrap_err(),
            "Invalid thinking level \"loud\". Valid values: off, minimal, low, medium, high, xhigh, max"
        );
    }

    #[test]
    fn model_flags_override_settings_and_split_provider_prefixes() {
        let mk = || super::Config {
            agent_dir: std::path::PathBuf::from("/nonexistent"),
            cwd: std::path::PathBuf::from("/"),
            provider: "p".into(),
            model: "m".into(),
            model_choices: vec!["m".into()],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: vec![],
        };
        let mut config = mk();
        super::apply_model_override(&mut config, None, Some("opencode-cf/gpt-x")).unwrap();
        assert_eq!(
            (config.provider.as_str(), config.model.as_str()),
            ("opencode-cf", "gpt-x")
        );
        let mut config = mk();
        super::apply_model_override(&mut config, Some("q"), Some("org/model")).unwrap();
        assert_eq!(
            (config.provider.as_str(), config.model.as_str()),
            ("q", "org/model")
        );
        let mut config = mk();
        super::apply_model_override(&mut config, None, None).unwrap();
        assert_eq!(config.model, "m");
    }

    #[test]
    fn export_session_file_renders_html_without_a_provider() {
        let dir = std::env::temp_dir().join(format!("mu-export-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("s.jsonl");
        let mut session = SessionDocument::new("/tmp/p");
        session.append_user_text("hello <b>there</b>");
        session.write_atomic(&input).unwrap();
        let out = super::export_session_file(&input, Some(&dir.join("o.html"))).unwrap();
        let html = std::fs::read_to_string(out).unwrap();
        assert!(html.contains("hello &lt;b&gt;there&lt;/b&gt;"));
        assert!(super::export_session_file(&dir.join("missing.jsonl"), None).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_api_keys_name_the_provider_and_how_to_fix_it() {
        let message = super::no_api_key_message("acme");
        assert!(message.starts_with("No API key found for acme."));
        assert!(message.contains("/login") && message.contains("models.json"));
    }

    #[test]
    fn missing_model_settings_explain_how_to_log_in_and_pick_a_model() {
        let message = super::no_model_selected_message("Pi settings have no defaultModel");
        assert!(message.starts_with("Pi settings have no defaultModel. No model selected."));
        assert!(message.contains("/login") && message.ends_with("select a model."));
    }

    #[test]
    fn http_failures_include_a_bounded_error_body_so_quota_errors_are_not_retried() {
        let status = reqwest::StatusCode::TOO_MANY_REQUESTS;
        assert_eq!(
            super::http_failure_message(status, ""),
            "model request returned HTTP 429 Too Many Requests"
        );
        let quota = super::http_failure_message(status, r#"{"error":"insufficient_quota"}"#);
        assert!(quota.ends_with(r#"{"error":"insufficient_quota"}"#));
        assert!(!super::retryable_provider_error(
            &crate::AgentError::Provider(quota)
        ));
        let long = super::http_failure_message(status, &"x".repeat(900));
        assert!(long.ends_with('…') && long.len() < 600);
    }

    /// Ported from Pi's `tool-system-prompt-contributions.test.ts`: the snippets and guidelines
    /// are exactly the ones in each tool's `*ToolSystemPromptContribution`.
    #[test]
    fn builtin_tool_prompt_contributions_match_pi() {
        let snippets = [
            ("read", "Read file contents"),
            ("bash", "Execute bash commands (ls, grep, find, etc.)"),
            ("powershell", "Execute PowerShell commands"),
            (
                "edit",
                "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
            ),
            ("write", "Create or overwrite files"),
            (
                "grep",
                "Search file contents for patterns (respects .gitignore)",
            ),
            ("find", "Find files by glob pattern (respects .gitignore)"),
            ("ls", "List directory contents"),
        ];
        for (name, snippet) in snippets {
            assert_eq!(super::tool_prompt_snippet(name), Some(snippet), "{name}");
        }
        let env =
            ["You can inspect PI_* environment variables for current model and session details."];
        assert_eq!(super::tool_prompt_guidelines("bash"), env);
        assert_eq!(super::tool_prompt_guidelines("powershell"), env);
        assert_eq!(
            super::tool_prompt_guidelines("read"),
            ["Use read to examine files instead of cat or sed."]
        );
        assert_eq!(
            super::tool_prompt_guidelines("write"),
            ["Use write only for new files or complete rewrites."]
        );
        assert_eq!(super::tool_prompt_guidelines("edit").len(), 4);
        for name in ["grep", "find", "ls"] {
            assert!(super::tool_prompt_guidelines(name).is_empty(), "{name}");
        }
    }

    #[test]
    fn sessions_with_a_vanished_cwd_are_refused_with_pis_message() {
        let dir = std::env::temp_dir().join(format!("mu-cwd-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("s.jsonl");
        let gone = dir.join("deleted-project");
        let existing = SessionDocument::new(dir.to_string_lossy().into_owned());
        assert_eq!(
            super::missing_session_cwd_error(&existing, &file, &dir),
            None
        );
        let missing = SessionDocument::new(gone.to_string_lossy().into_owned());
        let message = super::missing_session_cwd_error(&missing, &file, &dir).unwrap();
        assert_eq!(
            message,
            format!(
                "Stored session working directory does not exist: {}\nSession file: {}\nCurrent working directory: {}",
                gone.display(),
                file.display(),
                dir.display()
            )
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn importing_a_session_with_a_vanished_cwd_leaves_the_current_session_untouched() {
        let dir = std::env::temp_dir().join(format!("mu-import-cwd-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("s.jsonl");
        let doc = SessionDocument::new(dir.join("gone").to_string_lossy().into_owned());
        std::fs::write(&file, doc.to_jsonl().unwrap()).unwrap();
        let mut agent = super::MuAgent::demo().unwrap();
        let before = agent.session_path().to_owned();
        let error = agent.import_session(&file).unwrap_err().to_string();
        assert!(
            error.contains("Stored session working directory does not exist"),
            "{error}"
        );
        assert_eq!(agent.session_path(), before);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn session_dir_precedence_is_flag_then_env_then_setting() {
        let flag = Path::new("/flag");
        assert_eq!(
            super::configured_session_dir(Some(flag), Some("/env"), Some("/set")),
            Some(std::path::PathBuf::from("/flag"))
        );
        assert_eq!(
            super::configured_session_dir(None, Some("/env"), Some("/set")),
            Some(std::path::PathBuf::from("/env"))
        );
        assert_eq!(
            super::configured_session_dir(None, Some(""), Some("/set")),
            Some(std::path::PathBuf::from("/set"))
        );
        assert_eq!(super::configured_session_dir(None, None, None), None);
        if let Some(home) = std::env::var_os("HOME") {
            assert_eq!(
                super::expand_tilde("~/x"),
                std::path::PathBuf::from(home).join("x")
            );
        }
        assert_eq!(
            super::expand_tilde("/abs"),
            std::path::PathBuf::from("/abs")
        );
    }

    #[test]
    fn branch_summaries_project_into_context_as_a_framed_user_message() {
        let mut session = SessionDocument::new("/w".to_owned());
        session.append_user_text("first");
        session.append_branch_summary("explored X", "abandoned-leaf");
        let messages = super::runtime::session_messages(&session);
        let inputs = super::runtime::responses_input(&messages);
        let text = inputs.last().unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(
            text,
            "The following is a summary of a branch that this conversation came back from:\n\n<summary>\nexplored X</summary>"
        );
        assert_eq!(inputs.len(), 2);
    }

    #[test]
    fn vertex_base_url_is_built_from_project_and_location() {
        let url =
            super::vertex_base_url(&serde_json::json!({"project":"p1","location":"europe-west4"}))
                .unwrap();
        assert_eq!(
            url,
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/p1/locations/europe-west4/publishers/google"
        );
        let global =
            super::vertex_base_url(&serde_json::json!({"project":"p1","location":"global"}))
                .unwrap();
        assert!(
            global
                .starts_with("https://aiplatform.googleapis.com/v1/projects/p1/locations/global/")
        );
        let default = super::vertex_base_url(&serde_json::json!({"project":"p1"})).unwrap();
        assert!(default.contains("us-central1-aiplatform"));
    }

    /// Pi's `transformMessages` orphan handling: unanswered calls get a synthetic error result
    /// before the next turn begins, and answered calls are left alone.
    #[test]
    fn orphaned_tool_calls_get_synthetic_results_before_the_next_turn() {
        let call = |id: &str| serde_json::json!({"type":"function_call","call_id":id,"name":"read","arguments":"{}"});
        let output = |id: &str| serde_json::json!({"type":"function_call_output","call_id":id,"output":"ok"});
        let user =
            serde_json::json!({"role":"user","content":[{"type":"input_text","text":"next"}]});
        let items = super::runtime::close_orphaned_calls(vec![
            call("a"),
            call("b"),
            output("a"),
            user.clone(),
            call("c"),
        ]);
        let shape = items
            .iter()
            .map(|i| {
                format!(
                    "{}:{}",
                    i["type"].as_str().unwrap_or("message"),
                    i["call_id"].as_str().unwrap_or("")
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            shape,
            [
                "function_call:a",
                "function_call:b",
                "function_call_output:a",
                "function_call_output:b",
                "message:",
                "function_call:c",
                "function_call_output:c"
            ]
        );
        assert_eq!(items[3]["output"], "No result provided");
        // A complete history is unchanged.
        let complete = vec![call("x"), output("x"), user];
        assert_eq!(
            super::runtime::close_orphaned_calls(complete.clone()),
            complete
        );
    }

    #[test]
    fn image_tool_outputs_are_replaced_for_text_only_models() {
        let item = serde_json::json!({"type":"function_call_output","call_id":"c","output":[
            {"type":"input_text","text":"Read image file [image/png]"},
            {"type":"input_image","image_url":"data:a"},
            {"type":"input_image","image_url":"data:b"},
            {"type":"input_text","text":"after"}]});
        let out = super::runtime::downgrade_images(&[
            item,
            serde_json::json!({"type":"function_call_output","call_id":"d","output":"plain"}),
        ]);
        let texts = out[0]["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["text"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            texts,
            [
                "Read image file [image/png]",
                "(tool image omitted: model does not support images)",
                "after"
            ]
        );
        assert_eq!(out[1]["output"], "plain");
    }

    #[test]
    fn compaction_and_branch_summaries_appear_in_the_transcript() {
        let mut session = SessionDocument::new("/w".to_owned());
        session.append_user_text("one");
        let kept = session.entries().last().unwrap().id.clone().unwrap();
        session.append_compaction("COMPACTED", kept.as_str(), 77);
        session.append_branch_summary("BRANCHED", "x");
        let transcript = super::session_transcript(&session);
        let roles = transcript.iter().map(|m| m.role).collect::<Vec<_>>();
        assert_eq!(
            roles,
            [
                super::SessionTranscriptRole::User,
                super::SessionTranscriptRole::CompactionSummary(77),
                super::SessionTranscriptRole::BranchSummary
            ]
        );
        assert_eq!(transcript[1].text, "COMPACTED");
    }

    #[test]
    fn user_images_round_trip_through_the_session_into_responses_input() {
        let mut session = SessionDocument::new("/w".to_owned());
        session.append_user_content("see", &[("image/png".to_owned(), "QUJD".to_owned())]);
        let reloaded = SessionDocument::parse_jsonl(&session.to_jsonl().unwrap()).unwrap();
        let input = super::runtime::responses_input(&super::runtime::session_messages(&reloaded));
        assert_eq!(input[0]["content"][0]["text"], "see");
        assert_eq!(input[0]["content"][1]["type"], "input_image");
        assert_eq!(
            input[0]["content"][1]["image_url"],
            "data:image/png;base64,QUJD"
        );
    }

    #[test]
    fn codemode_only_hides_every_declaration_except_tool_search() {
        let tools = || {
            vec![
                serde_json::json!({"name":"read"}),
                serde_json::json!({"name":"mcp__x__y"}),
                serde_json::json!({"name":"tool_search"}),
            ]
        };
        let mut on = tools();
        super::runtime::hide_declarations_for_codemode_only(&mut on, false);
        assert_eq!(on.len(), 3);
        let mut only = tools();
        super::runtime::hide_declarations_for_codemode_only(&mut only, true);
        assert_eq!(only, vec![serde_json::json!({"name":"tool_search"})]);
    }

    #[test]
    fn user_images_reach_every_adapter_in_its_own_format() {
        let user = serde_json::json!({"role":"user","content":[
            {"type":"input_text","text":"look"},
            {"type":"input_image","image_url":"data:image/png;base64,QUJD"}]});
        let anthropic = super::anthropic::messages(std::slice::from_ref(&user));
        assert_eq!(anthropic[0]["content"][1]["source"]["data"], "QUJD");
        let completions = super::completions::messages("sys", std::slice::from_ref(&user));
        assert_eq!(
            completions[1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,QUJD"
        );
        let google = super::google::contents(std::slice::from_ref(&user));
        assert_eq!(google[0]["parts"][1]["inlineData"]["mimeType"], "image/png");
        let bedrock = super::bedrock::messages(std::slice::from_ref(&user));
        assert_eq!(bedrock[0]["content"][1]["image"]["format"], "png");
        let down = super::runtime::downgrade_images(&[user]);
        assert_eq!(
            down[0]["content"][1]["text"],
            "(image omitted: model does not support images)"
        );
    }

    #[test]
    fn models_accept_images_unless_input_excludes_them() {
        assert!(super::model_accepts_images(&serde_json::json!({})));
        assert!(super::model_accepts_images(
            &serde_json::json!({"input":["text","image"]})
        ));
        assert!(!super::model_accepts_images(
            &serde_json::json!({"input":["text"]})
        ));
    }

    #[test]
    fn bedrock_region_comes_from_the_runtime_host() {
        assert_eq!(
            super::bedrock_region_from_host("bedrock-runtime.eu-west-1.amazonaws.com").as_deref(),
            Some("eu-west-1")
        );
        assert_eq!(super::bedrock_region_from_host("example.com"), None);
    }

    #[tokio::test]
    async fn provider_headers_are_resolved_and_sent_on_every_request() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0; 4096];
            let read = socket.read(&mut buffer).await.unwrap();
            socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                .await
                .unwrap();
            String::from_utf8_lossy(&buffer[..read]).to_lowercase()
        });
        let headers = serde_json::json!({"X-Static":"one","X-Command":"!echo two","X-Unset":"$MU_SURELY_UNSET_VAR"});
        let client = super::provider_http_client(Some(&headers)).unwrap();
        client
            .get(format!("http://127.0.0.1:{port}/"))
            .send()
            .await
            .unwrap();
        let request = server.await.unwrap();
        assert!(request.contains("x-static: one"), "{request}");
        assert!(request.contains("x-command: two"), "{request}");
        assert!(!request.contains("x-unset"), "{request}");
        assert!(super::provider_http_client(Some(&serde_json::json!({"bad name":"x"}))).is_err());
    }

    #[test]
    fn custom_providers_come_from_models_json_without_gateway_headers() {
        let dir = std::env::temp_dir().join(format!("mu-custom-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("models.json"),
            r#"{"providers":{"acme":{"api":"openai-responses","baseUrl":"https://api.acme.test/v1","apiKey":"literal-key","models":[{"id":"m1","reasoning":true}]},"bad":{"api":"google-genai","baseUrl":"x","apiKey":"k"}}}"#,
        )
        .unwrap();
        let config = |provider: &str| super::Config {
            agent_dir: dir.clone(),
            cwd: dir.clone(),
            provider: provider.into(),
            model: "m1".into(),
            model_choices: vec![],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: vec![],
        };
        let prompt = super::PromptOptions::default();
        let provider =
            super::connect_custom_provider(&config("acme"), String::new(), &prompt).unwrap();
        let super::ProviderTransport::OpenCode {
            endpoint,
            token,
            gateway_headers,
            ..
        } = provider.transport
        else {
            panic!("expected an HTTP transport");
        };
        assert_eq!(endpoint.as_str(), "https://api.acme.test/v1/responses");
        assert_eq!(token, "literal-key");
        assert!(!gateway_headers && provider.supports_reasoning);
        let error = super::connect_custom_provider(&config("bad"), String::new(), &prompt)
            .err()
            .unwrap();
        assert!(error.to_string().contains("unsupported api `google-genai`"));
        assert!(super::connect_custom_provider(&config("nope"), String::new(), &prompt).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn api_key_flag_overrides_the_models_json_key() {
        let dir = std::env::temp_dir().join(format!("mu-apikey-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("models.json"),
            r#"{"providers":{"acme":{"api":"openai-responses","baseUrl":"https://a.test/v1","apiKey":"from-file","models":[{"id":"m"}]}}}"#,
        )
        .unwrap();
        let config = super::Config {
            agent_dir: dir.clone(),
            cwd: dir.clone(),
            provider: "acme".into(),
            model: "m".into(),
            model_choices: vec![],
            hide_thinking_block: false,
            retry: super::RetryPolicy::default(),
            enabled_tools: vec![],
        };
        let prompt = super::PromptOptions::from_session_options(&SessionOptions {
            api_key: Some("runtime".into()),
            ..SessionOptions::default()
        });
        let provider = super::connect_custom_provider(&config, String::new(), &prompt).unwrap();
        let super::ProviderTransport::OpenCode { token, .. } = provider.transport else {
            panic!()
        };
        assert_eq!(token, "runtime");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn logout_removes_the_stored_credential() {
        let root = std::env::temp_dir().join(format!("mu-logout-{}", uuid::Uuid::new_v4()));
        let agent = scripted_agent(&root, vec![]);
        let auth = root.join("agent/auth.json");
        crate::auth::set_api_key(&auth, "acme", "k").unwrap();
        assert!(agent.logout("acme").unwrap());
        assert!(!agent.logout("acme").unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fork_before_a_user_message_truncates_history_and_returns_its_text() {
        let root = std::env::temp_dir().join(format!("mu-fork-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(&root, vec![]);
        agent.session.append_user_text("first");
        agent.session.append_user_text("second");
        agent.session.append_user_text("third");
        let messages = agent.user_messages();
        assert_eq!(messages.len(), 3);
        let old_path = agent.session_path().to_owned();
        let text = agent.fork_before_user_message(&messages[1].0).unwrap();
        assert_eq!(text, "second");
        assert_eq!(agent.user_messages().len(), 1);
        assert_ne!(agent.session_path(), old_path);
        let saved = std::fs::read_to_string(agent.session_path()).unwrap();
        assert!(saved.contains("parentSession") && !saved.contains("second"));
        assert!(agent.fork_before_user_message("nope").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn strict_schemas_reject_unsupported_shapes_like_pi() {
        let strict = |schema: Value| super::strict_json_schema(&schema);
        let ok = strict(serde_json::json!({"type":"object","properties":{"a":{"type":"string"},"b":{"type":"number"}},"required":["a"]})).unwrap();
        assert_eq!(ok["required"], serde_json::json!(["a", "b"]));
        assert_eq!(ok["additionalProperties"], false);
        assert!(ok["properties"]["b"]["anyOf"].is_array());
        for (schema, message) in [
            (
                serde_json::json!({"type":"string"}),
                "root schema must have type object",
            ),
            (
                serde_json::json!({"type":"object","oneOf":[{}]}),
                "oneOf schemas are unsupported",
            ),
            (
                serde_json::json!({"type":"object","additionalProperties":true}),
                "schema-valued or true additionalProperties is unsupported",
            ),
            (
                serde_json::json!({"type":"object","properties":{"t":{"type":"array","items":[{}]}}}),
                "tuple schemas are unsupported",
            ),
            (
                serde_json::json!({"type":"object","properties":{"u":{"anyOf":[{"type":"object"},{"type":"null"}]}}}),
                "object and array unions are unsupported",
            ),
            (
                serde_json::json!({"type":"object","required":["zz"]}),
                "required contains an unknown property",
            ),
            (
                serde_json::json!({"type":"object","properties":{"p":true}}),
                "boolean schemas are unsupported",
            ),
        ] {
            assert_eq!(strict(schema).unwrap_err(), message);
        }
        // An unsupported schema leaves the tool non-strict instead of failing the request.
        let tool = serde_json::json!({"name":"read","parameters":{"type":"object","oneOf":[{}]}});
        assert!(
            super::mark_strict_builtin_tool(tool)
                .get("strict")
                .is_none()
        );
    }

    #[test]
    fn tree_navigation_branches_new_messages_from_the_chosen_entry() {
        let root = std::env::temp_dir().join(format!("mu-tree-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(&root, vec![]);
        agent.session.append_user_text("one");
        agent.session.append_user_text("two");
        let rows = agent.tree_rows();
        assert_eq!(rows.len(), 2);
        assert!(rows[1].is_leaf && rows[1].on_branch && rows[1].depth == 1);
        agent.navigate_to(&rows[0].id).unwrap();
        assert_eq!(agent.user_messages().len(), 1);
        agent.session.append_user_text("alt");
        let rows = agent.tree_rows();
        assert_eq!(rows.len(), 3);
        assert!(
            !rows[1].on_branch,
            "the abandoned branch is shown but off-branch"
        );
        assert_eq!(rows[2].depth, 1);
        assert!(agent.navigate_to("nope").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn session_summary_counts_messages_and_tokens() {
        let root = std::env::temp_dir().join(format!("mu-summary-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(&root, vec![]);
        agent.session.append_user_text("hi");
        agent.session.append_assistant_response(
            "a",
            "p",
            "m",
            mu_session::AssistantTurnResponse {
                thinking: String::new(),
                thinking_signature: None,
                text: "yo".into(),
                usage: mu_session::MessageUsage {
                    input: 7,
                    output: 3,
                    ..Default::default()
                },
            },
        );
        let summary = agent.session_summary();
        assert!(
            summary.contains("Messages: 1 user, 1 assistant, 0 tool calls, 0 tool results"),
            "{summary}"
        );
        assert!(summary.contains("Tokens: 7 input, 3 output"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn enabled_models_scope_the_cycle_list() {
        let choices = ["gpt-a", "gpt-b", "claude-x"].map(String::from).to_vec();
        let pats = |p: &[&str]| p.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(super::scope_models(&[], &choices, "p"), choices);
        assert_eq!(
            super::scope_models(&pats(&["claude-*"]), &choices, "p"),
            ["claude-x"]
        );
        assert_eq!(
            super::scope_models(&pats(&["P/GPT-B:high", "gpt-a"]), &choices, "p"),
            ["gpt-b", "gpt-a"]
        );
        assert_eq!(
            super::scope_models(&pats(&["nomatch"]), &choices, "p"),
            choices
        );
    }

    #[test]
    fn reload_picks_up_new_prompt_templates() {
        let root = std::env::temp_dir().join(format!("mu-reload-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(&root, vec![]);
        assert!(agent.reload_resources().contains("0 prompt templates"));
        let prompts = root.join(".pi/prompts");
        std::fs::create_dir_all(&prompts).unwrap();
        std::fs::write(prompts.join("hi.md"), "Hello $1").unwrap();
        assert!(agent.reload_resources().contains("1 prompt templates"));
        assert_eq!(
            crate::prompt_templates::expand_prompt_template("/hi bob", &agent.templates),
            "Hello bob"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn startup_summary_lists_the_resolved_configuration() {
        let root = std::env::temp_dir().join(format!("mu-verbose-{}", uuid::Uuid::new_v4()));
        let agent = scripted_agent(&root, vec![]);
        let summary = agent.startup_summary();
        assert!(
            summary.starts_with("provider: test-provider\nmodel: test-model\nthinking: medium")
        );
        assert!(summary.contains("tools: read, bash, edit, write"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn queue_modes_drain_one_or_all_queued_messages() {
        let queue = AgentPromptQueue::default();
        for prompt in ["one", "two", "three"] {
            queue.enqueue(prompt.into(), PromptBehavior::Steer);
            queue.enqueue(prompt.into(), PromptBehavior::FollowUp);
        }
        assert_eq!(
            queue.take_steering_batch(super::QueueMode::OneAtATime),
            ["one"]
        );
        assert_eq!(
            queue.take_steering_batch(super::QueueMode::All),
            ["two", "three"]
        );
        assert!(queue.take_steering_batch(super::QueueMode::All).is_empty());
        assert_eq!(queue.take_follow_ups(super::QueueMode::OneAtATime), ["one"]);
        assert_eq!(
            queue.take_follow_ups(super::QueueMode::All),
            ["two", "three"]
        );
        assert_eq!(
            super::QueueMode::from_setting("all"),
            Some(super::QueueMode::All)
        );
        assert_eq!(super::QueueMode::from_setting("bogus"), None);
        assert_eq!(super::QueueMode::default(), super::QueueMode::OneAtATime);
    }

    #[tokio::test]
    async fn compact_summarizes_old_history_and_rebuilds_the_transcript() {
        let root = std::env::temp_dir().join(format!("mu-compact-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(&root, vec![Ok(scripted_reply("## Goal\nship it", vec![]))]);
        for index in 0..6 {
            agent
                .session
                .append_user_text(format!("question {index} {}", "x".repeat(20_000)));
            agent.session.append_assistant_response(
                "openai-responses",
                "p",
                "m",
                mu_session::AssistantTurnResponse {
                    thinking: String::new(),
                    thinking_signature: None,
                    text: format!("answer {index} {}", "y".repeat(20_000)),
                    usage: mu_session::MessageUsage::default(),
                },
            );
        }
        let outcome = agent
            .compact()
            .await
            .unwrap()
            .expect("something to compact");
        assert_eq!(outcome.summary, "## Goal\nship it");
        assert!(outcome.tokens_before > 20_000);

        let input = responses_input(&agent.transcript);
        let first = input[0]["content"][0]["text"].as_str().unwrap();
        assert!(first.starts_with("The conversation history before this point was compacted"));
        assert!(first.contains("ship it"));
        assert!(
            input.len() < 12,
            "old messages were dropped: {}",
            input.len()
        );
        assert!(
            input.last().unwrap()["content"][0]["text"]
                .as_str()
                .unwrap()
                .starts_with("answer 5")
        );
        let saved = std::fs::read_to_string(agent.session_path()).unwrap();
        assert!(saved.contains("\"type\":\"compaction\""));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn auto_compaction_runs_after_a_turn_that_fills_the_context_window() {
        let root = std::env::temp_dir().join(format!("mu-auto-compact-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(&root, vec![Ok(scripted_reply("summary text", vec![]))]);
        std::fs::create_dir_all(root.join("agent")).unwrap();
        std::fs::write(
            root.join("agent/models.json"),
            r#"{"providers":{"test-provider":{"models":[{"id":"test-model","contextWindow":30000}]}}}"#,
        )
        .unwrap();
        for index in 0..4 {
            agent
                .session
                .append_user_text(format!("q{index} {}", "x".repeat(20_000)));
            agent.session.append_assistant_response(
                "openai-responses",
                "p",
                "m",
                mu_session::AssistantTurnResponse {
                    thinking: String::new(),
                    thinking_signature: None,
                    text: format!("a{index} {}", "y".repeat(20_000)),
                    usage: mu_session::MessageUsage {
                        total_tokens: 29_000,
                        ..mu_session::MessageUsage::default()
                    },
                },
            );
        }
        assert!(agent.needs_auto_compaction());
        agent.compact().await.unwrap();
        // The compaction entry carries no usage, but the last assistant usage is still full;
        // a disabled setting must suppress the trigger.
        std::fs::write(
            root.join("agent/settings.json"),
            r#"{"compaction":{"enabled":false}}"#,
        )
        .unwrap();
        assert!(!agent.needs_auto_compaction());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn compact_does_nothing_for_short_sessions_and_keeps_failed_summaries_out() {
        let root = std::env::temp_dir().join(format!("mu-compact-short-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(&root, vec![]);
        agent.session.append_user_text("hi");
        assert_eq!(agent.compact().await.unwrap(), None);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn unknown_tools_become_error_results_the_model_can_see() {
        let root = std::env::temp_dir().join(format!("mu-unknown-tool-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(
            &root,
            vec![
                Ok(scripted_reply(
                    "",
                    vec![("call-1", "missing_tool", serde_json::json!({}))],
                )),
                Ok(scripted_reply("recovered", vec![])),
            ],
        );
        let mut finished = Vec::new();
        agent
            .submit("use a tool".into(), &mut |event| {
                if let super::AgentEvent::ToolFinished {
                    output, is_error, ..
                } = event
                {
                    finished.push((output, is_error));
                }
            })
            .await
            .expect("the loop continues after a failed tool call");

        assert_eq!(finished, [("Tool missing_tool not found".to_owned(), true)]);
        let input = responses_input(&agent.transcript);
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[3]["content"][0]["text"], "recovered");
        std::fs::remove_dir_all(root).expect("remove temporary project");
    }

    #[tokio::test]
    async fn invalid_tool_arguments_report_pi_validation_errors_without_executing() {
        let root = std::env::temp_dir().join(format!("mu-invalid-args-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(
            &root,
            vec![
                Ok(scripted_reply(
                    "",
                    vec![("call-1", "write", serde_json::json!({"path": 7}))],
                )),
                Ok(scripted_reply("done", vec![])),
            ],
        );
        let mut outputs = Vec::new();
        agent
            .submit("write".into(), &mut |event| {
                if let super::AgentEvent::ToolFinished { output, .. } = event {
                    outputs.push(output);
                }
            })
            .await
            .unwrap();

        assert!(
            outputs[0].starts_with("Validation failed for tool \"write\":"),
            "{}",
            outputs[0]
        );
        assert!(outputs[0].contains("Received arguments:"));
        std::fs::remove_dir_all(root).expect("remove temporary project");
    }

    #[tokio::test]
    async fn a_cancelled_turn_ends_cleanly_without_persisting_an_assistant_message() {
        let root = std::env::temp_dir().join(format!("mu-cancelled-turn-{}", uuid::Uuid::new_v4()));
        let mut agent = scripted_agent(&root, vec![Ok(scripted_reply("never sent", vec![]))]);
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();

        agent
            .submit_with_cancel("hello".into(), &mut |_| {}, &cancel)
            .await
            .expect("an aborted turn is not an error");

        let session = std::fs::read_to_string(agent.session_path()).expect("read session");
        let parsed = SessionDocument::parse_jsonl(&session).expect("parse session");
        let roles = parsed
            .active_branch_entries()
            .into_iter()
            .filter_map(|entry| entry.extra_fields().get("message"))
            .map(|message| message["role"].as_str().unwrap_or_default().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(roles, ["user"]);
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
        assert!(responses_input(&agent.transcript).is_empty());
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

        assert_eq!(
            queue.take_steering_batch(QueueMode::OneAtATime),
            ["steer one"]
        );
        assert_eq!(
            queue.take_steering_batch(QueueMode::OneAtATime),
            ["steer two"]
        );
        assert!(queue.take_steering_batch(QueueMode::OneAtATime).is_empty());
        assert_eq!(
            queue.take_next(),
            Some(("later".into(), PromptBehavior::FollowUp))
        );
        assert_eq!(queue.take_next(), None);
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
        let input = responses_input(&agent.transcript);
        let user_text = input
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
