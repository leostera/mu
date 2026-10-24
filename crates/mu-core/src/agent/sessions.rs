//! Session files: discovery, naming, loading, and reconstruction of the active branch.

use std::{
    collections::HashSet,
    io::BufRead,
    path::{Path, PathBuf},
};

use mu_session::SessionDocument;
use serde_json::{Value, json};

use super::{AgentError, Config, config, runtime};
use crate::agent_loop::AgentMessage;

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

pub(super) fn session_active_mcp_tools(session: &SessionDocument) -> HashSet<String> {
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

pub(super) fn session_codemode_store(session: &SessionDocument) -> Value {
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
pub(super) fn missing_session_cwd_error(
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

pub(super) fn load_session(
    path: &Path,
) -> Result<(SessionDocument, Vec<AgentMessage>), AgentError> {
    let contents =
        std::fs::read_to_string(path).map_err(|error| AgentError::Session(error.to_string()))?;
    let session = SessionDocument::parse_jsonl(&contents)
        .map_err(|error| AgentError::Session(error.to_string()))?;
    let history = runtime::session_messages(&session);
    Ok((session, history))
}

pub(super) fn session_transcript(session: &SessionDocument) -> Vec<SessionTranscriptMessage> {
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

pub(super) fn session_preview(session: &SessionDocument) -> Option<String> {
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
pub(super) fn expand_tilde(path: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    match (path, home) {
        ("~", Some(home)) => home,
        (rest, Some(home)) if rest.starts_with("~/") => home.join(&rest[2..]),
        _ => PathBuf::from(path),
    }
}

/// Pi's session-directory precedence: `--session-dir`, then `PI_CODING_AGENT_SESSION_DIR`, then
/// the `sessionDir` setting.
pub(super) fn configured_session_dir(
    flag: Option<&Path>,
    env: Option<&str>,
    setting: Option<&str>,
) -> Option<PathBuf> {
    flag.map(Path::to_owned)
        .or_else(|| env.filter(|v| !v.is_empty()).map(expand_tilde))
        .or_else(|| setting.filter(|v| !v.is_empty()).map(expand_tilde))
}

pub(super) fn session_directory(
    agent_dir: &Path,
    cwd: &Path,
    override_dir: Option<&Path>,
) -> PathBuf {
    let setting = crate::settings::Settings::load_for_project(agent_dir, cwd).get("sessionDir");
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

pub(super) fn resolved_path(path: &Path) -> PathBuf {
    crate::path_utils::canonicalize_path(path)
}

pub(super) fn session_header(path: &Path) -> Option<Value> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let mut first_line = String::new();
    reader.read_line(&mut first_line).ok()?;
    let header = serde_json::from_str::<Value>(&first_line).ok()?;
    (header.get("type").and_then(Value::as_str) == Some("session")).then_some(header)
}

pub(super) fn find_session_id_path(
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

pub(super) fn normalize_session_name(name: Option<String>) -> Result<Option<String>, AgentError> {
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

pub(super) fn validate_session_id(id: &str) -> Result<(), AgentError> {
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

pub(super) fn list_session_paths(
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

pub(super) fn latest_session_path(
    agent_dir: &Path,
    cwd: &Path,
    override_dir: Option<&Path>,
) -> Option<PathBuf> {
    list_session_paths(agent_dir, cwd, override_dir)
        .into_iter()
        .next()
        .map(|(_, path, _)| path)
}

pub(super) fn session_path(c: &Config, override_dir: Option<&Path>) -> PathBuf {
    new_session_path(&c.agent_dir, &c.cwd, override_dir)
}

pub(super) fn new_session_path(
    agent_dir: &Path,
    cwd: &Path,
    override_dir: Option<&Path>,
) -> PathBuf {
    let dir = session_directory(agent_dir, cwd, override_dir);
    let timestamp = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
        .replace([':', '.'], "-");
    dir.join(format!("{timestamp}_{}.jsonl", uuid::Uuid::new_v4()))
}

pub(super) fn write_session(session: &SessionDocument, path: &Path) -> Result<(), AgentError> {
    session
        .write_atomic(path)
        .map_err(|error| AgentError::Session(error.to_string()))
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
