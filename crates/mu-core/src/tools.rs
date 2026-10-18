use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    process::{Output, Stdio},
    sync::{Arc, Mutex as StdMutex, OnceLock, Weak},
    time::{Duration, Instant},
};

use globset::{Glob, GlobMatcher};
use ignore::WalkBuilder;
use regex::{Regex, RegexBuilder};
use serde::Deserialize;
use serde_json::{Value, json};
use similar::{ChangeTag, TextDiff};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    sync::Mutex as AsyncMutex,
    task::JoinHandle,
};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    Arguments(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Command(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    /// Plain model-facing text until the agent's content-block bridge is complete.
    pub content: String,
    pub details: Option<Value>,
    pub structured_content: Option<Value>,
    pub is_error: bool,
}

impl ToolResult {
    fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            details: None,
            structured_content: None,
            is_error: false,
        }
    }
}

/// Pi's default coding tool set. Schemas use the same names and argument shapes.
pub fn definitions() -> Vec<Value> {
    vec![
        json!({"type":"function","name":"read","description":"Read a file. Supports line offset and limit.","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer","minimum":1},"limit":{"type":"integer","minimum":1}},"required":["path"],"additionalProperties":false}}),
        json!({"type":"function","name":"bash","description":"Run a shell command in the working directory.","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"number","description":"Timeout in seconds (optional, no default timeout)"}},"required":["command"],"additionalProperties":false}}),
        json!({"type":"function","name":"powershell","description":"Execute PowerShell commands in the working directory.","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"number","description":"Timeout in seconds (optional, no default timeout)"}},"required":["command"],"additionalProperties":false}}),
        json!({"type":"function","name":"edit","description":"Edit a single file using exact text replacements. Each edits[].oldText must match a unique, non-overlapping region of the original file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","minItems":1,"items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"],"additionalProperties":false}}},"required":["path","edits"],"additionalProperties":false}}),
        json!({"type":"function","name":"write","description":"Write content to a file, creating parent directories as needed.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":false}}),
        json!({"type":"function","name":"grep","description":"Search file contents for a pattern. Respects .gitignore and returns matching lines with file paths and line numbers.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"glob":{"type":"string"},"ignoreCase":{"type":"boolean"},"literal":{"type":"boolean"},"context":{"type":"number"},"limit":{"type":"number"}},"required":["pattern"],"additionalProperties":false}}),
        json!({"type":"function","name":"find","description":"Find files by glob pattern. Includes hidden files and respects .gitignore.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"limit":{"type":"number"}},"required":["pattern"],"additionalProperties":false}}),
        json!({"type":"function","name":"ls","description":"List directory contents, including dotfiles. Entries are sorted alphabetically; directories have a trailing slash.","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"number","description":"Maximum number of entries to return (default: 500)"}},"additionalProperties":false}}),
    ]
}

pub async fn execute(cwd: &Path, name: &str, arguments: &Value) -> Result<ToolResult, ToolError> {
    execute_with_environment(cwd, name, arguments, &HashMap::new()).await
}

pub async fn execute_with_environment(
    cwd: &Path,
    name: &str,
    arguments: &Value,
    environment: &HashMap<String, String>,
) -> Result<ToolResult, ToolError> {
    match name {
        "read" => read_file(cwd, arguments),
        "write" => write_file(cwd, arguments).await,
        "edit" => edit_file(cwd, arguments).await,
        "bash" => run_bash(cwd, arguments, environment).await,
        "powershell" => run_powershell(cwd, arguments, environment).await,
        "ls" => list_directory(cwd, arguments),
        "grep" => grep_files(cwd, arguments),
        "find" => find_files(cwd, arguments),
        other => Err(ToolError::Arguments(format!("unknown tool `{other}`"))),
    }
}

fn string_arg<'a>(args: &'a Value, name: &str) -> Result<&'a str, ToolError> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::Arguments(format!("`{name}` must be a string")))
}

fn resolve(cwd: &Path, args: &Value) -> Result<PathBuf, ToolError> {
    resolve_input_path(cwd, string_arg(args, "path")?)
}

fn resolve_input_path(cwd: &Path, input: &str) -> Result<PathBuf, ToolError> {
    let normalized = normalize_path_input(input);
    let path = if normalized.starts_with("file://") {
        url::Url::parse(&normalized)
            .ok()
            .and_then(|url| url.to_file_path().ok())
            .ok_or_else(|| ToolError::Arguments(format!("invalid file URL `{normalized}`")))?
    } else {
        PathBuf::from(normalized)
    };
    Ok(if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    })
}

fn normalize_path_input(input: &str) -> String {
    let mut normalized = input
        .chars()
        .map(|character| match character {
            '\u{00a0}' | '\u{2000}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
            character => character,
        })
        .collect::<String>();
    if normalized.starts_with('@') {
        normalized.remove(0);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    if let Some(home) = home {
        if normalized == "~" {
            return home.to_string_lossy().into_owned();
        }
        if normalized.starts_with("~/") || normalized.starts_with("~\\") {
            return home.join(&normalized[2..]).to_string_lossy().into_owned();
        }
    }
    normalized
}

fn resolve_read_path(cwd: &Path, args: &Value) -> Result<PathBuf, ToolError> {
    let input = string_arg(args, "path")?;
    let resolved = resolve_input_path(cwd, input)?;
    if resolved.exists() {
        return Ok(resolved);
    }
    for candidate in read_path_fallbacks(&resolved) {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Ok(resolved)
}

fn read_path_fallbacks(path: &Path) -> Vec<PathBuf> {
    let path = path.to_string_lossy();
    let screenshot = replace_screenshot_space(&path);
    let decomposed = path.nfd().collect::<String>();
    let curly_quotes = path.replace('\'', "’");
    let decomposed_curly_quotes = decomposed.replace('\'', "’");
    [
        screenshot,
        decomposed,
        curly_quotes,
        decomposed_curly_quotes,
    ]
    .into_iter()
    .filter(|candidate| candidate != &*path)
    .map(PathBuf::from)
    .collect()
}

fn replace_screenshot_space(path: &str) -> String {
    let mut output = path.to_owned();
    for token in ["AM", "PM", "Am", "Pm", "aM", "pM", "am", "pm"] {
        output = output.replace(&format!(" {token}."), &format!("\u{202f}{token}."));
    }
    output
}

const READ_MAX_LINES: usize = 2000;
const READ_MAX_BYTES: usize = 50 * 1024;

fn read_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let requested_path = string_arg(args, "path")?;
    let path = resolve_read_path(cwd, args)?;
    let bytes = std::fs::read(&path)?;
    // Node's Buffer.toString("utf-8") replaces malformed sequences instead of
    // rejecting the entire text file; mirror that decoding boundary here.
    let text = String::from_utf8_lossy(&bytes);
    let offset = read_number_argument(args, "offset", 1.0)?;
    let start_line = if offset == 0.0 {
        0.0
    } else {
        (offset - 1.0).max(0.0)
    };
    let all_lines = text.split('\n').collect::<Vec<_>>();
    if start_line >= all_lines.len() as f64 {
        return Err(ToolError::Arguments(format!(
            "Offset {} is beyond end of file ({} lines total)",
            format_json_number(offset),
            all_lines.len()
        )));
    }

    let start = js_slice_index(start_line, all_lines.len());
    let limit = args
        .get("limit")
        .map(|_| read_number_argument(args, "limit", 0.0))
        .transpose()?;
    let end_line = limit.map_or(all_lines.len() as f64, |limit| {
        (start_line + limit).min(all_lines.len() as f64)
    });
    let end = js_slice_index(end_line, all_lines.len()).max(start);
    let selected = all_lines[start..end].join("\n");
    let total_file_lines = all_lines.len();
    let user_limited_lines = end_line - start_line;
    let (truncated, truncated_by, output_lines, first_line_exceeds_limit) =
        truncate_read_head(&selected);
    let start_line = start_line + 1.0;
    let content = if first_line_exceeds_limit {
        let first_line_size = format_size(all_lines[start].len());
        format!(
            "[Line {start_line} is {first_line_size}, exceeds {} limit. Use bash: sed -n '{start_line}p' {} | head -c {}]",
            format_size(READ_MAX_BYTES),
            requested_path,
            READ_MAX_BYTES
        )
    } else if let Some(limit_name) = truncated {
        let end_line = start_line + output_lines as f64 - 1.0;
        let next_offset = end_line + 1.0;
        let mut output = truncated_by.clone();
        if limit_name == "bytes" {
            output.push_str(&format!(
                "\n\n[Showing lines {}-{} of {total_file_lines} ({} limit). Use offset={} to continue.]",
                format_json_number(start_line),
                format_json_number(end_line),
                format_size(READ_MAX_BYTES),
                format_json_number(next_offset)
            ));
        } else {
            output.push_str(&format!(
                "\n\n[Showing lines {}-{} of {total_file_lines}. Use offset={} to continue.]",
                format_json_number(start_line),
                format_json_number(end_line),
                format_json_number(next_offset)
            ));
        }
        output
    } else if limit.is_some() && start_line - 1.0 + user_limited_lines < all_lines.len() as f64 {
        let remaining = all_lines.len() as f64 - (start_line - 1.0 + user_limited_lines);
        let next_offset = start_line - 1.0 + user_limited_lines + 1.0;
        format!(
            "{truncated_by}\n\n[{} more lines in file. Use offset={} to continue.]",
            format_json_number(remaining),
            format_json_number(next_offset)
        )
    } else {
        truncated_by.clone()
    };
    let details = truncated.map(|truncation_kind| {
        let mut total_lines = selected.split('\n').count();
        if selected.ends_with('\n') {
            total_lines = total_lines.saturating_sub(1);
        }
        json!({
            "truncation": {
                "content": truncated_by,
                "truncated": true,
                "truncatedBy": truncation_kind,
                "totalLines": total_lines,
                "totalBytes": selected.len(),
                "outputLines": output_lines,
                "outputBytes": truncated_by.len(),
                "lastLinePartial": false,
                "firstLineExceedsLimit": first_line_exceeds_limit,
                "maxLines": READ_MAX_LINES,
                "maxBytes": READ_MAX_BYTES,
            }
        })
    });
    let mut result = ToolResult::text(content);
    result.details = details;
    Ok(result)
}

fn read_number_argument(args: &Value, name: &str, default: f64) -> Result<f64, ToolError> {
    let Some(value) = args.get(name) else {
        return Ok(default);
    };
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| ToolError::Arguments(format!("`{name}` must be a finite number")))
}

fn format_json_number(number: f64) -> String {
    number.to_string()
}

fn js_slice_index(index: f64, length: usize) -> usize {
    let integer = index.trunc();
    if integer < 0.0 {
        (length as f64 + integer).max(0.0) as usize
    } else {
        integer.min(length as f64) as usize
    }
}

/// Apply Pi's head truncation to complete lines only, using UTF-8 byte counts.
/// The returned marker is `Some("lines")` or `Some("bytes")` when truncated.
fn truncate_read_head(content: &str) -> (Option<&'static str>, String, usize, bool) {
    let mut lines = if content.is_empty() {
        Vec::new()
    } else {
        content.split('\n').collect::<Vec<_>>()
    };
    if content.ends_with('\n') {
        lines.pop();
    }
    if lines.len() <= READ_MAX_LINES && content.len() <= READ_MAX_BYTES {
        return (None, content.to_owned(), lines.len(), false);
    }
    if lines
        .first()
        .is_some_and(|line| line.len() > READ_MAX_BYTES)
    {
        return (Some("bytes"), String::new(), 0, true);
    }

    let mut output = Vec::new();
    let mut output_bytes = 0;
    let mut truncated_by = "lines";
    for line in lines.iter().take(READ_MAX_LINES) {
        let line_bytes = line.len() + usize::from(!output.is_empty());
        if output_bytes + line_bytes > READ_MAX_BYTES {
            truncated_by = "bytes";
            break;
        }
        output.push(*line);
        output_bytes += line_bytes;
    }
    if output.len() >= READ_MAX_LINES && output_bytes <= READ_MAX_BYTES {
        truncated_by = "lines";
    }
    (Some(truncated_by), output.join("\n"), output.len(), false)
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

// Coordinate write/edit by canonical path without retaining unused locks forever.
static FILE_MUTATION_LOCKS: OnceLock<StdMutex<HashMap<PathBuf, Weak<tokio::sync::Mutex<()>>>>> =
    OnceLock::new();

fn mutation_lock(path: &Path) -> Arc<tokio::sync::Mutex<()>> {
    let key = canonical_mutation_path(path);
    let locks = FILE_MUTATION_LOCKS.get_or_init(|| StdMutex::new(HashMap::new()));
    let mut locks = locks.lock().unwrap_or_else(|error| error.into_inner());
    locks.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(tokio::sync::Mutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    lock
}

fn canonical_mutation_path(path: &Path) -> PathBuf {
    let mut ancestor = path.to_owned();
    let mut missing_components = Vec::new();
    while !ancestor.exists() {
        let Some(name) = ancestor.file_name() else {
            break;
        };
        missing_components.push(name.to_owned());
        if !ancestor.pop() {
            break;
        }
    }
    let mut canonical = std::fs::canonicalize(&ancestor).unwrap_or(ancestor);
    for component in missing_components.iter().rev() {
        canonical.push(component);
    }
    canonical
}

async fn write_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let requested_path = string_arg(args, "path")?;
    let path = resolve(cwd, args)?;
    let content = string_arg(args, "content")?;
    let lock = mutation_lock(&path);
    let _guard = lock.lock().await;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, content)?;
    Ok(ToolResult::text(format!(
        "Successfully wrote to {requested_path}"
    )))
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextEdit {
    #[serde(rename = "oldText")]
    pub old_text: String,
    #[serde(rename = "newText")]
    pub new_text: String,
}

fn parse_edits(args: &Value) -> Result<Vec<TextEdit>, ToolError> {
    let mut args = args.clone();
    let object = args.as_object_mut().ok_or_else(|| {
        ToolError::Command(
            "Edit tool input is invalid. edits must contain at least one replacement.".into(),
        )
    })?;

    // Pi accepts legacy oldText/newText fields and a few model-emitted edits forms.
    if let Some(edits) = object.get_mut("edits") {
        if let Some(encoded) = edits.as_str() {
            if let Ok(parsed) = serde_json::from_str::<Value>(encoded) {
                *edits = parsed;
            }
        }
        if edits.get("oldText").and_then(Value::as_str).is_some()
            && edits.get("newText").and_then(Value::as_str).is_some()
        {
            *edits = Value::Array(vec![edits.take()]);
        }
    }
    let legacy = match (
        object.get("oldText").and_then(Value::as_str),
        object.get("newText").and_then(Value::as_str),
    ) {
        (Some(old), Some(new)) => {
            let legacy = json!({"oldText":old,"newText":new});
            object.remove("oldText");
            object.remove("newText");
            Some(legacy)
        }
        _ => None,
    };
    if let Some(legacy) = legacy {
        match object.remove("edits") {
            Some(Value::Array(mut edits)) => {
                edits.push(legacy);
                object.insert("edits".into(), Value::Array(edits));
            }
            Some(_) | None => {
                object.insert("edits".into(), Value::Array(vec![legacy]));
            }
        }
    }

    let edits = object
        .get("edits")
        .and_then(Value::as_array)
        .filter(|edits| !edits.is_empty())
        .ok_or_else(|| {
            ToolError::Command(
                "Edit tool input is invalid. edits must contain at least one replacement.".into(),
            )
        })?;
    edits
        .iter()
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|error| ToolError::Arguments(format!("invalid edit: {error}")))
        })
        .collect()
}

async fn edit_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let requested_path = string_arg(args, "path")?;
    let display_path = Path::new(requested_path);
    let path = resolve(cwd, args)?;
    let edits = parse_edits(args)?;
    let lock = mutation_lock(&path);
    let _guard = lock.lock().await;
    let original = std::fs::read(&path)
        .map_err(|error| edit_file_io_error(display_path, error))
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())?;
    let (bom, content) = original
        .strip_prefix('\u{feff}')
        .map_or(("", original.as_str()), |content| ("\u{feff}", content));
    let line_ending = if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    let normalized_edits = edits
        .into_iter()
        .map(|edit| TextEdit {
            old_text: edit.old_text.replace("\r\n", "\n").replace('\r', "\n"),
            new_text: edit.new_text.replace("\r\n", "\n").replace('\r', "\n"),
        })
        .collect::<Vec<_>>();

    let updated = apply_edits(&normalized, &normalized_edits, display_path)?;
    let details = edit_details(&normalized, &updated, display_path);
    let restored = if line_ending == "\r\n" {
        updated.replace('\n', "\r\n")
    } else {
        updated
    };
    std::fs::write(&path, format!("{bom}{restored}"))
        .map_err(|error| edit_file_io_error(display_path, error))?;
    let mut result = ToolResult::text(format!(
        "Successfully replaced {} block(s) in {}.",
        normalized_edits.len(),
        display_path.display()
    ));
    result.details = Some(details);
    Ok(result)
}

fn edit_file_io_error(path: &Path, error: std::io::Error) -> ToolError {
    ToolError::Command(edit_file_io_message(path, error))
}

fn edit_file_io_message(path: &Path, error: std::io::Error) -> String {
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => "ENOENT",
        std::io::ErrorKind::PermissionDenied => "EACCES",
        _ => {
            return format!("Could not edit file: {}. {error}.", path.display());
        }
    };
    format!(
        "Could not edit file: {}. Error code: {code}.",
        path.display()
    )
}

/// Compute a non-mutating edit preview, matching Pi's `computeEditsDiff` result shape.
/// Returns `{ diff, firstChangedLine }` on success or `{ error }` on failure.
pub fn compute_edit_diff(cwd: &Path, requested_path: &str, edits: &[TextEdit]) -> Value {
    let path = match resolve_input_path(cwd, requested_path) {
        Ok(path) => path,
        Err(error) => return json!({"error": error.to_string()}),
    };
    let original = match std::fs::read(&path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(error) => {
            return json!({"error": edit_file_io_message(Path::new(requested_path), error)});
        }
    };
    let content = original
        .strip_prefix('\u{feff}')
        .unwrap_or(original.as_str());
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    let normalized_edits = edits
        .iter()
        .cloned()
        .map(|mut edit| {
            edit.old_text = edit.old_text.replace("\r\n", "\n").replace('\r', "\n");
            edit.new_text = edit.new_text.replace("\r\n", "\n").replace('\r', "\n");
            edit
        })
        .collect::<Vec<_>>();
    let updated = match apply_edits(&normalized, &normalized_edits, Path::new(requested_path)) {
        Ok(updated) => updated,
        Err(ToolError::Arguments(message)) | Err(ToolError::Command(message)) => {
            return json!({"error":message});
        }
        Err(ToolError::Io(error)) => return json!({"error":error.to_string()}),
    };
    let details = edit_details(&normalized, &updated, Path::new(requested_path));
    json!({
        "diff": details["diff"],
        "firstChangedLine": details["firstChangedLine"],
    })
}

fn edit_details(old_content: &str, new_content: &str, path: &Path) -> Value {
    let diff = TextDiff::from_lines(old_content, new_content);
    // Pi numbers against `String.split("\\n")`, which keeps the trailing empty
    // segment when content ends in a newline. `str::lines` drops that segment.
    let old_lines = old_content.split('\n').count();
    let new_lines = new_content.split('\n').count();
    let line_number_width = old_lines.max(new_lines).to_string().len();
    let mut display_lines = Vec::new();
    let mut first_changed_line = None;
    let groups = diff.grouped_ops(4);

    for (group_index, group) in groups.iter().enumerate() {
        if group_index > 0 {
            display_lines.push(format!(" {} ...", " ".repeat(line_number_width)));
        }
        for op in group {
            for change in op.iter_changes(diff.old_slices(), diff.new_slices()) {
                let (marker, line_number) = match change.tag() {
                    ChangeTag::Equal => (' ', change.old_index().unwrap_or(0) + 1),
                    ChangeTag::Delete => ('-', change.old_index().unwrap_or(0) + 1),
                    ChangeTag::Insert => {
                        let line_number = change.new_index().unwrap_or(0) + 1;
                        first_changed_line.get_or_insert(line_number);
                        ('+', line_number)
                    }
                };
                if change.tag() == ChangeTag::Delete {
                    first_changed_line.get_or_insert(change.new_index().unwrap_or(0) + 1);
                }
                let text = change.value_ref().trim_end_matches(['\n', '\r']);
                display_lines.push(format!("{marker}{line_number:>line_number_width$} {text}"));
            }
        }
    }
    let patch = diff
        .unified_diff()
        .context_radius(4)
        .header(&path.display().to_string(), &path.display().to_string())
        .to_string();
    json!({
        "diff": display_lines.join("\n"),
        "patch": patch,
        "firstChangedLine": first_changed_line,
    })
}

#[derive(Clone)]
struct MatchedEdit {
    edit_index: usize,
    start: usize,
    length: usize,
    replacement: String,
}

struct EditLineGroup {
    first_line: usize,
    last_line_exclusive: usize,
    edits: Vec<MatchedEdit>,
}

fn apply_edits(content: &str, edits: &[TextEdit], path: &Path) -> Result<String, ToolError> {
    let fuzzy = edits.iter().any(|edit| !content.contains(&edit.old_text));
    let replacement_base = if fuzzy {
        normalize_for_fuzzy_match(content)
    } else {
        content.to_owned()
    };
    let mut matches = Vec::with_capacity(edits.len());

    for (edit_index, edit) in edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            let error = if edits.len() == 1 {
                format!("oldText must not be empty in {}.", path.display())
            } else {
                format!(
                    "edits[{edit_index}].oldText must not be empty in {}.",
                    path.display()
                )
            };
            return Err(ToolError::Command(error));
        }
        let fuzzy_old_text = normalize_for_fuzzy_match(&edit.old_text);
        let search_text = if replacement_base.contains(&edit.old_text) {
            edit.old_text.as_str()
        } else {
            fuzzy_old_text.as_str()
        };
        let positions = replacement_base
            .match_indices(search_text)
            .map(|(start, _)| start)
            .collect::<Vec<_>>();
        if positions.is_empty() {
            let error = if edits.len() == 1 {
                format!(
                    "Could not find the exact text in {}. The old text must match exactly including all whitespace and newlines.",
                    path.display()
                )
            } else {
                format!(
                    "Could not find edits[{edit_index}] in {}. The oldText must match exactly including all whitespace and newlines.",
                    path.display()
                )
            };
            return Err(ToolError::Command(error));
        }
        let normalized_base = normalize_for_fuzzy_match(&replacement_base);
        let occurrences = normalized_base.match_indices(&fuzzy_old_text).count();
        if occurrences > 1 {
            let error = if edits.len() == 1 {
                format!(
                    "Found {occurrences} occurrences of the text in {}. The text must be unique. Please provide more context to make it unique.",
                    path.display()
                )
            } else {
                format!(
                    "Found {occurrences} occurrences of edits[{edit_index}] in {}. Each oldText must be unique. Please provide more context to make it unique.",
                    path.display()
                )
            };
            return Err(ToolError::Command(error));
        }
        let [start] = positions.as_slice() else {
            continue;
        };
        matches.push(MatchedEdit {
            edit_index,
            start: *start,
            length: search_text.len(),
            replacement: edit.new_text.clone(),
        });
    }

    matches.sort_by_key(|edit| edit.start);
    for pair in matches.windows(2) {
        if pair[0].start + pair[0].length > pair[1].start {
            return Err(ToolError::Command(format!(
                "edits[{}] and edits[{}] overlap in {}. Merge them into one edit or target disjoint regions.",
                pair[0].edit_index,
                pair[1].edit_index,
                path.display()
            )));
        }
    }

    let updated = if fuzzy {
        apply_fuzzy_edits_preserving_untouched_lines(content, &replacement_base, &matches)
    } else {
        apply_matched_edits(&replacement_base, &matches)
    };
    if updated == content {
        let error = if edits.len() == 1 {
            format!(
                "No changes made to {}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected.",
                path.display()
            )
        } else {
            format!(
                "No changes made to {}. The replacements produced identical content.",
                path.display()
            )
        };
        return Err(ToolError::Command(error));
    }
    Ok(updated)
}

fn normalize_for_fuzzy_match(text: &str) -> String {
    text.nfkc()
        .collect::<String>()
        .split('\n')
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .map(|character| match character {
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => '\'',
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' => '"',
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            '\u{00a0}' | '\u{2002}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
            character => character,
        })
        .collect()
}

fn apply_matched_edits(content: &str, edits: &[MatchedEdit]) -> String {
    let mut updated = content.to_owned();
    for edit in edits.iter().rev() {
        updated.replace_range(edit.start..edit.start + edit.length, &edit.replacement);
    }
    updated
}

fn apply_fuzzy_edits_preserving_untouched_lines(
    original: &str,
    fuzzy_content: &str,
    edits: &[MatchedEdit],
) -> String {
    let line_starts = line_start_offsets(fuzzy_content);
    let original_line_starts = line_start_offsets(original);
    let mut groups: Vec<EditLineGroup> = Vec::new();
    for edit in edits {
        let first_line = line_index_at(&line_starts, edit.start);
        let last_line = line_index_at(&line_starts, edit.start + edit.length - 1);
        if let Some(group) = groups.last_mut()
            && first_line < group.last_line_exclusive
        {
            group.last_line_exclusive = group.last_line_exclusive.max(last_line + 1);
            group.edits.push(edit.clone());
        } else {
            groups.push(EditLineGroup {
                first_line,
                last_line_exclusive: last_line + 1,
                edits: vec![edit.clone()],
            });
        }
    }

    let mut output = String::new();
    let mut original_offset = 0;
    for group in groups {
        let start = line_starts[group.first_line];
        let end = line_starts
            .get(group.last_line_exclusive)
            .copied()
            .unwrap_or(fuzzy_content.len());
        let original_start = original_line_starts[group.first_line];
        let original_end = original_line_starts
            .get(group.last_line_exclusive)
            .copied()
            .unwrap_or(original.len());
        output.push_str(&original[original_offset..original_start]);
        let mut changed_lines = fuzzy_content[start..end].to_owned();
        for edit in group.edits.iter().rev() {
            let local_start = edit.start - start;
            changed_lines.replace_range(local_start..local_start + edit.length, &edit.replacement);
        }
        output.push_str(&changed_lines);
        original_offset = original_end;
    }
    output.push_str(&original[original_offset..]);
    output
}

fn line_start_offsets(content: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (index, byte) in content.bytes().enumerate() {
        if byte == b'\n' {
            starts.push(index + 1);
        }
    }
    starts
}

fn line_index_at(line_starts: &[usize], offset: usize) -> usize {
    line_starts.partition_point(|start| *start <= offset) - 1
}

const LS_DEFAULT_LIMIT: usize = 500;
const TOOL_OUTPUT_MAX_BYTES: usize = 50 * 1024;
const BASH_OUTPUT_MAX_LINES: usize = 2_000;

struct ListDirectoryInput {
    path: PathBuf,
    limit: f64,
}

fn parse_list_directory_input(cwd: &Path, args: &Value) -> Result<ListDirectoryInput, ToolError> {
    let object = args
        .as_object()
        .ok_or_else(|| ToolError::Arguments("ls arguments must be an object".into()))?;
    let path = match object.get("path") {
        None => cwd.to_owned(),
        Some(Value::String(path)) => resolve_input_path(cwd, path)?,
        Some(_) => return Err(ToolError::Arguments("`path` must be a string".into())),
    };
    let limit = match object.get("limit") {
        None => LS_DEFAULT_LIMIT as f64,
        Some(value) => value
            .as_f64()
            .filter(|limit| limit.is_finite())
            .ok_or_else(|| ToolError::Arguments("`limit` must be a finite number".into()))?,
    };
    Ok(ListDirectoryInput { path, limit })
}

fn list_directory(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let ListDirectoryInput { path, limit } = parse_list_directory_input(cwd, args)?;
    if !path.exists() {
        return Err(ToolError::Arguments(format!(
            "Path not found: {}",
            path.display()
        )));
    }
    let metadata = std::fs::metadata(&path)?;
    if !metadata.is_dir() {
        return Err(ToolError::Arguments(format!(
            "Not a directory: {}",
            path.display()
        )));
    }

    let mut entries = std::fs::read_dir(&path)
        .map_err(|error| ToolError::Arguments(format!("Cannot read directory: {error}")))?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by(|left, right| {
        left.to_lowercase()
            .cmp(&right.to_lowercase())
            .then_with(|| left.cmp(right))
    });

    let mut names = Vec::new();
    let mut limit_reached = false;
    for name in entries {
        if names.len() as f64 >= limit {
            limit_reached = true;
            break;
        }
        if let Ok(metadata) = std::fs::metadata(path.join(&name)) {
            names.push(if metadata.is_dir() {
                format!("{name}/")
            } else {
                name
            });
        }
    }
    if names.is_empty() {
        return Ok(ToolResult::text("(empty directory)"));
    }

    let raw_output = names.join("\n");
    let (mut output, truncated) = truncate_tool_output(&raw_output);
    let mut details = serde_json::Map::new();
    let mut notices = Vec::new();
    if limit_reached {
        notices.push(format!(
            "{limit} entries limit reached. Use limit={} for more",
            limit * 2.0
        ));
        details.insert("entryLimitReached".into(), json!(limit));
    }
    if truncated {
        notices.push(format!(
            "{} limit reached",
            format_size(TOOL_OUTPUT_MAX_BYTES)
        ));
        details.insert(
            "truncation".into(),
            tool_truncation_details(&raw_output, &output, TOOL_OUTPUT_MAX_BYTES),
        );
    }
    if !notices.is_empty() {
        output.push_str(&format!("\n\n[{}]", notices.join(". ")));
    }
    let mut result = ToolResult::text(output);
    result.details = (!details.is_empty()).then_some(Value::Object(details));
    Ok(result)
}

fn truncate_tool_output(content: &str) -> (String, bool) {
    let lines = content.split('\n').collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut byte_count = 0;
    for line in &lines {
        let line_bytes = line.len() + usize::from(!output.is_empty());
        if byte_count + line_bytes > TOOL_OUTPUT_MAX_BYTES {
            break;
        }
        byte_count += line_bytes;
        output.push(*line);
    }
    let truncated = output.len() < lines.len();
    (output.join("\n"), truncated)
}

fn json_number(value: f64) -> Value {
    if value.fract() == 0.0 && (0.0..18_446_744_073_709_551_616.0).contains(&value) {
        json!(value as u64)
    } else {
        json!(value)
    }
}

fn tool_truncation_details(content: &str, truncated_content: &str, max_bytes: usize) -> Value {
    let count_lines = |text: &str| {
        if text.is_empty() {
            0
        } else {
            text.split('\n').count() - usize::from(text.ends_with('\n'))
        }
    };
    json!({
        "content": truncated_content,
        "truncated": true,
        "truncatedBy": "bytes",
        "totalLines": count_lines(content),
        "totalBytes": content.len(),
        "outputLines": count_lines(truncated_content),
        "outputBytes": truncated_content.len(),
        "lastLinePartial": false,
        "firstLineExceedsLimit": content
            .split('\n')
            .next()
            .is_some_and(|line| line.len() > max_bytes),
        "maxLines": 9_007_199_254_740_991u64,
        "maxBytes": max_bytes,
    })
}

const GREP_DEFAULT_LIMIT: usize = 100;
const GREP_MAX_LINE_CHARS: usize = 500;

struct GrepInput {
    regex: Regex,
    path: PathBuf,
    glob: Option<GlobMatcher>,
    context: usize,
    limit: f64,
}

fn parse_grep_input(cwd: &Path, args: &Value) -> Result<GrepInput, ToolError> {
    let object = args
        .as_object()
        .ok_or_else(|| ToolError::Arguments("grep arguments must be an object".into()))?;
    let pattern = object
        .get("pattern")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::Arguments("`pattern` must be a string".into()))?
        .to_owned();
    let path = match object.get("path") {
        None => cwd.to_owned(),
        Some(Value::String(path)) => resolve_input_path(cwd, path)?,
        Some(_) => return Err(ToolError::Arguments("`path` must be a string".into())),
    };
    let glob = match object.get("glob") {
        None => None,
        Some(Value::String(glob)) => Some(
            Glob::new(glob)
                .map_err(|error| ToolError::Arguments(error.to_string()))?
                .compile_matcher(),
        ),
        Some(_) => return Err(ToolError::Arguments("`glob` must be a string".into())),
    };
    let ignore_case = match object.get("ignoreCase") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(ToolError::Arguments(
                "`ignoreCase` must be a boolean".into(),
            ));
        }
    };
    let literal = match object.get("literal") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => return Err(ToolError::Arguments("`literal` must be a boolean".into())),
    };
    let context = optional_nonnegative_integer(object.get("context"), "context")?.unwrap_or(0);
    let limit = match object.get("limit") {
        None => GREP_DEFAULT_LIMIT as f64,
        Some(value) => value
            .as_f64()
            .filter(|limit| limit.is_finite())
            .ok_or_else(|| ToolError::Arguments("`limit` must be a finite number".into()))?,
    }
    .max(1.0);
    let pattern = if literal {
        regex::escape(&pattern)
    } else {
        pattern
    };
    let regex = RegexBuilder::new(&pattern)
        .case_insensitive(ignore_case)
        .build()
        .map_err(|error| ToolError::Arguments(error.to_string()))?;
    Ok(GrepInput {
        regex,
        path,
        glob,
        context,
        limit,
    })
}

fn optional_nonnegative_integer(
    value: Option<&Value>,
    name: &str,
) -> Result<Option<usize>, ToolError> {
    value
        .map(|value| {
            value
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| {
                    ToolError::Arguments(format!("`{name}` must be a non-negative integer"))
                })
        })
        .transpose()
}

fn grep_files(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let input = parse_grep_input(cwd, args)?;
    let is_directory = input.path.is_dir();
    if !is_directory && !input.path.is_file() {
        return Err(ToolError::Arguments(format!(
            "Path not found: {}",
            input.path.display()
        )));
    }
    let regex = input.regex;
    let files = collect_search_files(&input.path, is_directory);
    let mut formatted_matches = Vec::new();
    let mut match_count = 0;
    let mut match_limit_reached = false;
    let mut lines_truncated = false;

    'files: for file in files {
        let display_path = search_display_path(&input.path, &file, is_directory);
        if input.glob.as_ref().is_some_and(|glob| {
            let basename = file.file_name().unwrap_or_default();
            !glob.is_match(&display_path) && !glob.is_match(Path::new(basename))
        }) {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&file) else {
            continue;
        };
        let lines = content
            .split_terminator('\n')
            .map(|line| line.trim_end_matches('\r'))
            .collect::<Vec<_>>();
        for (index, line) in lines.iter().enumerate() {
            if !regex.is_match(line) {
                continue;
            }
            if match_count as f64 >= input.limit {
                match_limit_reached = true;
                break 'files;
            }
            match_count += 1;
            let first = index.saturating_sub(input.context);
            let last = (index + input.context).min(lines.len().saturating_sub(1));
            for (context_index, text) in lines.iter().enumerate().take(last + 1).skip(first) {
                let (text, was_truncated) = truncate_grep_line(text);
                lines_truncated |= was_truncated;
                let line_number = context_index + 1;
                if context_index == index {
                    formatted_matches.push(format!("{display_path}:{line_number}: {text}"));
                } else {
                    formatted_matches.push(format!("{display_path}-{line_number}- {text}"));
                }
            }
        }
    }

    if match_count == 0 {
        return Ok(ToolResult::text("No matches found"));
    }
    let raw_output = formatted_matches.join("\n");
    let (mut output, byte_truncated) = truncate_tool_output(&raw_output);
    let mut details = serde_json::Map::new();
    let mut notices = Vec::new();
    if match_limit_reached {
        notices.push(format!(
            "{} matches limit reached. Use limit={} for more, or refine pattern",
            input.limit,
            input.limit * 2.0
        ));
        details.insert("matchLimitReached".into(), json_number(input.limit));
    }
    if byte_truncated {
        notices.push(format!(
            "{} limit reached",
            format_size(TOOL_OUTPUT_MAX_BYTES)
        ));
        details.insert(
            "truncation".into(),
            tool_truncation_details(&raw_output, &output, TOOL_OUTPUT_MAX_BYTES),
        );
    }
    if lines_truncated {
        notices.push(format!(
            "Some lines truncated to {GREP_MAX_LINE_CHARS} chars. Use read tool to see full lines"
        ));
        details.insert("linesTruncated".into(), json!(true));
    }
    if !notices.is_empty() {
        output.push_str(&format!("\n\n[{}]", notices.join(". ")));
    }
    let mut result = ToolResult::text(output);
    result.details = (!details.is_empty()).then_some(Value::Object(details));
    Ok(result)
}

fn collect_search_files(path: &Path, is_directory: bool) -> Vec<PathBuf> {
    if !is_directory {
        return vec![path.to_owned()];
    }
    let mut files = collect_candidates_with_nested_repositories(path)
        .into_iter()
        .filter(|entry| std::fs::symlink_metadata(entry).is_ok_and(|metadata| metadata.is_file()))
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn search_display_path(root: &Path, file: &Path, is_directory: bool) -> String {
    if is_directory {
        file.strip_prefix(root)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/")
    } else {
        file.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }
}

fn truncate_grep_line(line: &str) -> (String, bool) {
    let mut chars = line.chars();
    let truncated = chars.clone().count() > GREP_MAX_LINE_CHARS;
    let text = if truncated {
        format!(
            "{}... [truncated]",
            chars.by_ref().take(GREP_MAX_LINE_CHARS).collect::<String>()
        )
    } else {
        line.to_owned()
    };
    (text, truncated)
}

const FIND_DEFAULT_LIMIT: usize = 1000;

struct FindInput {
    pattern: GlobMatcher,
    path: PathBuf,
    limit: usize,
}

fn parse_find_input(cwd: &Path, args: &Value) -> Result<FindInput, ToolError> {
    let object = args
        .as_object()
        .ok_or_else(|| ToolError::Arguments("find arguments must be an object".into()))?;
    let pattern = object
        .get("pattern")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::Arguments("`pattern` must be a string".into()))?;
    let pattern = Glob::new(pattern)
        .map_err(|error| ToolError::Arguments(error.to_string()))?
        .compile_matcher();
    let path = match object.get("path") {
        None => cwd.to_owned(),
        Some(Value::String(path)) => resolve_input_path(cwd, path)?,
        Some(_) => return Err(ToolError::Arguments("`path` must be a string".into())),
    };
    let limit =
        optional_nonnegative_integer(object.get("limit"), "limit")?.unwrap_or(FIND_DEFAULT_LIMIT);
    Ok(FindInput {
        pattern,
        path,
        limit,
    })
}

fn collect_candidates_with_nested_repositories(search_root: &Path) -> Vec<PathBuf> {
    let mut pending_roots = vec![(search_root.to_owned(), true)];
    let mut visited_roots = HashSet::from([search_root.to_owned()]);
    let mut candidates = HashSet::new();

    while let Some((repository_root, is_search_root)) = pending_roots.pop() {
        let mut walker = WalkBuilder::new(&repository_root);
        walker
            .hidden(false)
            .require_git(false)
            .sort_by_file_name(|left, right| left.cmp(right));
        if !is_search_root {
            walker.parents(false);
        }
        for entry in walker.build() {
            let Ok(entry) = entry else {
                continue;
            };
            let entry_path = entry.path();
            if entry_path == repository_root {
                continue;
            }
            candidates.insert(entry_path.to_owned());
            if entry.file_type().is_some_and(|kind| kind.is_dir())
                && entry_path.join(".git").exists()
                && visited_roots.insert(entry_path.to_owned())
            {
                pending_roots.push((entry_path.to_owned(), false));
            }
        }
    }

    candidates.into_iter().collect()
}

fn find_files(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let FindInput {
        pattern,
        path,
        limit,
    } = parse_find_input(cwd, args)?;
    if !path.is_dir() {
        return Err(ToolError::Arguments(format!(
            "Path not found: {}",
            path.display()
        )));
    }

    let mut results = Vec::new();
    let mut candidates = collect_candidates_with_nested_repositories(&path);
    candidates.sort();
    for entry_path in candidates {
        let relative = entry_path.strip_prefix(&path).unwrap_or(&entry_path);
        let mut display = relative.to_string_lossy().replace('\\', "/");
        let matches = pattern.is_match(&display)
            || relative
                .file_name()
                .is_some_and(|name| pattern.is_match(Path::new(name)));
        if !matches {
            continue;
        }
        if results.len() >= limit {
            break;
        }
        if std::fs::symlink_metadata(&entry_path).is_ok_and(|metadata| metadata.is_dir()) {
            display.push('/');
        }
        results.push(display);
    }
    if results.is_empty() {
        return Ok(ToolResult::text("No files found matching pattern"));
    }

    let limit_reached = results.len() >= limit;
    let raw_output = results.join("\n");
    let (mut output, byte_truncated) = truncate_tool_output(&raw_output);
    let mut details = serde_json::Map::new();
    let mut notices = Vec::new();
    if limit_reached {
        notices.push(format!("{limit} results limit reached"));
        details.insert("resultLimitReached".into(), json!(limit));
    }
    if byte_truncated {
        notices.push(format!(
            "{} limit reached",
            format_size(TOOL_OUTPUT_MAX_BYTES)
        ));
        details.insert(
            "truncation".into(),
            tool_truncation_details(&raw_output, &output, TOOL_OUTPUT_MAX_BYTES),
        );
    }
    if !notices.is_empty() {
        output.push_str(&format!("\n\n[{}]", notices.join(". ")));
    }
    let mut result = ToolResult::text(output);
    result.details = (!details.is_empty()).then_some(Value::Object(details));
    Ok(result)
}

const MAX_TIMEOUT_SECONDS: f64 = 2_147_483_647.0 / 1000.0;

fn shell_timeout(args: &Value) -> Result<Option<Duration>, ToolError> {
    let Some(value) = args.get("timeout") else {
        return Ok(None);
    };
    let seconds = value.as_f64().ok_or_else(|| {
        ToolError::Command("Invalid timeout: must be a finite number of seconds".into())
    })?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(ToolError::Command(
            "Invalid timeout: must be a finite number of seconds".into(),
        ));
    }
    if seconds * 1000.0 > 2_147_483_647.0 {
        return Err(ToolError::Command(format!(
            "Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"
        )));
    }
    Duration::try_from_secs_f64(seconds)
        .map(Some)
        .map_err(|_| ToolError::Command("Invalid timeout: invalid duration".into()))
}

fn isolate_process_group(process: &mut tokio::process::Command) {
    #[cfg(unix)]
    process.process_group(0);
    #[cfg(not(unix))]
    let _ = process;
}

async fn terminate_process_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(process_id) = child.id() {
        // Each shell command runs in its own process group so descendants are killed too.
        if unsafe { libc::kill(-(process_id as i32), libc::SIGKILL) } == 0 {
            return;
        }
    }
    let _ = child.kill().await;
}

async fn capture_streams<Out, Err>(
    mut stdout: Out,
    mut stderr: Err,
    captured: Arc<AsyncMutex<Vec<u8>>>,
    updates: tokio::sync::mpsc::UnboundedSender<()>,
) where
    Out: AsyncRead + Unpin,
    Err: AsyncRead + Unpin,
{
    let mut stdout_open = true;
    let mut stderr_open = true;
    let mut stdout_chunk = [0; 8192];
    let mut stderr_chunk = [0; 8192];
    while stdout_open || stderr_open {
        tokio::select! {
            result = stdout.read(&mut stdout_chunk), if stdout_open => match result {
                Ok(0) | Err(_) => stdout_open = false,
                Ok(bytes_read) => {
                    captured.lock().await.extend_from_slice(&stdout_chunk[..bytes_read]);
                    let _ = updates.send(());
                },
            },
            result = stderr.read(&mut stderr_chunk), if stderr_open => match result {
                Ok(0) | Err(_) => stderr_open = false,
                Ok(bytes_read) => {
                    captured.lock().await.extend_from_slice(&stderr_chunk[..bytes_read]);
                    let _ = updates.send(());
                },
            },
        }
    }
}

async fn finish_capture(
    mut capture: JoinHandle<()>,
    updates: &mut tokio::sync::mpsc::UnboundedReceiver<()>,
) {
    loop {
        while updates.try_recv().is_ok() {}
        tokio::select! {
            _ = &mut capture => return,
            update = updates.recv() => {
                if update.is_none() {
                    return;
                }
            },
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                capture.abort();
                return;
            }
        }
    }
}

async fn run_shell_process(
    cwd: &Path,
    shell_name: &str,
    temp_file_prefix: &str,
    mut process: tokio::process::Command,
    timeout: Option<Duration>,
    environment: &HashMap<String, String>,
) -> Result<Output, ToolError> {
    if !cwd.exists() {
        return Err(ToolError::Command(format!(
            "Working directory does not exist: {}\nCannot execute {shell_name} commands.",
            cwd.display()
        )));
    }
    process
        .current_dir(cwd)
        .kill_on_drop(true)
        .env_remove("PI_SESSION_ID")
        .env_remove("PI_SESSION_FILE")
        .env_remove("PI_PROVIDER")
        .env_remove("PI_MODEL")
        .env_remove("PI_REASONING_LEVEL")
        .envs(environment)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    isolate_process_group(&mut process);
    let mut child = process.spawn().map_err(ToolError::Io)?;
    let captured = Arc::new(AsyncMutex::new(Vec::new()));
    let (updates_sender, mut updates_receiver) = tokio::sync::mpsc::unbounded_channel();
    let capture = tokio::spawn(capture_streams(
        child.stdout.take().expect("stdout is piped"),
        child.stderr.take().expect("stderr is piped"),
        Arc::clone(&captured),
        updates_sender,
    ));

    let status = match timeout {
        Some(timeout) => match tokio::time::timeout(timeout, child.wait()).await {
            Ok(status) => status.map_err(ToolError::Io)?,
            Err(_) => {
                terminate_process_tree(&mut child).await;
                let _ = child.wait().await;
                finish_capture(capture, &mut updates_receiver).await;
                let full_output =
                    sanitize_shell_output(&String::from_utf8_lossy(&captured.lock().await));
                let (mut message, truncation) = truncate_shell_output(
                    &full_output,
                    BASH_OUTPUT_MAX_LINES,
                    TOOL_OUTPUT_MAX_BYTES,
                );
                if let Some(truncation) = truncation {
                    let path = std::env::temp_dir()
                        .join(format!("{temp_file_prefix}-{}.txt", uuid::Uuid::new_v4()));
                    if std::fs::write(&path, &full_output).is_ok() {
                        append_shell_truncation_notice(
                            &mut message,
                            &truncation,
                            &full_output,
                            &path.to_string_lossy(),
                        );
                    }
                }
                if message.is_empty() {
                    message.push_str("(no output)");
                }
                message.push_str(&format!(
                    "\n\nCommand timed out after {} seconds",
                    timeout.as_secs_f64()
                ));
                return Err(ToolError::Command(message));
            }
        },
        None => child.wait().await.map_err(ToolError::Io)?,
    };
    finish_capture(capture, &mut updates_receiver).await;
    Ok(Output {
        status,
        stdout: captured.lock().await.clone(),
        stderr: Vec::new(),
    })
}

async fn run_bash(
    cwd: &Path,
    args: &Value,
    environment: &HashMap<String, String>,
) -> Result<ToolResult, ToolError> {
    let timeout = shell_timeout(args)?;
    let started_at = Instant::now();
    let command = string_arg(args, "command")?;
    let shell = default_bash_shell()?;
    let mut process = tokio::process::Command::new(shell);
    process.args(["-c", command]);
    let output = run_shell_process(cwd, "bash", "pi-bash", process, timeout, environment).await?;
    Ok(shell_result(output, started_at, "pi-bash"))
}

fn default_bash_shell() -> Result<PathBuf, ToolError> {
    #[cfg(not(windows))]
    {
        Ok(unix_bash_shell(
            Path::new("/bin/bash").is_file(),
            std::env::var_os("PATH"),
        ))
    }
    #[cfg(windows)]
    {
        windows_bash_shell(
            std::env::var_os("ProgramFiles").map(PathBuf::from),
            std::env::var_os("ProgramFiles(x86)").map(PathBuf::from),
            std::env::var_os("PATH"),
        )
        .map_err(ToolError::Command)
    }
}

#[cfg(any(windows, test))]
fn find_executable_on_path(path: Option<std::ffi::OsString>, names: &[&str]) -> Option<PathBuf> {
    let directories = path
        .map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default();
    names.iter().find_map(|name| {
        directories
            .iter()
            .map(|directory| directory.join(name))
            .find(|candidate| candidate.is_file())
    })
}

#[cfg(any(windows, test))]
fn find_powershell_executable(path: Option<std::ffi::OsString>) -> Option<PathBuf> {
    find_executable_on_path(path, &["pwsh.exe", "powershell.exe"])
}

#[cfg(any(windows, test))]
fn windows_bash_shell(
    program_files: Option<PathBuf>,
    program_files_x86: Option<PathBuf>,
    path: Option<std::ffi::OsString>,
) -> Result<PathBuf, String> {
    let git_bash_locations = program_files
        .into_iter()
        .chain(program_files_x86)
        .map(|directory| directory.join("Git/bin/bash.exe"))
        .collect::<Vec<_>>();
    if let Some(shell) = git_bash_locations
        .iter()
        .find(|candidate| candidate.is_file())
    {
        return Ok(shell.to_owned());
    }
    if let Some(shell) = find_executable_on_path(path, &["bash.exe"]) {
        return Ok(shell);
    }
    let searched = git_bash_locations
        .iter()
        .map(|path| format!("  {}", path.display()))
        .collect::<Vec<_>>()
        .join("\n");
    Err(format!(
        "No bash shell found. Options:\n  1. Install Git for Windows: https://git-scm.com/download/win\n  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n  3. Set shellPath in settings.json\n\nSearched Git Bash in:\n{searched}"
    ))
}

fn is_executable_file(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

fn executable_on_unix_path(path: Option<std::ffi::OsString>, name: &str) -> Option<PathBuf> {
    let directories = path
        .map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default();
    directories
        .iter()
        .map(|directory| directory.join(name))
        .find(|candidate| is_executable_file(candidate))
}

#[cfg(any(not(windows), test))]
fn unix_bash_shell(bin_bash_exists: bool, path: Option<std::ffi::OsString>) -> PathBuf {
    if bin_bash_exists {
        PathBuf::from("/bin/bash")
    } else {
        executable_on_unix_path(path, "bash").unwrap_or_else(|| PathBuf::from("sh"))
    }
}

#[cfg(any(windows, test))]
fn powershell_command_with_utf8_output(command: &str) -> String {
    format!(
        "try {{ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 }} catch {{}}\n{command}"
    )
}

async fn run_powershell(
    cwd: &Path,
    args: &Value,
    environment: &HashMap<String, String>,
) -> Result<ToolResult, ToolError> {
    let command = string_arg(args, "command")?;
    let timeout = shell_timeout(args)?;
    #[cfg(not(windows))]
    {
        let _ = (cwd, command, timeout, environment);
        Err(ToolError::Arguments(
            "The powershell tool is only available on Windows.".into(),
        ))
    }
    #[cfg(windows)]
    {
        let started_at = Instant::now();
        let command = powershell_command_with_utf8_output(command);
        let powershell = find_powershell_executable(std::env::var_os("PATH")).ok_or_else(|| {
            ToolError::Command(
                "No PowerShell executable found. Install PowerShell or add powershell.exe/pwsh.exe to PATH."
                    .into(),
            )
        })?;
        let mut process = tokio::process::Command::new(powershell);
        process.args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &command,
        ]);
        let output = run_shell_process(
            cwd,
            "PowerShell",
            "pi-powershell",
            process,
            timeout,
            environment,
        )
        .await?;
        Ok(shell_result(output, started_at, "pi-powershell"))
    }
}

fn sanitize_shell_output(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut characters = input.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\u{1b}' {
            strip_ansi_sequence(&mut characters);
            continue;
        }
        if character == '\u{9b}' {
            strip_csi_sequence(&mut characters);
            continue;
        }
        if character == '\r'
            || (character.is_control() && !matches!(character, '\t' | '\n'))
            || matches!(character, '\u{fff9}'..='\u{fffb}')
        {
            continue;
        }
        output.push(character);
    }
    output
}

fn strip_ansi_sequence(characters: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    let Some(introducer) = characters.next() else {
        return;
    };
    match introducer {
        '[' => strip_csi_sequence(characters),
        ']' => strip_string_sequence(characters, true),
        'P' | '^' | '_' | 'X' => strip_string_sequence(characters, false),
        _ => {}
    }
}

fn strip_csi_sequence(characters: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    for character in characters.by_ref() {
        if ('@'..='~').contains(&character) {
            break;
        }
    }
}

fn strip_string_sequence(
    characters: &mut std::iter::Peekable<std::str::Chars<'_>>,
    bell_terminated: bool,
) {
    let mut escaped = false;
    for character in characters.by_ref() {
        if bell_terminated && character == '\u{7}' {
            break;
        }
        if escaped && character == '\\' {
            break;
        }
        escaped = character == '\u{1b}';
    }
}

fn shell_result(output: Output, started_at: Instant, temp_file_prefix: &str) -> ToolResult {
    const STRUCTURED_OUTPUT_MAX_BYTES: usize = 1024 * 1024;
    let mut full_output = String::from_utf8_lossy(&output.stdout).into_owned();
    full_output.push_str(&String::from_utf8_lossy(&output.stderr));
    let full_output = sanitize_shell_output(&full_output);
    let exit_code = output.status.code();
    let (structured_output, structured_truncated) =
        truncate_utf8(&full_output, STRUCTURED_OUTPUT_MAX_BYTES);
    let (mut content, truncation) =
        truncate_shell_output(&full_output, BASH_OUTPUT_MAX_LINES, TOOL_OUTPUT_MAX_BYTES);
    let mut details = serde_json::Map::new();
    let mut full_output_path = None;
    if let Some(truncation) = truncation {
        let path =
            std::env::temp_dir().join(format!("{temp_file_prefix}-{}.txt", uuid::Uuid::new_v4()));
        if std::fs::write(&path, &full_output).is_ok() {
            let path = path.to_string_lossy().into_owned();
            append_shell_truncation_notice(&mut content, &truncation, &full_output, &path);
            full_output_path = Some(path.clone());
            details.insert("fullOutputPath".into(), Value::String(path));
        }
        details.insert("truncation".into(), truncation);
    } else if full_output.is_empty() {
        content = "(no output)".into();
    }

    let Some(exit_code) = exit_code else {
        content.push_str("\n\nCommand terminated without an exit code");
        return ToolResult {
            content,
            details: (!details.is_empty()).then_some(Value::Object(details)),
            structured_content: None,
            is_error: true,
        };
    };
    let is_error = exit_code != 0;
    if is_error {
        content.push_str(&format!("\n\nCommand exited with code {exit_code}"));
    }
    let wall_time_seconds = (started_at.elapsed().as_millis() as f64 / 100.0).round() / 10.0;
    let mut structured = json!({
        "output": structured_output,
        "truncated": structured_truncated,
        "exit_code": exit_code,
        "wall_time_seconds": wall_time_seconds,
    });
    if let Some(path) = full_output_path {
        structured["full_output_path"] = Value::String(path);
    }
    ToolResult {
        content,
        details: (!details.is_empty()).then_some(Value::Object(details)),
        structured_content: Some(structured),
        is_error,
    }
}

fn truncate_shell_output(
    content: &str,
    max_lines: usize,
    max_bytes: usize,
) -> (String, Option<Value>) {
    let mut lines = if content.is_empty() {
        Vec::new()
    } else {
        content.split('\n').collect::<Vec<_>>()
    };
    if content.ends_with('\n') {
        lines.pop();
    }
    let total_lines = lines.len();
    if total_lines <= max_lines && content.len() <= max_bytes {
        return (content.to_owned(), None);
    }

    let mut tail = Vec::new();
    let mut output_bytes = 0;
    let mut truncated_by = "lines";
    let mut last_line_partial = false;
    for line in lines.iter().rev().take(max_lines) {
        let line_bytes = line.len() + usize::from(!tail.is_empty());
        if output_bytes + line_bytes > max_bytes {
            truncated_by = "bytes";
            if tail.is_empty() {
                let partial = truncate_utf8_from_end(line, max_bytes);
                tail.push(partial);
                last_line_partial = true;
            }
            break;
        }
        output_bytes += line_bytes;
        tail.push((*line).to_owned());
    }
    let output = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
    let output_lines = if output.is_empty() {
        0
    } else {
        output.split('\n').count()
    };
    let truncation = json!({
        "content": output,
        "truncated": true,
        "truncatedBy": truncated_by,
        "totalLines": total_lines,
        "totalBytes": content.len(),
        "outputLines": output_lines,
        "outputBytes": output.len(),
        "lastLinePartial": last_line_partial,
        "firstLineExceedsLimit": false,
        "maxLines": max_lines,
        "maxBytes": max_bytes,
    });
    (output, Some(truncation))
}

fn truncate_utf8_from_end(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_owned();
    }
    let mut start = text.len() - max_bytes;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_owned()
}

fn append_shell_truncation_notice(
    content: &mut String,
    truncation: &Value,
    full_output: &str,
    path: &str,
) {
    let total_lines = truncation["totalLines"].as_u64().unwrap_or(0);
    let output_lines = truncation["outputLines"].as_u64().unwrap_or(0);
    let start = total_lines.saturating_sub(output_lines).saturating_add(1);
    let end = total_lines;
    let notice = if truncation["lastLinePartial"].as_bool() == Some(true) {
        format!(
            "[Showing last {} of line {end} (line is {}). Full output: {path}]",
            format_size(truncation["outputBytes"].as_u64().unwrap_or(0) as usize),
            format_size(
                full_output
                    .strip_suffix('\n')
                    .unwrap_or(full_output)
                    .rsplit('\n')
                    .next()
                    .unwrap_or_default()
                    .len(),
            ),
        )
    } else if truncation["truncatedBy"].as_str() == Some("lines") {
        format!("[Showing lines {start}-{end} of {total_lines}. Full output: {path}]")
    } else {
        format!(
            "[Showing lines {start}-{end} of {total_lines} ({} limit). Full output: {path}]",
            format_size(truncation["maxBytes"].as_u64().unwrap_or(0) as usize)
        )
    };
    content.push_str("\n\n");
    content.push_str(&notice);
}

fn truncate_utf8(text: &str, max_bytes: usize) -> (String, bool) {
    if text.len() <= max_bytes {
        return (text.to_owned(), false);
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_owned(), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn mutation_locks_share_symlink_aliases_and_separate_distinct_paths() {
        use std::os::unix::fs::symlink;

        let dir = std::env::temp_dir().join(format!("mu-mutation-lock-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("target.txt");
        let alias = dir.join("alias.txt");
        std::fs::write(&target, "content").unwrap();
        symlink(&target, &alias).unwrap();

        let target_lock = mutation_lock(&target);
        let alias_lock = mutation_lock(&alias);
        let other_lock = mutation_lock(&dir.join("other.txt"));
        assert!(Arc::ptr_eq(&target_lock, &alias_lock));
        assert!(!Arc::ptr_eq(&target_lock, &other_lock));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unix_bash_discovery_matches_pi_resolution_order_and_sh_fallback() {
        let root = std::env::temp_dir().join(format!("mu-bash-path-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let bash = root.join("bash");
        std::fs::write(&bash, "").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&bash, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let path = std::env::join_paths([&root]).unwrap();

        assert_eq!(unix_bash_shell(true, None), PathBuf::from("/bin/bash"));
        assert_eq!(unix_bash_shell(false, Some(path.clone())), bash);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&bash, std::fs::Permissions::from_mode(0o644)).unwrap();
            assert_eq!(unix_bash_shell(false, Some(path)), PathBuf::from("sh"));
        }
        assert_eq!(unix_bash_shell(false, None), PathBuf::from("sh"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn windows_bash_discovery_prefers_git_locations_then_path_and_reports_searches() {
        let root = std::env::temp_dir().join(format!("mu-windows-bash-{}", uuid::Uuid::new_v4()));
        let program_files = root.join("Program Files");
        let path_dir = root.join("path");
        std::fs::create_dir_all(program_files.join("Git/bin")).unwrap();
        std::fs::create_dir_all(&path_dir).unwrap();
        let git_bash = program_files.join("Git/bin/bash.exe");
        let path_bash = path_dir.join("bash.exe");
        let error = windows_bash_shell(Some(program_files.clone()), None, None).unwrap_err();
        assert!(error.contains("Set shellPath in settings.json"));
        assert!(error.contains(&git_bash.display().to_string()));
        std::fs::write(&path_bash, "").unwrap();
        let path = std::env::join_paths([&path_dir]).unwrap();

        assert_eq!(
            windows_bash_shell(Some(program_files.clone()), None, Some(path.clone())).unwrap(),
            path_bash
        );
        std::fs::write(&git_bash, "").unwrap();
        assert_eq!(
            windows_bash_shell(Some(program_files.clone()), None, Some(path)).unwrap(),
            git_bash
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn powershell_discovery_prefers_pwsh_and_requires_an_executable_on_path() {
        let root = std::env::temp_dir().join(format!("mu-powershell-{}", uuid::Uuid::new_v4()));
        let first = root.join("first");
        let second = root.join("second");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let powershell = first.join("powershell.exe");
        let pwsh = second.join("pwsh.exe");
        std::fs::write(&powershell, "").unwrap();
        std::fs::write(&pwsh, "").unwrap();
        let path = std::env::join_paths([&first, &second]).unwrap();

        assert_eq!(find_powershell_executable(Some(path)), Some(pwsh));
        assert_eq!(find_powershell_executable(None), None);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn powershell_sets_process_local_utf8_output_before_user_commands() {
        assert_eq!(
            powershell_command_with_utf8_output("Get-Location"),
            "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\nGet-Location"
        );
    }

    #[test]
    fn path_inputs_expand_tilde_strip_at_prefix_and_normalize_unicode_spaces() {
        let cwd = Path::new("/tmp/mu-path-test");
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .unwrap();
        assert_eq!(
            resolve_input_path(cwd, "~/folder/file.txt").unwrap(),
            home.join("folder/file.txt")
        );
        assert_eq!(
            resolve_input_path(cwd, "@folder\u{00a0}name/file.txt").unwrap(),
            cwd.join("folder name/file.txt")
        );
    }

    #[test]
    fn read_fallbacks_cover_screenshot_spaces_decomposed_names_and_quotes() {
        let candidates =
            read_path_fallbacks(Path::new("/tmp/2025-01-01 at 4.32. PM. Élodie's file.png"));
        assert!(
            candidates
                .iter()
                .any(|candidate| { candidate.to_string_lossy().contains("4.32.\u{202f}PM.") })
        );
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.to_string_lossy().contains("Élodie’s"))
        );
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.to_string_lossy().contains("E\u{301}lodie"))
        );
    }

    #[tokio::test]
    async fn read_retries_mac_screenshot_filename_spacing() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let requested = dir.join("shot 4.32. PM.png");
        let fallback = read_path_fallbacks(&requested).remove(0);
        std::fs::write(fallback, "image placeholder").unwrap();

        let result = execute(&dir, "read", &json!({"path":"shot 4.32. PM.png"}))
            .await
            .unwrap();
        assert_eq!(result.content, "image placeholder");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn read_write_and_exact_edit_operate_in_working_directory() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        let write = json!({"path":"nested/file.txt","content":"one\ntwo\n"});
        let result = execute(&dir, "write", &write).await.unwrap();
        assert_eq!(result.content, "Successfully wrote to nested/file.txt");
        let read = execute(&dir, "read", &json!({"path":"nested/file.txt","offset":2}))
            .await
            .unwrap();
        assert_eq!(read.content, "two\n");
        execute(
            &dir,
            "edit",
            &json!({"path":"nested/file.txt","oldText":"two","newText":"three"}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("nested/file.txt")).unwrap(),
            "one\nthree\n"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn read_matches_pi_line_and_byte_truncation_notices() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many-lines.txt");
        let lines = (1..=2500)
            .map(|line| format!("Line {line}"))
            .collect::<Vec<_>>();
        std::fs::write(&path, lines.join("\n")).unwrap();

        let result = execute(&dir, "read", &json!({"path":"many-lines.txt"}))
            .await
            .unwrap();

        assert!(result.content.starts_with("Line 1\n"));
        assert!(result.content.contains("Line 2000"));
        assert!(!result.content.contains("Line 2001"));
        assert!(
            result
                .content
                .ends_with("[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]")
        );
        assert_eq!(
            result.details.as_ref().unwrap()["truncation"]["truncatedBy"],
            "lines"
        );
        assert_eq!(
            result.details.as_ref().unwrap()["truncation"]["outputLines"],
            2000
        );
        assert_eq!(
            result.details.as_ref().unwrap()["truncation"]["totalLines"],
            2500
        );

        let byte_lines = (1..=500)
            .map(|line| format!("Line {line}: {}", "x".repeat(200)))
            .collect::<Vec<_>>();
        std::fs::write(&path, byte_lines.join("\n")).unwrap();
        let result = execute(&dir, "read", &json!({"path":"many-lines.txt"}))
            .await
            .unwrap();
        assert!(result.content.starts_with("Line 1:"));
        assert!(result.content.contains("(50.0KB limit). Use offset="));
        assert_eq!(
            result.details.as_ref().unwrap()["truncation"]["truncatedBy"],
            "bytes"
        );

        std::fs::write(&path, "x".repeat(READ_MAX_BYTES + 1)).unwrap();
        let result = execute(&dir, "read", &json!({"path":"many-lines.txt"}))
            .await
            .unwrap();
        assert!(result.content.contains("Line 1 is 50.0KB"));
        assert_eq!(
            result.details.as_ref().unwrap()["truncation"]["firstLineExceedsLimit"],
            true
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn read_offset_and_limit_return_only_text_and_pi_continuation_notices() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let lines = (1..=100)
            .map(|line| format!("Line {line}"))
            .collect::<Vec<_>>();
        std::fs::write(dir.join("file"), lines.join("\n")).unwrap();

        let offset = execute(&dir, "read", &json!({"path":"file","offset":51}))
            .await
            .unwrap();
        assert!(!offset.content.contains("Line 50"));
        assert!(offset.content.starts_with("Line 51\n"));
        assert!(offset.content.ends_with("Line 100"));
        assert!(!offset.content.contains("Use offset="));

        let limited = execute(&dir, "read", &json!({"path":"file","limit":10}))
            .await
            .unwrap();
        assert!(limited.content.starts_with("Line 1\n"));
        assert!(limited.content.contains("Line 10"));
        assert!(!limited.content.contains("Line 11"));
        assert!(
            limited
                .content
                .ends_with("[90 more lines in file. Use offset=11 to continue.]")
        );

        let offset_limited = execute(&dir, "read", &json!({"path":"file","offset":41,"limit":20}))
            .await
            .unwrap();
        assert!(offset_limited.content.starts_with("Line 41\n"));
        assert!(offset_limited.content.contains("Line 60"));
        assert!(!offset_limited.content.contains("Line 61"));
        assert!(
            offset_limited
                .content
                .ends_with("[40 more lines in file. Use offset=61 to continue.]")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn read_rejects_offsets_beyond_the_end_of_file() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("file"), "one\ntwo").unwrap();

        let error = execute(&dir, "read", &json!({"path":"file","offset":3}))
            .await
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "invalid arguments: Offset 3 is beyond end of file (2 lines total)"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn edit_rejects_ambiguous_replacements() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file");
        std::fs::write(&path, "x x").unwrap();
        let result = execute(
            &dir,
            "edit",
            &json!({"path":"file","oldText":"x","newText":"y"}),
        )
        .await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Found 2 occurrences")
        );
        assert_eq!(std::fs::read_to_string(path).unwrap(), "x x");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn edit_rejects_missing_or_empty_text_without_mutating_the_file() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file");
        std::fs::write(&path, "before\nafter\n").unwrap();

        let missing = execute(
            &dir,
            "edit",
            &json!({"path":"file","oldText":"absent","newText":"replacement"}),
        )
        .await
        .unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("Could not find the exact text")
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "before\nafter\n");

        let empty = execute(
            &dir,
            "edit",
            &json!({"path":"file","oldText":"","newText":"replacement"}),
        )
        .await
        .unwrap_err();
        assert!(empty.to_string().contains("oldText must not be empty"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "before\nafter\n");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn edit_preserves_bom_and_crlf_and_accepts_legacy_single_edit_arguments() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file");
        std::fs::write(&path, "\u{feff}alpha\r\nbeta\r\n").unwrap();

        let result = execute(
            &dir,
            "edit",
            &json!({"path":"file","oldText":"beta","newText":"BETA"}),
        )
        .await
        .unwrap();

        assert!(result.content.contains("Successfully replaced 1 block(s)"));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "\u{feff}alpha\r\nBETA\r\n"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn edit_multi_rejects_overlap_and_missing_edits_without_partial_mutation() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file");
        let original = "alpha\nbeta\ngamma\n";
        std::fs::write(&path, original).unwrap();

        let overlap = execute(
            &dir,
            "edit",
            &json!({"path":"file","edits":[
                {"oldText":"alpha\nbeta\n","newText":"A\nB\n"},
                {"oldText":"beta\ngamma\n","newText":"B\nG\n"}
            ]}),
        )
        .await
        .unwrap_err();
        assert!(overlap.to_string().contains("overlap"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);

        let missing = execute(
            &dir,
            "edit",
            &json!({"path":"file","edits":[
                {"oldText":"alpha\n","newText":"ALPHA\n"},
                {"oldText":"missing\n","newText":"MISSING\n"}
            ]}),
        )
        .await
        .unwrap_err();
        assert!(missing.to_string().contains("Could not find edits[1]"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn edit_fuzzy_matches_unicode_and_preserves_untouched_lines() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file");
        let original = "keep before  \nline one   \nline two  \nkeep after  \n";
        std::fs::write(&path, original).unwrap();

        execute(
            &dir,
            "edit",
            &json!({"path":"file","edits":[{"oldText":"line one\nline two\n","newText":"replaced\n"}]}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "keep before  \nreplaced\nkeep after  \n"
        );

        std::fs::write(&path, "ＡＢＣ１２３\ncafe\u{0301}\n").unwrap();
        execute(
            &dir,
            "edit",
            &json!({"path":"file","edits":[{"oldText":"ABC123\ncafé\n","newText":"XYZ789\ncoffee\n"}]}),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "XYZ789\ncoffee\n");

        std::fs::write(
            &path,
            "console.log(‘hello’);\nrange: 1–5\nhello\u{00a0}world\n",
        )
        .unwrap();
        execute(
            &dir,
            "edit",
            &json!({"path":"file","edits":[
                {"oldText":"console.log('hello');\n","newText":"console.log('world');\n"},
                {"oldText":"range: 1-5\n","newText":"range: 10-50\n"},
                {"oldText":"hello world\n","newText":"hello universe\n"}
            ]}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "console.log('world');\nrange: 10-50\nhello universe\n"
        );

        std::fs::write(&path, "replace me   \nafter   \n").unwrap();
        execute(
            &dir,
            "edit",
            &json!({"path":"file","edits":[{"oldText":"replace me\n","newText":"after\n"}]}),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after\nafter   \n");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn edit_fuzzy_matching_rejects_normalized_duplicates_without_mutation() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file");
        let original = "hello world  \nhello world\n";
        std::fs::write(&path, original).unwrap();

        let error = execute(
            &dir,
            "edit",
            &json!({"path":"file","oldText":"hello world","newText":"replacement"}),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("Found 2 occurrences"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn ls_sorts_entries_and_reports_limit_continuation() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["gamma", "beta", "alpha"] {
            std::fs::write(dir.join(name), "").unwrap();
        }

        let result = execute(&dir, "ls", &json!({"limit":2})).await.unwrap();

        assert_eq!(
            result.content,
            "alpha\nbeta\n\n[2 entries limit reached. Use limit=4 for more]"
        );
        assert_eq!(
            result.details.as_ref().unwrap()["entryLimitReached"].as_f64(),
            Some(2.0)
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn ls_reports_empty_and_invalid_directories() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            execute(&dir, "ls", &json!({})).await.unwrap().content,
            "(empty directory)"
        );
        let file = dir.join("file");
        std::fs::write(&file, "text").unwrap();
        let error = execute(&dir, "ls", &json!({"path":"file"}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("Not a directory"));
        let missing = execute(&dir, "ls", &json!({"path":"missing"}))
            .await
            .unwrap_err();
        assert!(missing.to_string().contains("Path not found"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_validates_duration_and_terminates_the_process() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let invalid = execute(&dir, "bash", &json!({"command":"true","timeout":0}))
            .await
            .unwrap_err();
        assert_eq!(
            invalid.to_string(),
            "Invalid timeout: must be a finite number of seconds"
        );
        let too_long = execute(
            &dir,
            "bash",
            &json!({"command":"true","timeout":2_147_483.648}),
        )
        .await
        .unwrap_err();
        assert_eq!(
            too_long.to_string(),
            "Invalid timeout: maximum is 2147483.647 seconds"
        );

        let timeout = execute(
            &dir,
            "bash",
            &json!({"command":"echo started; while :; do :; done","timeout":2}),
        )
        .await
        .unwrap_err();
        assert!(
            timeout
                .to_string()
                .contains("Command timed out after 2 seconds")
        );
        assert!(timeout.to_string().contains("started\n\n"), "{timeout}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_collects_output_from_recently_exited_shell_descendants() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let result = execute(
            &dir,
            "bash",
            &json!({"command":"(for i in {1..30}; do sleep 0.01; printf x; done) &"}),
        )
        .await
        .unwrap();

        assert_eq!(result.structured_content.unwrap()["output"], "x".repeat(30));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_preserves_interleaved_stdout_and_stderr_order() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let result = execute(
            &dir,
            "bash",
            &json!({"command":"printf out1; sleep 0.05; printf err1 >&2; sleep 0.05; printf out2"}),
        )
        .await
        .unwrap();

        assert_eq!(result.structured_content.unwrap()["output"], "out1err1out2");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_signal_termination_matches_pi_error_result() {
        let result = execute(Path::new("."), "bash", &json!({"command":"kill -TERM $$"}))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(
            result
                .content
                .ends_with("Command terminated without an exit code")
        );
        assert!(result.structured_content.is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_persists_full_truncated_output_like_pi() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let error = execute(
            &dir,
            "bash",
            &json!({"command":"i=0; while [ $i -lt 3000 ]; do printf 'line\\n'; i=$((i + 1)); done; while :; do :; done","timeout":1}),
        )
        .await
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("Command timed out after 1 seconds"));
        assert!(
            message.contains("Showing lines 1001-3000 of 3000"),
            "{message}"
        );
        let full_path = message
            .split("Full output: ")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap();
        assert!(
            Path::new(full_path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("pi-bash-")
        );
        assert_eq!(
            std::fs::read_to_string(full_path).unwrap().lines().count(),
            3000
        );
        std::fs::remove_file(full_path).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_terminates_descendants_in_the_command_process_group() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("late-child-output");
        let command = format!("(sleep 0.5; printf alive > '{}') & wait", marker.display());

        let error = execute(&dir, "bash", &json!({"command":command,"timeout":0.1}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("Command timed out"));
        tokio::time::sleep(Duration::from_millis(650)).await;
        assert!(
            !marker.exists(),
            "the killed child must not write after timeout"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn search_tool_truncation_details_match_pi_byte_boundaries() {
        let first_line = "x".repeat(TOOL_OUTPUT_MAX_BYTES + 1);
        let content = format!("{first_line}\nsecond line");
        let (truncated, was_truncated) = truncate_tool_output(&content);
        assert!(was_truncated);
        assert!(truncated.is_empty());
        let details = tool_truncation_details(&content, &truncated, TOOL_OUTPUT_MAX_BYTES);
        assert_eq!(details["firstLineExceedsLimit"], true);
        assert_eq!(details["truncatedBy"], "bytes");
        assert_eq!(details["outputBytes"], 0);
        assert_eq!(details["totalBytes"], content.len());
    }

    #[test]
    fn shell_output_removes_ansi_and_binary_controls_but_keeps_tabs_and_newlines() {
        assert_eq!(
            super::sanitize_shell_output(
                "\u{1b}[31mred\u{1b}[0m\r\n\u{1}x\t!\u{fff9}a\u{fffa}b\u{fffb}",
            ),
            "red\nx\t!ab"
        );
        assert_eq!(
            super::sanitize_shell_output(
                "\u{1b}]8;;https://example.test\u{7}linked\u{1b}]8;;\u{7}"
            ),
            "linked"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_receives_session_metadata_environment() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let environment = HashMap::from([
            ("PI_SESSION_ID".to_owned(), "session-test".to_owned()),
            ("PI_MODEL".to_owned(), "model-test".to_owned()),
        ]);

        let result = execute_with_environment(
            &dir,
            "bash",
            &json!({"command":"printf '%s/%s' \"$PI_SESSION_ID\" \"$PI_MODEL\""}),
            &environment,
        )
        .await
        .unwrap();

        assert_eq!(result.content, "session-test/model-test");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_runs_with_bash_semantics_instead_of_posix_sh() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let result = execute(
            &dir,
            "bash",
            &json!({"command":"[[ -n bash ]] && printf bash-version"}),
        )
        .await
        .unwrap();

        assert!(result.content.starts_with("bash-version"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn bash_runs_in_the_supplied_working_directory() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        #[cfg(windows)]
        let command = "cd";
        #[cfg(not(windows))]
        let command = "pwd";

        let result = execute(&dir, "bash", &json!({"command":command}))
            .await
            .unwrap();

        assert!(
            result
                .content
                .trim()
                .ends_with(dir.file_name().unwrap().to_str().unwrap())
        );
        let empty = execute(&dir, "bash", &json!({"command":"true"}))
            .await
            .unwrap();
        assert_eq!(empty.content, "(no output)");
        assert_eq!(empty.structured_content.as_ref().unwrap()["output"], "");
        assert_eq!(empty.structured_content.as_ref().unwrap()["exit_code"], 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_decodes_utf8_split_across_writes_after_collecting_output() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let result = execute(
            &dir,
            "bash",
            &json!({"command":"printf '\\303'; sleep 0.01; printf '\\251'"}),
        )
        .await
        .unwrap();

        assert_eq!(result.content, "é");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn bash_tail_truncation_keeps_recent_lines_and_respects_utf8_byte_boundaries() {
        let (lines, line_details) = truncate_shell_output("one\ntwo\nthree\n", 2, 100);
        assert_eq!(lines, "two\nthree");
        let line_details = line_details.unwrap();
        assert_eq!(line_details["truncatedBy"], "lines");
        assert_eq!(line_details["totalLines"], 3);
        assert_eq!(line_details["outputLines"], 2);

        let (bytes, byte_details) = truncate_shell_output("start\nééé", 10, 3);
        assert_eq!(bytes, "é");
        let byte_details = byte_details.unwrap();
        assert_eq!(byte_details["truncatedBy"], "bytes");
        assert_eq!(byte_details["lastLinePartial"], true);
        assert_eq!(byte_details["outputBytes"], 2);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_tail_truncation_persists_full_output_for_retrieval() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let result = execute(
            &dir,
            "bash",
            &json!({"command":"i=0; while [ $i -lt 2001 ]; do printf 'line\\n'; i=$((i + 1)); done"}),
        )
        .await
        .unwrap();
        let details = result.details.unwrap();
        assert_eq!(details["truncation"]["truncatedBy"], "lines");
        assert_eq!(details["truncation"]["totalLines"], 2001);
        assert_eq!(details["truncation"]["outputLines"], 2000);
        assert_eq!(result.content.lines().count(), 2002);
        let full_path = PathBuf::from(details["fullOutputPath"].as_str().unwrap());
        assert!(
            full_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("pi-bash-")
        );
        let full_output = std::fs::read_to_string(&full_path).unwrap();
        assert_eq!(full_output.lines().count(), 2001);
        assert_eq!(
            result.structured_content.unwrap()["full_output_path"],
            full_path.to_string_lossy().as_ref()
        );
        std::fs::remove_file(full_path).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn bash_caps_structured_output_at_one_megabyte() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let result = execute(&dir, "bash", &json!({"command":"yes x | head -c 1048577"}))
            .await
            .unwrap();

        let structured = result.structured_content.unwrap();
        assert!(structured["truncated"].as_bool().unwrap());
        assert!(structured["output"].as_str().unwrap().len() <= 1024 * 1024);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_error_result_preserves_stdout_and_stderr() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let result = execute(
            &dir,
            "bash",
            &json!({"command":"printf stdout; printf stderr >&2; exit 7"}),
        )
        .await
        .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("Command exited with code 7"));
        assert!(result.content.contains("stdout"));
        assert!(result.content.contains("stderr"));
        assert_eq!(result.structured_content.as_ref().unwrap()["exit_code"], 7);
        assert_eq!(
            result.structured_content.as_ref().unwrap()["truncated"],
            false
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
