use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    process::{Output, Stdio},
    sync::{Arc, Mutex as StdMutex, OnceLock, Weak},
    time::{Duration, Instant},
};

use globset::{Glob, GlobBuilder, GlobMatcher};
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

pub(crate) const SHELL_COMMAND_PREFIX_ENV: &str = "MU_INTERNAL_SHELL_COMMAND_PREFIX";
pub(crate) const BASH_SHELL_PATH_ENV: &str = "MU_INTERNAL_BASH_SHELL_PATH";

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    Arguments(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Command(String),
}

/// A base64 image attached to a tool result (Pi's `ImageContent`).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolImage {
    pub mime_type: String,
    pub data: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    pub images: Vec<ToolImage>,
    /// Plain model-facing text until the agent's content-block bridge is complete.
    pub content: String,
    pub details: Option<Value>,
    pub structured_content: Option<Value>,
    pub is_error: bool,
}

impl ToolResult {
    fn text(content: impl Into<String>) -> Self {
        Self {
            images: Vec::new(),
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
        json!({"type":"function","name":"read","description":"Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to read (relative or absolute)"},"offset":{"type":"number","description":"Line number to start reading from (1-indexed)"},"limit":{"type":"number","description":"Maximum number of lines to read"}},"required":["path"],"additionalProperties":false}}),
        json!({"type":"function","name":"bash","description":"Execute a Bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Shell command to execute"},"timeout":{"type":"number","description":"Timeout in seconds (optional, no default timeout)"}},"required":["command"],"additionalProperties":false}}),
        json!({"type":"function","name":"powershell","description":"Execute a PowerShell command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Shell command to execute"},"timeout":{"type":"number","description":"Timeout in seconds (optional, no default timeout)"}},"required":["command"],"additionalProperties":false}}),
        json!({"type":"function","name":"edit","description":"Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to edit (relative or absolute)"},"edits":{"type":"array","description":"One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.","items":{"type":"object","properties":{"oldText":{"type":"string","description":"Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."},"newText":{"type":"string","description":"Replacement text for this targeted edit."}},"required":["oldText","newText"],"additionalProperties":false}}},"required":["path","edits"],"additionalProperties":false}}),
        json!({"type":"function","name":"write","description":"Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to write (relative or absolute)"},"content":{"type":"string","description":"Content to write to the file"}},"required":["path","content"],"additionalProperties":false}}),
        json!({"type":"function","name":"grep","description":"Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Search pattern (regex or literal string)"},"path":{"type":"string","description":"Directory or file to search (default: current directory)"},"glob":{"type":"string","description":"Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"},"ignoreCase":{"type":"boolean","description":"Case-insensitive search (default: false)"},"literal":{"type":"boolean","description":"Treat pattern as literal string instead of regex (default: false)"},"context":{"type":"number","description":"Number of lines to show before and after each match (default: 0)"},"limit":{"type":"number","description":"Maximum number of matches to return (default: 100)"}},"required":["pattern"],"additionalProperties":false}}),
        json!({"type":"function","name":"find","description":"Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated to 1000 results or 50KB (whichever is hit first).","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"},"path":{"type":"string","description":"Directory to search in (default: current directory)"},"limit":{"type":"number","description":"Maximum number of results (default: 1000)"}},"required":["pattern"],"additionalProperties":false}}),
        json!({"type":"function","name":"ls","description":"List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first).","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory to list (default: current directory)"},"limit":{"type":"number","description":"Maximum number of entries to return (default: 500)"}},"additionalProperties":false}}),
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
    #[cfg(windows)]
    let normalized = crate::path_utils::normalize_windows_shell_path(&normalized);
    let path = if normalized.starts_with("file://") {
        url::Url::parse(&normalized)
            .ok()
            .and_then(|url| url.to_file_path().ok())
            .ok_or_else(|| ToolError::Arguments(format!("invalid file URL `{normalized}`")))?
    } else {
        PathBuf::from(normalized)
    };
    let absolute = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    Ok(normalize_absolute_path(absolute))
}

fn normalize_absolute_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                normalized.push(component.as_os_str());
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() && !path.is_absolute() {
                    normalized.push(component.as_os_str());
                }
            }
            std::path::Component::Normal(_) => normalized.push(component.as_os_str()),
        }
    }
    normalized
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

fn read_file_io_error(path: &Path, error: std::io::Error) -> ToolError {
    if error.kind() == std::io::ErrorKind::IsADirectory {
        return ToolError::Command(format!(
            "EISDIR: illegal operation on a directory, read '{}'",
            path.display()
        ));
    }
    let (code, message) = match error.kind() {
        std::io::ErrorKind::NotFound => ("ENOENT", "no such file or directory"),
        std::io::ErrorKind::PermissionDenied => ("EACCES", "permission denied"),
        _ => return ToolError::Io(error),
    };
    ToolError::Command(format!("{code}: {message}, access '{}'", path.display()))
}

/// Image type by file magic (Pi's `detectSupportedImageMimeTypeFromFile`, minus BMP conversion).
fn detect_image_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") && bytes.len() > 26 {
        Some("image/bmp")
    } else {
        None
    }
}

fn read_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    read_file_with(&LocalFiles, cwd, args)
}

/// The read tool over any [`FileOperations`] (Pi's `ReadOperations`).
pub fn read_file_with(
    files: &dyn FileOperations,
    cwd: &Path,
    args: &Value,
) -> Result<ToolResult, ToolError> {
    let requested_path = string_arg(args, "path")?;
    let path = resolve_read_path(cwd, args)?;
    let bytes = files
        .read(&path)
        .map_err(|error| read_file_io_error(&path, error))?;
    if let Some(mime_type) = detect_image_mime(&bytes) {
        let options = crate::image_process::ResizeOptions::default();
        return Ok(
            match crate::image_process::process_image(&bytes, mime_type, true, options) {
                Ok(processed) => {
                    let mut note = format!("Read image file [{}]", processed.mime_type);
                    if !processed.hints.is_empty() {
                        note = format!(
                            "Read image file [{mime_type}]\n{}",
                            processed.hints.join("\n")
                        );
                    }
                    ToolResult {
                        images: vec![ToolImage {
                            mime_type: processed.mime_type,
                            data: processed.data,
                        }],
                        ..ToolResult::text(note)
                    }
                }
                Err(message) => {
                    ToolResult::text(format!("Read image file [{mime_type}]\n{message}"))
                }
            },
        );
    }
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

fn write_file_io_error(path: &Path, operation: &str, error: std::io::Error) -> ToolError {
    if operation == "mkdir"
        && error.kind() == std::io::ErrorKind::NotADirectory
        && path.exists()
        && !path.is_dir()
    {
        return ToolError::Command(format!(
            "EEXIST: file already exists, mkdir '{}'",
            path.display()
        ));
    }
    let (code, message) = match error.kind() {
        std::io::ErrorKind::NotFound => ("ENOENT", "no such file or directory"),
        std::io::ErrorKind::PermissionDenied => ("EACCES", "permission denied"),
        std::io::ErrorKind::NotADirectory => ("ENOTDIR", "not a directory"),
        std::io::ErrorKind::IsADirectory => ("EISDIR", "illegal operation on a directory"),
        std::io::ErrorKind::AlreadyExists => ("EEXIST", "file already exists"),
        _ => return ToolError::Io(error),
    };
    ToolError::Command(format!(
        "{code}: {message}, {operation} '{}'",
        path.display()
    ))
}

async fn write_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    write_file_with(&LocalFiles, cwd, args).await
}

/// The write tool over any [`FileOperations`] (Pi's `WriteOperations`).
pub async fn write_file_with(
    files: &dyn FileOperations,
    cwd: &Path,
    args: &Value,
) -> Result<ToolResult, ToolError> {
    let requested_path = string_arg(args, "path")?;
    let path = resolve(cwd, args)?;
    let content = string_arg(args, "content")?;
    let lock = mutation_lock(&path);
    let _guard = lock.lock().await;
    if let Some(parent) = path.parent() {
        files
            .create_dir_all(parent)
            .map_err(|error| write_file_io_error(parent, "mkdir", error))?;
    }
    files
        .write(&path, content.as_bytes())
        .map_err(|error| write_file_io_error(&path, "open", error))?;
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

/// Injectable file access for the edit tool (Pi's `EditOperations`).
pub trait FileOperations: Send + Sync {
    fn read(&self, path: &Path) -> std::io::Result<Vec<u8>>;
    fn write(&self, path: &Path, contents: &[u8]) -> std::io::Result<()>;
    fn create_dir_all(&self, path: &Path) -> std::io::Result<()>;
}

/// The real filesystem.
pub struct LocalFiles;

impl FileOperations for LocalFiles {
    fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        std::fs::read(path)
    }

    fn write(&self, path: &Path, contents: &[u8]) -> std::io::Result<()> {
        std::fs::write(path, contents)
    }

    fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(path)
    }
}

async fn edit_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    edit_file_with(&LocalFiles, cwd, args).await
}

/// The edit tool over any [`FileOperations`].
pub async fn edit_file_with(
    files: &dyn FileOperations,
    cwd: &Path,
    args: &Value,
) -> Result<ToolResult, ToolError> {
    let requested_path = string_arg(args, "path")?;
    let display_path = Path::new(requested_path);
    let path = resolve(cwd, args)?;
    let edits = parse_edits(args)?;
    let lock = mutation_lock(&path);
    let _guard = lock.lock().await;
    let original = files
        .read(&path)
        .map_err(|error| ToolError::Command(edit_read_io_message(display_path, &path, error)))
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
    files
        .write(&path, format!("{bom}{restored}").as_bytes())
        .map_err(|error| write_file_io_error(&path, "open", error))?;
    let mut result = ToolResult::text(format!(
        "Successfully replaced {} block(s) in {}.",
        normalized_edits.len(),
        display_path.display()
    ));
    result.details = Some(details);
    Ok(result)
}

fn edit_read_io_message(
    requested_path: &Path,
    absolute_path: &Path,
    error: std::io::Error,
) -> String {
    let path = if error.kind() == std::io::ErrorKind::IsADirectory {
        absolute_path
    } else {
        requested_path
    };
    edit_file_io_message(path, error)
}

fn edit_file_io_message(path: &Path, error: std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::IsADirectory {
        return format!(
            "EISDIR: illegal operation on a directory, read '{}'",
            path.display()
        );
    }
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => "ENOENT",
        std::io::ErrorKind::PermissionDenied => "EACCES",
        _ => {
            return format!("Could not edit file: {}. Error: {error}.", path.display());
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
            return json!({
                "error": edit_read_io_message(Path::new(requested_path), &path, error)
            });
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
        None | Some(Value::Null) => cwd.to_owned(),
        Some(Value::String(path)) => resolve_input_path(cwd, path)?,
        Some(_) => return Err(ToolError::Arguments("`path` must be a string".into())),
    };
    let limit = match object.get("limit") {
        None | Some(Value::Null) => LS_DEFAULT_LIMIT as f64,
        Some(value) => value
            .as_f64()
            .filter(|limit| limit.is_finite())
            .ok_or_else(|| ToolError::Arguments("`limit` must be a finite number".into()))?,
    };
    Ok(ListDirectoryInput { path, limit })
}

fn trim_javascript_whitespace(line: &str) -> &str {
    line.trim_matches(is_javascript_trim_whitespace)
}

fn find_path_to_posix(path: &str) -> String {
    #[cfg(windows)]
    {
        path.replace('\\', "/")
    }
    #[cfg(not(windows))]
    {
        path.to_owned()
    }
}

fn is_javascript_trim_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'..='\u{000d}'
            | '\u{0020}'
            | '\u{00a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'..='\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
            | '\u{feff}'
    )
}

fn node_stat_error(path: &Path, error: std::io::Error) -> String {
    let (code, message) = match error.kind() {
        std::io::ErrorKind::NotFound => ("ENOENT", "no such file or directory"),
        std::io::ErrorKind::PermissionDenied => ("EACCES", "permission denied"),
        std::io::ErrorKind::NotADirectory => ("ENOTDIR", "not a directory"),
        _ => return error.to_string(),
    };
    format!("{code}: {message}, stat '{}'", path.display())
}

fn node_scandir_error(path: &Path, error: std::io::Error) -> String {
    let (code, message) = match error.kind() {
        std::io::ErrorKind::NotFound => ("ENOENT", "no such file or directory"),
        std::io::ErrorKind::PermissionDenied => ("EACCES", "permission denied"),
        std::io::ErrorKind::NotADirectory => ("ENOTDIR", "not a directory"),
        _ => return error.to_string(),
    };
    format!("{code}: {message}, scandir '{}'", path.display())
}

fn relative_find_path(search_root: &Path, result_path: &Path) -> PathBuf {
    if let Ok(relative) = result_path.strip_prefix(search_root) {
        return relative.to_owned();
    }
    if !search_root.is_absolute() || !result_path.is_absolute() {
        return result_path.to_owned();
    }

    let search_prefix = search_root
        .components()
        .find_map(|component| match component {
            std::path::Component::Prefix(prefix) => Some(prefix.as_os_str()),
            _ => None,
        });
    let result_prefix = result_path
        .components()
        .find_map(|component| match component {
            std::path::Component::Prefix(prefix) => Some(prefix.as_os_str()),
            _ => None,
        });
    if search_prefix != result_prefix {
        return result_path.to_owned();
    }

    let root_components = search_root
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(name) => Some(name.to_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let result_components = result_path
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(name) => Some(name.to_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let common = root_components
        .iter()
        .zip(&result_components)
        .take_while(|(root, result)| root == result)
        .count();
    let mut relative = PathBuf::new();
    for _ in common..root_components.len() {
        relative.push("..");
    }
    for component in &result_components[common..] {
        relative.push(component);
    }
    relative
}

fn relativize_fd_find_output(
    entry_path: &Path,
    search_root: &Path,
    is_directory: bool,
) -> Vec<String> {
    let mut raw_output_path = entry_path.to_string_lossy().into_owned();
    if is_directory && !raw_output_path.ends_with(std::path::MAIN_SEPARATOR) {
        raw_output_path.push(std::path::MAIN_SEPARATOR);
    }

    raw_output_path
        .split('\n')
        .filter_map(|raw_line| {
            let line = trim_javascript_whitespace(raw_line);
            if line.is_empty() {
                return None;
            }
            let had_trailing_separator = line.ends_with('/') || line.ends_with('\\');
            let path = Path::new(line);
            let relative = if path.is_absolute() {
                relative_find_path(search_root, path)
            } else {
                path.to_owned()
            };
            let mut result = find_path_to_posix(&relative.to_string_lossy());
            if had_trailing_separator && !result.ends_with('/') {
                result.push('/');
            }
            Some(result)
        })
        .collect()
}

fn sort_directory_entries(entries: &mut [String]) {
    entries.sort_by_key(|entry| entry.to_lowercase());
}

/// Injectable directory access for the ls tool (Pi's `LsOperations`).
pub trait DirectoryOperations: Send + Sync {
    fn exists(&self, path: &Path) -> bool;
    /// Whether `path` is a directory, following symlinks.
    fn is_dir(&self, path: &Path) -> std::io::Result<bool>;
    /// Entry names directly under `path`.
    fn read_dir(&self, path: &Path) -> std::io::Result<Vec<String>>;
}

impl DirectoryOperations for LocalFiles {
    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn is_dir(&self, path: &Path) -> std::io::Result<bool> {
        std::fs::metadata(path).map(|metadata| metadata.is_dir())
    }

    fn read_dir(&self, path: &Path) -> std::io::Result<Vec<String>> {
        std::fs::read_dir(path)?
            .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
            .collect()
    }
}

fn list_directory(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    list_directory_with(&LocalFiles, cwd, args)
}

/// The ls tool over any [`DirectoryOperations`].
pub fn list_directory_with(
    files: &dyn DirectoryOperations,
    cwd: &Path,
    args: &Value,
) -> Result<ToolResult, ToolError> {
    let ListDirectoryInput { path, limit } = parse_list_directory_input(cwd, args)?;
    if !files.exists(&path) {
        return Err(ToolError::Command(format!(
            "Path not found: {}",
            path.display()
        )));
    }
    let is_dir = files
        .is_dir(&path)
        .map_err(|error| ToolError::Command(node_stat_error(&path, error)))?;
    if !is_dir {
        return Err(ToolError::Command(format!(
            "Not a directory: {}",
            path.display()
        )));
    }

    let mut entries = files.read_dir(&path).map_err(|error| {
        ToolError::Command(format!(
            "Cannot read directory: {}",
            node_scandir_error(&path, error)
        ))
    })?;
    sort_directory_entries(&mut entries);

    let mut names = Vec::new();
    let mut limit_reached = false;
    for name in entries {
        if names.len() as f64 >= limit {
            limit_reached = true;
            break;
        }
        if let Ok(is_dir) = files.is_dir(&path.join(&name)) {
            names.push(if is_dir { format!("{name}/") } else { name });
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
    context: f64,
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
        None | Some(Value::Null) => cwd.to_owned(),
        Some(Value::String(path)) => resolve_input_path(cwd, path)?,
        Some(_) => return Err(ToolError::Arguments("`path` must be a string".into())),
    };
    let glob = match object.get("glob") {
        None | Some(Value::Null) => None,
        Some(Value::String(glob)) if glob.is_empty() => None,
        Some(Value::String(glob)) => Some(
            Glob::new(glob)
                .map_err(|error| ToolError::Command(format!("rg: {error}")))?
                .compile_matcher(),
        ),
        Some(_) => return Err(ToolError::Arguments("`glob` must be a string".into())),
    };
    let ignore_case = match object.get("ignoreCase") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(ToolError::Arguments(
                "`ignoreCase` must be a boolean".into(),
            ));
        }
    };
    let literal = match object.get("literal") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => return Err(ToolError::Arguments("`literal` must be a boolean".into())),
    };
    let context = match object.get("context") {
        None | Some(Value::Null) => 0.0,
        Some(value) => {
            let context = value
                .as_f64()
                .filter(|context| context.is_finite())
                .ok_or_else(|| ToolError::Arguments("`context` must be a finite number".into()))?;
            context.max(0.0)
        }
    };
    let limit = match object.get("limit") {
        None | Some(Value::Null) => GREP_DEFAULT_LIMIT as f64,
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
        .map_err(|error| ToolError::Command(format!("rg: {error}")))?;
    Ok(GrepInput {
        regex,
        path,
        glob,
        context,
        limit,
    })
}

fn grep_files(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let input = parse_grep_input(cwd, args)?;
    let is_directory = input.path.is_dir();
    if !is_directory && !input.path.is_file() {
        return Err(ToolError::Command(format!(
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
        let Ok(bytes) = std::fs::read(&file) else {
            continue;
        };
        // In JSON mode ripgrep treats NUL bytes as line boundaries and emits
        // match events from binary files; Node decodes other invalid UTF-8 with
        // replacement when formatting the matched text.
        let decoded = String::from_utf8_lossy(&bytes);
        let content = decoded.replace('\0', "\n");
        let match_lines = content
            .split_terminator('\n')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let context_lines = decoded
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .split('\n')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for (index, line) in match_lines.iter().enumerate() {
            if !regex.is_match(line) {
                continue;
            }
            match_count += 1;
            let reached_match_limit = match_count as f64 >= input.limit;
            if input.context == 0.0 {
                let sanitized = line.replace('\r', "");
                let (text, was_truncated) = truncate_grep_line(&sanitized);
                lines_truncated |= was_truncated;
                formatted_matches.push(format!("{display_path}:{}: {text}", index + 1));
                if reached_match_limit {
                    match_limit_reached = true;
                    break 'files;
                }
                continue;
            }

            let match_line = (index + 1) as f64;
            let start_line = (match_line - input.context).max(1.0);
            let end_line = (match_line + input.context).min(context_lines.len() as f64);
            let mut current_line = start_line;
            while current_line <= end_line {
                let text = if current_line.fract() == 0.0 {
                    context_lines
                        .get(current_line as usize - 1)
                        .map(String::as_str)
                        .unwrap_or("")
                } else {
                    ""
                };
                let sanitized = text.replace('\r', "");
                let (text, was_truncated) = truncate_grep_line(&sanitized);
                lines_truncated |= was_truncated;
                let line_number = format_json_number(current_line);
                if current_line == match_line {
                    formatted_matches.push(format!("{display_path}:{line_number}: {text}"));
                } else {
                    formatted_matches.push(format!("{display_path}-{line_number}- {text}"));
                }
                current_line += 1.0;
            }
            if reached_match_limit {
                match_limit_reached = true;
                break 'files;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FindPatternTarget {
    Basename,
    RelativePath,
    AbsolutePath,
}

struct FindInput {
    pattern: GlobMatcher,
    path: PathBuf,
    limit: usize,
    target: FindPatternTarget,
}

fn parse_find_input(cwd: &Path, args: &Value) -> Result<FindInput, ToolError> {
    let object = args
        .as_object()
        .ok_or_else(|| ToolError::Arguments("find arguments must be an object".into()))?;
    let pattern = object
        .get("pattern")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::Arguments("`pattern` must be a string".into()))?;
    let path = match object.get("path") {
        None | Some(Value::Null) => cwd.to_owned(),
        Some(Value::String(path)) => resolve_input_path(cwd, path)?,
        Some(_) => return Err(ToolError::Arguments("`path` must be a string".into())),
    };
    let target = if !pattern.contains('/') {
        FindPatternTarget::Basename
    } else if Path::new(pattern).is_absolute() {
        FindPatternTarget::AbsolutePath
    } else {
        FindPatternTarget::RelativePath
    };
    let effective_pattern = if pattern.is_empty() {
        "**".to_owned()
    } else if target == FindPatternTarget::RelativePath
        && !pattern.starts_with("**/")
        && pattern != "**"
    {
        format!("**/{pattern}")
    } else {
        pattern.to_owned()
    };
    let case_insensitive = !effective_pattern.chars().any(char::is_uppercase);
    let pattern = GlobBuilder::new(&effective_pattern)
        .case_insensitive(case_insensitive)
        .build()
        .map_err(|error| ToolError::Command(format!("[fd error]: {error}")))?
        .compile_matcher();
    let limit = parse_find_limit(object.get("limit"))?;
    Ok(FindInput {
        pattern,
        path,
        limit,
        target,
    })
}

fn parse_find_limit(value: Option<&Value>) -> Result<usize, ToolError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(FIND_DEFAULT_LIMIT);
    };
    let Some(limit) = value.as_f64().filter(|limit| limit.is_finite()) else {
        return Err(ToolError::Arguments(
            "`limit` must be a finite number".into(),
        ));
    };
    let rendered = value.to_string();
    let fd_error = |message: &str| {
        ToolError::Command(format!(
            "error: {message}\n\nFor more information, try '--help'."
        ))
    };
    if limit < 0.0 {
        return Err(fd_error(
            "a value is required for '--max-results <count>' but none was supplied",
        ));
    }
    if limit.fract() != 0.0 {
        return Err(fd_error(&format!(
            "invalid value '{rendered}' for '--max-results <count>': invalid digit found in string"
        )));
    }
    if let Some(integer) = value.as_u64() {
        return usize::try_from(integer).map_err(|_| {
            fd_error(&format!(
                "invalid value '{rendered}' for '--max-results <count>': number too large to fit in target type"
            ))
        });
    }
    if limit >= u64::MAX as f64 || limit > usize::MAX as f64 {
        return Err(fd_error(&format!(
            "invalid value '{rendered}' for '--max-results <count>': number too large to fit in target type"
        )));
    }
    Ok(limit as usize)
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
        target,
    } = parse_find_input(cwd, args)?;
    if !path.is_dir() {
        return Err(ToolError::Command(format!(
            "[fd error]: Search path '{}' is not a directory.\n[fd error]: No valid search paths given.",
            path.display()
        )));
    }

    let mut results = Vec::new();
    let mut matched_entries = 0;
    let mut candidates = collect_candidates_with_nested_repositories(&path);
    candidates.sort();
    for entry_path in candidates {
        let relative = entry_path.strip_prefix(&path).unwrap_or(&entry_path);
        let mut display = find_path_to_posix(&relative.to_string_lossy());
        let matches = match target {
            FindPatternTarget::Basename => relative
                .file_name()
                .is_some_and(|name| pattern.is_match(Path::new(name))),
            FindPatternTarget::RelativePath => pattern.is_match(&display),
            FindPatternTarget::AbsolutePath => {
                pattern.is_match(find_path_to_posix(&entry_path.to_string_lossy()))
            }
        };
        if !matches {
            continue;
        }
        if limit != 0 && matched_entries >= limit {
            break;
        }
        matched_entries += 1;
        let is_directory =
            std::fs::symlink_metadata(&entry_path).is_ok_and(|metadata| metadata.is_dir());
        if is_directory {
            display.push('/');
        }
        results.extend(relativize_fd_find_output(&entry_path, &path, is_directory));
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

struct ProcessGroupGuard {
    process_id: Option<u32>,
}

impl ProcessGroupGuard {
    fn new(process_id: Option<u32>) -> Self {
        Self { process_id }
    }

    fn disarm(&mut self) {
        self.process_id = None;
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(process_id) = self.process_id {
            // An aborted tool future drops this guard; stop descendants with the shell.
            unsafe {
                libc::kill(-(process_id as i32), libc::SIGKILL);
            }
        }
        #[cfg(not(unix))]
        let _ = self.process_id;
    }
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

fn shell_spawn_error(error: std::io::Error) -> ToolError {
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => Some("ENOENT"),
        std::io::ErrorKind::PermissionDenied => Some("EACCES"),
        std::io::ErrorKind::NotADirectory => Some("ENOTDIR"),
        _ => None,
    };
    code.map(|code| ToolError::Command(format!("spawn {code}")))
        .unwrap_or(ToolError::Io(error))
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
    let mut child = process.spawn().map_err(shell_spawn_error)?;
    let mut process_group = ProcessGroupGuard::new(child.id());
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
                process_group.disarm();
                finish_capture(capture, &mut updates_receiver).await;
                let captured = captured.lock().await.clone();
                return Err(interrupted_error(
                    &captured,
                    temp_file_prefix,
                    &format!("Command timed out after {} seconds", timeout.as_secs_f64()),
                ));
            }
        },
        None => child.wait().await.map_err(ToolError::Io)?,
    };
    finish_capture(capture, &mut updates_receiver).await;
    process_group.disarm();
    Ok(Output {
        status,
        stdout: captured.lock().await.clone(),
        stderr: Vec::new(),
    })
}

/// Pi's timeout/abort error: the (tail-truncated) output so far, a full-output path when it was
/// truncated, then `reason` (`Command timed out after N seconds` or `Command aborted`).
fn interrupted_error(captured: &[u8], temp_file_prefix: &str, reason: &str) -> ToolError {
    let full_output = sanitize_shell_output(&String::from_utf8_lossy(captured));
    let (mut message, truncation) =
        truncate_shell_output(&full_output, BASH_OUTPUT_MAX_LINES, TOOL_OUTPUT_MAX_BYTES);
    if let Some(truncation) = truncation {
        let path =
            std::env::temp_dir().join(format!("{temp_file_prefix}-{}.txt", uuid::Uuid::new_v4()));
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
    message.push_str(&format!("\n\n{reason}"));
    ToolError::Command(message)
}

/// How a shell command ended, independent of how it was run (what Pi's injectable
/// `BashOperations.exec` reports).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellEnd {
    Exited(Option<i32>),
    TimedOut(Duration),
    Aborted,
}

/// Turn captured output and an ending into the tool result or error Pi's bash tool produces.
pub fn shell_outcome(
    output: &[u8],
    end: &ShellEnd,
    started_at: Instant,
    temp_file_prefix: &str,
) -> Result<ToolResult, ToolError> {
    match end {
        ShellEnd::Exited(code) => Ok(shell_result_from(
            output,
            *code,
            started_at,
            temp_file_prefix,
        )),
        ShellEnd::TimedOut(timeout) => Err(interrupted_error(
            output,
            temp_file_prefix,
            &format!("Command timed out after {} seconds", timeout.as_secs_f64()),
        )),
        ShellEnd::Aborted => Err(interrupted_error(
            output,
            temp_file_prefix,
            "Command aborted",
        )),
    }
}

fn shell_child_environment(environment: &HashMap<String, String>) -> HashMap<String, String> {
    environment
        .iter()
        .filter(|(name, _)| {
            !matches!(
                name.as_str(),
                SHELL_COMMAND_PREFIX_ENV | BASH_SHELL_PATH_ENV
            )
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// What a shell run produced: combined output and how it ended.
pub struct ShellRun {
    pub output: Vec<u8>,
    pub end: ShellEnd,
}

/// Injectable command execution (Pi's `BashOperations`): tests and remote backends can replace
/// the local shell without touching the tool's formatting and truncation.
#[async_trait::async_trait]
pub trait ShellOperations: Send + Sync {
    async fn exec(
        &self,
        command: &str,
        cwd: &Path,
        environment: &HashMap<String, String>,
        timeout: Option<Duration>,
    ) -> Result<ShellRun, ToolError>;
}

/// Runs commands with the local bash.
pub struct LocalBash;

#[async_trait::async_trait]
impl ShellOperations for LocalBash {
    async fn exec(
        &self,
        command: &str,
        cwd: &Path,
        environment: &HashMap<String, String>,
        timeout: Option<Duration>,
    ) -> Result<ShellRun, ToolError> {
        let child_environment = shell_child_environment(environment);
        let shell = default_bash_shell(environment.get(BASH_SHELL_PATH_ENV).map(String::as_str))?;
        let mut process = tokio::process::Command::new(shell);
        process.args(["-c", command]);
        let output =
            run_shell_process(cwd, "bash", "pi-bash", process, timeout, &child_environment).await?;
        let mut bytes = output.stdout;
        bytes.extend_from_slice(&output.stderr);
        Ok(ShellRun {
            output: bytes,
            end: ShellEnd::Exited(output.status.code()),
        })
    }
}

async fn run_bash(
    cwd: &Path,
    args: &Value,
    environment: &HashMap<String, String>,
) -> Result<ToolResult, ToolError> {
    run_bash_with(&LocalBash, cwd, args, environment).await
}

/// The bash tool over any [`ShellOperations`]: argument parsing, the command prefix, and result
/// formatting (truncation, full-output file, exit codes, timeout/abort errors).
pub async fn run_bash_with(
    operations: &dyn ShellOperations,
    cwd: &Path,
    args: &Value,
    environment: &HashMap<String, String>,
) -> Result<ToolResult, ToolError> {
    let timeout = shell_timeout(args)?;
    let started_at = Instant::now();
    let command = string_arg(args, "command")?;
    let command = environment
        .get(SHELL_COMMAND_PREFIX_ENV)
        .filter(|prefix| !prefix.is_empty())
        .map_or_else(
            || command.to_owned(),
            |prefix| format!("{prefix}\n{command}"),
        );
    let run = operations.exec(&command, cwd, environment, timeout).await?;
    shell_outcome(&run.output, &run.end, started_at, "pi-bash")
}

fn default_bash_shell(custom_path: Option<&str>) -> Result<PathBuf, ToolError> {
    if let Some(custom_path) = custom_path {
        let path = PathBuf::from(custom_path);
        return path.exists().then_some(path).ok_or_else(|| {
            ToolError::Command(format!("Custom shell path not found: {custom_path}"))
        });
    }
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
fn powershell_arguments(command: &str) -> [&str; 6] {
    [
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        command,
    ]
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
        process.args(powershell_arguments(&command));
        let output = run_shell_process(
            cwd,
            "PowerShell",
            "pi-powershell",
            process,
            timeout,
            &shell_child_environment(environment),
        )
        .await?;
        let mut bytes = output.stdout;
        bytes.extend_from_slice(&output.stderr);
        shell_outcome(
            &bytes,
            &ShellEnd::Exited(output.status.code()),
            started_at,
            "pi-powershell",
        )
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

fn shell_result_from(
    output: &[u8],
    exit_code: Option<i32>,
    started_at: Instant,
    temp_file_prefix: &str,
) -> ToolResult {
    const STRUCTURED_OUTPUT_MAX_BYTES: usize = 1024 * 1024;
    let full_output = sanitize_shell_output(&String::from_utf8_lossy(output));
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
            images: Vec::new(),
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
        images: Vec::new(),
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
    fn read_returns_image_files_as_attachments_by_magic_not_extension() {
        let dir = std::env::temp_dir().join(format!("mu-read-img-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut real_png = Vec::new();
        image::DynamicImage::new_rgb8(2, 2)
            .write_to(
                &mut std::io::Cursor::new(&mut real_png),
                image::ImageFormat::Png,
            )
            .unwrap();
        std::fs::write(dir.join("shot.txt"), &real_png).unwrap();
        let result = read_file(&dir, &json!({"path":"shot.txt"})).unwrap();
        assert_eq!(result.content, "Read image file [image/png]");
        assert_eq!(result.images.len(), 1);
        assert_eq!(result.images[0].mime_type, "image/png");
        // A PNG signature with no valid image behind it is reported, not attached.
        std::fs::write(
            dir.join("corrupt.png"),
            [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 1],
        )
        .unwrap();
        let corrupt = read_file(&dir, &json!({"path":"corrupt.png"})).unwrap();
        assert!(corrupt.images.is_empty());
        assert!(
            corrupt
                .content
                .starts_with("Read image file [image/png]\n[Image omitted:"),
            "{}",
            corrupt.content
        );
        std::fs::write(dir.join("plain.png"), "just text").unwrap();
        let plain = read_file(&dir, &json!({"path":"plain.png"})).unwrap();
        assert!(plain.images.is_empty() && plain.content.contains("just text"));
        assert_eq!(detect_image_mime(b"GIF89a.."), Some("image/gif"));
        assert_eq!(
            detect_image_mime(&[0xff, 0xd8, 0xff, 0xe0]),
            Some("image/jpeg")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn find_relative_path_matches_node_for_absolute_results_outside_root() {
        assert_eq!(
            relative_find_path(Path::new("/workspace/project"), Path::new("/outside.txt")),
            PathBuf::from("../../outside.txt")
        );
    }

    #[test]
    fn find_output_trim_matches_javascript_bom_whitespace_behavior() {
        assert_eq!(trim_javascript_whitespace(" file \u{feff}"), "file");
        assert_eq!(trim_javascript_whitespace("file\u{0085}"), "file\u{0085}");
    }

    #[test]
    fn ls_stat_errors_match_node_common_error_codes() {
        let error = node_stat_error(
            Path::new("private-dir"),
            std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        );
        assert_eq!(error, "EACCES: permission denied, stat 'private-dir'");
    }

    #[test]
    fn ls_scandir_errors_match_node_common_error_codes() {
        let path = Path::new("private-dir");
        let error = node_scandir_error(
            path,
            std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        );
        assert_eq!(error, "EACCES: permission denied, scandir 'private-dir'");
    }

    #[test]
    fn directory_sort_preserves_case_variant_order_like_stable_locale_compare() {
        let mut entries = vec![
            "a".to_owned(),
            "A".to_owned(),
            "b".to_owned(),
            "B".to_owned(),
        ];
        sort_directory_entries(&mut entries);
        assert_eq!(entries, ["a", "A", "b", "B"]);
    }

    #[test]
    fn write_permission_errors_match_node_fs_open() {
        assert_eq!(
            write_file_io_error(
                Path::new("private.txt"),
                "open",
                std::io::ErrorKind::PermissionDenied.into()
            )
            .to_string(),
            "EACCES: permission denied, open 'private.txt'"
        );
    }

    #[test]
    fn edit_permission_errors_match_pi_access_error() {
        assert_eq!(
            edit_file_io_message(
                Path::new("private.txt"),
                std::io::ErrorKind::PermissionDenied.into()
            ),
            "Could not edit file: private.txt. Error code: EACCES."
        );
    }

    #[test]
    fn read_access_errors_match_node_common_error_codes() {
        let path = Path::new("/tmp/private.txt");
        let missing = read_file_io_error(path, std::io::ErrorKind::NotFound.into());
        let denied = read_file_io_error(path, std::io::ErrorKind::PermissionDenied.into());
        assert_eq!(
            missing.to_string(),
            "ENOENT: no such file or directory, access '/tmp/private.txt'"
        );
        assert_eq!(
            denied.to_string(),
            "EACCES: permission denied, access '/tmp/private.txt'"
        );
    }

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

    #[tokio::test]
    async fn same_path_write_then_edit_preserves_fifo_mutation_order() {
        let cwd = std::env::temp_dir().join(format!("mu-mutation-order-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&cwd).unwrap();
        let path = cwd.join("shared.txt");
        std::fs::write(&path, "initial").unwrap();

        let lock = mutation_lock(&path);
        let held_guard = lock.lock().await;
        let write_cwd = cwd.clone();
        let write_path = path.clone();
        let write = tokio::spawn(async move {
            write_file(&write_cwd, &json!({"path":write_path,"content":"written"})).await
        });
        tokio::task::yield_now().await;
        let edit_cwd = cwd.clone();
        let edit_path = path.clone();
        let edit = tokio::spawn(async move {
            edit_file(
                &edit_cwd,
                &json!({
                    "path":edit_path,
                    "edits":[{"oldText":"written","newText":"edited"}]
                }),
            )
            .await
        });
        tokio::task::yield_now().await;

        assert!(!write.is_finished());
        assert!(!edit.is_finished());
        drop(held_guard);

        write.await.unwrap().unwrap();
        edit.await.unwrap().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "edited");
        std::fs::remove_dir_all(cwd).unwrap();
    }

    #[tokio::test]
    async fn cancelled_mutation_waiter_does_not_block_later_file_operations() {
        let path = std::env::temp_dir().join(format!(
            "mu-cancelled-mutation-{}.txt",
            uuid::Uuid::new_v4()
        ));
        let lock = mutation_lock(&path);
        let held_guard = lock.lock().await;
        let started = Arc::new(tokio::sync::Notify::new());
        let waiter_lock = Arc::clone(&lock);
        let waiter_started = Arc::clone(&started);
        let waiter = tokio::spawn(async move {
            waiter_started.notify_one();
            let _guard = waiter_lock.lock().await;
        });

        started.notified().await;
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        drop(held_guard);

        let _guard = tokio::time::timeout(std::time::Duration::from_secs(1), lock.lock())
            .await
            .expect("cancelled waiter must not strand the mutation lock");
    }

    #[cfg(unix)]
    #[test]
    fn explicit_bash_shell_path_is_used_or_reports_pi_error() {
        assert_eq!(
            default_bash_shell(Some("/bin/bash")).unwrap(),
            PathBuf::from("/bin/bash")
        );
        assert_eq!(
            default_bash_shell(Some("/missing/custom/bash"))
                .unwrap_err()
                .to_string(),
            "Custom shell path not found: /missing/custom/bash"
        );
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
    fn powershell_uses_pi_noninteractive_command_arguments() {
        assert_eq!(
            powershell_arguments("Get-Location"),
            [
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "Get-Location",
            ]
        );
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
        assert_eq!(
            resolve_input_path(cwd, "nested/.././file.txt").unwrap(),
            cwd.join("file.txt")
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
        // Output is collected while a descendant keeps writing, and ends after 100ms of silence, so
        // a machine stall longer than that can cut it short: allow a few attempts.
        let mut last = Value::Null;
        for _ in 0..6 {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let result = execute(
                &dir,
                "bash",
                &json!({"command":"(for i in {1..30}; do sleep 0.01; printf x; done) &"}),
            )
            .await
            .unwrap();
            last = result.structured_content.unwrap()["output"].clone();
            if last == "x".repeat(30).as_str() {
                break;
            }
        }
        assert_eq!(last, "x".repeat(30));
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
    #[cfg(unix)]
    #[test]
    fn dropping_process_group_guard_terminates_shell_and_descendants() {
        use std::os::unix::process::CommandExt;

        let mut command = std::process::Command::new("/bin/bash");
        command.args(["-c", "sleep 60"]).process_group(0);
        let mut child = command.spawn().expect("start bash process");
        let process_id = child.id();
        drop(ProcessGroupGuard::new(Some(process_id)));
        child.wait().expect("reap terminated bash process");
        assert_ne!(
            unsafe { libc::kill(process_id as i32, 0) },
            0,
            "the process group signal must not report success for an already reaped shell"
        );
    }

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
    async fn bash_prepends_configured_command_prefix_without_leaking_internal_option() {
        let directory = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut environment = HashMap::new();
        environment.insert(
            SHELL_COMMAND_PREFIX_ENV.to_owned(),
            "printf prefix".to_owned(),
        );
        environment.insert(BASH_SHELL_PATH_ENV.to_owned(), "/bin/bash".to_owned());
        let result = execute_with_environment(
            &directory,
            "bash",
            &json!({"command": r#"printf command; test -z "$MU_INTERNAL_SHELL_COMMAND_PREFIX""#}),
            &environment,
        )
        .await
        .unwrap();
        assert_eq!(result.content, "prefixcommand");
        assert!(!result.is_error);
        std::fs::remove_dir_all(directory).unwrap();
    }

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
    /// Ported from Pi's `tools.test.ts`: a trailing newline is not an extra truncated line.
    /// Ported from Pi's `tools.test.ts` ("should include full output path for truncated timeout
    /// and abort errors" and "should reject a null exit code from custom operations"), driven by
    /// fake output instead of Pi's injected `BashOperations`.
    struct Fake(Vec<u8>, ShellEnd);

    #[async_trait::async_trait]
    impl ShellOperations for Fake {
        async fn exec(
            &self,
            command: &str,
            _cwd: &Path,
            _environment: &HashMap<String, String>,
            _timeout: Option<Duration>,
        ) -> Result<ShellRun, ToolError> {
            assert_eq!(
                command, "remote",
                "the command reaches the operations unchanged"
            );
            Ok(ShellRun {
                output: self.0.clone(),
                end: self.1.clone(),
            })
        }
    }

    /// Pi's tests drive the bash tool through injected `BashOperations`; so can ours now.
    #[tokio::test]
    async fn bash_tool_runs_over_injected_operations() {
        let args = json!({"command":"remote"});
        let env = HashMap::new();
        let cwd = std::env::temp_dir();
        let ok = run_bash_with(
            &Fake(b"hi\n".to_vec(), ShellEnd::Exited(Some(0))),
            &cwd,
            &args,
            &env,
        )
        .await
        .unwrap();
        assert_eq!(ok.content, "hi\n");
        assert!(!ok.is_error);
        let failed = run_bash_with(
            &Fake(b"oops\n".to_vec(), ShellEnd::Exited(Some(3))),
            &cwd,
            &args,
            &env,
        )
        .await
        .unwrap();
        assert!(failed.is_error && failed.content.ends_with("Command exited with code 3"));
        let null_exit = run_bash_with(
            &Fake(b"partial\n".to_vec(), ShellEnd::Exited(None)),
            &cwd,
            &args,
            &env,
        )
        .await
        .unwrap();
        assert!(
            null_exit.is_error
                && null_exit
                    .content
                    .contains("Command terminated without an exit code")
        );
        let timed_out = run_bash_with(
            &Fake(b"x\n".to_vec(), ShellEnd::TimedOut(Duration::from_secs(5))),
            &cwd,
            &args,
            &env,
        )
        .await;
        assert!(
            timed_out
                .err()
                .unwrap()
                .to_string()
                .contains("Command timed out after 5 seconds")
        );
        let aborted = run_bash_with(&Fake(Vec::new(), ShellEnd::Aborted), &cwd, &args, &env).await;
        assert!(
            aborted
                .err()
                .unwrap()
                .to_string()
                .starts_with("(no output)\n\nCommand aborted")
        );
        // The command prefix is applied before the operations see the command.
        let mut prefixed = HashMap::new();
        prefixed.insert(SHELL_COMMAND_PREFIX_ENV.to_owned(), "set -e".to_owned());
        struct Echo;
        #[async_trait::async_trait]
        impl ShellOperations for Echo {
            async fn exec(
                &self,
                command: &str,
                _: &Path,
                _: &HashMap<String, String>,
                _: Option<Duration>,
            ) -> Result<ShellRun, ToolError> {
                Ok(ShellRun {
                    output: command.as_bytes().to_vec(),
                    end: ShellEnd::Exited(Some(0)),
                })
            }
        }
        let echoed = run_bash_with(&Echo, &cwd, &args, &prefixed).await.unwrap();
        assert_eq!(echoed.content, "set -e\nremote");
    }

    struct MemoryFiles(std::sync::Mutex<Vec<u8>>, bool);

    impl FileOperations for MemoryFiles {
        fn read(&self, _path: &Path) -> std::io::Result<Vec<u8>> {
            if self.1 {
                return Err(std::io::Error::other("disk offline"));
            }
            Ok(self.0.lock().unwrap().clone())
        }

        fn write(&self, _path: &Path, contents: &[u8]) -> std::io::Result<()> {
            *self.0.lock().unwrap() = contents.to_vec();
            Ok(())
        }

        fn create_dir_all(&self, _path: &Path) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Ported from Pi's `tools.test.ts` ("unknown edit access errors") and its in-memory edit
    /// operations.
    struct FakeDirectory;

    impl DirectoryOperations for FakeDirectory {
        fn exists(&self, path: &Path) -> bool {
            !path.ends_with("missing")
        }

        fn is_dir(&self, path: &Path) -> std::io::Result<bool> {
            Ok(path.ends_with("sub") || path.ends_with("root"))
        }

        fn read_dir(&self, _path: &Path) -> std::io::Result<Vec<String>> {
            Ok(vec![
                "b.txt".into(),
                "Sub".into(),
                "sub".into(),
                "a.txt".into(),
            ])
        }
    }

    #[test]
    fn ls_tool_runs_over_injected_directory_operations() {
        let cwd = Path::new("/virtual");
        let listed = list_directory_with(&FakeDirectory, cwd, &json!({"path":"root"})).unwrap();
        // `Sub` is not a directory in the fake, only `sub` is; names sort case-insensitively.
        assert_eq!(listed.content, "a.txt\nb.txt\nSub\nsub/");
        let missing =
            list_directory_with(&FakeDirectory, cwd, &json!({"path":"missing"})).unwrap_err();
        assert!(missing.to_string().starts_with("Path not found: "));
        let file = list_directory_with(&FakeDirectory, cwd, &json!({"path":"a.txt"})).unwrap_err();
        assert!(file.to_string().starts_with("Not a directory: "));
    }

    #[test]
    fn read_tool_runs_over_injected_file_operations() {
        let memory = MemoryFiles(std::sync::Mutex::new(b"one\ntwo\nthree\n".to_vec()), false);
        let cwd = std::env::temp_dir();
        let slice =
            read_file_with(&memory, &cwd, &json!({"path":"x.txt","offset":2,"limit":1})).unwrap();
        assert!(slice.content.starts_with("two"), "{}", slice.content);
        let offline = MemoryFiles(std::sync::Mutex::new(Vec::new()), true);
        assert!(read_file_with(&offline, &cwd, &json!({"path":"x.txt"})).is_err());
    }

    #[tokio::test]
    async fn write_tool_runs_over_injected_file_operations() {
        let memory = MemoryFiles(std::sync::Mutex::new(Vec::new()), false);
        let result = write_file_with(
            &memory,
            &std::env::temp_dir(),
            &json!({"path":"a/b.txt","content":"data"}),
        )
        .await
        .unwrap();
        assert_eq!(result.content, "Successfully wrote to a/b.txt");
        assert_eq!(*memory.0.lock().unwrap(), b"data");
    }

    #[tokio::test]
    async fn edit_tool_runs_over_injected_file_operations() {
        let args = json!({"path":"broken.txt","edits":[{"oldText":"hello","newText":"world"}]});
        let cwd = std::env::temp_dir();
        let memory = MemoryFiles(std::sync::Mutex::new(b"hello\n".to_vec()), false);
        let result = edit_file_with(&memory, &cwd, &args).await.unwrap();
        assert!(
            result
                .content
                .starts_with("Successfully replaced 1 block(s) in broken.txt")
        );
        assert_eq!(*memory.0.lock().unwrap(), b"world\n");
        let offline = MemoryFiles(std::sync::Mutex::new(Vec::new()), true);
        let error = edit_file_with(&offline, &cwd, &args)
            .await
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "Could not edit file: broken.txt. Error: disk offline."
        );
    }

    #[test]
    fn shell_outcome_formats_timeout_abort_and_null_exit_like_pi() {
        let chatty = (1..=3000)
            .map(|i| format!("{i}\n"))
            .collect::<String>()
            .into_bytes();
        for (end, expected) in [
            (
                ShellEnd::TimedOut(Duration::from_secs(5)),
                "Command timed out after 5 seconds",
            ),
            (ShellEnd::Aborted, "Command aborted"),
        ] {
            let error = shell_outcome(&chatty, &end, Instant::now(), "pi-bash")
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains(expected), "{error}");
            assert!(!error.contains("Full output: undefined"));
            let path = error
                .split("Full output: ")
                .nth(1)
                .unwrap()
                .split(']')
                .next()
                .unwrap();
            let full = std::fs::read_to_string(path).unwrap();
            assert!(full.contains("1\n2\n3") && full.contains("2998\n2999\n3000"));
            assert!(error.contains("[Showing lines "));
            std::fs::remove_file(path).unwrap();
        }
        let quiet = shell_outcome(b"", &ShellEnd::Aborted, Instant::now(), "pi-bash")
            .err()
            .unwrap();
        assert!(
            quiet
                .to_string()
                .starts_with("(no output)\n\nCommand aborted")
        );
        let null_exit = shell_outcome(
            b"partial\n",
            &ShellEnd::Exited(None),
            Instant::now(),
            "pi-bash",
        )
        .unwrap();
        assert!(null_exit.is_error);
        // Pi matches /partial\s+Command terminated without an exit code$/.
        let body = null_exit
            .content
            .strip_suffix("Command terminated without an exit code")
            .unwrap();
        assert!(
            body.starts_with("partial")
                && body[7..].chars().all(char::is_whitespace)
                && body.len() > 7
        );
    }

    /// Ported from Pi's "tool cwd resolution" suite: every built-in resolves relative paths
    /// against the cwd it is executed with, not the process's.
    #[tokio::test]
    async fn every_builtin_resolves_relative_paths_against_the_execution_cwd() {
        let dir = std::env::temp_dir().join(format!("mu-ctx-cwd-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ctx-read.txt"), "hello from ctx.cwd").unwrap();
        let read = execute(&dir, "read", &json!({"path":"ctx-read.txt"}))
            .await
            .unwrap();
        assert!(read.content.contains("hello from ctx.cwd"));
        execute(
            &dir,
            "write",
            &json!({"path":"ctx-write.txt","content":"written via ctx.cwd"}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("ctx-write.txt")).unwrap(),
            "written via ctx.cwd"
        );
        std::fs::write(dir.join("ctx-edit.txt"), "old text").unwrap();
        execute(
            &dir,
            "edit",
            &json!({"path":"ctx-edit.txt","edits":[{"oldText":"old text","newText":"new text"}]}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("ctx-edit.txt")).unwrap(),
            "new text"
        );
        let grep = execute(&dir, "grep", &json!({"pattern":"hello"}))
            .await
            .unwrap();
        assert!(grep.content.contains("ctx-read.txt"));
        let find = execute(&dir, "find", &json!({"pattern":"ctx-edit.txt"}))
            .await
            .unwrap();
        assert!(find.content.contains("ctx-edit.txt"));
        let ls = execute(&dir, "ls", &json!({})).await.unwrap();
        assert!(ls.content.contains("ctx-write.txt"));
        let pwd = execute(&dir, "bash", &json!({"command":"pwd"}))
            .await
            .unwrap();
        let real = std::fs::canonicalize(&dir).unwrap();
        assert!(
            pwd.content.contains(real.to_str().unwrap())
                || pwd.content.contains(dir.to_str().unwrap())
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn bash_trailing_newline_is_not_counted_as_an_extra_line() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let result = execute(
            &dir,
            "bash",
            &json!({"command":"i=1; while [ $i -le 4000 ]; do printf 'line-%04d\\n' $i; i=$((i + 1)); done"}),
        )
        .await
        .unwrap();
        let details = result.details.unwrap();
        assert_eq!(details["truncation"]["totalLines"], 4000);
        assert_eq!(details["truncation"]["outputLines"], 2000);
        assert!(result.content.contains("line-2001") && result.content.contains("line-4000"));
        assert!(
            result
                .content
                .contains("[Showing lines 2001-4000 of 4000. Full output: "),
            "{}",
            &result.content[result.content.len().saturating_sub(200)..]
        );
        assert!(!result.content.contains("4001"));
        let _ = std::fs::remove_file(details["fullOutputPath"].as_str().unwrap());
        std::fs::remove_dir_all(dir).unwrap();
    }

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
