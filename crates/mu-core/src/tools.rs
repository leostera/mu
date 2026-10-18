use std::{
    collections::HashMap,
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
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    Arguments(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("command failed: {0}")]
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
    match name {
        "read" => read_file(cwd, arguments),
        "write" => write_file(cwd, arguments).await,
        "edit" => edit_file(cwd, arguments).await,
        "bash" => run_bash(cwd, arguments).await,
        "powershell" => run_powershell(cwd, arguments).await,
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
    let text = std::fs::read_to_string(&path)?;
    let offset = args
        .get("offset")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as usize;
    let all_lines = text.split('\n').collect::<Vec<_>>();
    if offset > all_lines.len() {
        return Err(ToolError::Arguments(format!(
            "Offset {offset} is beyond end of file ({} lines total)",
            all_lines.len()
        )));
    }

    let start = offset - 1;
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map(|n| n as usize);
    let end = limit.map_or(all_lines.len(), |limit| {
        start.saturating_add(limit).min(all_lines.len())
    });
    let selected = all_lines[start..end].join("\n");
    let total_file_lines = all_lines.len();
    let (truncated, truncated_by, output_lines, first_line_exceeds_limit) =
        truncate_read_head(&selected);
    let start_line = start + 1;
    let content = if first_line_exceeds_limit {
        let first_line_size = format_size(all_lines[start].len());
        format!(
            "[Line {start_line} is {first_line_size}, exceeds {} limit. Use bash: sed -n '{start_line}p' {} | head -c {}]",
            format_size(READ_MAX_BYTES),
            requested_path,
            READ_MAX_BYTES
        )
    } else if let Some(limit_name) = truncated {
        let end_line = start_line + output_lines - 1;
        let next_offset = end_line + 1;
        let mut output = truncated_by.clone();
        if limit_name == "bytes" {
            output.push_str(&format!(
                "\n\n[Showing lines {start_line}-{end_line} of {total_file_lines} ({} limit). Use offset={next_offset} to continue.]",
                format_size(READ_MAX_BYTES)
            ));
        } else {
            output.push_str(&format!(
                "\n\n[Showing lines {start_line}-{end_line} of {total_file_lines}. Use offset={next_offset} to continue.]"
            ));
        }
        output
    } else if let Some(limit) = limit.filter(|_| end < all_lines.len()) {
        let remaining = all_lines.len() - end;
        let next_offset = start + limit + 1;
        format!(
            "{truncated_by}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TextEdit {
    #[serde(rename = "oldText")]
    old_text: String,
    #[serde(rename = "newText")]
    new_text: String,
}

fn parse_edits(args: &Value) -> Result<Vec<TextEdit>, ToolError> {
    let mut args = args.clone();
    let object = args
        .as_object_mut()
        .ok_or_else(|| ToolError::Arguments("edit arguments must be an object".into()))?;

    // Pi accepts legacy oldText/newText fields and a few model-emitted edits forms.
    if let Some(edits) = object.get_mut("edits") {
        if let Some(encoded) = edits.as_str() {
            if let Ok(parsed) = serde_json::from_str::<Value>(encoded) {
                *edits = parsed;
            }
        }
        if edits.is_object() {
            *edits = Value::Array(vec![edits.take()]);
        }
    }
    if let (Some(old), Some(new)) = (object.remove("oldText"), object.remove("newText")) {
        let legacy = json!({"oldText":old,"newText":new});
        match object.remove("edits") {
            Some(Value::Array(mut edits)) => {
                edits.push(legacy);
                object.insert("edits".into(), Value::Array(edits));
            }
            Some(Value::Null) | None => {
                object.insert("edits".into(), Value::Array(vec![legacy]));
            }
            Some(_) => {
                return Err(ToolError::Arguments("`edits` must be an array".into()));
            }
        }
    }

    let edits = object
        .get("edits")
        .and_then(Value::as_array)
        .filter(|edits| !edits.is_empty())
        .ok_or_else(|| {
            ToolError::Arguments(
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
    let original =
        std::fs::read_to_string(&path).map_err(|error| edit_file_io_error(display_path, error))?;
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
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => "ENOENT",
        std::io::ErrorKind::PermissionDenied => "EACCES",
        _ => {
            return ToolError::Arguments(format!(
                "Could not edit file: {}. {error}.",
                path.display()
            ));
        }
    };
    ToolError::Arguments(format!(
        "Could not edit file: {}. Error code: {code}.",
        path.display()
    ))
}

fn edit_details(old_content: &str, new_content: &str, path: &Path) -> Value {
    let diff = TextDiff::from_lines(old_content, new_content);
    let old_lines = old_content.lines().count().max(1);
    let new_lines = new_content.lines().count().max(1);
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
            return Err(ToolError::Arguments(error));
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
            return Err(ToolError::Arguments(error));
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
            return Err(ToolError::Arguments(error));
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
            return Err(ToolError::Arguments(format!(
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
        return Err(ToolError::Arguments(error));
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
        "firstLineExceedsLimit": false,
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
    limit: usize,
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
    let limit = optional_nonnegative_integer(object.get("limit"), "limit")?
        .unwrap_or(GREP_DEFAULT_LIMIT)
        .max(1);
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
            if match_count >= input.limit {
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
            input.limit.saturating_mul(2)
        ));
        details.insert("matchLimitReached".into(), json!(input.limit));
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
    let mut walker = WalkBuilder::new(path);
    walker.hidden(false).require_git(false);
    walker
        .build()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_some_and(|kind| kind.is_file()))
        .map(|entry| entry.into_path())
        .collect()
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
    let mut walker = WalkBuilder::new(&path);
    walker.hidden(false).require_git(false);
    for entry in walker.build() {
        let Ok(entry) = entry else {
            continue;
        };
        let entry_path = entry.path();
        if entry_path == path {
            continue;
        }
        let relative = entry_path.strip_prefix(&path).unwrap_or(entry_path);
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
        if entry.file_type().is_some_and(|kind| kind.is_dir()) {
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

fn shell_timeout(args: &Value) -> Result<Option<Duration>, ToolError> {
    let Some(value) = args.get("timeout") else {
        return Ok(None);
    };
    let seconds = value.as_f64().ok_or_else(|| {
        ToolError::Arguments("Invalid timeout: must be a finite number of seconds".into())
    })?;
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 2_147_483.647 {
        return Err(ToolError::Arguments(
            "Invalid timeout: must be a finite number of seconds".into(),
        ));
    }
    Duration::try_from_secs_f64(seconds)
        .map(Some)
        .map_err(|_| ToolError::Arguments("Invalid timeout: invalid duration".into()))
}

async fn run_shell_process(
    cwd: &Path,
    mut process: tokio::process::Command,
    timeout: Option<Duration>,
) -> Result<Output, ToolError> {
    process
        .current_dir(cwd)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = process.output();
    match timeout {
        Some(timeout) => tokio::time::timeout(timeout, output)
            .await
            .map_err(|_| {
                ToolError::Command(format!(
                    "Command timed out after {} seconds",
                    timeout.as_secs_f64()
                ))
            })?
            .map_err(ToolError::Io),
        None => output.await.map_err(ToolError::Io),
    }
}

async fn run_bash(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let timeout = shell_timeout(args)?;
    let started_at = Instant::now();
    let command = string_arg(args, "command")?;
    #[cfg(windows)]
    let mut process = tokio::process::Command::new("cmd");
    #[cfg(windows)]
    process.args(["/C", command]);
    #[cfg(not(windows))]
    let mut process = tokio::process::Command::new("sh");
    #[cfg(not(windows))]
    process.args(["-c", command]);
    let output = run_shell_process(cwd, process, timeout).await?;
    Ok(shell_result(output, started_at))
}

#[cfg(any(windows, test))]
fn powershell_command_with_utf8_output(command: &str) -> String {
    format!(
        "try {{ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 }} catch {{}}\n{command}"
    )
}

async fn run_powershell(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let command = string_arg(args, "command")?;
    let timeout = shell_timeout(args)?;
    #[cfg(not(windows))]
    {
        let _ = (cwd, command, timeout);
        Err(ToolError::Arguments(
            "The powershell tool is only available on Windows.".into(),
        ))
    }
    #[cfg(windows)]
    {
        let started_at = Instant::now();
        let command = powershell_command_with_utf8_output(command);
        let powershell = ["pwsh.exe", "powershell.exe"]
            .into_iter()
            .find_map(|name| {
                std::env::var_os("PATH")
                    .into_iter()
                    .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
                    .map(|path| path.join(name))
                    .find(|path| path.is_file())
            })
            .unwrap_or_else(|| PathBuf::from("powershell.exe"));
        let mut process = tokio::process::Command::new(powershell);
        process.args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &command,
        ]);
        let output = run_shell_process(cwd, process, timeout).await?;
        Ok(shell_result(output, started_at))
    }
}

fn shell_result(output: Output, started_at: Instant) -> ToolResult {
    const STRUCTURED_OUTPUT_MAX_BYTES: usize = 1024 * 1024;
    let mut full_output = String::from_utf8_lossy(&output.stdout).into_owned();
    full_output.push_str(&String::from_utf8_lossy(&output.stderr));
    let exit_code = output.status.code().unwrap_or(1);
    let (structured_output, truncated) = truncate_utf8(&full_output, STRUCTURED_OUTPUT_MAX_BYTES);
    let wall_time_seconds = (started_at.elapsed().as_millis() as f64 / 100.0).round() / 10.0;
    let mut content = if full_output.is_empty() {
        "(no output)".to_owned()
    } else {
        full_output
    };
    let is_error = exit_code != 0;
    if is_error {
        content.push_str(&format!("\n\nCommand exited with code {exit_code}"));
    }
    ToolResult {
        content,
        details: None,
        structured_content: Some(json!({
            "output": structured_output,
            "truncated": truncated,
            "exit_code": exit_code,
            "wall_time_seconds": wall_time_seconds,
        })),
        is_error,
    }
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
        assert!(invalid.to_string().contains("Invalid timeout"));

        let timeout = execute(
            &dir,
            "bash",
            &json!({"command":"while :; do :; done","timeout":0.05}),
        )
        .await
        .unwrap_err();
        assert!(
            timeout
                .to_string()
                .contains("Command timed out after 0.05 seconds")
        );
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
