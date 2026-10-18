use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Instant,
};

use globset::{Glob, GlobMatcher};
use ignore::WalkBuilder;
use regex::{Regex, RegexBuilder};
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;

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
        json!({"type":"function","name":"write","description":"Write content to a file, creating parent directories as needed.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":false}}),
        json!({"type":"function","name":"edit","description":"Edit a single file using exact text replacements. Each edits[].oldText must match a unique, non-overlapping region of the original file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","minItems":1,"items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"],"additionalProperties":false}}},"required":["path","edits"],"additionalProperties":false}}),
        json!({"type":"function","name":"bash","description":"Run a shell command in the working directory.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}),
        json!({"type":"function","name":"ls","description":"List directory contents, including dotfiles. Entries are sorted alphabetically; directories have a trailing slash.","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"number","description":"Maximum number of entries to return (default: 500)"}},"additionalProperties":false}}),
        json!({"type":"function","name":"grep","description":"Search file contents for a pattern. Respects .gitignore and returns matching lines with file paths and line numbers.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"glob":{"type":"string"},"ignoreCase":{"type":"boolean"},"literal":{"type":"boolean"},"context":{"type":"number"},"limit":{"type":"number"}},"required":["pattern"],"additionalProperties":false}}),
        json!({"type":"function","name":"find","description":"Find files by glob pattern. Includes hidden files and respects .gitignore.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"limit":{"type":"number"}},"required":["pattern"],"additionalProperties":false}}),
    ]
}

pub async fn execute(cwd: &Path, name: &str, arguments: &Value) -> Result<ToolResult, ToolError> {
    match name {
        "read" => read_file(cwd, arguments),
        "write" => write_file(cwd, arguments),
        "edit" => edit_file(cwd, arguments),
        "bash" => run_bash(cwd, arguments).await,
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
    let path = Path::new(string_arg(args, "path")?);
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        cwd.join(path)
    })
}

const READ_MAX_LINES: usize = 2000;
const READ_MAX_BYTES: usize = 50 * 1024;

fn read_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let path = resolve(cwd, args)?;
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
            path.display(),
            READ_MAX_BYTES
        )
    } else if let Some(limit_name) = truncated {
        let end_line = start_line + output_lines - 1;
        let next_offset = end_line + 1;
        let mut output = truncated_by;
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
        truncated_by
    };
    Ok(ToolResult::text(content))
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

fn write_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let path = resolve(cwd, args)?;
    let content = string_arg(args, "content")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, content)?;
    Ok(ToolResult::text(format!(
        "Successfully wrote to {}",
        path.display()
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
        .ok_or_else(|| ToolError::Arguments("`edits` must be an array".into()))?;
    if edits.is_empty() {
        return Err(ToolError::Arguments(
            "edits must contain at least one replacement".into(),
        ));
    }
    edits
        .iter()
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|error| ToolError::Arguments(format!("invalid edit: {error}")))
        })
        .collect()
}

fn edit_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let path = resolve(cwd, args)?;
    let edits = parse_edits(args)?;
    let original = std::fs::read_to_string(&path)?;
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

    let mut replacements = Vec::with_capacity(normalized_edits.len());
    for (index, edit) in normalized_edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(ToolError::Arguments(format!(
                "edits[{index}].oldText must not be empty in {}.",
                path.display()
            )));
        }
        let matches = normalized
            .match_indices(&edit.old_text)
            .map(|(start, _)| start)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => {
                return Err(ToolError::Arguments(format!(
                    "Could not find edits[{index}] in {}. The oldText must match exactly including all whitespace and newlines.",
                    path.display()
                )));
            }
            [start] => {
                replacements.push((*start, edit.old_text.len(), edit.new_text.as_str(), index))
            }
            _ => {
                return Err(ToolError::Arguments(format!(
                    "Found {} occurrences of edits[{index}] in {}. Each oldText must be unique. Please provide more context to make it unique.",
                    matches.len(),
                    path.display()
                )));
            }
        }
    }
    replacements.sort_by_key(|(start, _, _, _)| *start);
    for pair in replacements.windows(2) {
        let (start, length, _, index) = pair[0];
        let (next_start, _, _, next_index) = pair[1];
        if start + length > next_start {
            return Err(ToolError::Arguments(format!(
                "edits[{index}] and edits[{next_index}] overlap in {}. Merge them into one edit or target disjoint regions.",
                path.display()
            )));
        }
    }

    let mut updated = normalized.clone();
    for (start, length, replacement, _) in replacements.into_iter().rev() {
        updated.replace_range(start..start + length, replacement);
    }
    if updated == normalized {
        return Err(ToolError::Arguments(format!(
            "No changes made to {}. The replacements produced identical content.",
            path.display()
        )));
    }
    let restored = if line_ending == "\r\n" {
        updated.replace('\n', "\r\n")
    } else {
        updated
    };
    std::fs::write(&path, format!("{bom}{restored}"))?;
    Ok(ToolResult::text(format!(
        "Successfully replaced {} block(s) in {}.",
        normalized_edits.len(),
        path.display()
    )))
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
        None => PathBuf::from("."),
        Some(Value::String(path)) => PathBuf::from(path),
        Some(_) => return Err(ToolError::Arguments("`path` must be a string".into())),
    };
    let limit = match object.get("limit") {
        None => LS_DEFAULT_LIMIT as f64,
        Some(value) => value
            .as_f64()
            .filter(|limit| limit.is_finite())
            .ok_or_else(|| ToolError::Arguments("`limit` must be a finite number".into()))?,
    };
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
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

    let (mut output, truncated) = truncate_tool_output(&names.join("\n"));
    let mut notices = Vec::new();
    if limit_reached {
        notices.push(format!(
            "{limit} entries limit reached. Use limit={} for more",
            limit * 2.0
        ));
    }
    if truncated {
        notices.push(format!(
            "{} limit reached",
            format_size(TOOL_OUTPUT_MAX_BYTES)
        ));
    }
    if !notices.is_empty() {
        output.push_str(&format!("\n\n[{}]", notices.join(". ")));
    }
    Ok(ToolResult::text(output))
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
        None => PathBuf::from("."),
        Some(Value::String(path)) => PathBuf::from(path),
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
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
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
            if match_count >= input.limit {
                break 'files;
            }
        }
    }

    if match_count == 0 {
        return Ok(ToolResult::text("No matches found"));
    }
    let (mut output, byte_truncated) = truncate_tool_output(&formatted_matches.join("\n"));
    let mut notices = Vec::new();
    if match_count >= input.limit {
        notices.push(format!(
            "{} matches limit reached. Use limit={} for more, or refine pattern",
            input.limit,
            input.limit.saturating_mul(2)
        ));
    }
    if byte_truncated {
        notices.push(format!(
            "{} limit reached",
            format_size(TOOL_OUTPUT_MAX_BYTES)
        ));
    }
    if lines_truncated {
        notices.push(format!(
            "Some lines truncated to {GREP_MAX_LINE_CHARS} chars. Use read tool to see full lines"
        ));
    }
    if !notices.is_empty() {
        output.push_str(&format!("\n\n[{}]", notices.join(". ")));
    }
    Ok(ToolResult::text(output))
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
        None => PathBuf::from("."),
        Some(Value::String(path)) => PathBuf::from(path),
        Some(_) => return Err(ToolError::Arguments("`path` must be a string".into())),
    };
    let limit =
        optional_nonnegative_integer(object.get("limit"), "limit")?.unwrap_or(FIND_DEFAULT_LIMIT);
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
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
    let mut limit_reached = false;
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
            limit_reached = true;
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

    let (mut output, byte_truncated) = truncate_tool_output(&results.join("\n"));
    let mut notices = Vec::new();
    if limit_reached {
        notices.push(format!(
            "{limit} results limit reached. Use limit={} for more, or refine pattern",
            limit.saturating_mul(2)
        ));
    }
    if byte_truncated {
        notices.push(format!(
            "{} limit reached",
            format_size(TOOL_OUTPUT_MAX_BYTES)
        ));
    }
    if !notices.is_empty() {
        output.push_str(&format!("\n\n[{}]", notices.join(". ")));
    }
    Ok(ToolResult::text(output))
}

async fn run_bash(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    const STRUCTURED_OUTPUT_MAX_BYTES: usize = 1024 * 1024;
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
    let output = process
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await?;
    let mut full_output = String::from_utf8_lossy(&output.stdout).into_owned();
    full_output.push_str(&String::from_utf8_lossy(&output.stderr));
    let exit_code = output.status.code().unwrap_or(1);
    let (structured_output, truncated) = truncate_utf8(&full_output, STRUCTURED_OUTPUT_MAX_BYTES);
    let wall_time_seconds = (started_at.elapsed().as_millis() as f64 / 100.0).round() / 10.0;
    let mut content = if full_output.is_empty() {
        "(no output)".to_owned()
    } else {
        full_output.clone()
    };
    let is_error = exit_code != 0;
    if is_error {
        content.push_str(&format!("\n\nCommand exited with code {exit_code}"));
    }
    Ok(ToolResult {
        content,
        details: None,
        structured_content: Some(json!({
            "output": structured_output,
            "truncated": truncated,
            "exit_code": exit_code,
            "wall_time_seconds": wall_time_seconds,
        })),
        is_error,
    })
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

    #[tokio::test]
    async fn read_write_and_exact_edit_operate_in_working_directory() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        let write = json!({"path":"nested/file.txt","content":"one\ntwo\n"});
        let result = execute(&dir, "write", &write).await.unwrap();
        assert_eq!(
            result.content,
            format!(
                "Successfully wrote to {}",
                dir.join("nested/file.txt").display()
            )
        );
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

        let byte_lines = (1..=500)
            .map(|line| format!("Line {line}: {}", "x".repeat(200)))
            .collect::<Vec<_>>();
        std::fs::write(&path, byte_lines.join("\n")).unwrap();
        let result = execute(&dir, "read", &json!({"path":"many-lines.txt"}))
            .await
            .unwrap();
        assert!(result.content.starts_with("Line 1:"));
        assert!(result.content.contains("(50.0KB limit). Use offset="));
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
        assert!(missing.to_string().contains("Could not find edits[0]"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "before\nafter\n");

        let empty = execute(
            &dir,
            "edit",
            &json!({"path":"file","oldText":"","newText":"replacement"}),
        )
        .await
        .unwrap_err();
        assert!(
            empty
                .to_string()
                .contains("edits[0].oldText must not be empty")
        );
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
