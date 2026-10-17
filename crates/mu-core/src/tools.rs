use std::{
    path::{Path, PathBuf},
    process::Stdio,
};

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
    pub content: String,
}

/// Pi's default coding tool set. Schemas use the same names and argument shapes.
pub fn definitions() -> Vec<Value> {
    vec![
        json!({"type":"function","name":"read","description":"Read a file. Supports line offset and limit.","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer","minimum":1},"limit":{"type":"integer","minimum":1}},"required":["path"],"additionalProperties":false}}),
        json!({"type":"function","name":"write","description":"Write content to a file, creating parent directories as needed.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":false}}),
        json!({"type":"function","name":"edit","description":"Replace one exact text occurrence in a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["path","oldText","newText"],"additionalProperties":false}}),
        json!({"type":"function","name":"bash","description":"Run a shell command in the working directory.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false}}),
    ]
}

pub async fn execute(cwd: &Path, name: &str, arguments: &Value) -> Result<ToolResult, ToolError> {
    match name {
        "read" => read_file(cwd, arguments),
        "write" => write_file(cwd, arguments),
        "edit" => edit_file(cwd, arguments),
        "bash" => run_bash(cwd, arguments).await,
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
    Ok(ToolResult { content })
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
    Ok(ToolResult {
        content: format!("Successfully wrote to {}", path.display()),
    })
}

fn edit_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let path = resolve(cwd, args)?;
    let old = string_arg(args, "oldText")?;
    let new = string_arg(args, "newText")?;
    if old.is_empty() {
        return Err(ToolError::Arguments("`oldText` must not be empty".into()));
    }
    let content = std::fs::read_to_string(&path)?;
    let mut matches = content.match_indices(old);
    let Some((index, _)) = matches.next() else {
        return Err(ToolError::Arguments("oldText was not found".into()));
    };
    if matches.next().is_some() {
        return Err(ToolError::Arguments(
            "oldText matched more than once".into(),
        ));
    }
    let mut updated = content;
    updated.replace_range(index..index + old.len(), new);
    std::fs::write(&path, updated)?;
    Ok(ToolResult {
        content: format!("Edited {}", path.display()),
    })
}

async fn run_bash(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
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
    let mut content = String::from_utf8_lossy(&output.stdout).into_owned();
    content.push_str(&String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(ToolError::Command(format!(
            "exit {}\n{content}",
            output.status
        )));
    }
    Ok(ToolResult { content })
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
        assert!(result.unwrap_err().to_string().contains("more than once"));
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
        assert!(missing.to_string().contains("oldText was not found"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "before\nafter\n");

        let empty = execute(
            &dir,
            "edit",
            &json!({"path":"file","oldText":"","newText":"replacement"}),
        )
        .await
        .unwrap_err();
        assert!(empty.to_string().contains("oldText` must not be empty"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "before\nafter\n");
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
        assert!(empty.content.is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_failure_preserves_stdout_and_stderr_in_the_error() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let error = execute(
            &dir,
            "bash",
            &json!({"command":"printf stdout; printf stderr >&2; exit 7"}),
        )
        .await
        .unwrap_err();

        let text = error.to_string();
        assert!(text.contains("command failed: exit"));
        assert!(text.contains("stdout"));
        assert!(text.contains("stderr"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
