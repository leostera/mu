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

fn read_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let path = resolve(cwd, args)?;
    let text = std::fs::read_to_string(&path)?;
    let offset = args
        .get("offset")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as usize;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(2000) as usize;
    let content = text
        .lines()
        .skip(offset - 1)
        .take(limit)
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{line}", offset + i))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(ToolResult {
        content: format!("{}\n{}", path.display(), content),
    })
}

fn write_file(cwd: &Path, args: &Value) -> Result<ToolResult, ToolError> {
    let path = resolve(cwd, args)?;
    let content = string_arg(args, "content")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, content)?;
    Ok(ToolResult {
        content: format!("Wrote {} bytes to {}", content.len(), path.display()),
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
        execute(&dir, "write", &write).await.unwrap();
        let read = execute(&dir, "read", &json!({"path":"nested/file.txt","offset":2}))
            .await
            .unwrap();
        assert!(read.content.ends_with("     2\ttwo"));
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
    async fn edit_rejects_ambiguous_replacements() {
        let dir = std::env::temp_dir().join(format!("mu-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("file"), "x x").unwrap();
        let result = execute(
            &dir,
            "edit",
            &json!({"path":"file","oldText":"x","newText":"y"}),
        )
        .await;
        assert!(result.unwrap_err().to_string().contains("more than once"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
