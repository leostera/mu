//! Pi's `@file` arguments and initial-message assembly (`cli/file-processor.ts`, `initial-message.ts`).

use std::path::{Path, PathBuf};

/// Positional arguments split into `@file` references and plain message text.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PromptArguments {
    pub files: Vec<String>,
    pub messages: Vec<String>,
}

impl PromptArguments {
    pub fn parse(arguments: &[String]) -> Self {
        let mut parsed = Self::default();
        for argument in arguments {
            match argument.strip_prefix('@') {
                Some(file) => parsed.files.push(file.to_owned()),
                None => parsed.messages.push(argument.clone()),
            }
        }
        parsed
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FileArgumentError {
    #[error("File not found: {0}")]
    NotFound(PathBuf),
    #[error("Could not read file {path}: {message}")]
    Unreadable { path: PathBuf, message: String },
    #[error("Image attachment {0} is not supported yet")]
    Image(PathBuf),
}

fn resolve(file: &str, cwd: &Path) -> PathBuf {
    let expanded = match file.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME").map_or_else(
            || PathBuf::from(file),
            |home| PathBuf::from(home).join(rest),
        ),
        None => PathBuf::from(file),
    };
    if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    }
}

/// Render `@file` arguments as Pi's `<file name="…">…</file>` blocks; empty files are skipped.
pub fn file_text(files: &[String], cwd: &Path) -> Result<String, FileArgumentError> {
    let mut text = String::new();
    for file in files {
        let path = resolve(file, cwd);
        let bytes = std::fs::read(&path).map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => FileArgumentError::NotFound(path.clone()),
            _ => FileArgumentError::Unreadable {
                path: path.clone(),
                message: error.to_string(),
            },
        })?;
        if bytes.is_empty() {
            continue;
        }
        if is_image(&bytes) {
            return Err(FileArgumentError::Image(path));
        }
        let content = String::from_utf8_lossy(&bytes);
        let content = content.strip_prefix('\u{feff}').unwrap_or(&content);
        text.push_str(&format!(
            "<file name=\"{}\">\n{content}\n</file>\n",
            path.display()
        ));
    }
    Ok(text)
}

fn is_image(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x89, b'P', b'N', b'G'])
        || bytes.starts_with(&[0xff, 0xd8, 0xff])
        || bytes.starts_with(b"GIF8")
        || bytes.starts_with(b"BM")
        || (bytes.len() > 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP")
}

/// Pi's `buildInitialMessage`: stdin, then `@file` text, then only the *first* message, joined
/// without separators. The remaining messages are returned for separate follow-up turns.
pub fn build(
    stdin: Option<&str>,
    file_text: &str,
    messages: &[String],
) -> (Option<String>, Vec<String>) {
    let mut parts = Vec::new();
    parts.extend(stdin);
    if !file_text.is_empty() {
        parts.push(file_text);
    }
    let (first, rest) = messages
        .split_first()
        .map_or((None, &[][..]), |(f, r)| (Some(f.as_str()), r));
    parts.extend(first);
    ((!parts.is_empty()).then(|| parts.concat()), rest.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn splits_file_references_from_messages() {
        let parsed = PromptArguments::parse(&args(&["@a.txt", "explain", "@b.md", "this"]));
        assert_eq!(parsed.files, ["a.txt", "b.md"]);
        assert_eq!(parsed.messages, ["explain", "this"]);
    }

    #[test]
    fn renders_text_files_skips_empty_ones_and_strips_bom() {
        let dir = std::env::temp_dir().join(format!("mu-initial-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "\u{feff}hello").unwrap();
        std::fs::write(dir.join("empty.txt"), "").unwrap();
        let text = file_text(&args(&["a.txt", "empty.txt"]), &dir).unwrap();
        assert_eq!(
            text,
            format!(
                "<file name=\"{}\">\nhello\n</file>\n",
                dir.join("a.txt").display()
            )
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_and_image_files_are_errors() {
        let dir = std::env::temp_dir().join(format!("mu-initial-err-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("p.png"), [0x89, b'P', b'N', b'G', 1]).unwrap();
        assert!(matches!(
            file_text(&args(&["nope"]), &dir),
            Err(FileArgumentError::NotFound(_))
        ));
        assert!(matches!(
            file_text(&args(&["p.png"]), &dir),
            Err(FileArgumentError::Image(_))
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn initial_message_orders_stdin_files_then_messages() {
        assert_eq!(
            build(Some("in\n"), "<f/>\n", &args(&["go"])),
            (Some("in\n<f/>\ngo".into()), vec![])
        );
        assert_eq!(build(None, "", &[]), (None, vec![]));
        // Only the first message joins the initial prompt; the rest become later turns.
        assert_eq!(
            build(None, "<f/>\n", &args(&["one", "two", "three"])),
            (Some("<f/>\none".into()), args(&["two", "three"]))
        );
    }
}
