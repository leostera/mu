//! Tab completion of file paths in the editor (Pi's `CombinedAutocompleteProvider`, paths only).

use std::path::{Path, PathBuf};

/// Result of completing the word before the cursor.
#[derive(Debug, PartialEq, Eq)]
pub enum Completion {
    /// Nothing matches.
    None,
    /// Replace the word with this text (a unique match, or the longest common prefix).
    Replace(String),
    /// Several matches and no further common prefix: show these.
    Choices(Vec<String>),
}

fn expand_home(prefix: &str) -> Option<PathBuf> {
    match prefix.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME").map(|home| PathBuf::from(home).join(rest)),
        None => Some(PathBuf::from(prefix)),
    }
}

fn common_prefix(names: &[String]) -> String {
    let first = names[0].as_str();
    let mut end = first.len();
    for name in &names[1..] {
        end = end.min(
            first
                .char_indices()
                .zip(name.chars())
                .take_while(|((_, a), b)| a == b)
                .last()
                .map_or(0, |((i, c), _)| i + c.len_utf8()),
        );
    }
    first[..end].to_owned()
}

/// Complete `word` (the text before the cursor, e.g. `src/tu` or `~/Dev`) against `cwd`.
/// Directories get a trailing `/`; dotfiles only complete when the typed name starts with `.`.
pub fn complete_path(word: &str, cwd: &Path) -> Completion {
    let (dir_part, name_part) = match word.rfind('/') {
        Some(at) => (&word[..=at], &word[at + 1..]),
        None => ("", word),
    };
    let Some(dir) = expand_home(dir_part) else {
        return Completion::None;
    };
    let dir = if dir_part.is_empty() {
        cwd.to_owned()
    } else if dir.is_absolute() {
        dir
    } else {
        cwd.join(dir)
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Completion::None;
    };
    let mut names = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with(name_part)
                || (name.starts_with('.') && !name_part.starts_with('.'))
            {
                return None;
            }
            let is_dir = entry.path().is_dir();
            Some(if is_dir { format!("{name}/") } else { name })
        })
        .collect::<Vec<_>>();
    names.sort();
    match names.len() {
        0 => Completion::None,
        1 => Completion::Replace(format!("{dir_part}{}", names[0])),
        _ => {
            let common = common_prefix(&names);
            if common.len() > name_part.len() {
                Completion::Replace(format!("{dir_part}{common}"))
            } else {
                Completion::Choices(names)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mu-complete-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src/tui")).unwrap();
        std::fs::write(dir.join("src/main.rs"), "").unwrap();
        std::fs::write(dir.join("src/mod.rs"), "").unwrap();
        std::fs::write(dir.join(".hidden"), "").unwrap();
        std::fs::write(dir.join("README.md"), "").unwrap();
        dir
    }

    #[test]
    fn unique_matches_complete_and_directories_get_a_slash() {
        let dir = fixture();
        assert_eq!(
            complete_path("REA", &dir),
            Completion::Replace("README.md".into())
        );
        assert_eq!(
            complete_path("sr", &dir),
            Completion::Replace("src/".into())
        );
        assert_eq!(
            complete_path("src/t", &dir),
            Completion::Replace("src/tui/".into())
        );
        assert_eq!(complete_path("zzz", &dir), Completion::None);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ambiguous_matches_extend_to_the_common_prefix_then_list_choices() {
        let dir = fixture();
        assert_eq!(
            complete_path("src/m", &dir),
            Completion::Choices(vec!["main.rs".into(), "mod.rs".into()])
        );
        assert_eq!(
            complete_path("src/ma", &dir),
            Completion::Replace("src/main.rs".into())
        );
        assert_eq!(
            complete_path("src/", &dir),
            Completion::Choices(vec!["main.rs".into(), "mod.rs".into(), "tui/".into()])
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn dotfiles_only_complete_when_asked_for() {
        let dir = fixture();
        assert_eq!(
            complete_path(".hi", &dir),
            Completion::Replace(".hidden".into())
        );
        assert!(
            !matches!(complete_path("", &dir), Completion::Replace(ref r) if r.starts_with('.'))
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
