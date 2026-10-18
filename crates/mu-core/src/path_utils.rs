use std::path::{Path, PathBuf};

/// Resolve an existing path to its symlink-free form, preserving the input when
/// the target is missing or cannot be resolved.
pub fn canonicalize_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned())
}

/// Revision token used to detect file replacement or modification.
pub fn file_revision(path: &Path) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    Some(file_revision_from_metadata(&metadata))
}

#[cfg(unix)]
fn file_revision_from_metadata(metadata: &std::fs::Metadata) -> String {
    use std::os::unix::fs::MetadataExt;

    let modified_ns =
        i128::from(metadata.mtime()) * 1_000_000_000 + i128::from(metadata.mtime_nsec());
    let changed_ns =
        i128::from(metadata.ctime()) * 1_000_000_000 + i128::from(metadata.ctime_nsec());
    format!(
        "{}:{}:{}:{modified_ns}:{changed_ns}",
        metadata.dev(),
        metadata.ino(),
        metadata.len()
    )
}

#[cfg(not(unix))]
fn file_revision_from_metadata(metadata: &std::fs::Metadata) -> String {
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let created_ns = metadata
        .created()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{}:{modified_ns}:{created_ns}", metadata.len())
}

/// Return the path relative to `cwd` when its lexical resolution stays inside it.
pub fn cwd_relative_path(file_path: &Path, cwd: &Path) -> Option<PathBuf> {
    let cwd = absolute_cwd(cwd);
    resolve_lexically(file_path, &cwd)
        .strip_prefix(cwd)
        .ok()
        .map(Path::to_owned)
}

/// Format a path relative to the working directory when possible, otherwise absolute.
pub fn format_path_relative_to_cwd_or_absolute(file_path: &Path, cwd: &Path) -> String {
    let cwd = absolute_cwd(cwd);
    let absolute_path = resolve_lexically(file_path, &cwd);
    let display_path = match absolute_path.strip_prefix(&cwd) {
        Ok(relative) if relative.as_os_str().is_empty() => PathBuf::from("."),
        Ok(relative) => relative.to_owned(),
        Err(_) => absolute_path,
    };
    display_path.to_string_lossy().replace('\\', "/")
}

fn absolute_cwd(cwd: &Path) -> PathBuf {
    let current_dir = std::env::current_dir().unwrap_or_default();
    resolve_lexically(cwd, &current_dir)
}

fn resolve_lexically(path: &Path, cwd: &Path) -> PathBuf {
    let absolute_path = if path.is_absolute() {
        path.to_owned()
    } else {
        cwd.join(path)
    };
    let mut resolved = PathBuf::new();
    for component in absolute_path.components() {
        match component {
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                resolved.push(component.as_os_str());
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !resolved.pop() && !absolute_path.is_absolute() {
                    resolved.push(component.as_os_str());
                }
            }
            std::path::Component::Normal(_) => resolved.push(component.as_os_str()),
        }
    }
    resolved
}

/// Whether a source reference is local, matching Pi's extension path rules.
pub fn is_local_path(value: &str) -> bool {
    let value = value.trim();
    ![
        "npm:", "git:", "github:", "http:", "https:", "ssh:", "builtin:",
    ]
    .iter()
    .any(|prefix| value.starts_with(prefix))
}

/// Convert Git Bash, MSYS, Cygwin, and WSL drive paths to native Windows form.
pub fn normalize_windows_shell_path(path: &str) -> String {
    let original = path;
    if !path.starts_with('/') || path.starts_with("//") || path.contains('\\') {
        return original.to_owned();
    }

    let path = path
        .strip_prefix("/mnt/")
        .or_else(|| path.strip_prefix("/cygdrive/"))
        .unwrap_or(path);
    let drive_path = path.strip_prefix('/').unwrap_or(path);
    let (drive, suffix) = drive_path.split_once('/').unwrap_or((drive_path, ""));
    if drive.len() != 1 || !drive.as_bytes()[0].is_ascii_alphabetic() {
        return original.to_owned();
    }
    format!(
        "{}:\\{}",
        drive.to_ascii_uppercase(),
        suffix.replace('/', "\\")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn file_revisions_exist_for_files_and_are_absent_for_missing_paths() {
        let root = std::env::temp_dir().join(format!("mu-file-revision-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("file.txt");
        let missing = root.join("missing.txt");
        std::fs::write(&file, "one").unwrap();
        let first = file_revision(&file).expect("file revision");
        std::fs::write(&file, "a longer version").unwrap();
        let second = file_revision(&file).expect("updated file revision");
        assert_ne!(first, second);
        assert_eq!(file_revision(&missing), None);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn canonicalization_follows_symlinks_and_preserves_unresolvable_paths() {
        let root = std::env::temp_dir().join(format!("mu-path-utils-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("target.txt");
        let missing = root.join("missing.txt");
        std::fs::write(&target, "contents").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let alias = root.join("alias.txt");
            symlink(&target, &alias).unwrap();
            assert_eq!(canonicalize_path(&alias), target.canonicalize().unwrap());
        }
        assert_eq!(canonicalize_path(&missing), missing);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cwd_relative_formatting_uses_lexically_resolved_paths() {
        let cwd = Path::new("/workspace/project");
        assert_eq!(
            cwd_relative_path(Path::new("src/../README.md"), cwd),
            Some(PathBuf::from("README.md"))
        );
        assert_eq!(cwd_relative_path(Path::new("."), cwd), Some(PathBuf::new()));
        assert_eq!(cwd_relative_path(Path::new("../sibling/file"), cwd), None);
        assert_eq!(
            format_path_relative_to_cwd_or_absolute(Path::new("."), cwd),
            "."
        );
        assert_eq!(
            format_path_relative_to_cwd_or_absolute(Path::new("../sibling/file"), cwd),
            "/workspace/sibling/file"
        );
    }

    #[test]
    fn local_path_classification_matches_pi_known_remote_prefixes() {
        for remote in [
            "npm:package",
            " git:repo",
            "github:owner/repo",
            "http://host",
            "https://host",
            "ssh://host",
            "builtin:tool",
        ] {
            assert!(!is_local_path(remote), "{remote} should be external");
        }
        for local in [
            "",
            "./extension",
            "../extension",
            "file:///tmp/extension",
            "ftp://host",
        ] {
            assert!(is_local_path(local), "{local} should be local");
        }
    }

    #[test]
    fn windows_shell_paths_are_normalized_without_touching_unc_or_native_paths() {
        assert_eq!(
            normalize_windows_shell_path("/c/project/file"),
            "C:\\project\\file"
        );
        assert_eq!(normalize_windows_shell_path("/c"), "C:\\");
        assert_eq!(normalize_windows_shell_path("/mnt/d/file"), "D:\\file");
        assert_eq!(normalize_windows_shell_path("/cygdrive/e/file"), "E:\\file");
        assert_eq!(normalize_windows_shell_path("/mnt/e"), "E:\\");
        for native in [
            "C:\\project",
            "//server/share",
            "/usr/local/bin",
            "/xy/path",
            "/mnt/xyz/path",
        ] {
            assert_eq!(normalize_windows_shell_path(native), native);
        }
    }
}
