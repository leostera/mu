use std::path::{Path, PathBuf};

/// Resolve an existing path to its symlink-free form, preserving the input when
/// the target is missing or cannot be resolved.
pub fn canonicalize_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned())
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
