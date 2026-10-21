//! Project context files (`AGENTS.md` and friends), following Pi's `loadProjectContextFiles`.
//!
//! The agent directory's file comes first, then one file per ancestor directory from the
//! filesystem root down to the working directory. Pi's git-worktree shadowing rule is not ported.

use std::path::{Path, PathBuf};

const CANDIDATES: [&str; 5] = [
    "AGENTS.override.md",
    "AGENTS.md",
    "AGENTS.MD",
    "CLAUDE.md",
    "CLAUDE.MD",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
}

fn load_from_dir(dir: &Path) -> Option<ContextFile> {
    CANDIDATES.iter().find_map(|name| {
        let path = dir.join(name);
        if !path.is_file() {
            return None;
        }
        let content = std::fs::read_to_string(&path).ok()?;
        Some(ContextFile {
            content: content
                .strip_prefix('\u{feff}')
                .unwrap_or(&content)
                .to_owned(),
            path,
        })
    })
}

pub fn load_project_context_files(cwd: &Path, agent_dir: &Path) -> Vec<ContextFile> {
    let mut files = Vec::new();
    let mut seen = std::collections::HashSet::new();
    if let Some(global) = load_from_dir(agent_dir) {
        seen.insert(global.path.clone());
        files.push(global);
    }
    let mut ancestors = Vec::new();
    let mut current = Some(cwd);
    while let Some(dir) = current {
        if let Some(file) = load_from_dir(dir) {
            if seen.insert(file.path.clone()) {
                ancestors.push(file);
            }
        }
        current = dir.parent();
    }
    ancestors.reverse();
    files.extend(ancestors);
    files
}

/// The `project_context` prompt section.
pub fn render_project_context(files: &[ContextFile]) -> String {
    let mut parts = vec!["Project-specific instructions and guidelines:".to_owned()];
    parts.extend(files.iter().map(|file| {
        format!(
            "<project_instructions path=\"{}\">\n{}\n</project_instructions>",
            file.path.display(),
            file.content
        )
    }));
    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_global_then_ancestors_outermost_first_with_candidate_precedence() {
        let root = std::env::temp_dir().join(format!("mu-ctx-{}", uuid::Uuid::new_v4()));
        let (agent, outer, inner) = (root.join("agent"), root.join("a"), root.join("a/b"));
        std::fs::create_dir_all(&agent).unwrap();
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(agent.join("AGENTS.md"), "global").unwrap();
        std::fs::write(outer.join("CLAUDE.md"), "outer").unwrap();
        std::fs::write(inner.join("AGENTS.md"), "inner agents").unwrap();
        std::fs::write(inner.join("CLAUDE.md"), "inner claude").unwrap();

        let files = load_project_context_files(&inner, &agent);
        let contents = files.iter().map(|f| f.content.as_str()).collect::<Vec<_>>();
        assert_eq!(contents, ["global", "outer", "inner agents"]);
        let rendered = render_project_context(&files);
        assert!(rendered.starts_with(
            "Project-specific instructions and guidelines:\n\n<project_instructions path="
        ));
        assert!(rendered.ends_with("inner agents\n</project_instructions>"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
