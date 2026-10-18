use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use ignore::gitignore::{Gitignore, GitignoreBuilder};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file_path: PathBuf,
    pub disable_model_invocation: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ParsedSkill {
    skill: Skill,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillDiagnosticKind {
    Warning,
    Collision,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillCollision {
    pub name: String,
    pub winner_path: PathBuf,
    pub loser_path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillDiagnostic {
    pub kind: SkillDiagnosticKind,
    pub path: PathBuf,
    pub message: String,
    pub collision: Option<SkillCollision>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SkillCatalog {
    pub skills: Vec<Skill>,
    pub diagnostics: Vec<SkillDiagnostic>,
}

impl SkillCatalog {
    /// Discover Pi-style skills in global and project locations. Project skills
    /// follow user skills, so name collisions retain Pi's first-wins behavior.
    pub fn discover(cwd: &Path, agent_dir: &Path) -> Self {
        Self::discover_with_home(cwd, agent_dir, &home_dir())
    }

    pub fn discover_with_paths(
        cwd: &Path,
        agent_dir: &Path,
        skill_paths: &[PathBuf],
        include_defaults: bool,
    ) -> Self {
        Self::discover_with_paths_and_home(
            cwd,
            agent_dir,
            skill_paths,
            include_defaults,
            &home_dir(),
        )
    }

    fn discover_with_home(cwd: &Path, agent_dir: &Path, home: &Path) -> Self {
        Self::discover_with_paths_and_home(cwd, agent_dir, &[], true, home)
    }

    fn discover_with_paths_and_home(
        cwd: &Path,
        agent_dir: &Path,
        skill_paths: &[PathBuf],
        include_defaults: bool,
        home: &Path,
    ) -> Self {
        let mut catalog = SkillCatalog::default();
        let mut seen_names = HashMap::new();
        let mut scanned_directories = HashSet::new();
        let mut scanned_skill_files = HashSet::new();
        if include_defaults {
            for root in [
                agent_dir.join("skills"),
                cwd.join(".pi/skills"),
                cwd.join(".agents/skills"),
                home.join(".agents/skills"),
            ] {
                scan(
                    &root,
                    &mut catalog,
                    &mut seen_names,
                    &mut scanned_directories,
                    &mut scanned_skill_files,
                    &[],
                );
            }
        }
        for path in skill_paths {
            let path = if path.is_absolute() {
                path.clone()
            } else {
                cwd.join(path)
            };
            if !path.exists() {
                catalog.diagnostics.push(SkillDiagnostic {
                    kind: SkillDiagnosticKind::Warning,
                    path: path.clone(),
                    message: "skill path does not exist".into(),
                    collision: None,
                });
            } else if path.is_dir() {
                scan(
                    &path,
                    &mut catalog,
                    &mut seen_names,
                    &mut scanned_directories,
                    &mut scanned_skill_files,
                    &[],
                );
            } else if path.extension().is_some_and(|extension| extension == "md") {
                match parse_skill(&path) {
                    Ok(parsed) => add_skill(
                        parsed,
                        &mut catalog,
                        &mut seen_names,
                        &mut scanned_skill_files,
                    ),
                    Err(message) => catalog.diagnostics.push(SkillDiagnostic {
                        kind: SkillDiagnosticKind::Warning,
                        path,
                        message,
                        collision: None,
                    }),
                }
            } else {
                catalog.diagnostics.push(SkillDiagnostic {
                    kind: SkillDiagnosticKind::Warning,
                    path: path.clone(),
                    message: "skill path is not a Markdown file or directory".into(),
                    collision: None,
                });
            }
        }
        catalog
    }

    /// Pi-style metadata only: the model is instructed to load each skill file
    /// with the read tool when relevant, rather than injecting full files.
    pub fn system_prompt(&self) -> String {
        let visible: Vec<_> = self
            .skills
            .iter()
            .filter(|skill| !skill.disable_model_invocation)
            .collect();
        if visible.is_empty() {
            return String::new();
        }
        let mut prompt = String::from(
            "\n\nThe following skills provide specialized instructions for specific tasks. Use the read tool to load a skill's file when relevant. Resolve relative paths mentioned by a skill against the directory containing its SKILL.md.\n\n<available_skills>\n",
        );
        for skill in visible {
            prompt.push_str("  <skill>\n    <name>");
            prompt.push_str(&escape_xml(&skill.name));
            prompt.push_str("</name>\n    <description>");
            prompt.push_str(&escape_xml(&skill.description));
            prompt.push_str("</description>\n    <location>");
            prompt.push_str(&escape_xml(&skill.file_path.to_string_lossy()));
            prompt.push_str("</location>\n  </skill>\n");
        }
        prompt.push_str("</available_skills>");
        prompt
    }
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
}

fn scan(
    root: &Path,
    catalog: &mut SkillCatalog,
    names: &mut HashMap<String, PathBuf>,
    scanned_directories: &mut HashSet<PathBuf>,
    scanned_skill_files: &mut HashSet<PathBuf>,
    inherited_ignore_rules: &[Gitignore],
) {
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_owned());
    if !scanned_directories.insert(canonical_root) {
        return;
    }
    let mut ignore_rules = inherited_ignore_rules.to_vec();
    if let Some(local_rules) = load_ignore_rules(root) {
        ignore_rules.push(local_rules);
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let entries: Vec<_> = entries.filter_map(Result::ok).collect();
    if entries.iter().any(|entry| entry.file_name() == "SKILL.md") {
        let file = root.join("SKILL.md");
        if !is_ignored(&file, false, &ignore_rules) {
            match parse_skill(&file) {
                Ok(parsed) => add_skill(parsed, catalog, names, scanned_skill_files),
                Err(message) => catalog.diagnostics.push(SkillDiagnostic {
                    kind: SkillDiagnosticKind::Warning,
                    path: file,
                    message,
                    collision: None,
                }),
            }
        }
        return;
    }
    for entry in entries {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        if is_ignored(&path, metadata.is_dir(), &ignore_rules) {
            continue;
        }
        if metadata.is_dir() {
            scan(
                &path,
                catalog,
                names,
                scanned_directories,
                scanned_skill_files,
                &ignore_rules,
            );
        } else if metadata.is_file() && path.extension().is_some_and(|extension| extension == "md")
        {
            // Pi also accepts described Markdown files directly in a skills root.
            // Unlike declared SKILL.md files, unrelated or malformed Markdown is ignored.
            if let Ok(parsed) = parse_skill(&path) {
                add_skill(parsed, catalog, names, scanned_skill_files);
            }
        }
    }
}

fn load_ignore_rules(root: &Path) -> Option<Gitignore> {
    let mut builder = GitignoreBuilder::new(root);
    let mut found = false;
    for file_name in [".gitignore", ".ignore", ".fdignore"] {
        let path = root.join(file_name);
        if path.is_file() {
            found = true;
            let _ = builder.add(path);
        }
    }
    found.then(|| builder.build().ok()).flatten()
}

fn is_ignored(path: &Path, is_directory: bool, rules: &[Gitignore]) -> bool {
    let mut ignored = false;
    for matcher in rules {
        let matched = matcher.matched(path, is_directory);
        if !matched.is_none() {
            ignored = matched.is_ignore();
        }
    }
    ignored
}

fn add_skill(
    parsed: ParsedSkill,
    catalog: &mut SkillCatalog,
    names: &mut HashMap<String, PathBuf>,
    scanned_files: &mut HashSet<PathBuf>,
) {
    let ParsedSkill { skill, warnings } = parsed;
    catalog
        .diagnostics
        .extend(warnings.into_iter().map(|message| SkillDiagnostic {
            kind: SkillDiagnosticKind::Warning,
            path: skill.file_path.clone(),
            message,
            collision: None,
        }));
    let canonical_file = skill
        .file_path
        .canonicalize()
        .unwrap_or_else(|_| skill.file_path.clone());
    if scanned_files.contains(&canonical_file) {
        return;
    }
    if let Some(winner_path) = names.get(&skill.name) {
        catalog.diagnostics.push(SkillDiagnostic {
            kind: SkillDiagnosticKind::Collision,
            path: skill.file_path.clone(),
            message: format!("name \"{}\" collision", skill.name),
            collision: Some(SkillCollision {
                name: skill.name,
                winner_path: winner_path.clone(),
                loser_path: skill.file_path,
            }),
        });
    } else {
        names.insert(skill.name.clone(), skill.file_path.clone());
        scanned_files.insert(canonical_file);
        catalog.skills.push(skill);
    }
}

fn parse_skill(path: &Path) -> Result<ParsedSkill, String> {
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let mut lines = text.lines().peekable();
    if lines.next() != Some("---") {
        return Err("missing YAML frontmatter".into());
    }
    let mut name = None;
    let mut description = None;
    let mut disabled = false;
    let mut closed = false;
    while let Some(line) = lines.next() {
        if line.trim() == "---" {
            closed = true;
            break;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_matches(['\"', '\'']);
        match key.trim() {
            "name" => {
                name = line
                    .split_once(':')
                    .and_then(|(_, raw)| parse_frontmatter_string(raw));
            }
            "description" if matches!(value, "|" | "|-" | "|+" | ">" | ">-" | ">+") => {
                let folded = value.starts_with('>');
                let mut block_lines = Vec::new();
                while lines
                    .peek()
                    .is_some_and(|next| next.trim().is_empty() || next.starts_with([' ', '\t']))
                {
                    let block_line = lines.next().expect("peeked description block line");
                    block_lines.push(block_line.trim_start().to_owned());
                }
                let separator = if folded { " " } else { "\n" };
                description = Some(block_lines.join(separator).trim().to_owned());
            }
            "description" => {
                description = line
                    .split_once(':')
                    .and_then(|(_, raw)| parse_frontmatter_string(raw));
            }
            "disable-model-invocation" => {
                disabled = line
                    .split_once(':')
                    .is_some_and(|(_, raw)| raw.trim() == "true");
            }
            _ => {}
        }
    }
    if !closed {
        return Err("unterminated YAML frontmatter".into());
    }
    let dir_name = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let name = name
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| dir_name.to_owned());
    let description = description
        .filter(|s| !s.trim().is_empty())
        .ok_or("description is required")?;
    let mut warnings = Vec::new();
    let name_length = name.encode_utf16().count();
    if name_length > 64 {
        warnings.push(format!("name exceeds 64 characters ({name_length})"));
    }
    if !name.chars().all(|character| {
        character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
    }) {
        warnings.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)".into(),
        );
    }
    if name.starts_with('-') || name.ends_with('-') {
        warnings.push("name must not start or end with a hyphen".into());
    }
    if name.contains("--") {
        warnings.push("name must not contain consecutive hyphens".into());
    }
    let description_length = description.encode_utf16().count();
    if description_length > 1024 {
        warnings.push(format!(
            "description exceeds 1024 characters ({description_length})"
        ));
    }
    Ok(ParsedSkill {
        skill: Skill {
            name,
            description,
            file_path: path.to_path_buf(),
            disable_model_invocation: disabled,
        },
        warnings,
    })
}

fn parse_frontmatter_string(raw_value: &str) -> Option<String> {
    let value = strip_yaml_comment(raw_value).trim();
    if value.is_empty() || matches!(value, "~" | "null" | "Null" | "NULL" | "true" | "false") {
        return None;
    }
    if let Some(value) = value.strip_prefix('\'') {
        return value
            .strip_suffix('\'')
            .map(|value| value.replace("''", "'"));
    }
    if value.starts_with('"') {
        return serde_json::from_str(value).ok();
    }
    if value.parse::<f64>().is_ok() || value.starts_with(['[', '{']) {
        return None;
    }
    Some(value.to_owned())
}

fn strip_yaml_comment(value: &str) -> &str {
    let mut quote = None;
    let mut escaped = false;
    for (index, character) in value.char_indices() {
        match (quote, character) {
            (Some('"'), '\\') if !escaped => escaped = true,
            (Some(active), current) if active == current && !escaped => quote = None,
            (None, '\'' | '"') => quote = Some(character),
            (None, '#') if index == 0 || value[..index].ends_with(char::is_whitespace) => {
                return &value[..index];
            }
            _ => escaped = false,
        }
        if character != '\\' {
            escaped = false;
        }
    }
    value
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn explicit_skill_paths_load_after_or_instead_of_default_discovery() {
        let root = std::env::temp_dir().join(format!("mu-skill-paths-{}", Uuid::new_v4()));
        let agent_dir = root.join("agent");
        let default_file = agent_dir.join("skills/default/SKILL.md");
        let explicit_file = root.join("custom/SKILL.md");
        std::fs::create_dir_all(default_file.parent().unwrap()).unwrap();
        std::fs::create_dir_all(explicit_file.parent().unwrap()).unwrap();
        std::fs::write(
            &default_file,
            "---\nname: default\ndescription: discovered\n---\n",
        )
        .unwrap();
        std::fs::write(
            &explicit_file,
            "---\nname: explicit\ndescription: selected directly\n---\n",
        )
        .unwrap();

        let defaults_disabled = SkillCatalog::discover_with_paths_and_home(
            &root,
            &agent_dir,
            std::slice::from_ref(&explicit_file),
            false,
            &root,
        );
        assert_eq!(defaults_disabled.skills.len(), 1);
        assert_eq!(defaults_disabled.skills[0].name, "explicit");

        let defaults_enabled = SkillCatalog::discover_with_paths_and_home(
            &root,
            &agent_dir,
            std::slice::from_ref(&explicit_file),
            true,
            &root,
        );
        assert_eq!(defaults_enabled.skills.len(), 2);
        assert_eq!(defaults_enabled.skills[0].name, "default");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn discovers_described_markdown_skills_at_the_root_but_ignores_other_markdown() {
        let root = std::env::temp_dir().join(format!("mu-skills-root-{}", Uuid::new_v4()));
        let skills_dir = root.join(".pi/skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("review.md"),
            "---\nname: review-guide\ndescription: Review source changes\n---\nInstructions",
        )
        .unwrap();
        std::fs::write(skills_dir.join("notes.md"), "Ordinary notes, not a skill").unwrap();

        let catalog = SkillCatalog::discover_with_home(&root, &root.join("global"), &root);
        assert_eq!(catalog.skills.len(), 1);
        assert_eq!(catalog.skills[0].name, "review-guide");
        assert_eq!(catalog.skills[0].file_path, skills_dir.join("review.md"));
        assert!(catalog.diagnostics.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn quoted_true_is_not_a_yaml_boolean_for_model_invocation() {
        let root = std::env::temp_dir().join(format!("mu-skills-bool-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let skill = root.join("quoted/SKILL.md");
        std::fs::create_dir_all(skill.parent().unwrap()).unwrap();
        std::fs::write(
            &skill,
            "---\nname: quoted\ndescription: Quoted flag\ndisable-model-invocation: 'true'\n---\n",
        )
        .unwrap();
        assert!(!parse_skill(&skill).unwrap().skill.disable_model_invocation);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn discovers_skills_through_symlinked_directories_without_following_cycles() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("mu-skills-links-{}", Uuid::new_v4()));
        let external_skill = root.join("external/review");
        let skills_dir = root.join(".pi/skills");
        std::fs::create_dir_all(&external_skill).unwrap();
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            external_skill.join("SKILL.md"),
            "---\nname: review\ndescription: Review skill\n---\n",
        )
        .unwrap();
        symlink(&external_skill, skills_dir.join("review-link")).unwrap();
        symlink(&skills_dir, external_skill.join("cycle")).unwrap();

        let catalog = SkillCatalog::discover_with_home(&root, &root.join("global"), &root);
        assert_eq!(catalog.skills.len(), 1);
        assert_eq!(catalog.skills[0].name, "review");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn skips_the_same_skill_file_reached_through_a_symlink() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("mu-skill-file-alias-{}", Uuid::new_v4()));
        let skill_root = root.join(".pi/skills");
        std::fs::create_dir_all(&skill_root).unwrap();
        let original = skill_root.join("original.md");
        std::fs::write(
            &original,
            "---\nname: shared\ndescription: same physical skill\n---\n",
        )
        .unwrap();
        symlink(&original, skill_root.join("alias.md")).unwrap();

        let catalog = SkillCatalog::discover_with_home(&root, &root.join("global"), &root);
        assert_eq!(catalog.skills.len(), 1);
        assert!(catalog.diagnostics.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn skill_name_collisions_keep_user_skill_and_report_both_paths() {
        let root = std::env::temp_dir().join(format!("mu-skill-collision-{}", Uuid::new_v4()));
        let agent_dir = root.join("agent");
        let user_file = agent_dir.join("skills/shared/SKILL.md");
        let project_file = root.join(".pi/skills/shared/SKILL.md");
        std::fs::create_dir_all(user_file.parent().unwrap()).unwrap();
        std::fs::create_dir_all(project_file.parent().unwrap()).unwrap();
        std::fs::write(
            &user_file,
            "---\nname: shared\ndescription: user definition\n---\n",
        )
        .unwrap();
        std::fs::write(
            &project_file,
            "---\nname: shared\ndescription: project definition\n---\n",
        )
        .unwrap();

        let catalog = SkillCatalog::discover_with_home(&root, &agent_dir, &root);
        assert_eq!(catalog.skills.len(), 1);
        assert_eq!(catalog.skills[0].description, "user definition");
        let collision = catalog
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.kind == SkillDiagnosticKind::Collision)
            .expect("skill collision diagnostic");
        let collision = collision.collision.as_ref().expect("collision details");
        assert_eq!(collision.winner_path, user_file);
        assert_eq!(collision.loser_path, project_file);
        assert_eq!(catalog.diagnostics[0].message, "name \"shared\" collision");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn respects_root_and_nested_ignore_rules_for_all_pi_ignore_files() {
        let root = std::env::temp_dir().join(format!("mu-skills-ignore-{}", Uuid::new_v4()));
        let skills_dir = root.join(".pi/skills");
        let nested = skills_dir.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(skills_dir.join(".gitignore"), "git-ignored.md\n").unwrap();
        std::fs::write(skills_dir.join(".ignore"), "ignore-ignored.md\n").unwrap();
        std::fs::write(skills_dir.join(".fdignore"), "fd-ignored.md\n").unwrap();
        std::fs::write(nested.join(".ignore"), "hidden.md\n").unwrap();
        for name in [
            "visible.md",
            "git-ignored.md",
            "ignore-ignored.md",
            "fd-ignored.md",
        ] {
            std::fs::write(
                skills_dir.join(name),
                format!(
                    "---\nname: {}\ndescription: test\n---\n",
                    name.trim_end_matches(".md")
                ),
            )
            .unwrap();
        }
        for name in ["kept.md", "hidden.md"] {
            std::fs::write(
                nested.join(name),
                format!(
                    "---\nname: {}\ndescription: test\n---\n",
                    name.trim_end_matches(".md")
                ),
            )
            .unwrap();
        }

        let catalog = SkillCatalog::discover_with_home(&root, &root.join("global"), &root);
        let names = catalog
            .skills
            .iter()
            .map(|skill| skill.name.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(names, HashSet::from(["visible", "kept"]));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parses_scalar_quotes_and_ignores_only_unquoted_yaml_comments() {
        assert_eq!(
            parse_frontmatter_string(" 'it''s # literal' # comment"),
            Some("it's # literal".into())
        );
        assert_eq!(
            parse_frontmatter_string("\"quoted # literal\" # comment"),
            Some("quoted # literal".into())
        );
    }

    #[test]
    fn parses_bom_crlf_and_folded_yaml_descriptions() {
        let root = std::env::temp_dir().join(format!("mu-skills-yaml-{}", Uuid::new_v4()));
        let skill_dir = root.join(".pi/skills");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "\u{feff}---\r\nname: yaml-skill # comment\r\ndescription: >-\r\n  first line\r\n  second line\r\ndisable-model-invocation: true\r\n---\r\nbody",
        )
        .unwrap();

        let catalog = SkillCatalog::discover_with_home(&root, &root.join("global"), &root);
        assert_eq!(catalog.skills.len(), 1);
        assert_eq!(catalog.skills[0].name, "yaml-skill");
        assert_eq!(catalog.skills[0].description, "first line second line");
        assert!(catalog.skills[0].disable_model_invocation);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_skill_metadata_warns_but_does_not_drop_the_skill() {
        let root = std::env::temp_dir().join(format!("mu-skills-warnings-{}", Uuid::new_v4()));
        let skill_dir = root.join(".pi/skills");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("invalid.md"),
            format!(
                "---\nname: Bad_Name\ndescription: {}\n---\n",
                "x".repeat(1025)
            ),
        )
        .unwrap();

        let catalog = SkillCatalog::discover_with_home(&root, &root.join("global"), &root);
        assert_eq!(catalog.skills.len(), 1);
        assert_eq!(catalog.skills[0].name, "Bad_Name");
        assert!(
            catalog
                .diagnostics
                .iter()
                .any(|warning| { warning.message.contains("name contains invalid characters") })
        );
        assert!(catalog.diagnostics.iter().any(|warning| {
            warning
                .message
                .contains("description exceeds 1024 characters (1025)")
        }));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn discovers_nested_skills_and_formats_metadata_without_injecting_contents() {
        let root = std::env::temp_dir().join(format!("mu-skills-{}", Uuid::new_v4()));
        let skill_dir = root.join(".pi/skills/rust");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: rust\ndescription: Rust <guidance>\n---\nsecret body",
        )
        .unwrap();
        let catalog = SkillCatalog::discover_with_home(&root, &root.join("global"), &root);
        assert_eq!(catalog.skills.len(), 1);
        let prompt = catalog.system_prompt();
        assert!(prompt.contains("Rust &lt;guidance&gt;"));
        assert!(!prompt.contains("secret body"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
