use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file_path: PathBuf,
    pub disable_model_invocation: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillDiagnostic {
    pub path: PathBuf,
    pub message: String,
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

    fn discover_with_home(cwd: &Path, agent_dir: &Path, home: &Path) -> Self {
        let mut catalog = Self::default();
        let mut seen_names = HashSet::new();
        for root in [
            agent_dir.join("skills"),
            cwd.join(".pi/skills"),
            cwd.join(".agents/skills"),
            home.join(".agents/skills"),
        ] {
            scan(&root, &mut catalog, &mut seen_names);
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

fn scan(root: &Path, catalog: &mut SkillCatalog, names: &mut HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let entries: Vec<_> = entries.filter_map(Result::ok).collect();
    if entries.iter().any(|entry| entry.file_name() == "SKILL.md") {
        let file = root.join("SKILL.md");
        match parse_skill(&file) {
            Ok(skill) => add_skill(skill, catalog, names),
            Err(message) => catalog.diagnostics.push(SkillDiagnostic {
                path: file,
                message,
            }),
        }
        return;
    }
    for entry in entries {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            scan(&path, catalog, names);
        } else if file_type.is_file() && path.extension().is_some_and(|extension| extension == "md")
        {
            // Pi also accepts described Markdown files directly in a skills root.
            // Unlike declared SKILL.md files, unrelated or malformed Markdown is ignored.
            if let Ok(skill) = parse_skill(&path) {
                add_skill(skill, catalog, names);
            }
        }
    }
}

fn add_skill(skill: Skill, catalog: &mut SkillCatalog, names: &mut HashSet<String>) {
    if names.insert(skill.name.clone()) {
        catalog.skills.push(skill);
    } else {
        catalog.diagnostics.push(SkillDiagnostic {
            path: skill.file_path,
            message: format!("duplicate skill name `{}` ignored", skill.name),
        });
    }
}

fn parse_skill(path: &Path) -> Result<Skill, String> {
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut lines = text.lines();
    if lines.next() != Some("---") {
        return Err("missing YAML frontmatter".into());
    }
    let mut name = None;
    let mut description = None;
    let mut disabled = false;
    let mut closed = false;
    for line in lines {
        if line.trim() == "---" {
            closed = true;
            break;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_matches(['\"', '\'']);
        match key.trim() {
            "name" => name = Some(value.to_owned()),
            "description" => description = Some(value.to_owned()),
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
    if name.len() > 64
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        || name.starts_with('-')
        || name.ends_with('-')
        || name.contains("--")
    {
        return Err(format!("invalid skill name `{name}`"));
    }
    let description = description
        .filter(|s| !s.trim().is_empty())
        .ok_or("description is required")?;
    if description.len() > 1024 {
        return Err("description exceeds 1024 characters".into());
    }
    Ok(Skill {
        name,
        description,
        file_path: path.to_path_buf(),
        disable_model_invocation: disabled,
    })
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
        assert!(!parse_skill(&skill).unwrap().disable_model_invocation);
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
