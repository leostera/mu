//! Prompt templates (`core/prompt-templates.ts`): `/name args` expands a Markdown file.

use std::path::{Path, PathBuf};

use regex::Regex;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptTemplate {
    pub name: String,
    pub description: String,
    pub argument_hint: Option<String>,
    pub content: String,
    pub file_path: PathBuf,
}

/// Bash-style argument splitting: whitespace separates, quotes group, `""` is an argument.
pub fn parse_command_args(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut has_token = false;
    let mut quote: Option<char> = None;
    for ch in input.chars() {
        match quote {
            Some(open) if ch == open => quote = None,
            Some(_) => current.push(ch),
            None if ch == '"' || ch == '\'' => {
                quote = Some(ch);
                has_token = true;
            }
            None if ch.is_whitespace() => {
                if has_token {
                    args.push(std::mem::take(&mut current));
                    has_token = false;
                }
            }
            None => {
                current.push(ch);
                has_token = true;
            }
        }
    }
    if has_token {
        args.push(current);
    }
    args
}

/// Substitute `$1`, `$@`, `$ARGUMENTS`, `${N:-default}`, `${@:N}` and `${@:N:L}`.
///
/// Replacement is single-pass: argument text containing `$1` is not substituted again.
pub fn substitute_args(content: &str, args: &[String]) -> String {
    let pattern = Regex::new(
        r"\$\{(\d+|ARGUMENTS|@):-([^}]*)\}|\$\{@:(\d+)(?::(\d+))?\}|\$(ARGUMENTS|@|\d+)",
    )
    .expect("static pattern");
    let all = args.join(" ");
    pattern
        .replace_all(content, |caps: &regex::Captures<'_>| {
            if let Some(target) = caps.get(1) {
                let value = match target.as_str() {
                    "@" | "ARGUMENTS" => all.clone(),
                    index => index
                        .parse::<usize>()
                        .ok()
                        .and_then(|n| n.checked_sub(1))
                        .and_then(|i| args.get(i))
                        .cloned()
                        .unwrap_or_default(),
                };
                return if value.is_empty() {
                    caps[2].to_owned()
                } else {
                    value
                };
            }
            if let Some(start) = caps.get(3) {
                let start = start
                    .as_str()
                    .parse::<usize>()
                    .unwrap_or(1)
                    .saturating_sub(1);
                let rest = args.get(start..).unwrap_or_default();
                return match caps
                    .get(4)
                    .and_then(|len| len.as_str().parse::<usize>().ok())
                {
                    Some(len) => rest.iter().take(len).cloned().collect::<Vec<_>>().join(" "),
                    None => rest.join(" "),
                };
            }
            match &caps[5] {
                "ARGUMENTS" | "@" => all.clone(),
                index => index
                    .parse::<usize>()
                    .ok()
                    .and_then(|n| n.checked_sub(1))
                    .and_then(|i| args.get(i))
                    .cloned()
                    .unwrap_or_default(),
            }
        })
        .into_owned()
}

fn load_template(path: &Path) -> Option<PromptTemplate> {
    let raw = std::fs::read_to_string(path).ok()?;
    let normalized = raw
        .strip_prefix('\u{feff}')
        .unwrap_or(&raw)
        .replace("\r\n", "\n");
    let (frontmatter, body) = match normalized.strip_prefix("---\n").and_then(|rest| {
        rest.find("\n---")
            .map(|end| (&rest[..end], rest[end + 4..].trim_start_matches('\n')))
    }) {
        Some((yaml, body)) => (
            yaml_rust2::YamlLoader::load_from_str(yaml)
                .ok()?
                .into_iter()
                .next(),
            body.to_owned(),
        ),
        None => (None, normalized.clone()),
    };
    let field = |name: &str| {
        frontmatter.as_ref().and_then(|doc| match &doc[name] {
            yaml_rust2::Yaml::String(value) => Some(value.clone()),
            _ => None,
        })
    };
    let description = field("description").unwrap_or_else(|| {
        body.lines()
            .find(|line| !line.trim().is_empty())
            .map(|line| {
                let head: String = line.chars().take(60).collect();
                if line.chars().count() > 60 {
                    format!("{head}...")
                } else {
                    head
                }
            })
            .unwrap_or_default()
    });
    Some(PromptTemplate {
        name: path.file_stem()?.to_string_lossy().into_owned(),
        description,
        argument_hint: field("argument-hint"),
        content: body,
        file_path: path.to_owned(),
    })
}

fn load_dir(dir: &Path) -> Vec<PromptTemplate> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "md"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
        .iter()
        .filter_map(|path| load_template(path))
        .collect()
}

/// Global (`<agent>/prompts`), project (`<cwd>/.pi/prompts`), then explicit paths.
pub fn load_prompt_templates(
    cwd: &Path,
    agent_dir: &Path,
    explicit: &[PathBuf],
    include_defaults: bool,
) -> Vec<PromptTemplate> {
    let mut templates = Vec::new();
    if include_defaults {
        templates.extend(load_dir(&agent_dir.join("prompts")));
        templates.extend(load_dir(&cwd.join(".pi/prompts")));
    }
    for path in explicit {
        let path = cwd.join(path);
        if path.is_dir() {
            templates.extend(load_dir(&path));
        } else if path.extension().is_some_and(|ext| ext == "md") {
            templates.extend(load_template(&path));
        }
    }
    templates
}

/// Expand `/name args` when a template matches; any other text is returned unchanged.
pub fn expand_prompt_template(text: &str, templates: &[PromptTemplate]) -> String {
    let Some(rest) = text.strip_prefix('/') else {
        return text.to_owned();
    };
    let (name, args) = match rest.split_once(char::is_whitespace) {
        Some((name, args)) => (name, args),
        None => (rest, ""),
    };
    if name.is_empty() {
        return text.to_owned();
    }
    match templates.iter().find(|template| template.name == name) {
        Some(template) => substitute_args(&template.content, &parse_command_args(args)),
        None => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn splits_arguments_like_bash_and_keeps_quoted_empty_strings() {
        assert_eq!(
            parse_command_args(r#"a "b c" 'd' "" e"#),
            ["a", "b c", "d", "", "e"]
        );
        assert!(parse_command_args("   ").is_empty());
    }

    #[test]
    fn substitutes_positional_all_default_and_slice_forms_once() {
        let a = args(&["one", "two", "three"]);
        assert_eq!(substitute_args("$1-$2-$9", &a), "one-two-");
        assert_eq!(
            substitute_args("$@|$ARGUMENTS", &a),
            "one two three|one two three"
        );
        assert_eq!(
            substitute_args("${4:-fallback} ${1:-x}", &a),
            "fallback one"
        );
        assert_eq!(substitute_args("${@:2}", &a), "two three");
        assert_eq!(substitute_args("${@:2:1}", &a), "two");
        assert_eq!(substitute_args("${@:0}", &a), "one two three");
        assert_eq!(substitute_args("${@:-none}", &[]), "none");
        // Argument text is never re-substituted.
        assert_eq!(substitute_args("$1", &args(&["$2", "z"])), "$2");
    }

    #[test]
    fn loads_templates_with_frontmatter_and_expands_matching_commands() {
        let root = std::env::temp_dir().join(format!("mu-prompts-{}", uuid::Uuid::new_v4()));
        let (agent, project) = (root.join("agent/prompts"), root.join("proj/.pi/prompts"));
        std::fs::create_dir_all(&agent).unwrap();
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(
            agent.join("review.md"),
            "---\ndescription: Review code\nargument-hint: <file>\n---\nReview $1 carefully.",
        )
        .unwrap();
        std::fs::write(
            project.join("plain.md"),
            "First line of a plain template\nmore",
        )
        .unwrap();
        let templates = load_prompt_templates(&root.join("proj"), &root.join("agent"), &[], true);
        assert_eq!(templates.len(), 2);
        assert_eq!(templates[0].description, "Review code");
        assert_eq!(templates[0].argument_hint.as_deref(), Some("<file>"));
        assert_eq!(templates[1].description, "First line of a plain template");
        assert_eq!(
            expand_prompt_template("/review \"a b.rs\"", &templates),
            "Review a b.rs carefully."
        );
        assert_eq!(
            expand_prompt_template("/unknown x", &templates),
            "/unknown x"
        );
        assert_eq!(expand_prompt_template("no slash", &templates), "no slash");
        assert!(
            load_prompt_templates(&root.join("proj"), &root.join("agent"), &[], false).is_empty()
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
