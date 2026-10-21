//! Pi's config-value resolution (`core/resolve-config-value.ts`): a value is a literal, a
//! `$VAR`/`${VAR}` template, or `!command` whose trimmed stdout is used (cached per process).

use std::{
    collections::HashMap,
    process::{Command, Stdio},
    sync::{LazyLock, Mutex},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum Part {
    Literal(String),
    Env(String),
}

fn is_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn push_literal(parts: &mut Vec<Part>, text: &str) {
    if text.is_empty() {
        return;
    }
    match parts.last_mut() {
        Some(Part::Literal(previous)) => previous.push_str(text),
        _ => parts.push(Part::Literal(text.to_owned())),
    }
}

fn parse_template(config: &str) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut rest = config;
    while let Some(dollar) = rest.find('$') {
        push_literal(&mut parts, &rest[..dollar]);
        let after = &rest[dollar + 1..];
        if let Some(escaped) = after.strip_prefix(['$', '!']) {
            push_literal(&mut parts, &after[..1]);
            rest = escaped;
        } else if let Some(braced) = after.strip_prefix('{')
            && let Some(end) = braced.find('}')
        {
            let name = &braced[..end];
            if is_env_name(name) {
                parts.push(Part::Env(name.to_owned()));
            } else {
                push_literal(&mut parts, &rest[dollar..dollar + end + 3]);
            }
            rest = &braced[end + 1..];
        } else {
            let length = after
                .char_indices()
                .take_while(|(i, c)| {
                    c.is_ascii_alphabetic() || *c == '_' || (*i > 0 && c.is_ascii_digit())
                })
                .count();
            if length > 0 {
                parts.push(Part::Env(after[..length].to_owned()));
                rest = &after[length..];
            } else {
                push_literal(&mut parts, "$");
                rest = after;
            }
        }
    }
    push_literal(&mut parts, rest);
    parts
}

static COMMAND_CACHE: LazyLock<Mutex<HashMap<String, Option<String>>>> =
    LazyLock::new(Mutex::default);

fn run_command(command: &str) -> Option<String> {
    let mut child = Command::new("sh")
        .args(["-c", command])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match child.try_wait().ok()? {
            Some(status) if status.success() => break,
            Some(_) => return None,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
    let output = child.wait_with_output().ok()?;
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!value.is_empty()).then_some(value)
}

/// Resolve `config`; `None` when a referenced variable is unset/empty or the command fails.
pub fn resolve(config: &str, env: &HashMap<String, String>) -> Option<String> {
    if let Some(command) = config.strip_prefix('!') {
        let mut cache = COMMAND_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        return cache
            .entry(command.to_owned())
            .or_insert_with(|| run_command(command))
            .clone();
    }
    let mut resolved = String::new();
    for part in parse_template(config) {
        match part {
            Part::Literal(text) => resolved.push_str(&text),
            Part::Env(name) => {
                let value = env
                    .get(&name)
                    .filter(|v| !v.is_empty())
                    .cloned()
                    .or_else(|| std::env::var(&name).ok().filter(|v| !v.is_empty()))?;
                resolved.push_str(&value);
            }
        }
    }
    Some(resolved)
}

/// Names of environment variables a template references (none for commands).
pub fn env_var_names(config: &str) -> Vec<String> {
    if config.starts_with('!') {
        return Vec::new();
    }
    let mut names = Vec::new();
    for part in parse_template(config) {
        if let Part::Env(name) = part
            && !names.contains(&name)
        {
            names.push(name);
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn literals_and_templates_resolve_with_escapes() {
        let e = env(&[("MU_A", "x"), ("MU_B", "y")]);
        assert_eq!(resolve("plain", &e).as_deref(), Some("plain"));
        assert_eq!(resolve("$MU_A-${MU_B}!", &e).as_deref(), Some("x-y!"));
        assert_eq!(
            resolve("$$MU_A $!x cost$", &e).as_deref(),
            Some("$MU_A !x cost$")
        );
        assert_eq!(
            resolve("${not valid} ${open", &e).as_deref(),
            Some("${not valid} ${open")
        );
        assert_eq!(resolve("$MU_MISSING_VAR_XYZ", &e), None);
    }

    #[test]
    fn commands_use_trimmed_stdout_and_failures_are_none() {
        let e = env(&[]);
        assert_eq!(resolve("!echo '  secret  '", &e).as_deref(), Some("secret"));
        assert_eq!(resolve("!exit 3", &e), None);
        assert_eq!(resolve("!true", &e), None, "empty output is unset");
    }

    #[test]
    fn lists_distinct_env_names() {
        assert_eq!(env_var_names("$A-${B}-$A"), ["A", "B"]);
        assert!(env_var_names("!echo $A").is_empty());
    }
}
