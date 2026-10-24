//! Tool descriptions in the system prompt: per-tool snippets and guidelines.

use serde_json::Value;

pub(super) fn tool_prompt_snippet(name: &str) -> Option<&'static str> {
    match name {
        "read" => Some("Read file contents"),
        "bash" => Some("Execute bash commands (ls, grep, find, etc.)"),
        "edit" => Some(
            "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
        ),
        "write" => Some("Create or overwrite files"),
        "grep" => Some("Search file contents for patterns (respects .gitignore)"),
        "find" => Some("Find files by glob pattern (respects .gitignore)"),
        "ls" => Some("List directory contents"),
        "powershell" => Some("Execute PowerShell commands"),
        _ => None,
    }
}

pub(super) fn tool_prompt_guidelines(name: &str) -> &'static [&'static str] {
    match name {
        "read" => &["Use read to examine files instead of cat or sed."],
        "edit" => &[
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
        ],
        "write" => &["Use write only for new files or complete rewrites."],
        "bash" | "powershell" => {
            &["You can inspect PI_* environment variables for current model and session details."]
        }
        _ => &[],
    }
}

pub(super) fn instructions_for_tools(base: &str, tools: &[Value]) -> String {
    let selected = tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    let snippets = selected
        .iter()
        .filter_map(|name| tool_prompt_snippet(name).map(|snippet| format!("- {name}: {snippet}")))
        .collect::<Vec<_>>();
    let mut guidelines = Vec::new();
    let mut add_guideline = |guideline: &str| {
        if !guidelines.iter().any(|existing| existing == guideline) {
            guidelines.push(guideline.to_owned());
        }
    };
    let has_shell = selected
        .iter()
        .any(|name| matches!(*name, "bash" | "powershell"));
    let has_search_or_list = selected
        .iter()
        .any(|name| matches!(*name, "grep" | "find" | "ls"));
    if has_shell && !has_search_or_list {
        if selected.contains(&"bash") && selected.contains(&"powershell") {
            add_guideline(
                "Use bash or PowerShell for file operations like listing, searching, and finding files",
            );
        } else if selected.contains(&"powershell") {
            add_guideline(
                "Use PowerShell for file operations like listing, searching, and finding files",
            );
        } else {
            add_guideline("Use bash for file operations like ls, rg, find");
        }
    }
    for name in &selected {
        for guideline in tool_prompt_guidelines(name) {
            add_guideline(guideline);
        }
    }
    add_guideline("Be concise in your responses");
    add_guideline("Show file paths clearly when working with files");

    let tool_section = if snippets.is_empty() {
        "(none)".to_owned()
    } else {
        snippets.join("\n")
    };
    let rules = guidelines
        .iter()
        .map(|guideline| format!("- {guideline}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{base}\n\n<tools>\n{tool_section}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>\n<rules>\n{rules}\n</rules>"
    )
}
