//! Deterministic core of Pi's compaction (`core/compaction/compaction.ts`): token estimates,
//! the trigger rule, and cut-point selection. Summary generation (an LLM call) is not ported.
//!
//! Entries are Pi session-entry JSON objects (`type: "message" | "compaction" | ...`).

use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactionSettings {
    pub enabled: bool,
    pub reserve_tokens: u64,
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        }
    }
}

/// Prompt appended to the serialized conversation when asking for a checkpoint summary.
/// Pi's `generateSummary` prompt: the base prompt plus the user's focus, when given.
pub fn summarization_prompt(custom_instructions: Option<&str>) -> String {
    summarization_prompt_for(custom_instructions, false)
}

/// With `has_previous_summary`, Pi's update prompt that folds new messages into the old summary.
pub fn summarization_prompt_for(
    custom_instructions: Option<&str>,
    has_previous_summary: bool,
) -> String {
    let base = if has_previous_summary {
        UPDATE_SUMMARIZATION_PROMPT
    } else {
        SUMMARIZATION_PROMPT
    };
    match custom_instructions.map(str::trim).filter(|c| !c.is_empty()) {
        Some(custom) => format!("{base}\n\nAdditional focus: {custom}"),
        None => base.to_owned(),
    }
}

/// Pi's prompt for the part of a turn that precedes a mid-turn cut.
pub const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "The messages above are earlier context from an ongoing conversation. Later messages are stored separately and do not need to be reconstructed.

Create a concise checkpoint of the user's request and the progress shown above. This checkpoint will be placed before the later messages so the conversation can continue with the necessary context.

## Original Request
[What did the user ask for?]

## Progress So Far
- [Key decisions and work completed in these messages]

## Context Needed to Continue
- [Information from these messages needed to understand the later work]

Only summarize information explicitly present above. Do not infer or recreate later messages.";

pub const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed
- UPDATE \"Next Steps\" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.";

/// File paths touched by `read`, `write` and `edit` tool calls (Pi's `FileOperations`).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct FileOperations {
    pub read: std::collections::BTreeSet<String>,
    pub written: std::collections::BTreeSet<String>,
    pub edited: std::collections::BTreeSet<String>,
}

impl FileOperations {
    /// Record the tool calls in assistant message entries (`type: "toolCall"` content blocks).
    pub fn add_entries(&mut self, entries: &[Value]) {
        for entry in entries {
            let Some(message) = entry.get("message") else {
                continue;
            };
            if message.get("role").and_then(Value::as_str) != Some("assistant") {
                continue;
            }
            for block in message
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if block.get("type").and_then(Value::as_str) != Some("toolCall") {
                    continue;
                }
                let Some(path) = block.pointer("/arguments/path").and_then(Value::as_str) else {
                    continue;
                };
                let set = match block.get("name").and_then(Value::as_str) {
                    Some("read") => &mut self.read,
                    Some("write") => &mut self.written,
                    Some("edit") => &mut self.edited,
                    _ => continue,
                };
                set.insert(path.to_owned());
            }
        }
    }

    /// Carry the lists a previous compaction appended to its summary (Pi keeps them in the
    /// entry's `details`; Mu reads them back from the summary text).
    pub fn add_previous_summary(&mut self, summary: &str) {
        let section = |tag: &str| {
            let (open, close) = (format!("<{tag}>\n"), format!("\n</{tag}>"));
            summary
                .split_once(&open)
                .and_then(|(_, rest)| rest.split_once(&close))
                .map(|(body, _)| body.lines().map(str::to_owned).collect::<Vec<_>>())
                .unwrap_or_default()
        };
        self.read.extend(section("read-files"));
        self.edited.extend(section("modified-files"));
    }

    /// Pi's `computeFileLists` + `formatFileOperations`: files only read, then modified files,
    /// as XML-ish sections appended to the summary (empty when nothing was touched).
    pub fn format(&self) -> String {
        let modified = self
            .edited
            .iter()
            .chain(&self.written)
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let read_only = self
            .read
            .iter()
            .filter(|f| !modified.contains(*f))
            .cloned()
            .collect::<Vec<_>>();
        let mut sections = Vec::new();
        if !read_only.is_empty() {
            sections.push(format!(
                "<read-files>\n{}\n</read-files>",
                read_only.join("\n")
            ));
        }
        if !modified.is_empty() {
            sections.push(format!(
                "<modified-files>\n{}\n</modified-files>",
                modified.into_iter().collect::<Vec<_>>().join("\n")
            ));
        }
        if sections.is_empty() {
            String::new()
        } else {
            format!("\n\n{}", sections.join("\n\n"))
        }
    }
}

pub const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or \"(none)\" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or \"(none)\" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.";

const ESTIMATED_IMAGE_CHARS: usize = 4800;

/// Compaction triggers when context exceeds the window minus the reserve.
pub fn should_compact(
    context_tokens: u64,
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    settings.enabled && context_tokens > context_window.saturating_sub(settings.reserve_tokens)
}

/// Total context tokens of a usage object (`totalTokens`, else the component sum).
pub fn calculate_context_tokens(usage: &Value) -> u64 {
    let field = |name: &str| usage.get(name).and_then(Value::as_u64).unwrap_or(0);
    match field("totalTokens") {
        0 => field("input") + field("output") + field("cacheRead") + field("cacheWrite"),
        total => total,
    }
}

fn content_chars(content: &Value) -> usize {
    match content {
        Value::String(text) => text.chars().count(),
        Value::Array(blocks) => blocks
            .iter()
            .map(|block| match block.get("type").and_then(Value::as_str) {
                Some("text") => block
                    .get("text")
                    .and_then(Value::as_str)
                    .map_or(0, |t| t.chars().count()),
                Some("image") => ESTIMATED_IMAGE_CHARS,
                _ => 0,
            })
            .sum(),
        _ => 0,
    }
}

/// `chars / 4` token estimate, rounded up (deliberately conservative).
pub fn estimate_tokens(message: &Value) -> u64 {
    let text = |key: &str| {
        message
            .get(key)
            .and_then(Value::as_str)
            .map_or(0, |t| t.chars().count())
    };
    let chars = match message.get("role").and_then(Value::as_str) {
        Some("user" | "custom" | "toolResult") => {
            content_chars(message.get("content").unwrap_or(&Value::Null))
        }
        Some("assistant") => message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|block| match block.get("type").and_then(Value::as_str) {
                Some("text") => block
                    .get("text")
                    .and_then(Value::as_str)
                    .map_or(0, |t| t.chars().count()),
                Some("thinking") => block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .map_or(0, |t| t.chars().count()),
                Some("toolCall") => {
                    block
                        .get("name")
                        .and_then(Value::as_str)
                        .map_or(0, |t| t.chars().count())
                        + block
                            .get("arguments")
                            .map_or(0, |a| a.to_string().chars().count())
                }
                _ => 0,
            })
            .sum(),
        Some("bashExecution") => text("command") + text("output"),
        Some("branchSummary" | "compactionSummary") => text("summary"),
        _ => 0,
    };
    chars.div_ceil(4) as u64
}

/// The context-visible message of an entry, if any.
fn entry_message(entry: &Value) -> Option<&Value> {
    (entry.get("type").and_then(Value::as_str) == Some("message"))
        .then(|| entry.get("message"))
        .flatten()
}

fn is_compaction(entry: &Value) -> bool {
    entry.get("type").and_then(Value::as_str) == Some("compaction")
}

fn role(entry: &Value) -> Option<&str> {
    entry_message(entry)?.get("role")?.as_str()
}

/// Valid cuts are user-like or assistant messages; never tool results.
fn is_cut_point(entry: &Value) -> bool {
    matches!(
        role(entry),
        Some(
            "user"
                | "assistant"
                | "bashExecution"
                | "custom"
                | "branchSummary"
                | "compactionSummary"
        )
    )
}

fn is_turn_start(entry: &Value) -> bool {
    !is_compaction(entry)
        && matches!(
            role(entry),
            Some("user" | "bashExecution" | "custom" | "branchSummary" | "compactionSummary")
        )
}

/// The entry starting the turn that contains `entry_index`, if one exists at or after `start`.
pub fn find_turn_start_index(entries: &[Value], entry_index: usize, start: usize) -> Option<usize> {
    (start..=entry_index)
        .rev()
        .find(|&index| is_turn_start(&entries[index]))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutPoint {
    pub first_kept_entry_index: usize,
    /// The user-like entry that began the turn being split, when the cut falls mid-turn.
    pub turn_start_index: Option<usize>,
    pub is_split_turn: bool,
}

/// Walk back from the newest entry until `keep_recent_tokens` are accumulated, then cut at the
/// nearest valid cut point at or after that entry (Pi's `findCutPoint`).
pub fn find_cut_point(
    entries: &[Value],
    start: usize,
    end: usize,
    keep_recent_tokens: u64,
) -> CutPoint {
    let cut_points = (start..end)
        .filter(|&i| !is_compaction(&entries[i]) && is_cut_point(&entries[i]))
        .collect::<Vec<_>>();
    let Some(&first_cut) = cut_points.first() else {
        return CutPoint {
            first_kept_entry_index: start,
            turn_start_index: None,
            is_split_turn: false,
        };
    };
    let mut accumulated = 0;
    let mut cut = first_cut;
    for i in (start..end).rev() {
        let tokens = entry_message(&entries[i]).map_or(0, estimate_tokens);
        if tokens == 0 {
            continue;
        }
        accumulated += tokens;
        if accumulated >= keep_recent_tokens {
            // Prefer the closest valid cut at or after this entry, else the last one.
            cut = cut_points
                .iter()
                .copied()
                .find(|&c| c >= i)
                .unwrap_or(*cut_points.last().expect("non-empty"));
            break;
        }
    }
    // Pull in adjacent metadata entries that do not affect the context.
    while cut > start
        && !is_compaction(&entries[cut - 1])
        && entry_message(&entries[cut - 1]).is_none()
    {
        cut -= 1;
    }
    let starts_turn = is_turn_start(&entries[cut]);
    let turn_start = if starts_turn {
        None
    } else {
        find_turn_start_index(entries, cut, start)
    };
    CutPoint {
        first_kept_entry_index: cut,
        turn_start_index: turn_start,
        is_split_turn: !starts_turn && turn_start.is_some(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn file_operations_list_read_only_and_modified_files_sorted() {
        let call = |name: &str, path: &str| json!({"type":"toolCall","id":"x","name":name,"arguments":{"path":path}});
        let entries = vec![
            json!({"message":{"role":"assistant","content":[call("read","b.rs"),call("read","a.rs"),call("edit","a.rs"),call("write","z.md"),call("bash","ignored")]}}),
            json!({"message":{"role":"user","content":[call("read","user-not-counted")]}}),
        ];
        let mut ops = FileOperations::default();
        ops.add_entries(&entries);
        assert_eq!(
            ops.format(),
            "\n\n<read-files>\nb.rs\n</read-files>\n\n<modified-files>\na.rs\nz.md\n</modified-files>"
        );
        assert_eq!(FileOperations::default().format(), "");
    }

    #[test]
    fn previous_summary_file_lists_carry_into_the_next_compaction() {
        let mut ops = FileOperations::default();
        ops.add_previous_summary("text\n\n<read-files>\nr.rs\n</read-files>\n\n<modified-files>\nm.rs\nn.rs\n</modified-files>");
        assert_eq!(
            ops.format(),
            "\n\n<read-files>\nr.rs\n</read-files>\n\n<modified-files>\nm.rs\nn.rs\n</modified-files>"
        );
        let mut none = FileOperations::default();
        none.add_previous_summary("no lists here");
        assert_eq!(none.format(), "");
    }

    #[test]
    fn turn_prefix_prompt_is_pis_checkpoint_prompt() {
        assert!(
            TURN_PREFIX_SUMMARIZATION_PROMPT.starts_with("The messages above are earlier context")
        );
        assert!(TURN_PREFIX_SUMMARIZATION_PROMPT.contains("## Original Request"));
        assert!(
            TURN_PREFIX_SUMMARIZATION_PROMPT.ends_with("Do not infer or recreate later messages.")
        );
    }

    #[test]
    fn custom_instructions_are_appended_as_additional_focus() {
        use super::*;
        assert_eq!(summarization_prompt(None), SUMMARIZATION_PROMPT);
        assert_eq!(summarization_prompt(Some("  ")), SUMMARIZATION_PROMPT);
        assert!(
            summarization_prompt(Some("keep API names"))
                .ends_with("\n\nAdditional focus: keep API names")
        );
    }

    use serde_json::json;

    use super::*;

    #[test]
    fn trigger_uses_window_minus_reserve_and_honors_the_enabled_flag() {
        let settings = CompactionSettings::default();
        assert!(!should_compact(100_000, 116_384, &settings));
        assert!(should_compact(100_001, 116_384, &settings));
        assert!(!should_compact(
            u64::MAX,
            116_384,
            &CompactionSettings {
                enabled: false,
                ..settings
            }
        ));
    }

    #[test]
    fn usage_prefers_total_tokens_then_sums_components() {
        assert_eq!(
            calculate_context_tokens(&json!({"totalTokens":7,"input":1})),
            7
        );
        assert_eq!(
            calculate_context_tokens(&json!({"input":1,"output":2,"cacheRead":3,"cacheWrite":4})),
            10
        );
    }

    #[test]
    fn estimates_round_chars_over_four_up_and_count_images() {
        assert_eq!(
            estimate_tokens(&json!({"role":"user","content":"abcde"})),
            2
        );
        assert_eq!(
            estimate_tokens(
                &json!({"role":"toolResult","content":[{"type":"image"},{"type":"text","text":"ab"}]})
            ),
            1201
        );
        assert_eq!(
            estimate_tokens(
                &json!({"role":"assistant","content":[{"type":"toolCall","name":"ab","arguments":{"a":1}}]})
            ),
            3
        );
    }
}

#[cfg(test)]
mod cut_tests {
    use serde_json::json;

    use super::*;

    fn message(role: &str, chars: usize) -> Value {
        let text = "x".repeat(chars);
        let content = if role == "assistant" {
            json!([{"type":"text","text":text}])
        } else {
            json!(text)
        };
        json!({"type":"message","message":{"role":role,"content":content}})
    }

    #[test]
    fn cuts_at_the_closest_valid_point_after_the_budget_is_met() {
        // Each message is 100 tokens; keep ~250 → budget met at the third-from-last entry.
        let entries = vec![
            message("user", 400),
            message("assistant", 400),
            message("user", 400),
            message("assistant", 400),
            message("user", 400),
            message("assistant", 400),
        ];
        let cut = find_cut_point(&entries, 0, entries.len(), 250);
        assert_eq!(cut.first_kept_entry_index, 3);
        assert!(
            cut.is_split_turn,
            "cutting at an assistant message splits its turn"
        );
        assert_eq!(cut.turn_start_index, Some(2));
    }

    #[test]
    fn never_cuts_at_tool_results_and_keeps_everything_when_under_budget() {
        let entries = vec![
            message("user", 40),
            message("assistant", 40),
            json!({"type":"message","message":{"role":"toolResult","content":"x".repeat(4000)}}),
        ];
        let cut = find_cut_point(&entries, 0, 3, 500);
        assert_eq!(
            cut.first_kept_entry_index, 1,
            "tool results pull the cut back to their call"
        );
        let small = find_cut_point(&entries[..2], 0, 2, 10_000);
        assert_eq!(small.first_kept_entry_index, 0);
        assert!(!small.is_split_turn);
    }

    #[test]
    fn user_message_cuts_do_not_split_turns_and_empty_ranges_keep_the_start() {
        let entries = vec![
            message("user", 400),
            message("assistant", 400),
            message("user", 400),
        ];
        let cut = find_cut_point(&entries, 0, 3, 50);
        assert_eq!(
            cut,
            CutPoint {
                first_kept_entry_index: 2,
                turn_start_index: None,
                is_split_turn: false
            }
        );
        assert_eq!(find_cut_point(&[], 0, 0, 10).first_kept_entry_index, 0);
    }
}

/// Compaction-aware context entries for a root-to-leaf `path` (Pi's `buildContextEntries`).
///
/// With a compaction on the path the result is: the newest compaction entry, the retained
/// entries from its `firstKeptEntryId` up to it (system messages dropped), then everything
/// after it. Without one the path is returned unchanged.
pub fn build_context_entries(path: &[Value]) -> Vec<Value> {
    let Some(index) = path.iter().rposition(is_compaction) else {
        return path.to_vec();
    };
    let compaction = &path[index];
    let first_kept = compaction.get("firstKeptEntryId").and_then(Value::as_str);
    let mut entries = vec![compaction.clone()];
    let mut found = false;
    for entry in &path[..index] {
        found |= first_kept.is_some() && entry.get("id").and_then(Value::as_str) == first_kept;
        if found && role(entry) != Some("system") {
            entries.push(entry.clone());
        }
    }
    entries.extend(path[index + 1..].iter().cloned());
    entries
}

#[cfg(test)]
mod context_tests {
    use serde_json::json;

    use super::*;

    fn entry(id: &str, role: &str) -> Value {
        json!({"type":"message","id":id,"message":{"role":role,"content":"x"}})
    }

    #[test]
    fn compaction_keeps_entries_from_the_first_kept_id_and_everything_after() {
        let path = vec![
            entry("a", "user"),
            entry("b", "assistant"),
            entry("sys", "system"),
            entry("c", "user"),
            json!({"type":"compaction","id":"k","firstKeptEntryId":"c","summary":"s"}),
            entry("d", "user"),
        ];
        let ids = build_context_entries(&path)
            .iter()
            .map(|e| e["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["k", "c", "d"]);
        assert_eq!(
            build_context_entries(&path[..2]).len(),
            2,
            "no compaction: unchanged"
        );
    }

    #[test]
    fn system_messages_inside_the_retained_range_are_dropped_and_newest_compaction_wins() {
        let path = vec![
            json!({"type":"compaction","id":"old","firstKeptEntryId":"a","summary":"o"}),
            entry("a", "user"),
            entry("sys", "system"),
            entry("b", "assistant"),
            json!({"type":"compaction","id":"new","firstKeptEntryId":"a","summary":"n"}),
        ];
        let ids = build_context_entries(&path)
            .iter()
            .map(|e| e["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["new", "a", "b"]);
    }
}
