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
