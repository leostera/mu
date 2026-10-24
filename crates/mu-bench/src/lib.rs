//! Deterministic inputs for the `hot_paths` benchmarks, shared so tests can check them.

use mu_session::{MessageUsage, SessionDocument};
use serde_json::{Value, json};

/// A session of `turns` user/assistant exchanges with realistic text sizes.
pub fn session(turns: usize) -> SessionDocument {
    let mut session = SessionDocument::new_with_id("/bench/project", "bench-session");
    for turn in 0..turns {
        session.append_user_text(format!(
            "Please look at file_{turn}.rs and explain {}",
            "the code ".repeat(20)
        ));
        session.append_assistant_text(
            "openai-responses",
            "openai",
            "gpt-bench",
            format!(
                "Here is what file_{turn}.rs does: {}",
                "it computes things. ".repeat(40)
            ),
            MessageUsage {
                input: 1_000 + turn as u64,
                output: 200,
                ..MessageUsage::default()
            },
        );
    }
    session
}

/// The session's entries as the JSON values compaction works on.
pub fn session_entries(session: &SessionDocument) -> Vec<Value> {
    let jsonl = session.to_jsonl().expect("bench session serializes");
    jsonl
        .lines()
        .skip(1)
        .map(|line| serde_json::from_str(line).expect("valid JSONL"))
        .collect()
}

/// An SSE byte stream of `frames` Chat Completions text chunks, then `[DONE]`.
pub fn completions_sse(frames: usize) -> Vec<u8> {
    let mut stream = Vec::new();
    for index in 0..frames {
        let chunk = json!({"choices":[{"index":0,"delta":{"content":format!("token {index} ")}}]});
        stream.extend_from_slice(format!("data: {chunk}\n\n").as_bytes());
    }
    stream.extend_from_slice(b"data: [DONE]\n\n");
    stream
}

/// Anthropic Messages stream events with `deltas` text deltas.
pub fn anthropic_events(deltas: usize) -> Vec<Value> {
    let mut events = vec![json!({"type":"message_start","message":{"usage":{"input_tokens":10}}})];
    events.extend((0..deltas).map(|index| {
        json!({"type":"content_block_delta","delta":{"type":"text_delta","text":format!("token {index} ")}})
    }));
    events.push(json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}));
    events.push(json!({"type":"message_stop"}));
    events
}

/// `count` tool definitions shaped like MCP tools.
pub fn tool_definitions(count: usize) -> Vec<Value> {
    (0..count)
        .map(|index| {
            json!({
                "type": "function",
                "name": format!("mcp__server{}__tool_{index}", index % 7),
                "description": format!("Operate resource {index}: list, read, search, or update records for project {}", index % 13),
                "parameters": {"type":"object","properties":{"query":{"type":"string","description":"search text"}}},
            })
        })
        .collect()
}

/// A markdown document with headings, lists, code, and long paragraphs.
pub fn markdown_document(sections: usize) -> String {
    (0..sections)
        .map(|index| {
            format!(
                "## Section {index}\n\n{}\n\n- item **bold** `code`\n- item [link](https://example.com/{index})\n\n```rust\nfn f{index}() -> u32 {{ {index} }}\n```\n\n> quoted text that wraps across lines\n",
                "A paragraph with *emphasis* and enough words to wrap several times on a narrow terminal. ".repeat(4)
            )
        })
        .collect()
}

/// A source file of `lines` lines for the edit tool.
pub fn source_file(lines: usize) -> String {
    (0..lines)
        .map(|index| format!("    let value_{index} = compute({index}); // “note” {index}\n"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_are_well_formed() {
        let session = session(3);
        assert_eq!(session_entries(&session).len(), 6);
        let stream = completions_sse(3);
        let mut frames = mu_core::sse::SseFrames::default();
        frames.push(&stream);
        let payloads = std::iter::from_fn(|| frames.next_data()).collect::<Vec<_>>();
        assert_eq!(payloads.len(), 4);
        assert_eq!(payloads[3], "[DONE]");
        assert!(mu_core::bench_internals::decode_anthropic_events(&anthropic_events(2)) > 0);
        assert_eq!(tool_definitions(5).len(), 5);
        assert!(markdown_document(1).contains("```rust"));
        assert_eq!(source_file(4).lines().count(), 4);
    }
}
