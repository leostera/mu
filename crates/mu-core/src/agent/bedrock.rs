//! Amazon Bedrock Converse adapter (`api: "bedrock-converse-stream"` in `models.json`).
//!
//! Uses the non-streaming `converse` endpoint with a Bedrock API key (bearer token). SigV4
//! request signing and the binary event stream (`converse-stream`) are not implemented yet.

use serde_json::{Value, json};

use super::{FunctionCall, Reply};

fn text_of(content: &Value) -> String {
    content
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Converse `messages`; consecutive same-role blocks merge (Converse requires alternation).
pub(super) fn messages(history: &[Value]) -> Vec<Value> {
    let mut out: Vec<(String, Vec<Value>)> = Vec::new();
    let mut push = |role: &str, block: Value| match out.last_mut() {
        Some((last, blocks)) if last == role => blocks.push(block),
        _ => out.push((role.to_owned(), vec![block])),
    };
    for item in history {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => push(
                "assistant",
                json!({"toolUse":{
                    "toolUseId": item["call_id"],
                    "name": item["name"],
                    "input": item["arguments"].as_str()
                        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                        .unwrap_or_else(|| json!({})),
                }}),
            ),
            Some("function_call_output") => push(
                "user",
                json!({"toolResult":{
                    "toolUseId": item["call_id"],
                    "content": [{"text": super::output_text(item)}],
                }}),
            ),
            _ => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let text = text_of(&item["content"]);
                if !text.trim().is_empty() {
                    push(role, json!({"text": text}));
                }
            }
        }
    }
    out.into_iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect()
}

pub(super) fn tool_config(definitions: &[Value]) -> Option<Value> {
    let tools = definitions
        .iter()
        .filter_map(|tool| {
            Some(json!({"toolSpec":{
                "name": tool.get("name")?,
                "description": tool.get("description").cloned().unwrap_or(json!("")),
                "inputSchema": {"json": tool.get("parameters").cloned().unwrap_or(json!({"type":"object"}))},
            }}))
        })
        .collect::<Vec<_>>();
    (!tools.is_empty()).then(|| json!({"tools": tools}))
}

fn is_claude(model: &str) -> bool {
    model.to_lowercase().contains("claude")
}

/// Claude models that use adaptive thinking instead of a token budget (Pi's
/// `supportsAdaptiveThinking`); Mu does not configure these yet.
fn uses_adaptive_thinking(model: &str) -> bool {
    let id = model.to_lowercase();
    [
        "opus-4-6",
        "opus-4-7",
        "opus-4-8",
        "opus-5",
        "sonnet-4-6",
        "sonnet-5",
        "haiku-5",
        "fable-5",
    ]
    .iter()
    .any(|needle| id.contains(needle))
}

/// `additionalModelRequestFields` for budget-based Claude extended thinking (Pi's
/// `buildAdditionalModelRequestFields`, non-adaptive branch) and its token budget.
pub(super) fn thinking_fields(model: &str, effort: Option<&str>) -> Option<(Value, u64)> {
    let effort = effort?;
    if !is_claude(model) {
        return None;
    }
    if uses_adaptive_thinking(model) {
        // Pi's `mapThinkingLevelToEffort` without a model `thinkingLevelMap`. Block binding (a
        // beta that drops stale signed thinking blocks) is not requested.
        let effort = match effort {
            "minimal" | "low" => "low",
            "medium" => "medium",
            _ => "high",
        };
        return Some((
            json!({
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": effort},
            }),
            0,
        ));
    }
    let budget = match effort {
        "minimal" => 1024,
        "low" => 2048,
        "medium" => 8192,
        _ => 16384,
    };
    Some((
        json!({
            "thinking": {"type": "enabled", "budget_tokens": budget, "display": "summarized"},
            "anthropic_beta": ["interleaved-thinking-2025-05-14"],
        }),
        budget,
    ))
}

pub(super) fn request_body(system: &str, history: &[Value], definitions: &[Value]) -> Value {
    let mut body = json!({
        "system": [{"text": system}],
        "messages": messages(history),
        "inferenceConfig": {"maxTokens": 16_384},
    });
    if let Some(config) = tool_config(definitions) {
        body["toolConfig"] = config;
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_thinking_uses_budgets_and_skips_adaptive_or_non_claude_models() {
        let (fields, budget) =
            thinking_fields("anthropic.claude-sonnet-4-5-v1:0", Some("medium")).unwrap();
        assert_eq!(budget, 8192);
        assert_eq!(
            fields["thinking"],
            json!({"type":"enabled","budget_tokens":8192,"display":"summarized"})
        );
        assert_eq!(
            fields["anthropic_beta"],
            json!(["interleaved-thinking-2025-05-14"])
        );
        assert_eq!(
            thinking_fields("anthropic.claude-3", Some("max"))
                .unwrap()
                .1,
            16384
        );
        let (adaptive, extra) =
            thinking_fields("anthropic.claude-opus-4-6", Some("minimal")).unwrap();
        assert_eq!(
            adaptive["thinking"],
            json!({"type":"adaptive","display":"summarized"})
        );
        assert_eq!(adaptive["output_config"]["effort"], "low");
        assert_eq!(
            extra, 0,
            "adaptive thinking needs no token-limit adjustment"
        );
        assert_eq!(
            thinking_fields("anthropic.claude-sonnet-5", Some("max"))
                .unwrap()
                .0["output_config"]["effort"],
            "high"
        );
        assert_eq!(thinking_fields("amazon.nova-pro", Some("high")), None);
        assert_eq!(thinking_fields("anthropic.claude-3", None), None);
    }

    #[test]
    fn history_becomes_alternating_converse_messages_with_tool_blocks() {
        let history = vec![
            json!({"role":"user","content":[{"type":"input_text","text":"hi"}]}),
            json!({"type":"function_call","call_id":"c1","name":"read","arguments":"{\"path\":\"a\"}"}),
            json!({"type":"function_call_output","call_id":"c1","output":"file"}),
        ];
        let out = messages(&history);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], json!({"role":"user","content":[{"text":"hi"}]}));
        assert_eq!(
            out[1]["content"][0]["toolUse"]["input"],
            json!({"path":"a"})
        );
        assert_eq!(out[2]["role"], "user");
        assert_eq!(
            out[2]["content"][0]["toolResult"]["content"][0]["text"],
            "file"
        );
    }

    #[test]
    fn tool_config_wraps_schemas_and_is_absent_without_tools() {
        assert_eq!(tool_config(&[]), None);
        let config =
            tool_config(&[json!({"name":"ls","description":"d","parameters":{"type":"object"}})])
                .unwrap();
        assert_eq!(
            config["tools"][0]["toolSpec"]["inputSchema"]["json"],
            json!({"type":"object"})
        );
    }
}

/// One decoded AWS event-stream message (`application/vnd.amazon.eventstream`).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct StreamMessage {
    /// `:event-type` (or `:exception-type`) header.
    pub kind: String,
    pub is_exception: bool,
    pub payload: Vec<u8>,
}

/// Incremental decoder for the binary framing Bedrock's `converse-stream` uses: a 12-byte prelude
/// (total length, headers length, CRC32), headers, payload, and a trailing message CRC32.
#[derive(Default)]
pub(super) struct EventStreamDecoder {
    buffer: Vec<u8>,
}

impl EventStreamDecoder {
    pub(super) fn push(&mut self, bytes: &[u8]) -> Result<Vec<StreamMessage>, String> {
        self.buffer.extend_from_slice(bytes);
        let mut messages = Vec::new();
        loop {
            if self.buffer.len() < 12 {
                return Ok(messages);
            }
            let be = |at: usize| {
                u32::from_be_bytes(self.buffer[at..at + 4].try_into().expect("4 bytes"))
            };
            let (total, headers_len) = (be(0) as usize, be(4) as usize);
            if crc32fast::hash(&self.buffer[..8]) != be(8) {
                return Err("event stream prelude checksum mismatch".into());
            }
            if total < 16 + headers_len {
                return Err("event stream message length is inconsistent".into());
            }
            if self.buffer.len() < total {
                return Ok(messages);
            }
            if crc32fast::hash(&self.buffer[..total - 4]) != be(total - 4) {
                return Err("event stream message checksum mismatch".into());
            }
            let headers = parse_headers(&self.buffer[12..12 + headers_len])?;
            let payload = self.buffer[12 + headers_len..total - 4].to_vec();
            self.buffer.drain(..total);
            let header = |name: &str| {
                headers
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, v)| v.clone())
            };
            let message_type = header(":message-type").unwrap_or_default();
            let is_exception = message_type == "exception";
            let kind = header(if is_exception {
                ":exception-type"
            } else {
                ":event-type"
            })
            .unwrap_or_default();
            messages.push(StreamMessage {
                kind,
                is_exception,
                payload,
            });
        }
    }
}

/// String-typed headers only; other value types are skipped by their fixed or prefixed size.
fn parse_headers(mut bytes: &[u8]) -> Result<Vec<(String, String)>, String> {
    let bad = || "event stream header is truncated".to_owned();
    let mut headers = Vec::new();
    while !bytes.is_empty() {
        let name_len = *bytes.first().ok_or_else(bad)? as usize;
        let name = bytes.get(1..1 + name_len).ok_or_else(bad)?;
        let name = String::from_utf8_lossy(name).into_owned();
        let kind = *bytes.get(1 + name_len).ok_or_else(bad)?;
        let rest = &bytes[2 + name_len..];
        let (value, used) = match kind {
            0 | 1 => (String::new(), 0),
            2 => (String::new(), 1),
            3 => (String::new(), 2),
            4 => (String::new(), 4),
            5 | 8 => (String::new(), 8),
            9 => (String::new(), 16),
            6 | 7 => {
                let len =
                    u16::from_be_bytes(rest.get(..2).ok_or_else(bad)?.try_into().expect("2 bytes"))
                        as usize;
                let value = rest.get(2..2 + len).ok_or_else(bad)?;
                (String::from_utf8_lossy(value).into_owned(), 2 + len)
            }
            other => return Err(format!("unknown event stream header type {other}")),
        };
        bytes = rest.get(used..).ok_or_else(bad)?;
        headers.push((name, value));
    }
    Ok(headers)
}

/// Accumulates `converse-stream` events into a [`Reply`], forwarding text and reasoning deltas.
#[derive(Default)]
pub(super) struct StreamAccumulator {
    text: String,
    thinking: String,
    calls: Vec<(String, String, String)>,
    stop_reason: Option<String>,
    usage: Value,
}

impl StreamAccumulator {
    pub(super) fn feed(
        &mut self,
        message: &StreamMessage,
        on_event: &mut impl FnMut(super::AgentEvent),
    ) -> Result<(), String> {
        let payload: Value = serde_json::from_slice(&message.payload).unwrap_or(Value::Null);
        if message.is_exception {
            let detail = payload["message"].as_str().unwrap_or("no details");
            return Err(format!("{}: {detail}", message.kind));
        }
        match message.kind.as_str() {
            "contentBlockStart" => {
                if let Some(tool) = payload["start"].get("toolUse") {
                    self.calls.push((
                        tool["toolUseId"].as_str().unwrap_or_default().to_owned(),
                        tool["name"].as_str().unwrap_or_default().to_owned(),
                        String::new(),
                    ));
                }
            }
            "contentBlockDelta" => {
                let delta = &payload["delta"];
                if let Some(text) = delta["text"].as_str() {
                    self.text.push_str(text);
                    on_event(super::AgentEvent::TextDelta(text.to_owned()));
                } else if let Some(text) = delta["reasoningContent"]["text"].as_str() {
                    self.thinking.push_str(text);
                    on_event(super::AgentEvent::ThinkingDelta(text.to_owned()));
                } else if let Some(input) = delta["toolUse"]["input"].as_str()
                    && let Some(call) = self.calls.last_mut()
                {
                    call.2.push_str(input);
                }
            }
            "messageStop" => self.stop_reason = payload["stopReason"].as_str().map(str::to_owned),
            "metadata" => self.usage = payload["usage"].clone(),
            _ => {}
        }
        Ok(())
    }

    pub(super) fn into_reply(self) -> Reply {
        let count = |key: &str| self.usage[key].as_u64().unwrap_or(0);
        let (input, output) = (count("inputTokens"), count("outputTokens"));
        let (cache_read, cache_write) = (
            count("cacheReadInputTokens"),
            count("cacheWriteInputTokens"),
        );
        Reply {
            text: self.text,
            thinking: self.thinking,
            thinking_signature: None,
            output_items: Vec::new(),
            tool_calls: self
                .calls
                .into_iter()
                .map(|(call_id, name, raw)| FunctionCall {
                    call_id,
                    name,
                    arguments: if raw.is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&raw).unwrap_or(Value::String(raw))
                    },
                })
                .collect(),
            input,
            output,
            cache_read,
            cache_write,
            reasoning: None,
            total_tokens: self.usage["totalTokens"]
                .as_u64()
                .unwrap_or(input + output + cache_read + cache_write),
            incomplete_reason: (self.stop_reason.as_deref() == Some("max_tokens"))
                .then(|| "max_output_tokens".to_owned()),
        }
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    /// Build one event-stream message the way AWS frames it.
    fn frame(kind: &str, message_type: &str, payload: &Value) -> Vec<u8> {
        let mut headers = Vec::new();
        let kind_header = if message_type == "exception" {
            ":exception-type"
        } else {
            ":event-type"
        };
        for (name, value) in [
            (kind_header, kind),
            (":message-type", message_type),
            (":content-type", "application/json"),
        ] {
            headers.push(name.len() as u8);
            headers.extend(name.as_bytes());
            headers.push(7);
            headers.extend((value.len() as u16).to_be_bytes());
            headers.extend(value.as_bytes());
        }
        let payload = serde_json::to_vec(payload).unwrap();
        let total = 12 + headers.len() + payload.len() + 4;
        let mut out = Vec::new();
        out.extend((total as u32).to_be_bytes());
        out.extend((headers.len() as u32).to_be_bytes());
        out.extend(crc32fast::hash(&out).to_be_bytes());
        out.extend(&headers);
        out.extend(&payload);
        out.extend(crc32fast::hash(&out).to_be_bytes());
        out
    }

    #[test]
    fn decoder_reassembles_messages_split_across_chunks() {
        let mut bytes = frame(
            "contentBlockDelta",
            "event",
            &json!({"delta":{"text":"hi"}}),
        );
        bytes.extend(frame(
            "messageStop",
            "event",
            &json!({"stopReason":"end_turn"}),
        ));
        let mut decoder = EventStreamDecoder::default();
        let mut messages = Vec::new();
        for chunk in bytes.chunks(7) {
            messages.extend(decoder.push(chunk).unwrap());
        }
        assert_eq!(
            messages.iter().map(|m| m.kind.as_str()).collect::<Vec<_>>(),
            ["contentBlockDelta", "messageStop"]
        );
        assert!(!messages[0].is_exception);
    }

    #[test]
    fn corrupted_frames_are_rejected() {
        let mut bytes = frame("messageStop", "event", &json!({}));
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        assert!(
            EventStreamDecoder::default()
                .push(&bytes)
                .unwrap_err()
                .contains("message checksum")
        );
        let mut bytes = frame("messageStop", "event", &json!({}));
        bytes[9] ^= 0xff;
        assert!(
            EventStreamDecoder::default()
                .push(&bytes)
                .unwrap_err()
                .contains("prelude checksum")
        );
    }

    #[test]
    fn stream_events_accumulate_text_reasoning_tool_calls_and_usage() {
        let events = [
            frame(
                "contentBlockDelta",
                "event",
                &json!({"delta":{"reasoningContent":{"text":"hmm"}}}),
            ),
            frame(
                "contentBlockDelta",
                "event",
                &json!({"delta":{"text":"he"}}),
            ),
            frame(
                "contentBlockDelta",
                "event",
                &json!({"delta":{"text":"llo"}}),
            ),
            frame(
                "contentBlockStart",
                "event",
                &json!({"start":{"toolUse":{"toolUseId":"t1","name":"bash"}}}),
            ),
            frame(
                "contentBlockDelta",
                "event",
                &json!({"delta":{"toolUse":{"input":"{\"command\":"}}}),
            ),
            frame(
                "contentBlockDelta",
                "event",
                &json!({"delta":{"toolUse":{"input":"\"ls\"}"}}}),
            ),
            frame("messageStop", "event", &json!({"stopReason":"tool_use"})),
            frame(
                "metadata",
                "event",
                &json!({"usage":{"inputTokens":4,"outputTokens":2,"totalTokens":6}}),
            ),
        ]
        .concat();
        let mut decoder = EventStreamDecoder::default();
        let mut accumulator = StreamAccumulator::default();
        let mut deltas = Vec::new();
        for message in decoder.push(&events).unwrap() {
            accumulator
                .feed(&message, &mut |event| deltas.push(format!("{event:?}")))
                .unwrap();
        }
        let reply = accumulator.into_reply();
        assert_eq!(
            (reply.text.as_str(), reply.thinking.as_str()),
            ("hello", "hmm")
        );
        assert_eq!(reply.tool_calls[0].arguments, json!({"command":"ls"}));
        assert_eq!((reply.input, reply.output, reply.total_tokens), (4, 2, 6));
        assert_eq!(
            deltas.len(),
            3,
            "reasoning and two text deltas stream out: {deltas:?}"
        );
    }

    #[test]
    fn max_tokens_stop_is_reported_as_truncation() {
        let mut accumulator = StreamAccumulator::default();
        let stop = StreamMessage {
            kind: "messageStop".into(),
            is_exception: false,
            payload: br#"{"stopReason":"max_tokens"}"#.to_vec(),
        };
        accumulator.feed(&stop, &mut |_| {}).unwrap();
        assert_eq!(
            accumulator.into_reply().incomplete_reason.as_deref(),
            Some("max_output_tokens")
        );
    }

    #[test]
    fn exception_messages_become_errors() {
        let frame = frame(
            "throttlingException",
            "exception",
            &json!({"message":"slow down"}),
        );
        let message = EventStreamDecoder::default()
            .push(&frame)
            .unwrap()
            .remove(0);
        let error = StreamAccumulator::default()
            .feed(&message, &mut |_| {})
            .unwrap_err();
        assert_eq!(error, "throttlingException: slow down");
    }
}
